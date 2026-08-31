import { mkdir, open, readFile, rename, unlink, lstat } from 'node:fs/promises';
import { dirname } from 'node:path';
import { randomUUID } from 'node:crypto';
import lockfile from 'proper-lockfile';

function validate(data) {
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error('Invalid credential file: expected an object');
  }
  for (const value of Object.values(data)) {
    if (!value || !['api_key', 'oauth'].includes(value.type)) {
      throw new Error('Invalid credential file: unsupported credential type');
    }
    if (value.type === 'oauth' && (
      typeof value.access !== 'string' || typeof value.refresh !== 'string' ||
      !Number.isFinite(value.expires)
    )) throw new Error('Invalid credential file: incomplete OAuth credential');
    if (value.type === 'api_key' && value.key !== undefined && typeof value.key !== 'string') {
      throw new Error('Invalid credential file: invalid API key');
    }
  }
  return data;
}

// Pi refreshes rotating OAuth tokens inside modify(). The lock must cover the
// network exchange, not just the final write, and work across Python processes.
export class FileCredentialStore {
  constructor(path, validator = validate) { this.path = path; this.validate = validator; }

  async load() {
    try {
      if ((await lstat(this.path)).isSymbolicLink()) throw new Error('Credential file must not be a symlink');
      return this.validate(JSON.parse(await readFile(this.path, 'utf8')));
    } catch (error) {
      if (error.code === 'ENOENT') return {};
      if (error instanceof SyntaxError) throw new Error('Invalid credential JSON; existing file was not changed');
      throw error;
    }
  }

  async read(id, options) {
    options?.signal?.throwIfAborted();
    const data = await this.load();
    return Object.hasOwn(data, id) ? data[id] : undefined;
  }

  async list(options) {
    options?.signal?.throwIfAborted();
    return Object.entries(await this.load()).map(([providerId, credential]) => ({ providerId, type: credential.type }));
  }

  async locked(fn, options) {
    options?.signal?.throwIfAborted();
    await mkdir(dirname(this.path), { recursive: true, mode: 0o700 });
    const release = await lockfile.lock(this.path, {
      realpath: false, stale: 120_000, update: 10_000,
      retries: { retries: 120, minTimeout: 50, maxTimeout: 1000 },
    });
    try {
      options?.signal?.throwIfAborted();
      return await fn(await this.load());
    } finally { await release(); }
  }

  async save(data) {
    this.validate(data);
    const temporary = `${this.path}.${randomUUID()}.tmp`;
    try {
      const file = await open(temporary, 'wx', 0o600);
      try { await file.writeFile(`${JSON.stringify(data, null, 2)}\n`); await file.sync(); }
      finally { await file.close(); }
      await rename(temporary, this.path);
    } finally { await unlink(temporary).catch(error => { if (error.code !== 'ENOENT') throw error; }); }
  }

  async modify(id, fn, options) {
    return this.locked(async data => {
      const current = Object.hasOwn(data, id) ? data[id] : undefined;
      const next = await fn(current);
      options?.signal?.throwIfAborted();
      if (next !== undefined) {
        Object.defineProperty(data, id, { value: next, enumerable: true, configurable: true, writable: true });
        await this.save(data);
      }
      return next ?? current;
    }, options);
  }

  async delete(id, options) {
    return this.locked(async data => { delete data[id]; await this.save(data); }, options);
  }
}

export class FileCatalogStore extends FileCredentialStore {
  constructor(path) {
    super(path, data => {
      if (!data || typeof data !== 'object' || Array.isArray(data) ||
          Object.values(data).some(entry => !entry || !Array.isArray(entry.models))) {
        throw new Error('Invalid model catalog cache');
      }
      return data;
    });
  }
  async write(id, entry, options) { await this.modify(id, async () => entry, options); }
}
