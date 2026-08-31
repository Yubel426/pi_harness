import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, stat, symlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';
import { once } from 'node:events';

const runtime = process.env.PI_HARNESS_RUNTIME_DIR;
if (!runtime) throw new Error('Set PI_HARNESS_RUNTIME_DIR to an installed test runtime');
const { FileCredentialStore, FileCatalogStore } = await import(pathToFileURL(resolve(runtime, 'credentials.mjs')));
const { createModels, createProvider } = await import(pathToFileURL(resolve(runtime, 'node_modules/@earendil-works/pi-ai/dist/index.js')));

async function temporary(t) {
  const directory = await mkdtemp(join(tmpdir(), 'pi-auth-test-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return join(directory, 'auth.json');
}

test('atomic private storage, metadata-only listing and provider-scoped logout', async t => {
  const file = await temporary(t);
  const store = new FileCredentialStore(file);
  await store.modify('openai', async () => ({ type: 'api_key', key: 'test-secret' }));
  await store.modify('anthropic', async () => ({ type: 'api_key', key: 'other-secret' }));
  assert.equal((await stat(file)).mode & 0o777, 0o600);
  assert.deepEqual(await store.list(), [{ providerId: 'openai', type: 'api_key' }, { providerId: 'anthropic', type: 'api_key' }]);
  await store.delete('openai');
  assert.equal(await store.read('openai'), undefined);
  assert.equal((await store.read('anthropic')).key, 'other-secret');
});

test('corrupt and symlink credentials fail closed without overwriting', async t => {
  const file = await temporary(t);
  await writeFile(file, '{broken');
  const store = new FileCredentialStore(file);
  await assert.rejects(store.modify('openai', async () => ({ type: 'api_key', key: 'new' })), /Invalid credential JSON/);
  assert.equal(await readFile(file, 'utf8'), '{broken');
  await symlink(file, `${file}.link`);
  await assert.rejects(new FileCredentialStore(`${file}.link`).read('openai'), /symlink/);
});

test('cross-process writes do not lose other provider credentials', async t => {
  const file = await temporary(t);
  const moduleUrl = pathToFileURL(resolve(runtime, 'credentials.mjs')).href;
  const jobs = Array.from({ length: 6 }, (_, i) => {
    const code = `import { FileCredentialStore } from ${JSON.stringify(moduleUrl)}; await new FileCredentialStore(${JSON.stringify(file)}).modify('p${i}', async () => ({ type: 'api_key', key: 'test' }));`;
    const process = spawn('node', ['--input-type=module', '-e', code], { stdio: 'pipe' });
    return once(process, 'exit').then(([code]) => assert.equal(code, 0));
  });
  await Promise.all(jobs);
  assert.equal((await new FileCredentialStore(file).list()).length, 6);
});

test('real Pi auth refresh is serialized and rotated token is persisted', async t => {
  const file = await temporary(t);
  const store = new FileCredentialStore(file);
  await store.modify('test', async () => ({ type: 'oauth', access: 'old', refresh: 'refresh-old', expires: 0 }));
  let refreshes = 0;
  const models = createModels({ credentials: store });
  models.setProvider(createProvider({ id: 'test', name: 'Test', models: [],
    api: { stream() {}, streamSimple() {} },
    auth: { oauth: {
      name: 'Test OAuth', async login() {},
      async refresh(current) {
        refreshes++;
        assert.equal(current.refresh, 'refresh-old');
        await new Promise(resolve => setTimeout(resolve, 20));
        return { type: 'oauth', access: 'new', refresh: 'refresh-new', expires: Date.now() + 3600000 };
      },
      async toAuth(current) { return { apiKey: current.access }; },
    } },
  }));
  const results = await Promise.all(Array.from({ length: 6 }, () => models.getAuth('test')));
  assert.equal(refreshes, 1);
  assert.ok(results.every(result => result.auth.apiKey === 'new'));
  assert.equal((await store.read('test')).refresh, 'refresh-new');
});

test('failed refresh preserves the previous credential', async t => {
  const file = await temporary(t);
  const store = new FileCredentialStore(file);
  const old = { type: 'oauth', access: 'old', refresh: 'keep', expires: 0 };
  await store.modify('test', async () => old);
  await assert.rejects(store.modify('test', async () => { throw new Error('refresh failed'); }));
  assert.deepEqual(await store.read('test'), old);
});

test('dynamic model catalog survives a fresh store instance', async t => {
  const file = `${await temporary(t)}.models.json`;
  const entry = { models: [{ provider: 'radius', id: 'test' }], checkedAt: 123 };
  await new FileCatalogStore(file).write('radius', entry);
  assert.deepEqual(await new FileCatalogStore(file).read('radius'), entry);
});
