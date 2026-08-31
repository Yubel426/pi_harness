import { createInterface } from 'node:readline/promises';
import { Writable } from 'node:stream';
import { once } from 'node:events';
import { builtinModels } from '@earendil-works/pi-ai/providers/all';
import { FileCredentialStore, FileCatalogStore } from './credentials.mjs';

const controller = new AbortController();
process.on('SIGTERM', () => controller.abort());
process.on('SIGINT', () => controller.abort());
const credentials = new FileCredentialStore(process.argv[2]);
const models = builtinModels({ credentials, modelsStore: new FileCatalogStore(`${process.argv[2]}.models.json`) });

async function send(value) {
  if (!process.stdout.write(`${JSON.stringify(value)}\n`)) await once(process.stdout, 'drain');
}

function provider(id) {
  const value = models.getProvider(id);
  if (!value) throw new Error(`Unknown provider: ${id}`);
  return value;
}

function methods(value) {
  return [value.auth.oauth && 'oauth', value.auth.apiKey?.login && 'api_key'].filter(Boolean);
}

function resolveModel(request) {
  const specification = request.model;
  if (typeof specification === 'object' && specification !== null) {
    provider(specification.provider);
    for (const key of ['id', 'api', 'baseUrl', 'input', 'contextWindow', 'maxTokens']) {
      if (specification[key] === undefined) throw new Error(`Custom model requires ${key}`);
    }
    return specification;
  }
  provider(request.provider);
  const value = models.getModel(request.provider, specification);
  if (!value) throw new Error(`Unknown model ${request.provider}/${specification}; use 'pi-harness models ${request.provider}' or supply a full Pi model definition`);
  return request.baseUrl ? { ...value, baseUrl: request.baseUrl } : value;
}

async function dispatch(request) {
  const options = { ...request.options, signal: controller.signal };
  if (['models', 'model', 'stream', 'stream_simple', 'complete', 'complete_simple'].includes(request.op)) {
    const restored = await models.refresh({ allowNetwork: false, signal: controller.signal });
    if (restored.errors.size) throw new Error(`Could not restore model catalogs for: ${[...restored.errors.keys()].join(', ')}`);
  }
  if (request.context) {
    const selected = resolveModel(request);
    const stored = await credentials.read(selected.provider);
    if (stored?.type === 'oauth' && !request.options?.apiKey &&
        (request.baseUrl || (typeof request.model === 'object' &&
          selected.baseUrl !== models.getModel(selected.provider, selected.id)?.baseUrl))) {
      throw new Error('Custom endpoints cannot use stored OAuth credentials; use a separate API-key provider');
    }
  }
  switch (request.op) {
    case 'providers':
      return models.getProviders().map(p => ({ id: p.id, name: p.name, authMethods: methods(p) }));
    case 'models':
      if (request.provider) provider(request.provider);
      if (request.refresh) {
        const refreshed = await models.refresh({ providers: request.provider ? [request.provider] : undefined, signal: controller.signal });
        if (refreshed.errors.size) throw new Error(`Model refresh failed for: ${[...refreshed.errors.keys()].join(', ')}`);
      }
      return request.available
        ? models.getAvailable(request.provider, { signal: controller.signal })
        : models.getModels(request.provider);
    case 'model': return resolveModel(request);
    case 'status': {
      const selected = request.provider ? [provider(request.provider)] : models.getProviders();
      const stored = new Map((await credentials.list()).map(c => [c.providerId, c.type]));
      return Promise.all(selected.map(async p => {
        const auth = await models.checkAuth(p.id, { signal: controller.signal });
        return { provider: p.id, stored: stored.get(p.id) ?? null, configured: !!auth, source: auth?.source ?? null };
      }));
    }
    case 'logout':
      provider(request.provider);
      await models.logout(request.provider, { signal: controller.signal });
      return { provider: request.provider, loggedOut: true };
    case 'api_key':
      if (!provider(request.provider).auth.apiKey) throw new Error('Provider does not accept API keys');
      if (typeof request.key !== 'string' || !request.key.trim()) throw new Error('API key cannot be empty');
      await credentials.modify(request.provider, async () => ({ type: 'api_key', key: request.key, ...(request.env ? { env: request.env } : {}) }), { signal: controller.signal });
      return { provider: request.provider, saved: true };
    case 'complete': case 'complete_simple': {
      const model = resolveModel(request);
      return request.op === 'complete'
        ? models.complete(model, request.context, options)
        : models.completeSimple(model, request.context, options);
    }
    case 'stream': case 'stream_simple': {
      const model = resolveModel(request);
      const stream = request.op === 'stream'
        ? models.stream(model, request.context, options)
        : models.streamSimple(model, request.context, options);
      for await (const event of stream) await send({ event });
      return null;
    }
    default: throw new Error(`Unknown bridge operation: ${request.op}`);
  }
}

// Login owns the terminal, so browser callbacks can cancel a pending manual-code
// prompt. Login credentials are persisted locally, never returned over JSONL.
async function login(id, method) {
  let muted = false;
  const output = new Writable({ write(chunk, _encoding, done) {
    if (!muted) process.stdout.write(chunk);
    done();
  } });
  output.columns = process.stdout.columns;
  const rl = createInterface({ input: process.stdin, output, terminal: !!process.stdin.isTTY });
  rl.on('SIGINT', () => controller.abort());
  const prompt = async p => {
    controller.signal.throwIfAborted();
    const signal = p.signal ? AbortSignal.any([p.signal, controller.signal]) : controller.signal;
    if (p.type === 'select') {
      process.stdout.write(`${p.message}\n`);
      p.options.forEach((o, i) => process.stdout.write(`  ${i + 1}. ${o.label}\n`));
      const answer = (await rl.question('Select number: ', { signal })).trim();
      const option = p.options[Number(answer) - 1] ?? p.options.find(o => o.id === answer);
      if (!option) throw new Error('Invalid selection');
      return option.id;
    }
    // Piped secret entry is supported; terminal entry is never echoed.
    process.stdout.write(`${p.message}${p.placeholder ? ` (${p.placeholder})` : ''}: `);
    muted = p.type === 'secret' || p.type === 'manual_code';
    try { return await rl.question('', { signal }); }
    finally { if (muted) process.stdout.write('\n'); muted = false; }
  };
  try {
    if (!id) id = await prompt({ type: 'select', message: 'Provider', options: models.getProviders()
      .filter(p => methods(p).length).map(p => ({ id: p.id, label: `${p.name} (${p.id})` })) });
    const p = provider(id);
    const available = methods(p);
    if (!method) method = available.length === 1 ? available[0] : await prompt({
      type: 'select', message: 'Authentication method', options: available.map(id => ({ id, label: id })),
    });
    if (!available.includes(method)) throw new Error(`${id} does not support ${method} login`);
    await models.login(id, method, {
      signal: controller.signal, prompt,
      notify(event) {
        if (event.type === 'auth_url') process.stdout.write(`Open in your browser:\n${event.url}\n${event.instructions ?? ''}\n`);
        else if (event.type === 'device_code') process.stdout.write(`Open ${event.verificationUri}\nEnter code: ${event.userCode}\n`);
        else {
          process.stdout.write(`${event.message}\n`);
          for (const link of event.links ?? []) process.stdout.write(`${link.label ?? 'More information'}: ${link.url}\n`);
        }
      },
    });
    process.stdout.write(`Logged in to ${id}. Credentials saved to ${credentials.path}\n`);
    const refreshed = await models.refresh({ providers: [id], signal: controller.signal });
    if (refreshed.errors.size) process.stdout.write('Login succeeded, but the model catalog could not be refreshed. Retry with models --refresh.\n');
  } finally { rl.close(); }
}

try {
  if (process.argv[3] === 'login') {
    await login(process.argv[4], process.argv[5]);
  } else {
    const rl = createInterface({ input: process.stdin, terminal: false });
    let request;
    for await (const line of rl) { request = JSON.parse(line); break; }
    if (!request) throw new Error('Missing bridge request');
    await send({ result: await dispatch(request) });
  }
} catch (error) {
  if (process.argv[3] === 'login') {
    // Upstream OAuth errors may contain token responses. Never echo them.
    console.error('Login failed or was cancelled. Check the provider setup and try again.');
  } else await send({ error: ['auth', 'oauth'].includes(error?.code)
    ? 'Provider authentication failed. Check auth-status or log in again.'
    : error instanceof Error ? error.message : 'Pi bridge failed' });
  process.exitCode = 1;
}
