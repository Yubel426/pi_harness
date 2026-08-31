import { socksDispatcher } from 'fetch-socks';
import { getProxyForUrl } from 'proxy-from-env';
import { ProxyAgent } from 'undici';

const dispatchers = new Map();

function applyPiOverrides() {
  const proxy = process.env.PI_PROXY?.trim();
  if (proxy) {
    for (const name of [
      'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
      'http_proxy', 'https_proxy', 'all_proxy',
    ]) process.env[name] = proxy;
  }
  if (process.env.PI_NO_PROXY !== undefined) {
    process.env.NO_PROXY = process.env.PI_NO_PROXY;
    process.env.no_proxy = process.env.PI_NO_PROXY;
  }
}

function requestUrl(input) {
  if (typeof input === 'string' || input instanceof URL) return input.toString();
  if (input && typeof input.url === 'string') return input.url;
  throw new TypeError('Proxy-aware fetch requires a URL or Request input');
}

function decode(value) {
  try { return decodeURIComponent(value); }
  catch { throw new Error('Proxy URL contains invalid percent-encoded credentials'); }
}

function socksOptions(url) {
  const types = new Map([
    ['socks:', 5], ['socks5:', 5], ['socks5h:', 5],
    ['socks4:', 4], ['socks4a:', 4],
  ]);
  const type = types.get(url.protocol);
  if (!type) return null;
  const port = url.port ? Number(url.port) : 1080;
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('Proxy URL contains an invalid port');
  }
  const host = url.hostname.replace(/^\[|\]$/g, '');
  if (!host) throw new Error('Proxy URL is missing a host');
  return {
    type,
    host,
    port,
    ...(url.username ? { userId: decode(url.username) } : {}),
    ...(url.password ? { password: decode(url.password) } : {}),
  };
}

function dispatcherFor(proxy) {
  if (dispatchers.has(proxy)) return dispatchers.get(proxy);
  let url;
  try { url = new URL(proxy); }
  catch { throw new Error('Invalid proxy URL; include a scheme such as http:// or socks5h://'); }
  const socks = socksOptions(url);
  let dispatcher;
  if (socks) dispatcher = socksDispatcher(socks);
  else if (url.protocol === 'http:' || url.protocol === 'https:') dispatcher = new ProxyAgent(url);
  else throw new Error(`Unsupported proxy protocol: ${url.protocol.replace(/:$/, '')}`);
  dispatchers.set(proxy, dispatcher);
  return dispatcher;
}

/** Install per-request proxy routing while preserving the standard fetch API. */
export function installProxySupport() {
  applyPiOverrides();
  const nativeFetch = globalThis.fetch;
  if (typeof nativeFetch !== 'function') throw new Error('Node.js fetch is unavailable');
  globalThis.fetch = (input, init) => {
    if (init?.dispatcher) return nativeFetch(input, init);
    const proxy = getProxyForUrl(requestUrl(input));
    return proxy
      ? nativeFetch(input, { ...init, dispatcher: dispatcherFor(proxy) })
      : nativeFetch(input, init);
  };
}
