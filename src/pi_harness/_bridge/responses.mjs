const terminalTypes = new Set([
  'response.completed', 'response.incomplete', 'response.failed',
]);

function isTerminal(record) {
  const data = record.split('\n')
    .filter(line => line.startsWith('data:'))
    .map(line => line.slice(5).replace(/^ /, ''))
    .join('\n');
  try { return terminalTypes.has(JSON.parse(data).type); }
  catch { return false; } // Let Pi report malformed or provider error events.
}

/**
 * End a Responses SSE body at its terminal event, even if a relay keeps it open.
 * See https://github.com/earendil-works/pi/issues/6808.
 */
export async function fetchResponses(input, init) {
  const response = await globalThis.fetch(input, init);
  if (!response.ok || !response.body ||
      !response.headers.get('content-type')?.toLowerCase().includes('text/event-stream')) {
    return response;
  }

  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let pending = '';
  let skipLF = false;
  const forward = (text, controller) => {
    // A split UTF-8 character can decode to nothing; keep the CRLF state intact.
    if (!text) return false;
    // CR ends a line immediately. Only suppress its optional LF in the next chunk,
    // so a terminal event ending in CR CR never needs another byte or HTTP EOF.
    if (skipLF && text.startsWith('\n')) text = text.slice(1);
    skipLF = text.endsWith('\r');
    pending += text.replace(/\r\n?/g, '\n');
    let boundary;
    while ((boundary = pending.indexOf('\n\n')) !== -1) {
      const record = pending.slice(0, boundary);
      pending = pending.slice(boundary + 2);
      controller.enqueue(encoder.encode(`${record}\n\n`));
      if (isTerminal(record)) {
        // Preserve the full terminal event (usage, signatures, failure status).
        // terminate() closes our readable side and cancels the upstream body.
        controller.terminate();
        return true;
      }
    }
    return false;
  };
  const body = response.body.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      forward(decoder.decode(chunk, { stream: true }), controller);
    },
    flush(controller) {
      if (!forward(decoder.decode(), controller) && pending) {
        controller.enqueue(encoder.encode(pending));
      }
    },
  }));
  const headers = new Headers(response.headers);
  headers.delete('content-length');
  headers.delete('content-encoding');
  return new Response(body, {
    status: response.status, statusText: response.statusText, headers,
  });
}
