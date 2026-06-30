import crypto from 'crypto';
import { NO_MULTIMODAL } from './config.js';

const IMAGE_FORMATS = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/gif']);
const DOC_FORMATS = new Set(['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']);
const TEXT_FORMATS = new Set(['text/plain', 'text/csv', 'text/html', 'text/markdown', 'application/json', 'text/xml']);

function isSupportedImage(mt: string) { return IMAGE_FORMATS.has(mt.toLowerCase()); }
function isTextFormat(mt: string) { return TEXT_FORMATS.has(mt.toLowerCase()); }

function convertMediaBlock(source: any, mt: string): any {
  const st = source.type || '';
  if (isTextFormat(mt)) {
    if (st === 'base64' && source.data) {
      try { return { type: 'text', text: Buffer.from(source.data, 'base64').toString('utf-8') }; } catch { return null; }
    }
    if (st === 'url' && source.url) return { type: 'text', text: source.url };
    return null;
  }
  if (DOC_FORMATS.has(mt.toLowerCase())) {
    const extMap: Record<string, string> = {
      'application/pdf': 'document.pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document.docx',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'document.xlsx',
    };
    const filename = extMap[mt.toLowerCase()] || 'document';
    if (st === 'base64' && source.data) return { type: 'file', file: { filename, file_data: `data:${mt};base64,${source.data}` } };
    if (st === 'url' && source.url) return { type: 'file', file: { filename, file_data: source.url } };
    return null;
  }
  return null;
}

export function extractText(content: any): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map(i => {
      if (typeof i === 'string') return i;
      if (i.type === 'text') return i.text || '';
      if (i.type === 'thinking') return i.thinking || '';
      if (i.type === 'image') return `[image:${i.source?.type || 'unknown'}]`;
      if (i.type === 'document') return `[document:${i.source?.media_type || 'unknown'}]`;
      if (i.type === 'redacted_thinking') return '[redacted]';
      return i.text || JSON.stringify(i);
    }).join('\n');
  }
  return content ? String(content) : '';
}

export function anthropicToOpenAI(body: any, model: string): any {
  const thinking = body.thinking?.type === 'enabled' || body.thinking?.type === 'adaptive';
  const messages: any[] = [];
  const system = body.system;

  if (typeof system === 'string') {
    if (system) messages.push({ role: 'system', content: system });
  } else if (Array.isArray(system)) {
    const sysBlocks: any[] = [];
    for (const block of system) {
      if (block.type === 'text') sysBlocks.push({ type: 'text', text: block.text || '' });
      else if (block.type === 'image') {
        const src = block.source || {}; const mt = src.media_type || '';
        if (isSupportedImage(mt)) {
          const url = src.type === 'base64' ? `data:${mt};base64,${src.data}` : (src.url || '');
          sysBlocks.push({ type: 'image_url', image_url: { url } });
        }
      } else if (block.type === 'document') {
        const src = block.source || {}; const mt = src.media_type || '';
        if (isSupportedImage(mt)) {
          const url = src.type === 'base64' ? `data:${mt};base64,${src.data}` : (src.url || '');
          sysBlocks.push({ type: 'image_url', image_url: { url } });
        } else {
          const conv = convertMediaBlock(src, mt);
          if (conv) sysBlocks.push(conv);
        }
      }
    }
    if (sysBlocks.length) {
      const hasMedia = sysBlocks.some((b: any) => b.type !== 'text');
      if (hasMedia && !NO_MULTIMODAL.has(model)) {
        messages.push({ role: 'system', content: sysBlocks });
      } else {
        messages.push({ role: 'system', content: sysBlocks.map((b: any) => b.text || '').join('\n') });
      }
    }
  }

  for (const msg of body.messages || []) {
    const role = msg.role;
    const content = msg.content;
    const isAsst = role === 'assistant';

    if (typeof content === 'string') {
      const out: any = { role, content };
      if (thinking && isAsst) out.reasoning_content = ' ';
      messages.push(out);
      continue;
    }
    if (!Array.isArray(content)) continue;

    const textParts: string[] = [];
    const contentBlocks: any[] = [];
    const toolCalls: any[] = [];
    const thinkingParts: string[] = [];
    const toolResults: any[] = [];
    let hasMedia = false;

    for (const block of content) {
      if (typeof block === 'string') { textParts.push(block); contentBlocks.push({ type: 'text', text: block }); continue; }
      const bt = block.type;
      if (bt === 'text') { textParts.push(block.text || ''); contentBlocks.push({ type: 'text', text: block.text || '' }); }
      else if (bt === 'image') {
        const src = block.source || {}; const mt = src.media_type || '';
        if (isSupportedImage(mt)) {
          hasMedia = true;
          const url = src.type === 'base64' ? `data:${mt};base64,${src.data}` : (src.url || '');
          contentBlocks.push({ type: 'image_url', image_url: { url } });
        }
      } else if (bt === 'document') {
        const src = block.source || {}; const mt = src.media_type || '';
        if (isSupportedImage(mt)) {
          hasMedia = true;
          const url = src.type === 'base64' ? `data:${mt};base64,${src.data}` : (src.url || '');
          contentBlocks.push({ type: 'image_url', image_url: { url } });
        } else {
          const conv = convertMediaBlock(src, mt);
          if (conv) { hasMedia = true; contentBlocks.push(conv); }
        }
      } else if (bt === 'thinking') { thinkingParts.push(block.thinking || ''); }
      else if (bt === 'tool_use') {
        toolCalls.push({ id: block.id || `call_${crypto.randomUUID().slice(0,8)}`, type: 'function', function: { name: block.name || '', arguments: JSON.stringify(block.input || {}) } });
      } else if (bt === 'tool_result') {
        toolResults.push({ role: 'tool', tool_call_id: block.tool_use_id || '', content: extractText(block.content) });
      }
    }

    messages.push(...toolResults);

    const joinedThinking = thinkingParts.join('\n');
    const forceString = NO_MULTIMODAL.has(model);

    if (toolCalls.length) {
      const out: any = { role, content: null, tool_calls: toolCalls };
      if (joinedThinking) out.reasoning_content = joinedThinking;
      else if (thinking && isAsst) out.reasoning_content = ' ';
      messages.push(out);
    } else if (hasMedia && !forceString) {
      const out: any = { role, content: contentBlocks };
      if (joinedThinking) out.reasoning_content = joinedThinking;
      else if (thinking && isAsst) out.reasoning_content = ' ';
      messages.push(out);
    } else if (textParts.length || thinkingParts.length || (thinking && isAsst) || hasMedia) {
      const out: any = { role, content: textParts.join('\n') || '' };
      if (joinedThinking) out.reasoning_content = joinedThinking;
      else if (thinking && isAsst) out.reasoning_content = ' ';
      messages.push(out);
    }
  }

  const oai: any = { model, messages, max_tokens: body.max_tokens || 16384, stream: body.stream || false };

  if (thinking && body.thinking?.budget_tokens) oai.max_completion_tokens = body.thinking.budget_tokens;
  if (body.temperature !== undefined) oai.temperature = body.temperature;
  if (body.top_p !== undefined) oai.top_p = body.top_p;
  if (body.stop_sequences) oai.stop = body.stop_sequences;

  if (body.tools) {
    oai.tools = body.tools.map((t: any) => ({
      type: 'function',
      function: { name: t.name, description: t.description || '', parameters: t.input_schema || {} },
    }));
    const tc = body.tool_choice || 'auto';
    if (typeof tc === 'object') {
      if (tc.type === 'tool') oai.tool_choice = { type: 'function', function: { name: tc.name } };
      else if (tc.type === 'any') oai.tool_choice = 'required';
      else oai.tool_choice = 'auto';
    } else oai.tool_choice = tc;
  }

  return oai;
}

export function openaiToAnthropic(resp: any, model: string): any {
  const choice = resp.choices?.[0] || {};
  const msg = choice.message || {};
  const usage = resp.usage || {};
  const blocks: any[] = [];

  const hasThinking = !!(msg.reasoning_content || msg.reasoning);
  if (msg.reasoning_content) blocks.push({ type: 'thinking', thinking: msg.reasoning_content });
  if (msg.reasoning) blocks.push({ type: 'thinking', thinking: msg.reasoning });
  const content = msg.content;
  const textContent = typeof content === 'string' ? content : (Array.isArray(content) ? content.map((p: any) => p.text || '').join('') : '');
  if (textContent) {
    blocks.push({ type: 'text', text: textContent });
  } else if (hasThinking) {
    // Model only returned reasoning -- duplicate as text so Claude Code shows it
    blocks.push({ type: 'text', text: msg.reasoning_content || msg.reasoning });
  }
  for (const tc of msg.tool_calls || []) {
    const fn = tc.function || {};
    let inp: any = {};
    try { inp = JSON.parse(fn.arguments || '{}'); } catch {}
    blocks.push({ type: 'tool_use', id: tc.id || `toolu_${crypto.randomUUID().slice(0,8)}`, name: fn.name || '', input: inp });
  }

  if (!blocks.length) blocks.push({ type: 'text', text: '' });

  const stopMap: Record<string, string> = { stop: 'end_turn', length: 'max_tokens', tool_calls: 'tool_use', content_filter: 'content_filter' };
  let stop = stopMap[choice.finish_reason || ''] || 'end_turn';
  if (msg.tool_calls?.length) stop = 'tool_use';

  return {
    id: `msg_${crypto.randomUUID().replace(/-/g, '').slice(0,24)}`,
    type: 'message', role: 'assistant', content: blocks, model,
    stop_reason: stop, stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens || 0,
      output_tokens: getOutputTokens(usage),
      cache_creation_input_tokens: usage.cache_creation_input_tokens || usage.prompt_tokens_details?.cache_creation || 0,
      cache_read_input_tokens: usage.prompt_tokens_details?.cached_tokens || 0,
    },
  };
}

export function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 3));
}

export function estimateInputTokens(body: any): number {
  const chunks: string[] = [];
  const system = body.system;
  if (typeof system === 'string') chunks.push(system);
  else if (Array.isArray(system)) {
    for (const s of system) {
      if (s.type === 'image' || s.type === 'document') chunks.push(s.source?.data || s.source?.url || '');
      else chunks.push(s.text || '');
    }
  }
  for (const tool of body.tools || []) { chunks.push(tool.name || ''); chunks.push(tool.description || ''); chunks.push(JSON.stringify(tool.input_schema || {})); }
  for (const msg of body.messages || []) {
    const c = msg.content;
    if (typeof c === 'string') chunks.push(c);
    else if (Array.isArray(c)) {
      for (const b of c) {
        if (typeof b === 'string') chunks.push(b);
        else if (b.type === 'tool_result') chunks.push(extractText(b.content));
        else if (b.type === 'thinking') chunks.push(b.thinking || '');
        else if (b.type === 'image' || b.type === 'document') chunks.push(b.source?.data || b.source?.url || '');
        else chunks.push(b.text || ''), chunks.push(JSON.stringify(b.input || ''));
      }
    }
  }
  return estimateTokens(chunks.join('\n'));
}

export function getOutputTokens(usage: any): number {
  return (usage.completion_tokens || 0) + (usage.completion_tokens_details?.reasoning_tokens || 0);
}

export function extractCacheTokens(usage: any): number {
  let total = 0;
  const details = usage.prompt_tokens_details || {};
  const read = details.cached_tokens || usage.cached_tokens || usage.cache_read_input_tokens || 0;
  const creation = details.cache_creation || usage.cache_creation_input_tokens || 0;
  return read + creation;
}

// --- SSE helpers ---

export function sse(event: string, payload: any): string {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

export function forwardHeaders(headers: Headers): Record<string, string> {
  const out: Record<string, string> = {};
  headers?.forEach((value, key) => {
    const kl = key.toLowerCase();
    if (kl.startsWith('x-request-id') || kl.startsWith('x-ratelimit') || kl.startsWith('openai-') || kl.startsWith('anthropic-') || kl.startsWith('cf-') || kl.startsWith('x-cache') || kl.startsWith('x-gg')) {
      out[key] = value;
    }
  });
  return out;
}
