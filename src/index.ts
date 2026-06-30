import express from 'express';
import cors from 'cors';
import crypto from 'crypto';
import chalk from 'chalk';
import { initDB, logRequest, getDashboardData, getUsageStats, getRecentLogs, getLogsPaginated, deleteLogs, getFilteredStats, closeDB } from './database.js';
import { anthropicToOpenAI, openaiToAnthropic, estimateInputTokens, estimateTokens, getOutputTokens, extractCacheTokens, extractText, sse, forwardHeaders } from './convert.js';
import { SECRET_KEY, API_KEY, PROXY, HOST, PORT, MODELS, ROUTES, getModelConfig, routeFor, FALLBACK_PROVIDERS } from './config.js';

initDB();

const app = express();
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.set('trust proxy', true);

const RETRYABLE = new Set([429, 502, 503, 504]);
const AUTH_ERRORS = new Set([401, 403]);
const QUOTA_ERRORS = new Set([402, 429]);

function getIP(req: express.Request): string {
  return (req.headers['x-forwarded-for'] as string)?.split(',')[0]?.trim() || req.ip || 'unknown';
}

function reqId(): string {
  return `msg_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`;
}

function elapsed(start: number): number {
  return Math.max(1, Date.now() - start);
}

function classifyError(status: number, body: string): string {
  if (AUTH_ERRORS.has(status)) return 'auth';
  if (QUOTA_ERRORS.has(status)) return 'quota';
  if (status === 400 && body.toLowerCase().includes('quota')) return 'quota';
  if (status === 400 && body.toLowerCase().includes('rate')) return 'rate_limit';
  if (RETRYABLE.has(status)) return 'retryable';
  if (status === 404) return 'retryable'; // Cloudflare routing blips, try fallbacks
  if (status === 400) return 'retryable'; // bad request on one provider may work on another
  return 'fatal';
}

async function sleep(ms: number) {
  return new Promise(r => setTimeout(r, ms));
}

function extractErrMsg(resp: Response, text: string): string {
  try {
    const d = JSON.parse(text);
    const e = d.error;
    if (typeof e === 'string') return e;
    if (e?.message) return e.message;
  } catch {}
  return text.slice(0, 200);
}

// Maps OpenCode model IDs to equivalent models supported by each fallback provider
const MODEL_FALLBACK: Record<string, Record<string, string>> = {
  // Free tier → cheap/small provider models
  'deepseek-v4-flash-free':  { openai: 'gpt-4o-mini', gemini: 'gemini-2.0-flash', groq: 'llama-3.1-8b-instant', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o-mini', mistral: 'mistral-small-latest' },
  'mimo-v2.5-free':          { openai: 'gpt-4o-mini', gemini: 'gemini-2.0-flash', groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest' },
  'nemotron-3-ultra-free':   { openai: 'gpt-4o-mini', gemini: 'gemini-2.0-flash', groq: 'meta-llama/llama-4-scout-17b-16e-instruct', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest' },
  'north-mini-code-free':    { openai: 'gpt-4o-mini', gemini: 'gemini-2.0-flash', groq: 'llama-3.1-8b-instant', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o-mini', mistral: 'codestral-latest' },
  // Paid chat models (OpenAI protocol)
  'glm-5.2':                 { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'glm-5.1':                 { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'glm-5':                   { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest' },
  'kimi-k2.5':               { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'kimi-k2.6':               { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'kimi-k2.7':               { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'deepseek-v4-pro':         { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'meta-llama/llama-4-scout-17b-16e-instruct', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'deepseek-v4-flash':       { openai: 'gpt-4o-mini',  gemini: 'gemini-2.0-flash', groq: 'llama-3.1-8b-instant', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o-mini', mistral: 'mistral-small-latest' },
  'mimo-v2-pro':             { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest' },
  'mimo-v2-omni':            { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest' },
  'mimo-v2.5-pro':           { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest' },
  'mimo-v2.5':               { openai: 'gpt-4o',       gemini: 'gemini-2.0-flash', groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest' },
  // Paid Anthropic protocol models
  'minimax-m3':              { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'openai/gpt-oss-120b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'minimax-m2.7':            { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'openai/gpt-oss-120b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'minimax-m2.5':            { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'openai/gpt-oss-120b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'qwen3.7-max':             { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'qwen3.7-plus':            { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'qwen3.6-plus':            { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
  'qwen3.5-plus':            { openai: 'gpt-4o',       gemini: 'gemini-2.0-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest' },
};

const DEFAULT_FALLBACKS: Record<string, string> = {
  openai: 'gpt-4o-mini',
  gemini: 'gemini-2.0-flash',
  groq: 'llama-3.3-70b-versatile',
  openrouter: 'auto',
  cerebras: 'gemma-4-31b',
  github: 'gpt-4o-mini',
  mistral: 'mistral-small-latest',
};

// Build provider chain: [OpenCode, ...fallbacks]
function buildChain(endpoint: string, isAnthropicProto: boolean, body: any) {
  const chain: { name: string; url: string; key: string; body: any; headers: Record<string, string> }[] = [];

  const opencodeHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
  if (isAnthropicProto) {
    opencodeHeaders['x-api-key'] = API_KEY;
    opencodeHeaders['anthropic-version'] = '2023-06-01';
  } else {
    opencodeHeaders['Authorization'] = `Bearer ${API_KEY}`;
  }
  if (PROXY) opencodeHeaders['OpenCode-Proxy'] = PROXY;
  chain.push({ name: `opencode`, url: endpoint, key: API_KEY, body, headers: opencodeHeaders });

  const fbBody = isAnthropicProto ? anthropicToOpenAI(body, body.model) : { ...body };
  for (const fb of FALLBACK_PROVIDERS) {
    const perModel = MODEL_FALLBACK[fbBody.model];
    const fbModel = (perModel && perModel[fb.name]) || DEFAULT_FALLBACKS[fb.name] || fbBody.model;
    const fbBodyClone = { ...fbBody, model: fbModel };
    chain.push({ name: fb.name, url: fb.chatEndpoint, key: fb.apiKey, body: fbBodyClone, headers: { 'Authorization': `Bearer ${fb.apiKey}`, 'Content-Type': 'application/json' } });
  }
  return chain;
}

// ── Streaming: Convert OpenAI SSE → Anthropic SSE ──────────────────────────
async function* openaiStreamToAnthropic(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  model: string,
  origModel: string,
  estInput: number,
): AsyncGenerator<string> {
  const msgId = `msg_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`;
  let started = false;
  let textBlockIdx: number | null = null;
  let reasoningBlockIdx: number | null = null;
  const toolBlockIdx: Record<number, number> = {};
  const openBlocks: number[] = [];
  let nextIdx = 0;
  let streamOut = 0;
  let actualUsage: any = null;
  let finishReason: string | null = null;

  let buf = '';
  const decoder = new TextDecoder();

  function emitCleanup(): string {
    let out = '';
    if (!started) {
      out += sse('message_start', { type: 'message_start', message: { id: msgId, type: 'message', role: 'assistant', content: [], model: origModel, stop_reason: null, stop_sequence: null, usage: { input_tokens: estInput, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 } } });
    }
    for (const idx of openBlocks) out += sse('content_block_stop', { type: 'content_block_stop', index: idx });
    out += sse('message_delta', { type: 'message_delta', delta: { stop_reason: 'max_tokens' }, usage: { output_tokens: streamOut, input_tokens: estInput } });
    out += sse('message_stop', { type: 'message_stop' });
    return out;
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') {
          let finalIn = estInput;
          let finalOut = streamOut;
          let finalCache = 0;
          let finalCacheCreation = 0;
          if (actualUsage) {
            finalIn = actualUsage.prompt_tokens ?? estInput;
            finalOut = getOutputTokens(actualUsage);
            if (finalOut == null) {
              const total = actualUsage.total_tokens;
              const prompt = actualUsage.prompt_tokens;
              if (total != null && prompt != null) finalOut = total - prompt;
            }
            if (finalOut == null) finalOut = streamOut;
            finalCache = extractCacheTokens(actualUsage);
            finalCacheCreation = actualUsage.cache_creation_input_tokens || actualUsage.prompt_tokens_details?.cache_creation || 0;
          }
          if (!started) {
            yield sse('message_start', { type: 'message_start', message: { id: msgId, type: 'message', role: 'assistant', content: [], model: origModel, stop_reason: null, stop_sequence: null, usage: { input_tokens: finalIn, output_tokens: 0, cache_creation_input_tokens: finalCacheCreation, cache_read_input_tokens: finalCache } } });
          }
          for (const idx of openBlocks) yield sse('content_block_stop', { type: 'content_block_stop', index: idx });
          const stopMap: Record<string, string> = { stop: 'end_turn', length: 'max_tokens', content_filter: 'content_filter' };
          let stop = stopMap[finishReason || ''] || 'end_turn';
          if (Object.keys(toolBlockIdx).length) stop = 'tool_use';
          yield sse('message_delta', { type: 'message_delta', delta: { stop_reason: stop }, usage: { output_tokens: finalOut, input_tokens: finalIn, cache_read_input_tokens: finalCache } });
          yield sse('message_stop', { type: 'message_stop' });
          openBlocks.length = 0;
          return { finalIn, finalOut, finalCache, provider: '' } as any;
        }

        let chunk: any;
        try { chunk = JSON.parse(data); } catch { continue; }

        const chunkUsage = chunk.usage;
        if (chunkUsage && typeof chunkUsage === 'object') actualUsage = chunkUsage;

        const choices = chunk.choices || [];
        const firstChoice = choices[0] || {};
        if (firstChoice.finish_reason) finishReason = firstChoice.finish_reason;
        const delta = firstChoice.delta || {};

        if (!started) {
          yield sse('message_start', { type: 'message_start', message: { id: msgId, type: 'message', role: 'assistant', content: [], model: origModel, stop_reason: null, stop_sequence: null, usage: { input_tokens: estInput, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 } } });
          started = true;
        }

        const text = typeof delta.content === 'string' ? delta.content : (Array.isArray(delta.content) ? delta.content.map((p: any) => p.text || '').join('') : '');
        if (text) {
          if (textBlockIdx === null) {
            textBlockIdx = nextIdx++;
            yield sse('content_block_start', { type: 'content_block_start', index: textBlockIdx, content_block: { type: 'text', text: '' } });
            openBlocks.push(textBlockIdx);
          }
          streamOut += estimateTokens(text);
          yield sse('content_block_delta', { type: 'content_block_delta', index: textBlockIdx, delta: { type: 'text_delta', text } });
        }

        const reasoning = delta.reasoning_content || delta.reasoning;
        if (typeof reasoning === 'string' && reasoning) {
          if (reasoningBlockIdx === null) {
            reasoningBlockIdx = nextIdx++;
            yield sse('content_block_start', { type: 'content_block_start', index: reasoningBlockIdx, content_block: { type: 'thinking', thinking: '' } });
            openBlocks.push(reasoningBlockIdx);
          }
          streamOut += estimateTokens(reasoning);
          yield sse('content_block_delta', { type: 'content_block_delta', index: reasoningBlockIdx, delta: { type: 'thinking_delta', thinking: reasoning } });
        }

        for (const tc of delta.tool_calls || []) {
          const apiIdx = tc.index || 0;
          if (toolBlockIdx[apiIdx] === undefined) {
            const bidx = nextIdx++;
            toolBlockIdx[apiIdx] = bidx;
            yield sse('content_block_start', { type: 'content_block_start', index: bidx, content_block: { type: 'tool_use', id: tc.id || `toolu_${crypto.randomUUID().slice(0, 8)}`, name: tc.function?.name || '', input: {} } });
            openBlocks.push(bidx);
          }
          const args = tc.function?.arguments || '';
          if (args) {
            streamOut += estimateTokens(args);
            yield sse('content_block_delta', { type: 'content_block_delta', index: toolBlockIdx[apiIdx], delta: { type: 'input_json_delta', partial_json: args } });
          }
        }
      }
    }
  } finally {
    if (openBlocks.length) {
      yield emitCleanup();
    }
  }
}

// ── Shared retry helper ──────────────────────────────────────────────────────
interface ChainResult { text: string; headers: Headers; status: number; name: string; resp?: Response; }

async function tryChain<T>(
  chain: { name: string; url: string; key: string; body: any; headers: Record<string, string> }[],
  opts: {
    start: number; rid: string; modelId: string; origModel: string;
    routeKey: string; protocol: string; isStream: boolean;
    thinking: string; effort: string; ip: string;
    isAnthropicProto: boolean;
  },
  onOk: (result: ChainResult) => Promise<T>,
  onAllFailed: () => T,
  isStreaming?: boolean,
): Promise<T> {
  for (let i = 0; i < chain.length; i++) {
    const { name, url, body, headers } = chain[i];
    const backoff = i > 0 ? Math.min(1000 * Math.pow(2, i - 1), 4000) : 0;
    if (backoff) await sleep(backoff);

    try {
      const resp = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });

      if (resp.ok && isStreaming) {
        return await onOk({ text: '', headers: resp.headers, status: resp.status, name, resp });
      }

      const respText = !isStreaming ? await resp.text() : (await resp.text().catch(() => ''));

      if (resp.ok) {
        return await onOk({ text: respText, headers: resp.headers, status: resp.status, name });
      }

      const errMsg = extractErrMsg(resp, respText);
      const errClass = classifyError(resp.status, respText);
      const sc = resp.status >= 500 ? C.bad(resp.status) : resp.status >= 400 ? C.warn(resp.status) : C.good(resp.status);
      console.log(`  ${chalk.dim(name)} ${chalk.dim('→')} ${sc} ${chalk.dim(`(${errClass})`)}: ${chalk.dim(errMsg.slice(0, 100))}`);
      logRequest({ request_id: opts.rid, model: opts.modelId, original_model: opts.origModel, route: opts.routeKey, provider: name, protocol: opts.protocol, is_stream: opts.isStream, thinking: opts.thinking, effort: opts.effort, status: resp.status, duration_ms: elapsed(opts.start), error: errMsg, ip: opts.ip });

      if (errClass === 'fatal' || i >= chain.length - 1) {
        return onAllFailed();
      }
      continue;
    } catch (e: any) {
      console.log(`  ${chalk.dim(name)} ${C.bad('✗')} ${chalk.dim(e.message)}`);
      logRequest({ request_id: opts.rid, model: opts.modelId, original_model: opts.origModel, route: opts.routeKey, provider: name, protocol: opts.protocol, is_stream: opts.isStream, thinking: opts.thinking, effort: opts.effort, status: 502, duration_ms: elapsed(opts.start), error: e.message, ip: opts.ip });
      if (i < chain.length - 1) continue;
      return onAllFailed();
    }
  }
  return onAllFailed();
}

function errResponse(status: number, msg: string): object {
  return { type: 'error', error: { type: 'api_error', message: msg } };
}

// ── Main handler ────────────────────────────────────────────────────────────
async function handleMessages(req: express.Request, res: express.Response) {
  const start = Date.now();
  const rid = reqId();

  if (req.headers['x-api-key'] !== SECRET_KEY) {
    res.status(401).json(errResponse(401, 'Invalid x-api-key'));
    return;
  }

  const body = req.body;
  const origModel = body.model || 'claude-sonnet-4-6';
  const route = routeFor(origModel);
  const modelId = route.model;
  const cfg = getModelConfig(modelId);
  const protocol = cfg.protocol;
  const endpoint = cfg.endpoint;
  const isStream = body.stream || false;

  const thinking = body.thinking || {};
  const thinkingType = typeof thinking === 'object' ? (thinking.type || 'none') : 'none';
  const effort = body.effort || (thinking.effort) || (body.output_config?.effort) || 'none';

  console.log(`  ${chalk.dim('[' + rid.slice(0, 12) + ']')} ${chalk.bold(origModel)} ${chalk.dim('→')} ${chalk.bold(modelId)} ${chalk.dim('| ' + protocol + ' | stream=' + isStream + (thinkingType !== 'none' ? ' | thinking=' + thinkingType : '') + (effort && effort !== 'none' ? ' | effort=' + effort : ''))}`);

  const reqBody = { ...body, model: modelId };
  const commonOpts = { start, rid, modelId, origModel, routeKey: route.match[0], protocol, isStream, thinking: thinkingType, effort, ip: getIP(req) };

  if (protocol === 'anthropic') {
    const chain = buildChain(endpoint, true, reqBody);

    if (!isStream) {
      await tryChain(chain, { ...commonOpts, isAnthropicProto: true }, async (r) => {
        const data = JSON.parse(r.text);
        const usage = data.usage || {};
        const inp = usage.input_tokens || usage.prompt_tokens || 0;
        const out = usage.output_tokens || getOutputTokens(usage) || 0;
        const cache = extractCacheTokens(usage);
        console.log(`  ${C.good('✓')} ${chalk.bold(modelId)} ${chalk.dim('| +' + inp + ' in | +' + out + ' out | +' + cache + ' cache')} ${chalk.dim('via ' + r.name)}`);
        logRequest({ ...commonOpts, model: modelId, provider: r.name, status: 200, duration_ms: elapsed(start), tokens_input: inp, tokens_output: out, tokens_cache: cache });
        for (const [k, v] of Object.entries(forwardHeaders(r.headers))) res.setHeader(k, v as string);
        res.json(data);
        return null as any;
      }, () => {
        res.status(502).json(errResponse(502, 'All upstream providers failed'));
        return null as any;
      });
    } else {
      const estInput = estimateInputTokens(reqBody);

      await tryChain(chain, { ...commonOpts, isAnthropicProto: true, isStream: true }, async (r) => {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('x-request-id', rid);

        const reader = (r.resp!.body as ReadableStream).getReader();
        let streamIn = estInput, streamOut = 0, streamCache = 0;
        let buf = '';
        const decoder = new TextDecoder();
        let openBlocks: number[] = [];

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const text = decoder.decode(value, { stream: true });
            buf += text;
            while (buf.includes('\n')) {
              const nl = buf.indexOf('\n');
              const line = buf.slice(0, nl).trim();
              buf = buf.slice(nl + 1);
              if (line.startsWith('data:')) {
                const ds = line.slice(5).trim();
                if (ds === '[DONE]') continue;
                try {
                  const ev = JSON.parse(ds);
                  if (ev.type === 'content_block_start') {
                    openBlocks.push(ev.index);
                  } else if (ev.type === 'content_block_stop') {
                    openBlocks = openBlocks.filter(i => i !== ev.index);
                  } else if (ev.type === 'message_start') {
                    const mu = ev.message?.usage || {};
                    streamIn = mu.input_tokens ?? estInput;
                    streamCache = extractCacheTokens(mu);
                  } else if (ev.type === 'message_delta') {
                    const du = ev.usage || {};
                    streamOut = du.output_tokens || 0;
                  }
                } catch {}
              }
            }
            res.write(text);
          }
        } catch (e: any) {
          console.error(`  ${C.bad('✗')} ${chalk.dim('stream error from ' + r.name + ': ' + e.message)}`);
          for (const idx of openBlocks) res.write(sse('content_block_stop', { type: 'content_block_stop', index: idx }));
          if (!res.headersSent) res.setHeader('Content-Type', 'text/event-stream');
          res.write(sse('message_delta', { type: 'message_delta', delta: { stop_reason: 'error' }, usage: { output_tokens: streamOut } }));
          res.write(sse('message_stop', { type: 'message_stop' }));
        }

        console.log(`  ${C.good('✓')} ${chalk.bold(modelId)} ${chalk.dim('| +' + streamIn + ' in | +' + streamOut + ' out | +' + streamCache + ' cache')} ${chalk.dim('via ' + r.name)}`);
        logRequest({ ...commonOpts, model: modelId, provider: r.name, status: 200, duration_ms: elapsed(start), tokens_input: streamIn, tokens_output: streamOut, tokens_cache: streamCache });
        res.end();
        return null as any;
      }, () => {
        res.status(502).json(errResponse(502, 'All upstream providers failed'));
        return null as any;
      }, true);
    }
  } else {
    // ── OpenAI protocol ─────────────────────────────────────────
    const oaiBody = anthropicToOpenAI(reqBody, modelId);
    const chain = buildChain(endpoint, false, oaiBody);

    if (!isStream) {
      await tryChain(chain, { ...commonOpts, isAnthropicProto: false }, async (r) => {
        const data = JSON.parse(r.text);
        const usage = data.usage || {};
        const inp = usage.prompt_tokens || 0;
        const out = getOutputTokens(usage);
        const cache = extractCacheTokens(usage);
        console.log(`  ${C.good('✓')} ${chalk.bold(modelId)} ${chalk.dim('| +' + inp + ' in | +' + out + ' out | +' + cache + ' cache')} ${chalk.dim('via ' + r.name)}`);
        logRequest({ ...commonOpts, model: modelId, provider: r.name, status: 200, duration_ms: elapsed(start), tokens_input: inp, tokens_output: out, tokens_cache: cache });
        const anthroResp = openaiToAnthropic(data, origModel);
        for (const [k, v] of Object.entries(forwardHeaders(r.headers))) res.setHeader(k, v as string);
        res.json(anthroResp);
        return null as any;
      }, () => {
        res.status(502).json(errResponse(502, 'All upstream providers failed'));
        return null as any;
      });
    } else {
      const estInput = estimateInputTokens(reqBody);
      const streamBody = { ...oaiBody, stream: true, stream_options: { include_usage: true } };

      // Rebuild chain with streaming body — preserve per-provider model mapping
      const streamChain = chain.map(c => ({ ...c, body: { ...streamBody, model: c.body.model } }));

      await tryChain(streamChain, { ...commonOpts, isAnthropicProto: false, isStream: true }, async (r) => {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('x-request-id', rid);

        const reader = (r.resp!.body as ReadableStream).getReader();
        const gen = openaiStreamToAnthropic(reader, modelId, origModel, estInput);

        try {
          for await (const chunk of gen) {
            res.write(chunk);
          }
        } catch (e: any) {
          console.error(`  ${C.bad('✗')} ${chalk.dim('stream error: ' + e.message)}`);
        }

        console.log(`  ${C.good('✓')} ${chalk.bold(modelId)} ${chalk.dim('via ' + r.name)}`);
        logRequest({ ...commonOpts, model: modelId, provider: r.name, status: 200, duration_ms: elapsed(start) });
        res.end();
        return null as any;
      }, () => {
        res.status(502).json(errResponse(502, 'All upstream providers failed'));
        return null as any;
      }, true);
    }
  }
}

// ── Routes ───────────────────────────────────────────────────────────────────
app.post('/v1/messages', handleMessages);
app.post('/anthropic/v1/messages', handleMessages);

app.get('/v1/models', (_req, res) => {
  res.json({ data: Object.keys(MODELS).map(id => ({ id, object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'opencode-proxy' })), object: 'list' });
});

app.get('/v1/providers', (_req, res) => {
  res.json({
    chain: FALLBACK_PROVIDERS.map(fb => ({ name: fb.name, endpoint: fb.chatEndpoint })),
    model_mappings: Object.fromEntries(
      Object.entries(MODEL_FALLBACK).map(([model, map]) => [model, map])
    ),
    default_fallbacks: DEFAULT_FALLBACKS,
    fallbacks_configured: FALLBACK_PROVIDERS.length,
  });
});

app.get('/v1/providers', (_req, res) => {
  res.json({
    chain: FALLBACK_PROVIDERS.map(fb => ({ name: fb.name, endpoint: fb.chatEndpoint })),
    model_mappings: Object.fromEntries(
      Object.entries(MODEL_FALLBACK).map(([model, map]) => [model, map])
    ),
    default_fallbacks: DEFAULT_FALLBACKS,
    fallbacks_configured: FALLBACK_PROVIDERS.length,
  });
});

app.get('/health', (_req, res) => {
  const s = getUsageStats();
  res.json({ status: 'ok', version: '1.0.0', timestamp: new Date().toISOString(), requests: s.total_requests, tokens_input: s.total_input_tokens, tokens_output: s.total_output_tokens });
});

app.post('/v1/messages/count_tokens', (req, res) => {
  res.json({ input_tokens: estimateInputTokens(req.body || {}) });
});

// ── Dashboard ────────────────────────────────────────────────────────────────
function buildDashboardHTML(authKey: string): string {
  const modelsJson = JSON.stringify(Object.entries(MODELS).map(([id, cfg]) => ({
    id, endpoint: cfg.endpoint.replace(/^.+\/(zen\/(?:go\/)?v[^/]+)/, '$1'), protocol: cfg.protocol
  })));
  const routesJson = JSON.stringify(Object.entries(ROUTES).map(([key, r]) => ({ key, model: r.model, match: r.match })));
  const fallbacksJson = JSON.stringify(FALLBACK_PROVIDERS.map(f => f.name));
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Dash</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23d97757'/%3E%3Ctext x='16' y='22' text-anchor='middle' font-family='system-ui' font-weight='800' font-size='18' fill='%230a0a0f'%3EC%3C/text%3E%3C/svg%3E">
<style>
:root{--bg:#0a0a0f;--bg2:#12121a;--bg3:#1a1a26;--border:#2a2a3a;--text:#e8e4e0;--muted:#6b6560;--accent:#d97757;--accent2:#e8957a;--green:#7bc47f;--red:#e05a5a;--code-bg:#0e0e16;--font-sans:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--font-mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{margin:0;padding:0;box-sizing:border-box}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:#2a2a3a}
::-webkit-scrollbar-thumb:hover{background:#3a3a4a}
*{scrollbar-width:thin;scrollbar-color:#2a2a3a var(--bg)}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);font-size:14px;line-height:1.55;min-height:100vh;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
code{font-family:var(--font-mono);font-size:.92em;word-break:break-all}

/* ── Topbar ── */
nav.topbar{position:sticky;top:0;z-index:50;background:rgba(10,10,15,.85);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-bottom:1px solid var(--border);padding:0 12px;height:52px;display:flex;align-items:center;justify-content:space-between;gap:8px}
.brand{display:flex;align-items:center;gap:8px;font-weight:700;font-size:13px;color:var(--text);white-space:nowrap}
.brand-mark{width:20px;height:20px;background:var(--accent);border-radius:4px;display:grid;place-items:center;color:#0a0a0f;font-size:10px;font-weight:800;flex-shrink:0}
.nav-tabs{display:flex;align-items:center;gap:2px;padding:3px;background:var(--bg2);border:1px solid var(--border);border-radius:6px;flex-wrap:wrap}
.nav-tabs button{display:inline-flex;align-items:center;background:transparent;color:var(--muted);border:none;font:500 11px var(--font-sans);padding:6px 12px;cursor:pointer;border-radius:4px;white-space:nowrap}
.nav-tabs button:hover{color:var(--text)}
.nav-tabs button.active{background:var(--accent);color:#0a0a0f;font-weight:600}
@media(min-width:600px){
  nav.topbar{padding:0 24px;height:56px}
  .brand{font-size:14px}
  .nav-tabs button{font-size:12px;padding:7px 14px}
}

/* ── Views ── */
.view{display:none}
.view.active{display:block}

/* ── Welcome ── */
.welcome-wrap{max-width:960px;margin:0 auto;padding:40px 16px 60px;text-align:center}
.welcome-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(217,119,87,.12);border:1px solid rgba(217,119,87,.25);color:var(--accent);font-size:10px;font-weight:600;letter-spacing:.08em;padding:4px 10px;border-radius:4px;margin-bottom:20px}
.welcome-wrap h1{font-size:clamp(24px,6vw,48px);font-weight:700;letter-spacing:-.02em;line-height:1.1;margin-bottom:16px;color:var(--text)}
.welcome-wrap h1 span{background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.welcome-sub{font-size:14px;color:var(--muted);max-width:600px;margin:0 auto 28px;line-height:1.6;padding:0 8px}
.welcome-cta{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:7px;font:500 13px var(--font-sans);padding:10px 18px;border:1px solid var(--border);background:var(--bg2);color:var(--text);cursor:pointer;text-decoration:none;border-radius:6px}
.btn:hover{background:var(--bg3);border-color:var(--muted);text-decoration:none}
.btn-primary{background:var(--accent);border-color:var(--accent);color:#0a0a0f;font-weight:600}
.btn-primary:hover{background:var(--accent2);border-color:var(--accent2)}

/* ── Feature grid ── */
.feature-grid{display:grid;grid-template-columns:1fr;gap:10px;margin-top:40px;text-align:left}
@media(min-width:600px){.feature-grid{grid-template-columns:repeat(3,1fr);gap:12px}}
.feature{background:var(--bg2);border:1px solid var(--border);padding:18px;border-radius:6px}
.feature h3{font-size:13px;font-weight:600;margin-bottom:4px}
.feature p{font-size:13px;color:var(--muted);line-height:1.5}

/* ── Docs ── */
.docs-page{max-width:900px;margin:0 auto;padding:28px 16px 60px;min-height:calc(100vh - 52px)}
.docs-page h2{font-size:20px;font-weight:700;letter-spacing:-.01em;margin-bottom:8px;color:var(--text)}
.docs-page .eyebrow{font-size:10px;font-weight:600;letter-spacing:.08em;color:var(--accent);margin-bottom:8px}
.docs-page p{color:var(--muted);margin-bottom:14px;line-height:1.6;font-size:13px}
@media(min-width:600px){
  .docs-page{padding:40px 24px 80px;min-height:calc(100vh - 56px)}
  .docs-page h2{font-size:22px}
}

/* ── Category toggles ── */
.category-toggle{display:flex;align-items:center;gap:8px;width:100%;padding:12px 14px;background:var(--bg2);border:1px solid var(--border);color:var(--text);font:500 13px var(--font-sans);cursor:pointer;text-align:left;margin-top:14px;user-select:none;border-radius:6px}
.category-toggle:hover{background:var(--bg3)}
.category-toggle .arrow{transition:transform .2s;font-size:11px;color:var(--muted);flex-shrink:0}
.category-toggle.open .arrow{transform:rotate(90deg)}
.category-body{display:none;border:1px solid var(--border);border-top:0;border-radius:0 0 6px 6px}
.category-body.open{display:block}

/* ── Endpoint lists ── */
.endpoint-list{display:flex;flex-direction:column;gap:5px;padding:10px}
.endpoint{background:var(--bg3);border:1px solid var(--border);padding:8px 10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;border-radius:4px}
.method{font:700 9px var(--font-mono);padding:2px 6px;flex-shrink:0;border-radius:3px}
.method.openai{background:rgba(217,119,87,.15);color:var(--accent)}
.method.anthropic{background:rgba(232,149,122,.15);color:var(--accent2)}
.method.fallback{background:rgba(107,101,96,.15);color:var(--muted)}
.ep-url{font:400 12px var(--font-mono);color:var(--text);word-break:break-all}
.ep-detail{font-size:11px;color:var(--muted);margin-left:auto;text-align:right}
.code-block{background:var(--code-bg);border:1px solid var(--border);margin:10px 0;overflow-x:auto;border-radius:6px}
pre{padding:12px;font:400 12px var(--font-mono);color:#c8c4c0;white-space:pre-wrap;word-break:break-all}
@media(min-width:600px){
  .endpoint-list{padding:14px;gap:6px}
  .endpoint{padding:10px 14px;gap:10px}
  .method{font-size:10px;padding:3px 7px}
  .ep-url{font-size:13px}
  pre{padding:14px;font-size:13px}
}

/* ── Logs ── */
.logs-wrap{padding:16px;max-width:1200px;margin:0 auto}
@media(min-width:600px){.logs-wrap{padding:24px}}

/* ── Stats grid ── */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:20px}
@media(min-width:600px){.stats-grid{grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}}
.stat-card{background:var(--bg2);border:1px solid var(--border);padding:12px;text-align:center;border-radius:6px}
.stat-card h3{font-size:10px;color:var(--muted);font-weight:600;letter-spacing:.04em;margin-bottom:4px}
.stat-card .value{font-size:22px;font-weight:700;color:var(--text)}
.stat-card .sub{font-size:11px;color:var(--muted);margin-top:3px}
@media(min-width:600px){
  .stat-card{padding:16px}
  .stat-card h3{font-size:11px}
  .stat-card .value{font-size:28px}
}

h2{font-size:16px;font-weight:700;margin:20px 0 10px;color:var(--text)}
@media(min-width:600px){h2{font-size:18px;margin:24px 0 12px}}

/* ── Table ── */
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:10px 0}
table{width:100%;border-collapse:collapse;background:var(--bg2);border:1px solid var(--border);min-width:600px;border-radius:6px;overflow:hidden}
th{background:var(--bg3);padding:6px 8px;text-align:left;font-size:10px;color:var(--muted);font-weight:600;letter-spacing:.03em;white-space:nowrap}
td{padding:6px 8px;border-top:1px solid var(--border);font-size:12px;white-space:nowrap}
tr:hover{background:rgba(217,119,87,.04)}
.status-ok{color:var(--green)}
.status-err{color:var(--red)}
@media(min-width:600px){
  th{padding:8px 12px;font-size:11px}
  td{padding:8px 12px;font-size:13px}
}

/* ── Model grid ── */
.model-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px;margin-bottom:16px}
@media(min-width:600px){.model-grid{grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px}}
.model-card{background:var(--bg2);border:1px solid var(--border);padding:10px;border-radius:6px}
.model-card h3{font-size:12px;font-weight:600;color:var(--text);margin-bottom:3px;word-break:break-all}
.model-card .detail{font-size:10px;color:var(--muted)}
.model-card .tokens{font-size:10px;color:var(--accent);margin-top:3px}
@media(min-width:600px){
  .model-card{padding:12px}
  .model-card h3{font-size:13px}
  .model-card .detail,.model-card .tokens{font-size:11px}
}

footer{border-top:1px solid var(--border);padding:16px;text-align:center;font-size:11px;color:var(--muted)}
@media(min-width:600px){footer{padding:22px;font-size:12px}}
</style>
</head>
<body>
<nav class="topbar">
  <div class="brand"><span class="brand-mark">C</span> Claude Dash</div>
  <div class="nav-tabs">
    <button type="button" class="tab-btn active" data-view="welcome">Welcome</button>
    <button type="button" class="tab-btn" data-view="logs">Logs</button>
    <button type="button" class="tab-btn" data-view="docs">Docs</button>
  </div>
</nav>

<main id="welcome" class="view active">
  <section class="welcome-wrap">
    <div class="welcome-badge">AI PROXY GATEWAY</div>
    <h1>Claude <span>Dash</span></h1>
    <p class="welcome-sub">
      Multi-provider proxy translating Anthropic-format requests to 7 upstream APIs.
      Routes Claude models to OpenCode equivalents with automatic format conversion and fallback chain.
    </p>
    <div class="welcome-cta">
      <button type="button" class="btn btn-primary js-go-logs">View Logs</button>
      <button type="button" class="btn js-go-docs">Read Docs</button>
    </div>
    <div class="stats-grid" id="welcome-stats" style="margin-top:40px;text-align:left"></div>
  </section>
</main>

<main id="logs" class="view">
  <section class="logs-wrap">
    <div class="stats-grid" id="stats"></div>
    <h2>Top Models</h2>
    <div class="model-grid" id="models"></div>
    <h2>Providers</h2>
    <div class="model-grid" id="providers"></div>
    <h2>Recent Requests</h2>
    <div id="logs-table"><div class="table-wrap"><table><thead><tr><th>Time</th><th>Model</th><th>Route</th><th>Provider</th><th>Status</th><th>Duration</th><th>Tokens</th><th>Error</th></tr></thead><tbody id="log-body"></tbody></table></div></div>
  </section>
</main>

<main id="docs" class="view">
  <article class="docs-page">
    <p class="eyebrow">Documentation</p>
    <h2>Claude Dash</h2>
    <p>
      Drop-in replacement for the Anthropic <code>/v1/messages</code> API. Send standard Claude-format requests and the proxy automatically routes them to the configured OpenCode model, converts formats, and forwards responses. Falls back through 7 providers.
    </p>
    <div class="code-block"><pre>Base URL: <span id="baseUrlPlaceholder">http://localhost:4000</span></pre></div>

    <h3 style="font-size:14px;font-weight:600;color:var(--text);margin:28px 0 10px;">Authentication</h3>
    <p>Pass your API key as the <code>x-api-key</code> header:</p>
    <div class="code-block"><pre>curl -H "x-api-key: your-secret-key" -H "Content-Type: application/json" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{"model":"sonnet","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}' \\
  <span id="baseUrlPlaceholder2">http://localhost:4000</span>/v1/messages</pre></div>

    <h3 style="font-size:14px;font-weight:600;color:var(--text);margin:28px 0 10px;">Available Models</h3>
    <p>The proxy supports 23 models across three tiers. Models with matching endpoint suffix share the same route.</p>
    <div id="docsModels"></div>

    <h3 style="font-size:14px;font-weight:600;color:var(--text);margin:28px 0 10px;">Route Mappings</h3>
    <p>Claude model names are mapped to OpenCode equivalents:</p>
    <div id="docsRoutes"></div>

    <h3 style="font-size:14px;font-weight:600;color:var(--text);margin:28px 0 10px;">Fallback Providers</h3>
    <p>When OpenCode fails, the proxy falls back through 7 providers in order. Each OpenCode model is mapped to the closest equivalent on each provider (e.g., <code>deepseek-v4-flash-free</code> → <code>gpt-4o-mini</code> on OpenAI, <code>llama-3.1-8b-instant</code> on Groq). Default model shown if no per-model mapping exists.</p>
    <div id="docsFallbacks"></div>

    <h3 style="font-size:14px;font-weight:600;color:var(--text);margin:28px 0 10px;">Request Flow</h3>
    <p>
      1. Incoming <code>/v1/messages</code> request is matched to a route by model name<br>
      2. Body is converted from Anthropic to OpenAI format if needed<br>
      3. Primary provider (OpenCode) is tried first<br>
      4. On failure, fallback providers are tried in sequence with exponential backoff<br>
      5. Successful response is converted back to Anthropic format and returned
    </p>

    <h3 style="font-size:14px;font-weight:600;color:var(--text);margin:28px 0 10px;">Endpoints</h3>
    <div id="docsEndpoints"></div>
  </article>
</main>

<footer>Claude Dash &mdash; Multi-provider AI API gateway</footer>

<script>
const K='${authKey}';
const MODELS_DATA = ${modelsJson};
const ROUTES_DATA = ${routesJson};
const FALLBACKS_DATA = ${fallbacksJson};

document.getElementById('baseUrlPlaceholder').textContent = location.origin;
document.getElementById('baseUrlPlaceholder2').textContent = location.origin;

// ── Tab Switcher ──
document.querySelectorAll('.tab-btn').forEach(function(b){b.addEventListener('click',function(){
  var t=b.getAttribute('data-view');
  document.querySelectorAll('.tab-btn').forEach(function(x){x.classList.remove('active')});
  document.querySelectorAll('.view').forEach(function(v){v.classList.remove('active')});
  b.classList.add('active');
  document.getElementById(t).classList.add('active');
})});
document.querySelector('.js-go-logs')&&document.querySelector('.js-go-logs').addEventListener('click',function(){document.querySelector('.tab-btn[data-view=\"logs\"]').click()});
document.querySelector('.js-go-docs')&&document.querySelector('.js-go-docs').addEventListener('click',function(){document.querySelector('.tab-btn[data-view=\"docs\"]').click()});

// ── Category toggle click handler (event delegation) ──
document.addEventListener('click', function(e){
  var btn=e.target;
  while(btn && !btn.classList.contains('category-toggle')) btn=btn.parentNode;
  if(!btn) return;
  e.preventDefault();
  var body=btn.nextElementSibling;
  if(body && body.classList.contains('category-body')){
    body.classList.toggle('open');
    btn.classList.toggle('open');
  }
});

// ── Docs: Models ──
(function(){
  var tiers={};
  MODELS_DATA.forEach(function(m){
    var t=m.endpoint.split('/').pop()||'other';
    if(!tiers[t])tiers[t]=[];
    tiers[t].push(m);
  });
  var html='',first=true;
  Object.keys(tiers).forEach(function(t){
    var label=t==='completions'?'Paid (Zen Go)':t==='messages'?'Anthropic Protocol':t==='chat/completions'||t==='completions'?'Paid':'Free';
    html+='<button class="category-toggle'+(first?' open':'')+'"><span class="arrow">&#9654;</span> '+label+' ('+tiers[t].length+' models)</button>';
    html+='<div class="category-body'+(first?' open':'')+'"><div class="endpoint-list">';
    tiers[t].forEach(function(m){
      var ep=m.endpoint;
      html+='<div class="endpoint"><span class="method '+m.protocol+'">'+m.protocol+'</span><span class="ep-url">'+m.id+'</span><span class="ep-detail">'+ep+'</span></div>';
    });
    html+='</div></div>';
    first=false;
  });
  document.getElementById('docsModels').innerHTML=html;
})();

// ── Docs: Routes ──
(function(){
  var html='<div class="endpoint-list">';
  ROUTES_DATA.forEach(function(r){
    html+='<div class="endpoint"><span class="ep-url">'+r.key+'</span><span class="ep-detail">→ '+r.model+'</span><span style="font-size:11px;color:var(--muted);width:100%">matches: '+r.match.join(', ')+'</span></div>';
  });
  html+='</div>';
  document.getElementById('docsRoutes').innerHTML=html;
})();

// ── Docs: Fallbacks ──
(function(){
  var pmeta={
    groq:{model:'llama-3.3-70b-versatile',url:'api.groq.com',models:'Llama 3.1/3.3, Qwen3, GPT-OSS, Llama 4 Scout'},
    openrouter:{model:'auto (routes to best free)',url:'openrouter.ai',models:'20+ free models via single endpoint'},
    cerebras:{model:'gemma-4-31b',url:'api.cerebras.ai',models:'Gemma 4, ZAI-GLM 4.7, GPT-OSS'},
    github:{model:'gpt-4o-mini',url:'models.inference.ai.azure.com',models:'GPT-4o, GPT-4o-mini, Llama, Phi'},
    mistral:{model:'mistral-small-latest',url:'api.mistral.ai',models:'Mistral Small/Large, Codestral'},
    openai:{model:'gpt-4o-mini',url:'api.openai.com',models:'GPT-4o, GPT-4o-mini'},
    gemini:{model:'gemini-2.0-flash',url:'generativelanguage.googleapis.com',models:'Gemini 2.0 Flash/Pro'},
  };
  var html='<div class="endpoint-list">';
  FALLBACKS_DATA.forEach(function(f){
    var m=pmeta[f]||{};
    html+='<div class="endpoint"><span class="method fallback">'+f+'</span><span class="ep-url">'+f+'</span><span class="ep-detail" style="font-size:10px">→ '+m.model+'</span><span style="font-size:10px;color:var(--muted);width:100%">'+m.models+'</span></div>';
  });
  if(!FALLBACKS_DATA.length)html+='<div class="endpoint"><span style="color:var(--muted)">No fallback providers configured</span></div>';
  html+='</div>';
  document.getElementById('docsFallbacks').innerHTML=html;
})();

// ── Docs: Endpoints ──
(function(){
  var eps=[
    {m:'POST',path:'/v1/messages',d:'Anthropic-format chat completions (main endpoint)'},
    {m:'POST',path:'/zen/v1/chat/completions',d:'OpenAI-format completions targeting free models'},
    {m:'GET',path:'/v1/models',d:'List available models'},
    {m:'GET',path:'/dash',d:'Dashboard UI'},
    {m:'GET',path:'/dash/data',d:'Dashboard stats + recent logs (JSON)'},
    {m:'GET',path:'/health',d:'Health check'},
  ];
  var html='<div class="endpoint-list">';
  eps.forEach(function(ep){
    html+='<div class="endpoint"><span class="method '+ep.m.toLowerCase()+'">'+ep.m+'</span><span class="ep-url">'+ep.path+'</span><span class="ep-detail">'+ep.d+'</span></div>';
  });
  html+='</div>';
  document.getElementById('docsEndpoints').innerHTML=html;
})();

// ── Dashboard data loading ──
function n(v){return v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(1)+'K':String(v)}

async function load(){
  try{
    var r=await fetch('/dash/data?key='+K);var d=await r.json();var s=d.stats;

    document.getElementById('stats').innerHTML=
      '<div class="stat-card"><h3>Total Requests</h3><div class="value">'+s.total_requests+'</div><div class="sub">'+s.today_requests+' today</div></div>'+
      '<div class="stat-card"><h3>Input Tokens</h3><div class="value">'+n(s.total_input_tokens)+'</div><div class="sub">'+n(s.today_input_tokens)+' today</div></div>'+
      '<div class="stat-card"><h3>Output Tokens</h3><div class="value">'+n(s.total_output_tokens)+'</div><div class="sub">'+n(s.today_output_tokens)+' today</div></div>'+
      '<div class="stat-card"><h3>Cache</h3><div class="value">'+n(s.total_cache)+'</div></div>';

    document.getElementById('welcome-stats').innerHTML=
      '<div class="stat-card"><h3>Total Requests</h3><div class="value">'+s.total_requests+'</div><div class="sub">'+s.today_requests+' today</div></div>'+
      '<div class="stat-card"><h3>Input Tokens</h3><div class="value">'+n(s.total_input_tokens)+'</div><div class="sub">'+n(s.today_input_tokens)+' today</div></div>'+
      '<div class="stat-card"><h3>Output Tokens</h3><div class="value">'+n(s.total_output_tokens)+'</div><div class="sub">'+n(s.today_output_tokens)+' today</div></div>'+
      '<div class="stat-card"><h3>Cache Tokens</h3><div class="value">'+n(s.total_cache)+'</div></div>'+
      '<div class="stat-card"><h3>Models Used</h3><div class="value">'+Object.keys(s.models).length+'</div><div class="sub">'+'${Object.keys(MODELS).length}'+' configured</div></div>'+
      '<div class="stat-card"><h3>Active Providers</h3><div class="value">'+(s.providers?Object.keys(s.providers).length:'0')+'</div><div class="sub">'+(FALLBACKS_DATA.length||'0')+' configured</div></div>';

    var mc='';
    Object.keys(s.models).forEach(function(k){
      var v=s.models[k];
      mc+='<div class="model-card"><h3>'+k+'</h3><div class="detail">'+v.requests+' req</div><div class="tokens">'+n(v.input)+' in / '+n(v.output)+' out / '+n(v.cache)+' cache</div></div>';
    });
    document.getElementById('models').innerHTML=mc||'<div class="stat-card">No data yet</div>';

    var pc='';
    if(s.providers){
      Object.keys(s.providers).forEach(function(k){
        var v=s.providers[k];
        pc+='<div class="model-card"><h3>'+k+'</h3><div class="detail">'+v.requests+' req</div><div class="tokens">'+n(v.input)+' in / '+n(v.output)+' out / '+n(v.cache)+' cache</div></div>';
      });
    }
    var pel=document.getElementById('providers');
    if(pel)pel.innerHTML=pc||'<div class="stat-card">No data yet</div>';

    var l='';
    (d.recent||[]).forEach(function(r){
      var ok=r.status>=200&&r.status<400;
      var err=(r.error||'').replace(/[\\n\\r]+/g,' ').slice(0,60);
      l+='<tr><td>'+(r.timestamp||'').slice(11,19)+'</td><td>'+(r.model||'')+'</td><td>'+(r.route||'')+'</td><td>'+(r.provider||'')+'</td><td class="'+(ok?'status-ok':'status-err')+'">'+r.status+'</td><td>'+(r.duration_ms||0)+'ms</td><td>'+(r.success?n((r.tokens_input||0)+(r.tokens_output||0)):'FAIL')+'</td><td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">'+err+'</td></tr>';
    });
    document.getElementById('log-body').innerHTML=l||'<tr><td colspan="8">No logs yet</td></tr>';
  }catch(e){document.getElementById('logs-table').innerHTML='<p style=\"color:var(--red)\">Error: '+e.message+'</p>';}
}
load();
setInterval(load,5000);
</script>
</body>
</html>`;
}


app.get('/dash', (_req, res) => res.type('html').send(buildDashboardHTML(SECRET_KEY)));

app.get('/dash/data', (req, res) => {
  if (req.headers['x-api-key'] !== SECRET_KEY && req.query.key !== SECRET_KEY) return res.status(401).json({ error: 'unauthorized' });
  res.json({ stats: getUsageStats(), recent: getRecentLogs(50) });
});

app.get('/dash/stats', (req, res) => {
  if (req.headers['x-api-key'] !== SECRET_KEY && req.query.key !== SECRET_KEY) return res.status(401).json({ error: 'unauthorized' });
  const from = req.query.from_date as string;
  const to = req.query.to_date as string;
  if (from || to) {
    res.json(getFilteredStats(from, to));
  } else {
    res.json(getUsageStats());
  }
});

app.get('/dash/logs', (req, res) => {
  if (req.headers['x-api-key'] !== SECRET_KEY && req.query.key !== SECRET_KEY) return res.status(401).json({ error: 'unauthorized' });
  const limit = parseInt(req.query.limit as string || '100', 10);
  const offset = parseInt(req.query.offset as string || '0', 10);
  res.json(getLogsPaginated(limit, offset));
});

app.delete('/dash/history', (req, res) => {
  if (req.headers['x-api-key'] !== SECRET_KEY && req.query.key !== SECRET_KEY) return res.status(401).json({ error: 'unauthorized' });
  const all = req.query.all === 'true';
  const before = req.query.before as string;
  const deleted = deleteLogs(all, before);
  res.json({ deleted });
});

// ── Terminal dashboard ───────────────────────────────────────────────────────
const C = {
  accent: chalk.hex('#d97757'),
  accentBold: chalk.hex('#d97757').bold,
  muted: chalk.hex('#6b6560'),
  good: chalk.hex('#7bc47f'),
  bad: chalk.hex('#e05a5a'),
  warn: chalk.hex('#d99757'),
  info: chalk.hex('#6b8595'),
  dim: chalk.dim,
  bold: chalk.bold,
  bar: chalk.hex('#d97757').dim,
};

function printHeader() {
  console.log('');
  console.log(C.bar('  ┌──────────────────────────────────────────────────────────┐'));
  console.log(C.bar('  │') + C.accentBold('                Claude Dash — Terminal                    ') + C.bar('│'));
  console.log(C.bar('  └──────────────────────────────────────────────────────────┘'));
  console.log('');
}

function statusColor(s: number): string {
  if (s >= 200 && s < 300) return C.good(String(s));
  if (s >= 300 && s < 400) return C.warn(String(s));
  if (s >= 400 && s < 500) return C.bad(String(s));
  return C.bad(String(s));
}

function printStats() {
  try {
    const data = getDashboardData();
    const s = data.stats;
    const pad = '  ';

    console.log(pad + C.muted('Requests') + '  ' + C.bold(String(s.total_requests)) + C.dim(` (${s.today_requests} today)`));
    console.log(pad + C.muted('Input') + '    ' + C.bold(fmt(s.total_input_tokens)) + C.dim(` (${fmt(s.today_input_tokens)} today)`));
    console.log(pad + C.muted('Output') + '   ' + C.bold(fmt(s.total_output_tokens)) + C.dim(` (${fmt(s.today_output_tokens)} today)`));
    console.log(pad + C.muted('Cache') + '    ' + C.bold(fmt(s.total_cache)));

    const keys = Object.keys(s.models);
    if (keys.length) {
      console.log('');
      console.log(pad + C.accent.underline('Top Models'));
      for (const k of keys.slice(0, 5)) {
        const v = s.models[k];
        console.log(pad + C.dim('│') + C.bold(' ' + k) + C.dim(` — ${v.requests} req · ${fmt(v.input)} in · ${fmt(v.output)} out`));
      }
    }

    if (s.providers) {
      const pk = Object.keys(s.providers);
      if (pk.length) {
        console.log('');
        console.log(pad + C.accent.underline('Providers'));
        for (const k of pk.slice(0, 7)) {
          const v = s.providers[k];
          const dot = v.requests > 0 ? C.good('●') : C.muted('○');
          console.log(pad + dot + C.bold(' ' + k) + C.dim(` — ${v.requests} req · ${fmt(v.input)} in · ${fmt(v.output)} out`));
        }
      }
    }

    const recent = data.recent?.slice(0, 5) || [];
    if (recent.length) {
      console.log('');
      console.log(pad + C.accent.underline('Recent Requests'));
      for (const r of recent) {
        const time = C.dim(r.timestamp?.slice(11, 19) || '--:--:--');
        const model = C.bold(r.model || '-');
        const arrow = C.dim('→');
        const prov = r.provider || '-';
        const st = statusColor(r.status);
        const dur = C.dim(`${r.duration_ms}ms`);
        const err = r.error ? C.dim(` ${r.error.replace(/\n/g, ' ').slice(0, 50)}`) : '';
        console.log(pad + time + ' ' + model + ' ' + arrow + ' ' + prov + ' [' + st + '] ' + dur + err);
      }
    }
  } catch {}
  console.log('');
}

function fmt(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

// ── Shutdown handlers ─────────────────────────────────────────────────────────
function shutdown() {
  console.log(C.dim('  Shutting down...'));
  closeDB();
  process.exit(0);
}
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

// ── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, HOST, () => {
  printHeader();
  const p = '  ';
  console.log(p + C.muted('API') + '      ' + C.accent(`http://${HOST}:${PORT}`));
  console.log(p + C.muted('Dash') + '     ' + C.accent(`http://${HOST}:${PORT}/dash`));
  console.log(p + C.muted('Models') + '    ' + C.bold(String(Object.keys(MODELS).length)) + C.dim(' configured'));
  const fb = FALLBACK_PROVIDERS.map(f => f.name).join(', ') || C.dim('none');
  console.log(p + C.muted('Chain') + '    ' + C.info(fb));
  console.log('');
  setInterval(printStats, 30000);
  printStats();
});
