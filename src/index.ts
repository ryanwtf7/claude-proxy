import express from 'express';
import cors from 'cors';
import crypto from 'crypto';
import chalk from 'chalk';
import { ProxyAgent } from 'undici';
import { initDB, logRequest, getDashboardData, getUsageStats, getRecentLogs, getLogsPaginated, deleteLogs, getFilteredStats, closeDB } from './database.js';
import { anthropicToOpenAI, openaiToAnthropic, estimateInputTokens, estimateTokens, getOutputTokens, extractCacheTokens, extractText, sse, forwardHeaders } from './convert.js';
import { SECRET_KEY, API_KEY, PROXY, HOST, PORT, MODELS, ROUTES, getModelConfig, routeFor, FALLBACK_PROVIDERS } from './config.js';

const proxyAgent = PROXY ? new ProxyAgent(PROXY) : undefined;

initDB();

const app = express();
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.set('trust proxy', true);

const PROVIDER_CONTEXT_LIMITS: Record<string, number> = {
  groq: 128000,
  openrouter: 200000,
  cerebras: 128000,
  github: 8000,
  mistral: 32000,
  pollinations: 128000,
  ovhcloud: 128000,
  openai: 128000,
  gemini: 1000000,
};

function truncateMessages(messages: any[], maxTokens: number): any[] {
  if (!messages.length) return messages;
  let total = 0;
  const result: any[] = [];
  const systemMsg = messages.find((m: any) => m.role === 'system');
  const chatMsgs = messages.filter((m: any) => m.role !== 'system');
  if (systemMsg) {
    const sysTokens = estimateTokens(typeof systemMsg.content === 'string' ? systemMsg.content : JSON.stringify(systemMsg.content));
    total += sysTokens;
    result.push(systemMsg);
  }
  const keepFrom = Math.max(0, chatMsgs.length - 40);
  for (let i = keepFrom; i < chatMsgs.length; i++) {
    const msg = chatMsgs[i];
    const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    const msgTokens = estimateTokens(content);
    if (total + msgTokens > maxTokens) {
      if (result.length > 1) break;
    }
    total += msgTokens;
    result.push(msg);
  }
  return result;
}
const RETRYABLE = new Set([413, 429, 502, 503, 504]);
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
  'deepseek-v4-flash-free':  { openai: 'gpt-4o-mini', gemini: 'gemini-2.0-flash', groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o-mini', mistral: 'mistral-small-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'mimo-v2.5-free':          { openai: 'gpt-4o-mini', gemini: 'gemini-2.0-flash', groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'nemotron-3-ultra-free':   { openai: 'gpt-4o-mini', gemini: 'gemini-2.0-flash', groq: 'meta-llama/llama-4-scout-17b-16e-instruct', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'north-mini-code-free':    { openai: 'gpt-4o-mini', gemini: 'gemini-2.0-flash', groq: 'llama-3.1-8b-instant', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o-mini', mistral: 'codestral-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  // Paid chat models (OpenAI protocol)
  'glm-5.2':                 { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'glm-5.1':                 { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'glm-5':                   { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'kimi-k2.5':               { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'kimi-k2.6':               { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'kimi-k2.7':               { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'deepseek-v4-pro':         { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'meta-llama/llama-4-scout-17b-16e-instruct', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'deepseek-v4-flash':       { openai: 'gpt-4o-mini',  gemini: 'gemini-2.0-flash', groq: 'llama-3.1-8b-instant', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o-mini', mistral: 'mistral-small-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'mimo-v2-pro':             { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'mimo-v2-omni':            { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'mimo-v2.5-pro':           { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'mimo-v2.5':               { openai: 'gpt-4o',       gemini: 'gemini-2.0-flash', groq: 'llama-3.3-70b-versatile', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-small-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  // Paid Anthropic protocol models
  'minimax-m3':              { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'openai/gpt-oss-120b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'minimax-m2.7':            { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'openai/gpt-oss-120b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'minimax-m2.5':            { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'openai/gpt-oss-120b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'qwen3.7-max':             { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'qwen3.7-plus':            { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'qwen3.6-plus':            { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
  'qwen3.5-plus':            { openai: 'gpt-4o',       gemini: 'gemini-2.5-pro',  groq: 'qwen/qwen3-32b', openrouter: 'auto', cerebras: 'gemma-4-31b', github: 'gpt-4o', mistral: 'mistral-large-latest', pollinations: 'openai', ovhcloud: 'gpt-oss-20b' },
};

const DEFAULT_FALLBACKS: Record<string, string> = {
  openai: 'gpt-4o-mini',
  gemini: 'gemini-2.0-flash',
  groq: 'llama-3.3-70b-versatile',
  openrouter: 'auto',
  cerebras: 'gemma-4-31b',
  github: 'gpt-4o-mini',
  mistral: 'mistral-small-latest',
  pollinations: 'openai',
  ovhcloud: 'gpt-oss-20b',
};

// Better models for opus-tier requests
const OPUS_FALLBACKS: Record<string, string> = {
  openai: 'gpt-4o',
  gemini: 'gemini-2.5-pro',
  groq: 'llama-3.3-70b-versatile',
  openrouter: 'auto',
  cerebras: 'gemma-4-31b',
  github: 'gpt-4o',
  mistral: 'mistral-large-latest',
  pollinations: 'openai',
  ovhcloud: 'Meta-Llama-3_3-70B-Instruct',
};

// Mid-range models for sonnet-tier requests
const SONNET_FALLBACKS: Record<string, string> = {
  openai: 'gpt-4o-mini',
  gemini: 'gemini-2.0-flash',
  groq: 'llama-3.3-70b-versatile',
  openrouter: 'auto',
  cerebras: 'gemma-4-31b',
  github: 'gpt-4o',
  mistral: 'mistral-small-latest',
  pollinations: 'openai',
  ovhcloud: 'Mistral-Small-3.2-24B-Instruct-2506',
};

// Build provider chain: [OpenCode, ...fallbacks]
function stripReasoning(messages: any[]): any[] {
  return messages.map((m: any) => {
    if (m.reasoning_content) {
      const { reasoning_content, ...rest } = m;
      return rest;
    }
    return m;
  });
}

const PROVIDERS_WITH_REASONING = new Set(['openai', 'gemini', 'openrouter']);

function modelForProvider(provider: string, routeTier: Record<string, string>, origModel: string): string {
  return routeTier[provider] || DEFAULT_FALLBACKS[provider] || (MODEL_FALLBACK[origModel]?.[provider]) || origModel;
}

function buildChain(endpoint: string, isAnthropicProto: boolean, body: any, routeKey?: string) {
  const chain: { name: string; url: string; key: string; body: any; headers: Record<string, string> }[] = [];

  const fbBody = isAnthropicProto ? anthropicToOpenAI(body, body.model) : { ...body };
  const routeTier = routeKey === 'opus' ? OPUS_FALLBACKS : routeKey === 'sonnet' ? SONNET_FALLBACKS : DEFAULT_FALLBACKS;

  // Check if a primary provider is configured for this model
  const modelCfg = getModelConfig(fbBody.model);
  const primary = modelCfg.primary;
  let primaryUsed = false;

  if (primary) {
    const fb = FALLBACK_PROVIDERS.find(p => p.name === primary);
    if (fb) {
      const fbModel = modelForProvider(fb.name, routeTier, fbBody.model);
      const ctxLimit = PROVIDER_CONTEXT_LIMITS[fb.name] || 128000;
      const fbBodyClone = { ...fbBody, model: fbModel, max_tokens: Math.min(fbBody.max_tokens || 8000, 8000) };
      if (!PROVIDERS_WITH_REASONING.has(fb.name) && fbBodyClone.messages) {
        fbBodyClone.messages = stripReasoning(fbBodyClone.messages);
      }
      if (fbBodyClone.messages) {
        fbBodyClone.messages = truncateMessages(fbBodyClone.messages, ctxLimit);
      }
      chain.push({ name: fb.name, url: fb.chatEndpoint, key: fb.apiKey, body: fbBodyClone, headers: { 'Authorization': `Bearer ${fb.apiKey}`, 'Content-Type': 'application/json' } });
      primaryUsed = true;
    } else {
      console.warn(`  ${C.warn('!')} primary provider "${primary}" not found in FALLBACK_PROVIDERS, falling back to opencode`);
    }
  }

  if (!primaryUsed) {
    // Default: try opencode first
    const opencodeHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
    if (isAnthropicProto) {
      opencodeHeaders['x-api-key'] = API_KEY;
      opencodeHeaders['anthropic-version'] = '2023-06-01';
    } else {
      opencodeHeaders['Authorization'] = `Bearer ${API_KEY}`;
    }
    if (PROXY) opencodeHeaders['OpenCode-Proxy'] = PROXY;
    const opencodeBody = { ...body };
    if (!opencodeBody.system) {
      opencodeBody.system = 'You are a helpful assistant. Always respond in English.';
    }
    chain.push({ name: `opencode`, url: endpoint, key: API_KEY, body: opencodeBody, headers: opencodeHeaders });
  }

  for (const fb of FALLBACK_PROVIDERS) {
    if (primary && fb.name === primary) continue;
    const fbModel = modelForProvider(fb.name, routeTier, fbBody.model);
    const ctxLimit = PROVIDER_CONTEXT_LIMITS[fb.name] || 128000;
    const fbBodyClone = { ...fbBody, model: fbModel, max_tokens: Math.min(fbBody.max_tokens || 8000, 8000) };
    if (!PROVIDERS_WITH_REASONING.has(fb.name) && fbBodyClone.messages) {
      fbBodyClone.messages = stripReasoning(fbBodyClone.messages);
    }
    if (fbBodyClone.messages) {
      fbBodyClone.messages = truncateMessages(fbBodyClone.messages, ctxLimit);
    }
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
  parallelCount?: number,
): Promise<T> {
  // Fire first `parallelCount` providers simultaneously, take the first success
  if (parallelCount && parallelCount > 1 && chain.length > 1) {
    const batch = chain.slice(0, parallelCount);
    const rest = chain.slice(parallelCount);
    const errors: { name: string; fatal: boolean; msg: string }[] = [];

    const results = await Promise.allSettled(batch.map(async (item) => {
      const { name, url, body, headers } = item;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 25000);
      try {
        const fetchOpts: any = { method: 'POST', headers, body: JSON.stringify(body), signal: controller.signal };
        if (proxyAgent) fetchOpts.dispatcher = proxyAgent;
        const resp = await fetch(url, fetchOpts);
        clearTimeout(timeout);
        if (resp.ok) {
          const respText = isStreaming ? '' : await resp.text();
          return { status: 'ok' as const, result: { text: respText, headers: resp.headers, status: resp.status, name, resp: isStreaming ? resp : undefined } };
        }
        const respText = await resp.text().catch(() => '');
        const errMsg = extractErrMsg(resp, respText);
        const errClass = classifyError(resp.status, respText);
        console.log(`  ${chalk.dim(name)} ${chalk.dim('→')} ${errClass === 'fatal' ? C.bad(resp.status) : C.warn(resp.status)} ${chalk.dim(`(${errClass})`)}: ${chalk.dim(errMsg.slice(0, 100))}`);
        logRequest({ request_id: opts.rid, model: body.model, original_model: opts.origModel, route: opts.routeKey, provider: name, protocol: opts.protocol, is_stream: opts.isStream, thinking: opts.thinking, effort: opts.effort, status: resp.status, duration_ms: elapsed(opts.start), error: errMsg, ip: opts.ip });
        return { status: 'error' as const, name, fatal: errClass === 'fatal', msg: errMsg };
      } catch (e: any) {
        clearTimeout(timeout);
        console.log(`  ${chalk.dim(name)} ${C.bad('✗')} ${chalk.dim(e.message)}`);
        logRequest({ request_id: opts.rid, model: body.model, original_model: opts.origModel, route: opts.routeKey, provider: name, protocol: opts.protocol, is_stream: opts.isStream, thinking: opts.thinking, effort: opts.effort, status: 502, duration_ms: elapsed(opts.start), error: e.message, ip: opts.ip });
        return { status: 'error' as const, name, fatal: false, msg: e.message };
      }
    }));

    for (const r of results) {
      if (r.status === 'fulfilled' && r.value.status === 'ok') {
        return await onOk(r.value.result);
      }
      if (r.status === 'fulfilled' && r.value.status === 'error' && r.value.fatal) {
        // All providers in this batch errored fatally? Only abort if ALL are fatal
      }
    }

    // All parallel failed — continue sequentially with remaining providers
    const allFailed = errors.length === batch.length;
    if (!rest.length) return onAllFailed();
    chain = rest;
  }

  // Sequential fallback for remaining providers
  for (let i = 0; i < chain.length; i++) {
    const { name, url, body, headers } = chain[i];
    const backoff = i > 0 ? Math.min(1000 * Math.pow(2, i - 1), 4000) : 0;
    if (backoff) await sleep(backoff);

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 25000);
      const fetchOpts: any = { method: 'POST', headers, body: JSON.stringify(body), signal: controller.signal };
      if (proxyAgent) fetchOpts.dispatcher = proxyAgent;
      const resp = await fetch(url, fetchOpts);
      clearTimeout(timeout);

      const retryAfter = resp.headers.get('retry-after');
      if (retryAfter && !resp.ok) {
        const secs = parseInt(retryAfter, 10);
        if (!isNaN(secs) && secs > 0 && secs <= 120) {
          console.log(`  ${chalk.dim(name)} ${C.warn('→')} ${chalk.dim(`retry-after: ${secs}s`)}`);
          await sleep(Math.min(secs * 1000, 30000));
        }
      }

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
      logRequest({ request_id: opts.rid, model: body.model, original_model: opts.origModel, route: opts.routeKey, provider: name, protocol: opts.protocol, is_stream: opts.isStream, thinking: opts.thinking, effort: opts.effort, status: resp.status, duration_ms: elapsed(opts.start), error: errMsg, ip: opts.ip });

      if (errClass === 'fatal' || i >= chain.length - 1) {
        return onAllFailed();
      }
      continue;
    } catch (e: any) {
      console.log(`  ${chalk.dim(name)} ${C.bad('✗')} ${chalk.dim(e.message)}`);
      logRequest({ request_id: opts.rid, model: body.model, original_model: opts.origModel, route: opts.routeKey, provider: name, protocol: opts.protocol, is_stream: opts.isStream, thinking: opts.thinking, effort: opts.effort, status: 502, duration_ms: elapsed(opts.start), error: e.message, ip: opts.ip });
      if (i < chain.length - 1) continue;
      return onAllFailed();
    }
  }
  return onAllFailed();
}

export const ROUTE_PARALLEL: Record<string, number> = {
  opus: 3,
  sonnet: 2,
  haiku: 0,
};

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
    const chain = buildChain(endpoint, true, reqBody, route.match[0]);
    const parallel = ROUTE_PARALLEL[route.match[0]] || 0;

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
      }, false, parallel);
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
      }, true, parallel);
    }
  } else {
    // ── OpenAI protocol ─────────────────────────────────────────
    const oaiBody = anthropicToOpenAI(reqBody, modelId);
    const chain = buildChain(endpoint, false, oaiBody, route.match[0]);
    const parallel = ROUTE_PARALLEL[route.match[0]] || 0;

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
      }, false, parallel);
    } else {
      const estInput = estimateInputTokens(reqBody);
      const streamBody = { ...oaiBody, stream: true, stream_options: { include_usage: true } };

      // Rebuild chain with streaming body — preserve per-provider model mapping,
      // strip reasoning for providers that don't support it, cap tokens for fallbacks
      const MAX_FALLBACK_TOKENS = 8000;
      const streamChain = chain.map(c => {
        const body: any = { ...streamBody, model: c.body.model, max_tokens: Math.min(streamBody.max_tokens || MAX_FALLBACK_TOKENS, MAX_FALLBACK_TOKENS) };
        if (!PROVIDERS_WITH_REASONING.has(c.name) && body.messages) {
          body.messages = stripReasoning(body.messages);
        }
        return { ...c, body };
      });

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
      }, true, parallel);
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
    <button type="button" class="tab-btn active" data-view="welcome">Status</button>
    <button type="button" class="tab-btn" data-view="logs">Logs</button>
    <button type="button" class="tab-btn" data-view="docs">Docs</button>
  </div>
</nav>

<main id="welcome" class="view active">
  <section class="welcome-wrap">
    <div class="welcome-badge">SYSTEM STATUS</div>
    <div id="welcome-grid" style="text-align:left"></div>
    <div id="provider-status" style="margin-top:20px;text-align:left"></div>
    <div id="recent-activity" style="margin-top:20px;text-align:left"></div>
  </section>
</main>

<main id="logs" class="view">
  <section class="logs-wrap">
    <div class="stats-grid" id="stats"></div>
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:14px;flex-wrap:wrap">
      <h2 style="margin:0">Recent Requests</h2>
      <input id="log-filter" type="text" placeholder="Filter by model, provider, status..." style="flex:1;min-width:180px;padding:8px 12px;background:var(--bg2);border:1px solid var(--border);color:var(--text);font:400 13px var(--font-sans);border-radius:6px;outline:none">
      <span id="log-count" style="font-size:12px;color:var(--muted)">0 records</span>
      <button id="log-refresh" class="btn" style="padding:6px 12px;font-size:12px">Refresh</button>
    </div>
    <div id="logs-table"><div class="table-wrap"><table><thead><tr><th>Time</th><th>Model</th><th>Route</th><th>Provider</th><th>Status</th><th>Duration</th><th>In</th><th>Out</th><th>Error</th></tr></thead><tbody id="log-body"></tbody></table></div></div>
    <div style="display:flex;gap:8px;justify-content:center;margin-top:12px;align-items:center">
      <button id="page-prev" class="btn" style="padding:5px 10px;font-size:11px">← Prev</button>
      <span id="page-info" style="font-size:12px;color:var(--muted)">Page 1</span>
      <button id="page-next" class="btn" style="padding:5px 10px;font-size:11px">Next →</button>
    </div>
    <h2>Top Models</h2>
    <div class="model-grid" id="models"></div>
    <h2>Providers</h2>
    <div class="model-grid" id="providers"></div>
  </section>
</main>

<main id="docs" class="view">
  <article class="docs-page">
    <p class="eyebrow">Documentation</p>
    <h2>Claude Dash</h2>
    <p>Drop-in replacement for the Anthropic <code>/v1/messages</code> API. Sends Claude-format requests to OpenCode and falls back through 7 providers.</p>
    <div class="code-block"><pre>Base URL: <span id="baseUrlPlaceholder">http://localhost:4000</span></pre></div>

    <h3>Authentication</h3>
    <p>Pass your API key as the <code>x-api-key</code> header:</p>
    <div class="code-block"><pre>curl -H "x-api-key: your-secret-key" -H "Content-Type: application/json" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{"model":"sonnet","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}' \\
  <span id="baseUrlPlaceholder2">http://localhost:4000</span>/v1/messages</pre></div>

    <h3>Available Models</h3>
    <p>The proxy supports <strong>${Object.keys(MODELS).length} models</strong> across three tiers.</p>
    <div id="docsModels"></div>

    <h3>Route Mappings</h3>
    <p>Short names are mapped to OpenCode model IDs:</p>
    <div id="docsRoutes"></div>

    <h3>Fallback Providers</h3>
    <p>When OpenCode fails, the chain tries each provider in order: Groq → OpenRouter → Cerebras → GitHub → Mistral → Pollinations → OVHcloud → OpenAI → Gemini.</p>
    <div id="docsFallbacks"></div>

    <h3>Request Flow</h3>
    <p>
      1. Model name matched to route → OpenCode model ID<br>
      2. Anthropic body converted to OpenAI format if needed<br>
      3. OpenCode tried first; on failure falls through providers with exponential backoff<br>
      4. Response converted back to Anthropic format and returned
    </p>

    <h3>Endpoints</h3>
    <div id="docsEndpoints"></div>
  </article>
</main>

<footer>Claude Dash &mdash; Multi-provider AI API gateway</footer>

<script>
var K='${authKey}';
var MODELS_DATA = ${modelsJson};
var ROUTES_DATA = ${routesJson};
var FALLBACKS_DATA = ${fallbacksJson};
var logPage = 0;
var logFilter = '';

document.getElementById('baseUrlPlaceholder').textContent = location.origin;
document.getElementById('baseUrlPlaceholder2').textContent = location.origin;

// ── Tab Switcher ──
document.querySelectorAll('.tab-btn').forEach(function(b){b.addEventListener('click',function(){
  var t=b.getAttribute('data-view');
  document.querySelectorAll('.tab-btn').forEach(function(x){x.classList.remove('active')});
  document.querySelectorAll('.view').forEach(function(v){v.classList.remove('active')});
  b.classList.add('active');
  document.getElementById(t).classList.add('active');
  if (t==='logs') loadLogs();
})});

// ── Category toggle ──
document.addEventListener('click', function(e){
  var btn=e.target;
  while(btn && !btn.classList.contains('category-toggle')) btn=btn.parentNode;
  if(!btn) return;
  var body=btn.nextElementSibling;
  if(body && body.classList.contains('category-body')){
    body.classList.toggle('open');
    btn.classList.toggle('open');
  }
});

// ── Docs Builders ──
(function(){
  var tiers={};
  MODELS_DATA.forEach(function(m){
    var t=m.endpoint.split('/').pop()||'other';
    if(!tiers[t])tiers[t]=[]; tiers[t].push(m);
  });
  var html='',first=true;
  Object.keys(tiers).forEach(function(t){
    var label=t==='completions'?'Paid (Zen Go)':t==='messages'?'Anthropic Protocol':'Free';
    html+='<button class="category-toggle'+(first?' open':'')+'"><span class="arrow">&#9654;</span> '+label+' ('+tiers[t].length+' models)</button>';
    html+='<div class="category-body'+(first?' open':'')+'"><div class="endpoint-list">';
    tiers[t].forEach(function(m){html+='<div class="endpoint"><span class="method '+m.protocol+'">'+m.protocol+'</span><span class="ep-url">'+m.id+'</span><span class="ep-detail">'+m.endpoint+'</span></div>';});
    html+='</div></div>'; first=false;
  });
  document.getElementById('docsModels').innerHTML=html;
})();

(function(){
  var html='<div class="endpoint-list">';
  ROUTES_DATA.forEach(function(r){html+='<div class="endpoint"><span class="ep-url">'+r.key+'</span><span class="ep-detail">→ '+r.model+'</span><span style="font-size:11px;color:var(--muted);width:100%">matches: '+r.match.join(', ')+'</span></div>';});
  html+='</div>';
  document.getElementById('docsRoutes').innerHTML=html;
})();

(function(){
  var pmeta={
    groq:{model:'llama-3.3-70b-versatile',models:'Llama 3.1/3.3, Qwen3, Llama 4 Scout'},
    openrouter:{model:'auto',models:'20+ free models'},
    cerebras:{model:'gemma-4-31b',models:'Gemma 4, ZAI-GLM 4.7'},
    github:{model:'gpt-4o-mini',models:'GPT-4o, GPT-4o-mini, Llama'},
    mistral:{model:'mistral-small-latest',models:'Mistral Small/Large, Codestral'},
    pollinations:{model:'openai',models:'GPT-OSS 20B (free, no key)'},
    ovhcloud:{model:'gpt-oss-20b',models:'Qwen3, Llama 3.3, Mistral (free, no key)'},
    openai:{model:'gpt-4o-mini',models:'GPT-4o, GPT-4o-mini'},
    gemini:{model:'gemini-2.0-flash',models:'Gemini 2.0 Flash/Pro'},
  };
  var html='<div class="endpoint-list">';
  FALLBACKS_DATA.forEach(function(f){
    var m=pmeta[f]||{};
    html+='<div class="endpoint"><span class="method fallback">'+f+'</span><span class="ep-url">'+f+'</span><span class="ep-detail">'+m.model+'</span><span style="font-size:10px;color:var(--muted);width:100%">'+m.models+'</span></div>';
  });
  if(!FALLBACKS_DATA.length)html+='<div class="endpoint"><span style="color:var(--muted)">None configured</span></div>';
  html+='</div>';
  document.getElementById('docsFallbacks').innerHTML=html;
})();

(function(){
  var eps=[
    {m:'POST',path:'/v1/messages',d:'Anthropic chat completions'},
    {m:'GET',path:'/v1/models',d:'List models'},
    {m:'GET',path:'/health',d:'Health + usage'},
    {m:'GET',path:'/dash',d:'Dashboard UI'},
    {m:'GET',path:'/dash/data',d:'Stats JSON'},
    {m:'DELETE',path:'/dash/history',d:'Delete logs'},
  ];
  var html='<div class="endpoint-list">';
  eps.forEach(function(ep){html+='<div class="endpoint"><span class="method">'+ep.m+'</span><span class="ep-url">'+ep.path+'</span><span class="ep-detail">'+ep.d+'</span></div>';});
  html+='</div>';
  document.getElementById('docsEndpoints').innerHTML=html;
})();

// ── Helpers ──
function n(v){return v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(1)+'K':String(v)}
function badge(status,label){
  var ok=status>=200&&status<400;
  var color=ok?'var(--green)':status>=400&&status<500?'var(--accent)':'var(--red)';
  return '<span style="display:inline-block;padding:1px 7px;font:600 10px var(--font-mono);background:'+color+'20;color:'+color+';border-radius:3px">'+(label||status)+'</span>';
}

// ── Load welcome data ──
async function loadWelcome(){
  try{
    var r=await fetch('/dash/data?key='+K);var d=await r.json();var s=d.stats;
    document.getElementById('welcome-grid').innerHTML=
      '<div class="stats-grid">'+
      '<div class="stat-card"><h3>Requests</h3><div class="value">'+s.total_requests+'</div><div class="sub">'+s.today_requests+' today</div></div>'+
      '<div class="stat-card"><h3>Input</h3><div class="value">'+n(s.total_input_tokens)+'</div><div class="sub">'+n(s.today_input_tokens)+' today</div></div>'+
      '<div class="stat-card"><h3>Output</h3><div class="value">'+n(s.total_output_tokens)+'</div><div class="sub">'+n(s.today_output_tokens)+' today</div></div>'+
      '<div class="stat-card"><h3>Cache</h3><div class="value">'+n(s.total_cache)+'</div><div class="sub">'+s.total_requests+' hits</div></div>'+
      '<div class="stat-card"><h3>Models</h3><div class="value">'+Object.keys(s.models).length+'</div><div class="sub">used of ${Object.keys(MODELS).length} configured</div></div>'+
      '<div class="stat-card"><h3>Providers</h3><div class="value">'+(s.providers?Object.keys(s.providers).length:'0')+'</div><div class="sub">active of '+(FALLBACKS_DATA.length||'0')+' configured</div></div>'+
      '</div>';

    var provHtml='<h2 style="margin-bottom:10px">Provider Status</h2><div class="endpoint-list">';
    var allProviders=['opencode','groq','openrouter','cerebras','github','mistral','pollinations','ovhcloud','openai','gemini'];
    allProviders.forEach(function(p){
      var v=s.providers&&s.providers[p];
      var active=v&&v.requests>0;
      var dot=active?'<span style="color:var(--green);font-size:14px">●</span>':'<span style="color:var(--muted);font-size:14px">○</span>';
      var stats=active?' '+v.requests+' req · '+n(v.input)+' in · '+n(v.output)+' out':' no activity';
      provHtml+='<div class="endpoint"><span>'+dot+'</span><span class="ep-url">'+p+'</span><span class="ep-detail" style="font-size:11px">'+stats+'</span></div>';
    });
    provHtml+='</div>';
    document.getElementById('provider-status').innerHTML=provHtml;

    var recent=s.total_requests>0&&d.recent?d.recent.slice(0,5):[];
    if(recent.length){
      var actHtml='<h2 style="margin-bottom:10px">Recent Activity</h2><div class="endpoint-list">';
      recent.forEach(function(r){
        var ok=r.status>=200&&r.status<400;
        actHtml+='<div class="endpoint"><span class="ep-url" style="font-size:11px;color:var(--muted)">'+(r.timestamp||'').slice(11,19)+'</span><span class="ep-url" style="font-size:12px">'+(r.model||'')+'</span><span style="font-size:11px;color:var(--muted)">→ '+(r.provider||'')+'</span>'+badge(r.status)+'<span class="ep-detail" style="font-size:11px">'+(r.duration_ms||0)+'ms</span></div>';
      });
      actHtml+='</div>';
      document.getElementById('recent-activity').innerHTML=actHtml;
    } else {
      document.getElementById('recent-activity').innerHTML='<div class="endpoint-list"><div class="endpoint"><span style="color:var(--muted)">No requests yet — send a test request to see activity here</span></div></div>';
    }
  }catch(e){document.getElementById('welcome-grid').innerHTML='<p style="color:var(--red)">Error loading: '+e.message+'</p>';}
}

// ── Load logs ──
async function loadLogs(){
  var limit=50;
  var offset=logPage*limit;
  try{
    var r=await fetch('/dash/logs?key='+K+'&limit='+limit+'&offset='+offset);var d=await r.json();
    var logs=d.logs||[];var hasMore=d.has_more;
    document.getElementById('page-info').textContent='Page '+(logPage+1)+(hasMore?'+':'');
    document.getElementById('page-prev').style.opacity=logPage>0?'1':'0.3';
    document.getElementById('page-next').style.opacity=hasMore?'1':'0.3';

    var filter=document.getElementById('log-filter').value.toLowerCase();
    if(filter) logs=logs.filter(function(r){return((r.model||'')+(r.provider||'')+(r.route||'')+String(r.status)).toLowerCase().indexOf(filter)>=0;});

    document.getElementById('log-count').textContent=logs.length+' records'+(filter?' filtered':'');
    var l='';
    logs.forEach(function(r){
      var ok=r.status>=200&&r.status<400;
      var err=(r.error||'').replace(/[\\n\\r]+/g,' ').slice(0,50);
      l+='<tr>'+
        '<td style="color:var(--muted);font-size:11px">'+(r.timestamp||'').slice(11,19)+'</td>'+
        '<td><span style="font-weight:600;font-size:12px">'+(r.model||'')+'</span></td>'+
        '<td style="color:var(--muted);font-size:11px">'+(r.route||'')+'</td>'+
        '<td>'+badge(r.status,r.provider||'')+'</td>'+
        '<td>'+badge(r.status)+'</td>'+
        '<td style="color:var(--muted);font-size:11px">'+(r.duration_ms||0)+'ms</td>'+
        '<td style="font-size:11px">'+n(r.tokens_input||0)+'</td>'+
        '<td style="font-size:11px">'+n(r.tokens_output||0)+'</td>'+
        '<td style="max-width:120px;overflow:hidden;text-overflow:ellipsis;color:var(--red);font-size:11px">'+err+'</td>'+
        '</tr>';
    });
    document.getElementById('log-body').innerHTML=l||'<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:20px">No logs to show</td></tr>';
  }catch(e){document.getElementById('logs-table').innerHTML='<p style="color:var(--red)">Error: '+e.message+'</p>';}
}

// ── Load all data ──
async function load(){
  try{
    var r=await fetch('/dash/data?key='+K);var d=await r.json();var s=d.stats;

    document.getElementById('stats').innerHTML=
      '<div class="stat-card"><h3>Requests</h3><div class="value">'+s.total_requests+'</div><div class="sub">'+s.today_requests+' today</div></div>'+
      '<div class="stat-card"><h3>Input</h3><div class="value">'+n(s.total_input_tokens)+'</div><div class="sub">'+n(s.today_input_tokens)+' today</div></div>'+
      '<div class="stat-card"><h3>Output</h3><div class="value">'+n(s.total_output_tokens)+'</div><div class="sub">'+n(s.today_output_tokens)+' today</div></div>'+
      '<div class="stat-card"><h3>Cache</h3><div class="value">'+n(s.total_cache)+'</div></div>';

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
    document.getElementById('providers').innerHTML=pc||'<div class="stat-card">No data yet</div>';

    loadWelcome();
  }catch(e){}
}
load();
setInterval(load,5000);

document.getElementById('log-filter').addEventListener('input',function(){logPage=0;loadLogs();});
document.getElementById('page-prev').addEventListener('click',function(){if(logPage>0){logPage--;loadLogs();}});
document.getElementById('page-next').addEventListener('click',function(){logPage++;loadLogs();});
document.getElementById('log-refresh').addEventListener('click',function(){loadLogs();});
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
};

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
console.clear();
const useTUI = process.argv.includes('--tui') || process.env.TUI === '1';
app.listen(PORT, HOST, () => {
  if (useTUI) {
    import('./tui.js').then(m => m.startTUI());
    return;
  }
  const p = '  ';
  console.log(`  ${C.accent('●')} ${C.bold('Claude Dash')} ${C.dim('running on')} ${C.accent(`http://${HOST}:${PORT}`)}`);
  console.log(`  ${C.dim('  dash')} ${C.dim('→')} ${C.muted(`http://${HOST}:${PORT}/dash`)}`);
  console.log(`  ${C.dim('  models')} ${C.dim('→')} ${C.bold(String(Object.keys(MODELS).length))} ${C.dim('· chain:')} ${C.info(FALLBACK_PROVIDERS.map(f => f.name).join(', ') || 'none')}`);
  console.log('');
  setInterval(printStats, 120000);
  printStats();
});
