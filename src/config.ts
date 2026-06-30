import fs from 'fs';
import path from 'path';

function loadEnv() {
  const envPath = path.join(process.cwd(), '.env');
  if (fs.existsSync(envPath)) {
    for (const line of fs.readFileSync(envPath, 'utf-8').split('\n')) {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith('#') && trimmed.includes('=')) {
        const idx = trimmed.indexOf('=');
        const key = trimmed.slice(0, idx).trim();
        const val = trimmed.slice(idx + 1).trim();
        if (!process.env[key]) process.env[key] = val;
      }
    }
  }
}

function loadClaudeSettings() {
  const settingsPath = path.join(process.env.HOME || '/root', '.claude', 'settings.json');
  if (fs.existsSync(settingsPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
      for (const [k, v] of Object.entries(data.env || {})) {
        if (!process.env[k]) process.env[k] = String(v);
      }
    } catch {}
  }
}

loadEnv();
loadClaudeSettings();

const _anthroBase = (process.env.ANTHROPIC_BASE_URL || 'https://opencode.ai').replace(/\/+$/, '');

export const SECRET_KEY = process.env.SECRET_KEY || 'ryanisyourpapa-nometterwhat';
export const API_KEY = process.env.OPENCODE_API_KEY || process.env.ANTHROPIC_API_KEY || '';
export const PROXY = process.env.OPENCODE_PROXY || '';
export const HOST = process.env.HOST || process.env.OPENCODE_HOST || '0.0.0.0';
export const PORT = parseInt(process.env.PORT || process.env.OPENCODE_PORT || '4000', 10);

export const API_BASE_OPENAI = `${_anthroBase}/zen/go/v1/chat/completions`;
export const API_BASE_ANTHROPIC = `${_anthroBase}/zen/go/v1/messages`;
export const API_BASE_ZEN_V1 = `${_anthroBase}/zen/v1/chat/completions`;

export interface ModelConfig {
  endpoint: string;
  protocol: 'openai' | 'anthropic';
}

export const MODELS: Record<string, ModelConfig> = {
  'glm-5.2':                { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'glm-5.1':                { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'glm-5':                  { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'kimi-k2.5':              { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'kimi-k2.6':              { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'kimi-k2.7':              { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'deepseek-v4-pro':        { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'deepseek-v4-flash':      { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'mimo-v2-pro':            { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'mimo-v2-omni':           { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'mimo-v2.5-pro':          { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'mimo-v2.5':              { endpoint: API_BASE_OPENAI,    protocol: 'openai' },
  'deepseek-v4-flash-free': { endpoint: API_BASE_ZEN_V1,    protocol: 'openai' },
  'mimo-v2.5-free':         { endpoint: API_BASE_ZEN_V1,    protocol: 'openai' },
  'nemotron-3-ultra-free':  { endpoint: API_BASE_ZEN_V1,    protocol: 'openai' },
  'north-mini-code-free':   { endpoint: API_BASE_ZEN_V1,    protocol: 'openai' },
  'minimax-m3':             { endpoint: API_BASE_ANTHROPIC, protocol: 'anthropic' },
  'minimax-m2.7':           { endpoint: API_BASE_ANTHROPIC, protocol: 'anthropic' },
  'minimax-m2.5':           { endpoint: API_BASE_ANTHROPIC, protocol: 'anthropic' },
  'qwen3.7-max':            { endpoint: API_BASE_ANTHROPIC, protocol: 'anthropic' },
  'qwen3.7-plus':           { endpoint: API_BASE_ANTHROPIC, protocol: 'anthropic' },
  'qwen3.6-plus':           { endpoint: API_BASE_ANTHROPIC, protocol: 'anthropic' },
  'qwen3.5-plus':           { endpoint: API_BASE_ANTHROPIC, protocol: 'anthropic' },
};

export const NO_MULTIMODAL = new Set(['glm-5.1', 'glm-5']);

export interface RouteConfig {
  match: string[];
  model: string;
}

const opusModel   = process.env.OPUS_MAP_MODEL   || 'deepseek-v4-flash-free';
const sonnetModel = process.env.SONNET_MAP_MODEL  || 'mimo-v2.5-free';
const haikuModel  = process.env.HAIKU_MAP_MODEL   || 'north-mini-code-free';

export const ROUTES: Record<string, RouteConfig> = {
  opus:   { match: ['opus', 'claude-opus-4-8', 'claude-opus-4-6', 'claude-opus-3-5'],   model: opusModel },
  sonnet: { match: ['sonnet', 'claude-sonnet-4-6', 'claude-sonnet-4-8', 'claude-sonnet-3-5'], model: sonnetModel },
  haiku:  { match: ['haiku', 'claude-haiku-4-5', 'claude-3-haiku', 'claude-3-5-haiku'],  model: haikuModel },
};

export function getModelConfig(modelId: string): ModelConfig {
  return MODELS[modelId] || { endpoint: API_BASE_OPENAI, protocol: 'openai' };
}

export function routeFor(modelName: string): RouteConfig {
  const name = modelName.toLowerCase().trim();
  if (!name) return ROUTES.sonnet;
  // Direct model key match - use it as-is
  if (MODELS[modelName]) {
    return { match: [modelName], model: modelName };
  }
  // Fuzzy match against route patterns
  for (const r of Object.values(ROUTES)) {
    if (r.match.some(m => name.includes(m))) return r;
  }
  return ROUTES.sonnet;
}

export interface FallbackProvider {
  name: string;
  apiKey: string;
  baseUrl: string;
  chatEndpoint: string;
}

export function loadFallbackProviders(): FallbackProvider[] {
  const raw = process.env.FALLBACK_PROVIDERS;
  if (raw) {
    try {
      return JSON.parse(raw).map((fb: any) => {
        const base = (fb.base_url || '').replace(/\/+$/, '');
        return { name: fb.provider || 'unknown', apiKey: fb.api_key || '', baseUrl: base, chatEndpoint: base + '/chat/completions' };
      });
    } catch {}
  }
  const providers: FallbackProvider[] = [];
  let i = 0;
  while (true) {
    const name = process.env[`OPENCODE_FALLBACK_${i}_NAME`];
    const key = process.env[`OPENCODE_FALLBACK_${i}_API_KEY`];
    const url = process.env[`OPENCODE_FALLBACK_${i}_BASE_URL`];
    if (!name || !key || !url) break;
    const base = url.replace(/\/+$/, '');
    providers.push({ name, apiKey: key, baseUrl: base, chatEndpoint: base + '/chat/completions' });
    i++;
  }
  // Working providers first
  const groqKey = process.env.GROQ_API_KEY;
  if (groqKey && !providers.some(p => p.name === 'groq')) {
    providers.push({ name: 'groq', apiKey: groqKey, baseUrl: 'https://api.groq.com/openai/v1', chatEndpoint: 'https://api.groq.com/openai/v1/chat/completions' });
  }
  const openrouterKey = process.env.OPENROUTER_KEY;
  if (openrouterKey && !providers.some(p => p.name === 'openrouter')) {
    providers.push({ name: 'openrouter', apiKey: openrouterKey, baseUrl: 'https://openrouter.ai/api/v1', chatEndpoint: 'https://openrouter.ai/api/v1/chat/completions' });
  }
  const cerebrasKey = process.env.CEREBRAS_KEY;
  if (cerebrasKey && !providers.some(p => p.name === 'cerebras')) {
    providers.push({ name: 'cerebras', apiKey: cerebrasKey, baseUrl: 'https://api.cerebras.ai/v1', chatEndpoint: 'https://api.cerebras.ai/v1/chat/completions' });
  }
  const githubKey = process.env.GITHUB_KEY;
  if (githubKey && !providers.some(p => p.name === 'github')) {
    providers.push({ name: 'github', apiKey: githubKey, baseUrl: 'https://models.inference.ai.azure.com', chatEndpoint: 'https://models.inference.ai.azure.com/chat/completions' });
  }
  const mistralKey = process.env.MISTRAL_KEY;
  if (mistralKey && !providers.some(p => p.name === 'mistral')) {
    providers.push({ name: 'mistral', apiKey: mistralKey, baseUrl: 'https://api.mistral.ai/v1', chatEndpoint: 'https://api.mistral.ai/v1/chat/completions' });
  }
  // Quota-exhausted providers last
  const openaiKey = process.env.OPENAI_KEY;
  if (openaiKey && !providers.some(p => p.name === 'openai')) {
    providers.push({ name: 'openai', apiKey: openaiKey, baseUrl: 'https://api.openai.com/v1', chatEndpoint: 'https://api.openai.com/v1/chat/completions' });
  }
  const geminiKey = process.env.GEMINI_KEY;
  if (geminiKey && !providers.some(p => p.name === 'gemini')) {
    providers.push({ name: 'gemini', apiKey: geminiKey, baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', chatEndpoint: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions' });
  }
  return providers;
}

export const FALLBACK_PROVIDERS = loadFallbackProviders();
