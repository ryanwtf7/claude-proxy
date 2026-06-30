# Claude Proxy

Multi-provider API proxy with an Anthropic-compatible endpoint. Designed primarily for use with Claude Code, but compatible with any client that speaks the Anthropic Messages API.

## Features

- **Anthropic-compatible API** -- drop-in replacement for `api.anthropic.com` via `claude-proxy` config
- **23 models** across OpenCode free and paid tiers, with automatic routing
- **7 fallback providers** -- Groq, OpenRouter, Cerebras, GitHub Models, Mistral, OpenAI, Gemini
- **Automatic fallback chain** -- if the primary model fails (auth, quota, rate limit), the request falls through to the next provider
- **Format conversion** -- transparent bidirectional conversion between Anthropic Messages format and OpenAI Chat Completions format
- **Streaming** -- full SSE streaming support with content block, thinking, and tool use deltas
- **Request logging** -- per-request tracking with model, provider, tokens, latency, and error details in SQLite
- **Web dashboard** -- live stats, per-model and per-provider breakdown, recent request log
- **Terminal dashboard** -- periodic stats printout in the running terminal
- **Token estimation** -- approximate input token counting for streaming usage display
- **Configurable model routing** -- map `opus`/`sonnet`/`haiku` to any model

## Prerequisites

- Node.js 20+
- npm
- API keys for at least OpenCode (others are optional)

## Quick Start

```bash
git clone https://github.com/ryanwtf7/claude-proxy
cd claude-proxy
cp .env.example .env
# edit .env with your API keys
npm install
npm start
```

The server starts on `http://0.0.0.0:4000` by default.

### Configure Claude Code

Point Claude Code at your local proxy:

```json
{
  "proxy": "http://localhost:4000",
  "apiKey": "YOUR_SECRET_KEY"
}
```

## Configuration

All configuration is via environment variables in `.env`:

### Required

| Variable | Description |
|---|---|
| `SECRET_KEY` | API key clients must send as `x-api-key` header |
| `OPENCODE_API_KEY` | OpenCode API key from https://opencode.ai |

### Fallback Provider Keys

| Variable | Provider | Base URL |
|---|---|---|
| `GROQ_API_KEY` | Groq | `https://api.groq.com/openai/v1` |
| `OPENROUTER_KEY` | OpenRouter | `https://openrouter.ai/api/v1` |
| `CEREBRAS_KEY` | Cerebras | `https://api.cerebras.ai/v1` |
| `GITHUB_KEY` | GitHub Models | `https://models.inference.ai.azure.com` |
| `MISTRAL_KEY` | Mistral | `https://api.mistral.ai/v1` |
| `OPENAI_KEY` | OpenAI | `https://api.openai.com/v1` |
| `GEMINI_KEY` | Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` |

### Model Route Overrides

| Variable | Default | Description |
|---|---|---|
| `OPUS_MAP_MODEL` | `deepseek-v4-flash-free` | Model used for `opus` requests |
| `SONNET_MAP_MODEL` | `mimo-v2.5-free` | Model used for `sonnet` requests |
| `HAIKU_MAP_MODEL` | `north-mini-code-free` | Model used for `haiku` requests |

### Server Config

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `4000` | Listen port |

## API Endpoints

### POST /v1/messages

Anthropic Messages API compatible endpoint. Accepts the same request body format as `api.anthropic.com/v1/messages`.

Headers:
- `x-api-key`: Your `SECRET_KEY`
- `anthropic-version`: `2023-06-01` (or any version string)
- `Content-Type`: `application/json`

### POST /anthropic/v1/messages

Alias for `/v1/messages`.

### GET /v1/models

Lists all configured models.

### GET /health

Health check endpoint.

### GET /dash

Web dashboard with live usage statistics.

## Model Routing

Route names (`opus`, `sonnet`, `haiku`) map to configurable models. Any request model that matches a route key (case-insensitive, partial match) will be routed accordingly. If the model string exactly matches a known model ID, it is used as-is.

### Free Models (OpenCode)

- `deepseek-v4-flash-free`
- `mimo-v2.5-free`
- `nemotron-3-ultra-free`
- `north-mini-code-free`

### Paid Models (OpenCode, OpenAI protocol)

- `glm-5.2`, `glm-5.1`, `glm-5`
- `kimi-k2.5`, `kimi-k2.6`, `kimi-k2.7`
- `deepseek-v4-pro`, `deepseek-v4-flash`
- `mimo-v2-pro`, `mimo-v2-omni`, `mimo-v2.5-pro`, `mimo-v2.5`

### Paid Models (Anthropic protocol)

- `minimax-m3`, `minimax-m2.7`, `minimax-m2.5`
- `qwen3.7-max`, `qwen3.7-plus`, `qwen3.6-plus`, `qwen3.5-plus`

## Fallback Chain

When the primary provider (OpenCode) returns an error, the request is retried against fallback providers in order with exponential backoff (1s, 2s, 4s):

1. OpenCode (primary)
2. Groq
3. OpenRouter
4. Cerebras
5. GitHub Models
6. Mistral
7. OpenAI
8. Gemini

Each model has a per-provider mapping defined in `MODEL_FALLBACK` that selects the closest equivalent model on each fallback provider.

## Architecture

```
Client (Claude Code)
  |  POST /v1/messages (Anthropic format)
  v
Express server
  |-- routeFor() -- map model name to model ID
  |-- getModelConfig() -- get endpoint and protocol
  |-- buildChain() -- build provider chain
  |-- tryChain() -- iterate with backoff
  |     |-- fetch from OpenCode
  |     |-- on failure: fetch from next fallback
  |     v
  |-- openaiToAnthropic() / anthropicToOpenAI() -- format conversion
  |-- logRequest() -- persist to SQLite
  v
Response (Anthropic format)
```

## Tech Stack

- **Runtime:** Node.js 20+
- **Language:** TypeScript (via tsx)
- **Framework:** Express 5
- **Database:** SQLite via better-sqlite3
- **Dev tools:** ESLint, TypeScript 6

## License

MIT
