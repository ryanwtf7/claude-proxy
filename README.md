# Claude Proxy

Multi-provider API proxy with an Anthropic-compatible endpoint. Drop-in replacement for `api.anthropic.com` — routes requests through OpenCode with automatic fallback across 7 providers.

## Quick Start

```bash
cp .env.example .env   # add your API keys
npm install
npm start              # starts on http://0.0.0.0:4000
```

Point Claude Code at the proxy (`~/.claude/settings.json`):

```json
{
  "env": {
    "ANTHROPIC_API_KEY": "YOUR_API_KEY",
    "ANTHROPIC_BASE_URL": "http://localhost:4000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

## Architecture

```
Client (Claude Code)
  |  POST /v1/messages (Anthropic format)
  v
Express server
  |-- routeFor()       — map model name to backend model ID
  |-- getModelConfig() — pick endpoint + protocol (OpenAI or Anthropic)
  |-- buildChain()     — build provider fallback chain
  |-- tryChain()       — iterate with exponential backoff (1s, 2s, 4s)
  |     |-- fetch from OpenCode (primary)
  |     |-- on failure: fall through to Groq, OpenRouter, Cerebras,
  |     |   GitHub Models, Mistral, OpenAI, Gemini
  |     v
  |-- openaiToAnthropic() / anthropicToOpenAI() — format conversion
  |-- logRequest()     — persist to SQLite
  v
Response (Anthropic format)
```

## Providers (fallback chain)

| # | Provider | Env Variable |
|---|----------|-------------|
| 1 | OpenCode (primary) | `OPENCODE_API_KEY` |
| 2 | Groq | `GROQ_API_KEY` |
| 3 | OpenRouter | `OPENROUTER_KEY` |
| 4 | Cerebras | `CEREBRAS_KEY` |
| 5 | GitHub Models | `GITHUB_KEY` |
| 6 | Mistral | `MISTRAL_KEY` |
| 7 | OpenAI | `OPENAI_KEY` |
| 8 | Gemini | `GEMINI_KEY` |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | — | API key clients send as `x-api-key` |
| `OPUS_MAP_MODEL` | `deepseek-v4-flash-free` | Model for `opus` requests |
| `SONNET_MAP_MODEL` | `mimo-v2.5-free` | Model for `sonnet` requests |
| `HAIKU_MAP_MODEL` | `north-mini-code-free` | Model for `haiku` requests |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `4000` | Listen port |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/messages` | Anthropic Messages API (main) |
| POST | `/anthropic/v1/messages` | Alias for `/v1/messages` |
| GET | `/v1/models` | List configured models |
| GET | `/health` | Health check |
| GET | `/dash` | Live web dashboard |
| GET | `/dash/data` | Dashboard stats (JSON) |

## Tech Stack

**Runtime:** Node.js 20+ · **Language:** TypeScript (via tsx) · **Framework:** Express 5 · **DB:** SQLite (better-sqlite3)

## License

MIT
