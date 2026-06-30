import Database from 'better-sqlite3';
import path from 'path';

const dbPath = path.join(process.cwd(), 'proxy.db');

let db: Database.Database;

export interface LogEntry {
  id: number; timestamp: string; request_id: string | null;
  model: string | null; original_model: string | null;
  route: string | null; provider: string | null;
  protocol: string | null; is_stream: number;
  thinking: string | null; effort: string | null;
  status: number; duration_ms: number;
  tokens_input: number; tokens_output: number; tokens_cache: number;
  success: number; error: string | null; ip: string | null;
}

export interface UsageStats {
  total_requests: number; today_requests: number;
  total_input_tokens: number; total_output_tokens: number;
  total_cache: number;
  today_input_tokens: number; today_output_tokens: number;
  models: Record<string, { requests: number; input: number; output: number; cache: number }>;
  providers: Record<string, { requests: number; input: number; output: number; cache: number }>;
}

export interface DashboardData { stats: UsageStats; recent: LogEntry[]; }

export function initDB() {
  db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  db.exec(`CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    request_id TEXT,
    model TEXT,
    original_model TEXT,
    route TEXT,
    provider TEXT,
    protocol TEXT,
    is_stream INTEGER DEFAULT 0,
    thinking TEXT,
    effort TEXT,
    status INTEGER,
    duration_ms INTEGER,
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    tokens_cache INTEGER DEFAULT 0,
    success INTEGER DEFAULT 1,
    error TEXT,
    ip TEXT
  )`);
  for (const col of ['route', 'provider', 'protocol', 'is_stream', 'thinking', 'effort']) {
    try { db.exec(`ALTER TABLE requests ADD COLUMN ${col} TEXT`); } catch {}
  }
}

export function logRequest(params: {
  request_id?: string; model: string; original_model?: string;
  route?: string; provider?: string; protocol?: string;
  is_stream?: boolean; thinking?: string; effort?: string;
  status: number; duration_ms: number;
  tokens_input?: number; tokens_output?: number; tokens_cache?: number;
  success?: boolean; error?: string; ip?: string;
}) {
  try {
    db.prepare(`INSERT INTO requests (request_id, model, original_model, route, provider, protocol, is_stream, thinking, effort, status, duration_ms, tokens_input, tokens_output, tokens_cache, success, error, ip) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`).run(
      params.request_id || null, params.model, params.original_model || null,
      params.route || null, params.provider || null, params.protocol || null,
      params.is_stream ? 1 : 0, params.thinking || null, params.effort || null,
      params.status, params.duration_ms,
      params.tokens_input || 0, params.tokens_output || 0, params.tokens_cache || 0,
      params.success !== false ? 1 : 0, params.error || null, params.ip || null,
    );
  } catch {}
}

export function getFilteredStats(from_date?: string, to_date?: string): UsageStats {
  const stats: UsageStats = {
    total_requests: 0, today_requests: 0,
    total_input_tokens: 0, total_output_tokens: 0, total_cache: 0,
    today_input_tokens: 0, today_output_tokens: 0,
    models: {}, providers: {},
  };
  try {
    const conditions: string[] = [];
    const params: string[] = [];
    if (from_date) { conditions.push('timestamp >= ?'); params.push(from_date); }
    if (to_date) { conditions.push('timestamp <= ?'); params.push(to_date + ' 23:59:59'); }
    const where = conditions.length ? ' WHERE ' + conditions.join(' AND ') : '';

    const totals = db.prepare(`SELECT COUNT(*) as count, COALESCE(SUM(tokens_input),0) as inp, COALESCE(SUM(tokens_output),0) as out, COALESCE(SUM(tokens_cache),0) as cc FROM requests${where}`).get(...params) as any;
    if (totals) { stats.total_requests = totals.count; stats.total_input_tokens = totals.inp; stats.total_output_tokens = totals.out; stats.total_cache = totals.cc; }

    const today = db.prepare(`SELECT COUNT(*) as count, COALESCE(SUM(tokens_input),0) as inp, COALESCE(SUM(tokens_output),0) as out FROM requests WHERE date(timestamp) = date('now')`).get() as any;
    if (today) { stats.today_requests = today.count; stats.today_input_tokens = today.inp; stats.today_output_tokens = today.out; }

    const perModel = db.prepare(`SELECT model, COUNT(*) as count, COALESCE(SUM(tokens_input),0) as inp, COALESCE(SUM(tokens_output),0) as out, COALESCE(SUM(tokens_cache),0) as cc FROM requests${where} GROUP BY model ORDER BY count DESC`).all(...params) as any[];
    for (const m of perModel) {
      if (m.model) stats.models[m.model] = { requests: m.count, input: m.inp, output: m.out, cache: m.cc };
    }

    const providerWhere = conditions.length ? conditions.concat(["provider IS NOT NULL"]).join(' AND ') : "provider IS NOT NULL";
    const perProvider = db.prepare(`SELECT provider, COUNT(*) as count, COALESCE(SUM(tokens_input),0) as inp, COALESCE(SUM(tokens_output),0) as out, COALESCE(SUM(tokens_cache),0) as cc FROM requests WHERE ${providerWhere} GROUP BY provider ORDER BY count DESC`).all(...params) as any[];
    stats.providers = {};
    for (const p of perProvider) {
      if (p.provider) stats.providers[p.provider] = { requests: p.count, input: p.inp, output: p.out, cache: p.cc };
    }
  } catch {}
  return stats;
}

export function getUsageStats(): UsageStats {
  const stats: UsageStats = {
    total_requests: 0, today_requests: 0,
    total_input_tokens: 0, total_output_tokens: 0, total_cache: 0,
    today_input_tokens: 0, today_output_tokens: 0,
    models: {}, providers: {},
  };
  try {
    const totals = db.prepare(`SELECT COUNT(*) as count, COALESCE(SUM(tokens_input),0) as inp, COALESCE(SUM(tokens_output),0) as out, COALESCE(SUM(tokens_cache),0) as cc FROM requests`).get() as any;
    if (totals) { stats.total_requests = totals.count; stats.total_input_tokens = totals.inp; stats.total_output_tokens = totals.out; stats.total_cache = totals.cc; }

    const today = db.prepare(`SELECT COUNT(*) as count, COALESCE(SUM(tokens_input),0) as inp, COALESCE(SUM(tokens_output),0) as out FROM requests WHERE date(timestamp) = date('now')`).get() as any;
    if (today) { stats.today_requests = today.count; stats.today_input_tokens = today.inp; stats.today_output_tokens = today.out; }

    const perModel = db.prepare(`SELECT model, COUNT(*) as count, COALESCE(SUM(tokens_input),0) as inp, COALESCE(SUM(tokens_output),0) as out, COALESCE(SUM(tokens_cache),0) as cc FROM requests GROUP BY model ORDER BY count DESC`).all() as any[];
    for (const m of perModel) {
      if (m.model) stats.models[m.model] = { requests: m.count, input: m.inp, output: m.out, cache: m.cc };
    }

    const perProvider = db.prepare(`SELECT provider, COUNT(*) as count, COALESCE(SUM(tokens_input),0) as inp, COALESCE(SUM(tokens_output),0) as out, COALESCE(SUM(tokens_cache),0) as cc FROM requests WHERE provider IS NOT NULL GROUP BY provider ORDER BY count DESC`).all() as any[];
    stats.providers = {};
    for (const p of perProvider) {
      if (p.provider) stats.providers[p.provider] = { requests: p.count, input: p.inp, output: p.out, cache: p.cc };
    }
  } catch {}
  return stats;
}

export function getRecentLogs(limit = 100): LogEntry[] {
  try {
    return db.prepare(`SELECT * FROM requests ORDER BY id DESC LIMIT ?`).all(limit) as LogEntry[];
  } catch { return []; }
}

export function getLogsPaginated(limit = 100, offset = 0): { logs: LogEntry[]; has_more: boolean } {
  try {
    const logs = db.prepare(`SELECT * FROM requests ORDER BY id DESC LIMIT ? OFFSET ?`).all(limit, offset) as LogEntry[];
    const total = (db.prepare(`SELECT COUNT(*) as count FROM requests`).get() as any)?.count || 0;
    return { logs, has_more: offset + limit < total };
  } catch { return { logs: [], has_more: false }; }
}

export function deleteLogs(all?: boolean, before?: string): number {
  try {
    if (all) {
      const r = db.prepare(`DELETE FROM requests`).run();
      return r.changes;
    }
    if (before) {
      const r = db.prepare(`DELETE FROM requests WHERE timestamp < ?`).run(before);
      return r.changes;
    }
    return 0;
  } catch { return 0; }
}

export function getDashboardData(): DashboardData {
  return { stats: getUsageStats(), recent: getRecentLogs(50) };
}

export function closeDB() {
  try {
    db.pragma('wal_checkpoint(TRUNCATE)');
    db.close();
  } catch {}
}
