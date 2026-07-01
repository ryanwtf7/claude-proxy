import blessed from 'blessed';
import { getDashboardData, getRecentLogs } from './database.js';
import { FALLBACK_PROVIDERS } from './config.js';
import { MODELS } from './config.js';

let screen: blessed.Widgets.Screen | null = null;
let statsBox: blessed.Widgets.BoxElement | null = null;
let logBox: blessed.Widgets.BoxElement | null = null;
let interval: ReturnType<typeof setInterval> | null = null;
let scrollOffset = 0;

function fmt(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

export function startTUI() {
  if (screen) return;

  screen = blessed.screen({
    smartCSR: true,
    title: 'Claude Dash — TUI',
    dockBorders: true,
    fullUnicode: true,
  });

  const titleBar = blessed.box({
    top: 0, left: 0, width: '100%', height: 1,
    content: ' Claude Dash — Terminal Dashboard ',
    style: { fg: '#d97757', bold: true },
    tags: true,
  });
  screen.append(titleBar);

  statsBox = blessed.box({
    top: 1, left: 0, width: '100%', height: '50%' as any,
    label: ' Stats ',
    border: { type: 'line' } as any,
    style: { fg: '#e8e4e0', border: { fg: '#d97757' } },
    scrollable: true,
    alwaysScroll: true,
    scrollbar: { ch: '│' } as any,
    padding: { left: 1, right: 1 },
    tags: true,
  });
  (statsBox as any).style.border.fg = '#d97757';
  screen.append(statsBox);

  logBox = blessed.box({
    top: '50%-1' as any, left: 0, width: '100%', height: '50%+1' as any,
    label: ' Recent Requests ',
    border: { type: 'line' } as any,
    style: { fg: '#e8e4e0', border: { fg: '#6b8595' } },
    scrollable: true,
    alwaysScroll: true,
    scrollbar: { ch: '│' } as any,
    padding: { left: 1, right: 1 },
    tags: true,
  });
  (logBox as any).style.border.fg = '#6b8595';
  screen.append(logBox);

  const helpBar = blessed.box({
    bottom: 0, left: 0, width: '100%', height: 1,
    content: ' {#6b6560}q:quit  j/k:scroll  g:top  G:bottom{/}',
    style: { fg: '#6b6560' },
    tags: true,
  });
  screen.append(helpBar);

  screen.key(['q', 'C-c'], () => {
    stopTUI();
    process.exit(0);
  });
  screen.key(['j', 'down'], () => {
    scrollOffset += 1;
    if (logBox) logBox.setScroll(scrollOffset);
    if (screen) screen.render();
  });
  screen.key(['k', 'up'], () => {
    scrollOffset = Math.max(0, scrollOffset - 1);
    if (logBox) logBox.setScroll(scrollOffset);
    if (screen) screen.render();
  });
  screen.key(['g'], () => {
    scrollOffset = 0;
    if (logBox) logBox.setScroll(0);
    if (screen) screen.render();
  });
  screen.key(['G'], () => {
    scrollOffset = 9999;
    if (logBox) logBox.setScroll(9999);
    if (screen) screen.render();
  });
  screen.key(['pageup'], () => {
    scrollOffset = Math.max(0, scrollOffset - 10);
    if (logBox) logBox.setScroll(scrollOffset);
    if (screen) screen.render();
  });
  screen.key(['pagedown'], () => {
    scrollOffset += 10;
    if (logBox) logBox.setScroll(scrollOffset);
    if (screen) screen.render();
  });

  refresh();
  interval = setInterval(refresh, 5000);
  screen.render();
}

function refresh() {
  if (!statsBox || !logBox || !screen) return;

  try {
    const data = getDashboardData();
    const s = data.stats;

    let statsContent = '';
    statsContent += `{bold}Requests:{/bold} ${s.total_requests}  {#6b6560}(${s.today_requests} today){/}\n`;
    statsContent += `{bold}Input:{/bold}    ${fmt(s.total_input_tokens)}  {#6b6560}(${fmt(s.today_input_tokens)} today){/}\n`;
    statsContent += `{bold}Output:{/bold}   ${fmt(s.total_output_tokens)}  {#6b6560}(${fmt(s.today_output_tokens)} today){/}\n`;
    statsContent += `{bold}Cache:{/bold}    ${fmt(s.total_cache)}\n`;

    const modelKeys = Object.keys(s.models);
    if (modelKeys.length) {
      statsContent += `\n{bold}{#d97757}Top Models{/}\n`;
      for (const k of modelKeys.slice(0, 5)) {
        const v = s.models[k];
        statsContent += ` {#6b6560}│{/} {bold}${k}{/} {#6b6560}— ${v.requests} req · ${fmt(v.input)} in · ${fmt(v.output)} out{/}\n`;
      }
    }

    const provKeys = Object.keys(s.providers || {});
    if (provKeys.length) {
      statsContent += `\n{bold}{#d97757}Providers{/}\n`;
      for (const k of provKeys.slice(0, 7)) {
        const v = s.providers![k];
        const dot = v.requests > 0 ? '{#7bc47f}●{/}' : '{#6b6560}○{/}';
        statsContent += ` ${dot} {bold}${k}{/} {#6b6560}— ${v.requests} req · ${fmt(v.input)} in · ${fmt(v.output)} out{/}\n`;
      }
    }

    statsBox.setContent(statsContent);

    const recent = data.recent?.slice(0, 20) || [];
    let logContent = '';
    if (recent.length) {
      for (const r of recent) {
        const time = r.timestamp?.slice(11, 19) || '--:--:--';
        const model = r.model || '-';
        const prov = r.provider || '-';
        const status = r.status;
        const ok = status >= 200 && status < 400;
        const st = ok ? `{#7bc47f}${status}{/}` : `{#e05a5a}${status}{/}`;
        const dur = `${r.duration_ms}ms`;
        const err = r.error ? ` {#6b6560}${r.error.replace(/\n/g, ' ').slice(0, 40)}{/}` : '';
        logContent += `{#6b6560}${time}{/} {bold}${model}{/} {#6b6560}→{/} ${prov} [${st}] {#6b6560}${dur}{/}${err}\n`;
      }
    } else {
      logContent = '{#6b6560}No requests yet{/}';
    }
    logBox.setContent(logContent);
    logBox.setScrollPerc(100);

    screen.render();
  } catch {}
}

export function stopTUI() {
  if (interval) { clearInterval(interval); interval = null; }
  if (screen) { screen.destroy(); screen = null; }
  statsBox = null;
  logBox = null;
}
