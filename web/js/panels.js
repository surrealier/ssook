/* ssook side panels & modals — Wave 5
 *
 * - TaskQueuePanel  (GET /api/tasks)
 * - LogViewerPanel  (GET /api/logs/tail)
 * - CheatsheetModal (Ctrl+1..9, ?, etc)
 *
 * Each panel is a singleton DOM node created lazily on first open.
 * Cleanup on close removes intervals so the panel costs nothing while
 * hidden.
 */

const _PANEL_CSS = `
.ssook-panel {
  position: fixed;
  top: 52px; right: 16px;
  width: 420px; max-height: 75vh;
  background: var(--background-tint-00);
  border: 1px solid var(--border-01);
  border-radius: var(--border-radius-12);
  z-index: 9990;
  display: flex; flex-direction: column;
  box-shadow: 0 8px 32px var(--shadow-02);
  overflow: hidden;
}
.ssook-panel-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.625rem 0.75rem;
  border-bottom: 1px solid var(--border-01);
  font-weight: 500;
}
.ssook-panel-body { overflow: auto; padding: 0.5rem 0.75rem; flex: 1; }
.ssook-panel-row {
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.4rem 0; border-bottom: 1px solid var(--border-01);
  font-size: 12px;
}
.ssook-panel-row:last-child { border-bottom: none; }
.ssook-panel-row .label { flex: 1; font-weight: 500; }
.ssook-panel-row .progress {
  width: 100px; height: 6px; background: var(--background-tint-02);
  border-radius: 3px; overflow: hidden;
}
.ssook-panel-row .progress > div {
  height: 100%; background: var(--action-link-05); transition: width 0.2s;
}
.ssook-panel pre {
  font-family: 'DM Mono', monospace; font-size: 11px;
  background: var(--background-tint-02); color: var(--text-04);
  padding: 0.5rem; border-radius: var(--border-radius-08);
  white-space: pre; overflow: auto; max-height: 60vh;
}
.ssook-modal-overlay {
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.55); z-index: 9991;
  display: flex; align-items: center; justify-content: center;
}
.ssook-modal {
  background: var(--background-tint-00);
  border: 1px solid var(--border-01);
  border-radius: var(--border-radius-12);
  padding: 1.25rem 1.5rem;
  width: min(560px, 90vw);
  max-height: 80vh; overflow: auto;
  box-shadow: 0 12px 36px var(--shadow-02);
}
.ssook-modal h3 { margin: 0 0 0.75rem; }
.kbd {
  display: inline-block;
  font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500;
  padding: 2px 6px; border-radius: 4px;
  background: var(--background-tint-02);
  border: 1px solid var(--border-01);
  color: var(--text-04);
}
.ssook-cheatsheet table { width: 100%; border-collapse: collapse; }
.ssook-cheatsheet td { padding: 6px 4px; font-size: 13px; }
.ssook-cheatsheet tr td:first-child { width: 130px; }
.ssook-header-pill {
  position: relative;
  background: var(--background-tint-02);
  color: var(--text-03);
  border: 1px solid var(--border-01);
  border-radius: 9999px;
  padding: 2px 10px;
  font-size: 11px;
  cursor: pointer;
}
.ssook-header-pill:hover { color: var(--text-05); }
.ssook-header-pill .count {
  display: inline-block; min-width: 14px;
  font-weight: 600; color: var(--action-link-05);
}
`;

function _injectPanelStyles() {
  if (document.getElementById('ssook-panel-style')) return;
  const s = document.createElement('style');
  s.id = 'ssook-panel-style';
  s.textContent = _PANEL_CSS;
  document.head.appendChild(s);
}

/* ── Task Queue Panel ───────────────────────────────────── */
const TaskQueuePanel = {
  _open: false,
  _interval: null,
  _pill: null,

  mountPill() {
    _injectPanelStyles();
    const host = document.querySelector('#page-actions');
    if (!host || this._pill) return;
    const pill = document.createElement('button');
    pill.className = 'ssook-header-pill';
    pill.id = 'ssook-task-pill';
    pill.title = 'Running tasks';
    pill.setAttribute('aria-label', 'Running background tasks');
    pill.innerHTML = `<span class="count">0</span> ${typeof I18n !== 'undefined' ? I18n.t('panels.tasks') : 'tasks'}`;
    pill.style.display = 'none';
    pill.onclick = () => this.toggle();
    host.appendChild(pill);
    this._pill = pill;
    // Poll header count even when panel is closed, so the pill reflects activity.
    this._headerPoll();
    setInterval(() => this._headerPoll(), 2000);
  },

  async _headerPoll() {
    if (!this._pill) return;
    try {
      const r = await API.tasks();
      const running = (r.tasks || []).filter(t => t.running);
      if (running.length === 0) {
        this._pill.style.display = 'none';
      } else {
        this._pill.style.display = '';
        this._pill.querySelector('.count').textContent = running.length;
      }
    } catch (e) { /* swallow — header pill stays as-is */ }
  },

  toggle() { this._open ? this.close() : this.open(); },

  open() {
    _injectPanelStyles();
    this._open = true;
    let panel = document.getElementById('ssook-task-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'ssook-task-panel';
      panel.className = 'ssook-panel';
      document.body.appendChild(panel);
    }
    this._render(panel);
    this._interval = setInterval(() => this._render(panel), 1000);
  },

  close() {
    this._open = false;
    if (this._interval) { clearInterval(this._interval); this._interval = null; }
    const p = document.getElementById('ssook-task-panel');
    if (p) p.remove();
  },

  async _render(panel) {
    let data;
    try { data = await API.tasks(); }
    catch (e) {
      panel.innerHTML = `<div class="ssook-panel-header"><span>Tasks</span>
        <button class="btn btn-ghost btn-sm" onclick="TaskQueuePanel.close()">${Icons.x(14)}</button></div>
        <div class="ssook-panel-body"><div class="text-muted">Server unreachable</div></div>`;
      return;
    }
    const running = (data.tasks || []).filter(t => t.running);
    const rows = running.length
      ? running.map(t => {
          const pct = t.total > 0 ? Math.round(t.progress / t.total * 100) : 0;
          return `<div class="ssook-panel-row">
            <span class="label">${t.id}</span>
            <span class="progress"><div style="width:${pct}%;"></div></span>
            <span class="text-secondary" style="min-width:34px;text-align:right;">${pct}%</span>
            <button class="btn btn-ghost btn-sm" title="Force stop" onclick="TaskQueuePanel.stop('${t.id}')">${Icons.stop(12)}</button>
          </div>
          <div class="text-secondary" style="font-size:11px;margin-bottom:0.25rem;">${t.msg || ''}</div>`;
        }).join('')
      : `<div class="text-muted" style="padding:1rem;text-align:center;">${typeof I18n !== 'undefined' ? I18n.t('panels.no_tasks') : 'No tasks running'}</div>`;
    panel.innerHTML = `<div class="ssook-panel-header">
        <span>${typeof I18n !== 'undefined' ? I18n.t('panels.tasks_title') : 'Running tasks'}</span>
        <div style="display:flex;gap:0.25rem;">
          <button class="btn btn-ghost btn-sm" onclick="TaskQueuePanel.stopAll()">${typeof I18n !== 'undefined' ? I18n.t('panels.stop_all') : 'Stop all'}</button>
          <button class="btn btn-ghost btn-sm" onclick="TaskQueuePanel.close()">${Icons.x(14)}</button>
        </div>
      </div>
      <div class="ssook-panel-body">${rows}</div>`;
  },

  async stop(id) {
    try { await fetch('/api/force-stop/' + encodeURIComponent(id), { method: 'POST' }); }
    catch (e) { App.toast('warn', 'Force-stop failed', e.message); }
  },

  async stopAll() {
    try { await fetch('/api/force-stop/all', { method: 'POST' }); }
    catch (e) { App.toast('warn', 'Force-stop all failed', e.message); }
  },
};

/* ── Log Viewer Panel ───────────────────────────────────── */
const LogViewerPanel = {
  _open: false,
  _interval: null,
  _filter: '',

  toggle() { this._open ? this.close() : this.open(); },

  open() {
    _injectPanelStyles();
    this._open = true;
    let panel = document.getElementById('ssook-log-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'ssook-log-panel';
      panel.className = 'ssook-panel';
      panel.style.width = '640px';
      document.body.appendChild(panel);
    }
    this._render(panel);
    this._interval = setInterval(() => this._render(panel), 3000);
  },

  close() {
    this._open = false;
    if (this._interval) { clearInterval(this._interval); this._interval = null; }
    const p = document.getElementById('ssook-log-panel');
    if (p) p.remove();
  },

  setFilter(v) {
    this._filter = (v || '').toLowerCase();
    const panel = document.getElementById('ssook-log-panel');
    if (panel) this._render(panel);
  },

  async _render(panel) {
    let data;
    try { data = await API.logsTail(300); }
    catch (e) {
      panel.innerHTML = `<div class="ssook-panel-header"><span>Logs</span>
        <button class="btn btn-ghost btn-sm" onclick="LogViewerPanel.close()">${Icons.x(14)}</button></div>
        <div class="ssook-panel-body"><div class="text-muted">Server unreachable</div></div>`;
      return;
    }
    const lines = (data.lines || []).filter(l => !this._filter || l.toLowerCase().includes(this._filter));
    panel.innerHTML = `<div class="ssook-panel-header">
        <span>${typeof I18n !== 'undefined' ? I18n.t('panels.logs_title') : 'Logs'} <span class="text-secondary" style="font-size:11px;">(${data.path})</span></span>
        <button class="btn btn-ghost btn-sm" onclick="LogViewerPanel.close()">${Icons.x(14)}</button>
      </div>
      <div class="ssook-panel-body">
        <input type="text" class="form-input input-normal" id="ssook-log-filter"
          placeholder="${typeof I18n !== 'undefined' ? I18n.t('panels.logs_filter_hint') : 'Filter (e.g. trace_id, ERROR, ssook.vlm)'}"
          style="margin-bottom:0.5rem;font-size:12px;"
          value="${this._filter}"
          oninput="LogViewerPanel.setFilter(this.value)">
        <pre>${lines.map(l => l.replace(/</g, '&lt;')).join('')}</pre>
      </div>`;
  },
};

/* ── Cheatsheet Modal ───────────────────────────────────── */
const CheatsheetModal = {
  _rows: () => [
    [`<span class="kbd">Ctrl</span> + <span class="kbd">1</span>..<span class="kbd">9</span>`,
      typeof I18n !== 'undefined' ? I18n.t('cheatsheet.switch_tab') : 'Switch to tab N (sidebar order)'],
    [`<span class="kbd">?</span>`,
      typeof I18n !== 'undefined' ? I18n.t('cheatsheet.open_cheatsheet') : 'Open this cheatsheet'],
    [`<span class="kbd">Esc</span>`,
      typeof I18n !== 'undefined' ? I18n.t('cheatsheet.close_overlay') : 'Close overlay / help'],
    [`<span class="kbd">Space</span>`,
      typeof I18n !== 'undefined' ? I18n.t('cheatsheet.toggle_play') : 'Toggle play/pause (Viewer)'],
    [`<span class="kbd">←</span> / <span class="kbd">→</span>`,
      typeof I18n !== 'undefined' ? I18n.t('cheatsheet.nav_image') : 'Prev / next image (Viewer)'],
  ],

  show() {
    _injectPanelStyles();
    if (document.getElementById('ssook-cheatsheet')) return;
    const overlay = document.createElement('div');
    overlay.id = 'ssook-cheatsheet';
    overlay.className = 'ssook-modal-overlay';
    overlay.onclick = (e) => { if (e.target === overlay) this.close(); };
    overlay.innerHTML = `<div class="ssook-modal ssook-cheatsheet" role="dialog" aria-labelledby="ssook-cs-title">
      <h3 id="ssook-cs-title">${typeof I18n !== 'undefined' ? I18n.t('cheatsheet.title') : 'Keyboard Shortcuts'}</h3>
      <table><tbody>
        ${this._rows().map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}
      </tbody></table>
      <div style="margin-top:1rem;text-align:right;">
        <button class="btn btn-secondary btn-sm" onclick="CheatsheetModal.close()">${typeof I18n !== 'undefined' ? I18n.t('close') : 'Close'}</button>
      </div>
    </div>`;
    document.body.appendChild(overlay);
    this._esc = (e) => { if (e.key === 'Escape') this.close(); };
    document.addEventListener('keydown', this._esc);
  },

  close() {
    const o = document.getElementById('ssook-cheatsheet');
    if (o) o.remove();
    if (this._esc) { document.removeEventListener('keydown', this._esc); this._esc = null; }
  },
};
