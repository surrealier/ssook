/* Notification system for ssook */
const Notify = {
  _items: [],   // {id, type, msg, detail, time, read}
  _id: 0,
  _unread: 0,

  /* ── Public API ─────────────────────────────────── */
  info(msg)              { this._add('info', msg); },
  success(msg)           { this._add('success', msg); },
  warn(msg, detail)      { this._add('warning', msg, detail); },
  error(msg, detail)     { this._add('error', msg, detail); },

  _add(type, msg, detail) {
    this._items.unshift({ id: ++this._id, type, msg, detail: detail || '', time: new Date(), read: false });
    if (this._items.length > 200) this._items.length = 200;
    this._unread++;
    this._updateBadge();
    if (this._panelOpen) this._renderList();
  },

  /* ── Badge ──────────────────────────────────────── */
  _updateBadge() {
    const b = document.getElementById('notify-badge');
    if (!b) return;
    if (this._unread > 0) { b.textContent = this._unread > 99 ? '99+' : this._unread; b.style.display = ''; }
    else b.style.display = 'none';
  },

  /* ── Panel toggle ───────────────────────────────── */
  _panelOpen: false,

  toggle() {
    this._panelOpen ? this.closePanel() : this.openPanel();
  },

  openPanel() {
    this._panelOpen = true;
    this._unread = 0;
    this._items.forEach(n => n.read = true);
    this._updateBadge();

    let panel = document.getElementById('notify-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'notify-panel';
      panel.className = 'notify-panel shadow-02';
      document.body.appendChild(panel);
      // close on outside click
      setTimeout(() => document.addEventListener('click', this._outsideClick), 0);
    }
    this._renderList();
  },

  _outsideClick(e) {
    const panel = document.getElementById('notify-panel');
    const btn = document.getElementById('notify-btn');
    if (panel && !panel.contains(e.target) && btn && !btn.contains(e.target)) Notify.closePanel();
  },

  closePanel() {
    this._panelOpen = false;
    const p = document.getElementById('notify-panel');
    if (p) p.remove();
    document.removeEventListener('click', this._outsideClick);
  },

  _renderList() {
    const panel = document.getElementById('notify-panel');
    if (!panel) return;
    const t = (k) => I18n.t(k);
    const lang = I18n.getLang();

    if (!this._items.length) {
      panel.innerHTML = `<div class="notify-header">
          <span class="text-action">${lang === 'ko' ? '알림' : 'Notifications'}</span>
          <button class="btn btn-ghost btn-sm" onclick="Notify.closePanel()">${Icons.x(14)}</button>
        </div>
        <div style="padding:2rem;text-align:center;" class="text-muted">${lang === 'ko' ? '알림이 없습니다' : 'No notifications'}</div>`;
      return;
    }

    const items = this._items.map(n => {
      const icon = n.type === 'error' ? '🔴' : n.type === 'warning' ? '🟡' : n.type === 'success' ? '🟢' : '🔵';
      const time = n.time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const detailHtml = n.detail ? `<pre class="notify-detail">${this._esc(n.detail)}</pre>` : '';
      const copyText = this._esc(n.msg + (n.detail ? '\n' + n.detail : ''));
      return `<div class="notify-item notify-${n.type}">
        <div class="notify-item-head">
          <span>${icon} <span class="text-secondary">${time}</span></span>
          <button class="btn btn-ghost btn-sm notify-copy-btn" data-nid="${n.id}" onclick="Notify._copy(this)" title="Copy">${Icons.copy(12)}</button>
        </div>
        <div class="notify-msg">${this._esc(n.msg)}</div>
        ${detailHtml}
      </div>`;
    }).join('');

    panel.innerHTML = `<div class="notify-header">
        <span class="text-action">${lang === 'ko' ? '알림' : 'Notifications'} (${this._items.length})</span>
        <div style="display:flex;gap:0.25rem;">
          <button class="btn btn-ghost btn-sm" onclick="Notify.clear()">${lang === 'ko' ? '모두 삭제' : 'Clear all'}</button>
          <button class="btn btn-ghost btn-sm" onclick="Notify.closePanel()">${Icons.x(14)}</button>
        </div>
      </div>
      <div class="notify-list">${items}</div>`;
  },

  clear() {
    this._items = [];
    this._unread = 0;
    this._updateBadge();
    this._renderList();
  },

  _copy(btn) {
    const n = this._items.find(i => i.id === +btn.dataset.nid);
    if (!n) return;
    const text = n.msg + (n.detail ? '\n' + n.detail : '');
    navigator.clipboard.writeText(text).then(() => {
      btn.innerHTML = Icons.check(12);
      setTimeout(() => { btn.innerHTML = Icons.copy(12); }, 1200);
    });
  },

  _esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; },

  /* ── Inject bell into header ────────────────────── */
  mount() {
    // inject styles
    const style = document.createElement('style');
    style.textContent = `
      #notify-btn{position:relative;background:none;border:none;cursor:pointer;color:var(--text-03);padding:4px;border-radius:var(--border-radius-04);transition:color .15s,background .15s}
      #notify-btn:hover{color:var(--text-05);background:var(--background-tint-02)}
      #notify-badge{position:absolute;top:-2px;right:-4px;background:var(--action-danger-05);color:#fff;font-size:10px;font-weight:700;min-width:16px;height:16px;line-height:16px;text-align:center;border-radius:8px;padding:0 4px;display:none}
      .notify-panel{position:fixed;top:52px;right:16px;width:380px;max-height:70vh;background:var(--background-tint-00);border:1px solid var(--border-01);border-radius:var(--border-radius-12);z-index:9990;display:flex;flex-direction:column;overflow:hidden}
      .notify-header{display:flex;align-items:center;justify-content:space-between;padding:0.625rem 0.75rem;border-bottom:1px solid var(--border-01)}
      .notify-list{overflow-y:auto;flex:1;max-height:calc(70vh - 48px)}
      .notify-item{padding:0.625rem 0.75rem;border-bottom:1px solid var(--border-01)}
      .notify-item:last-child{border-bottom:none}
      .notify-item-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:0.25rem}
      .notify-msg{font-size:13px;color:var(--text-04);line-height:1.4;word-break:break-word}
      .notify-detail{font-family:'DM Mono',monospace;font-size:11px;color:var(--text-03);background:var(--background-tint-02);border-radius:var(--border-radius-04);padding:0.375rem 0.5rem;margin-top:0.375rem;white-space:pre-wrap;word-break:break-all;max-height:120px;overflow-y:auto}
      .notify-error{border-left:3px solid var(--status-error-05)}
      .notify-warning{border-left:3px solid var(--status-warning-05)}
      .notify-success{border-left:3px solid var(--status-success-05)}
      .notify-info{border-left:3px solid var(--status-info-05)}
      .notify-copy-btn{padding:2px!important;min-width:auto!important}
    `;
    document.head.appendChild(style);
  }
};
