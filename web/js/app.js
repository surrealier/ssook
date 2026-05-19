/* Main application controller */
const App = {
  currentTab: 'viewer',

  /* Sidebar definition: [section_i18n_key, [[tab_id, icon_fn, nav_i18n_key], ...]] */
  _nav: [
    ['sec.inference',  [['viewer','viewer'],['settings','settings']]],
    ['sec.evaluation', [['evaluation','evaluation'],['benchmark','benchmark']]],
    ['sec.analysis',   [['analysis','analysis'],['model-compare','compare'],['error-analyzer','errorAnalyzer'],['conf-optimizer','confOptimizer'],['embedding-viewer','embeddingViewer']]],
    ['sec.tools',      [['diagnose','diagnose'],['calibration','calibration'],['inspector','inspector'],['profiler','profiler']]],
    ['sec.data',       [['explorer','explorer'],['splitter','splitter'],['converter','converter'],['remapper','remapper'],['merger','merger'],['sampler','sampler'],['augmentation','augmentation']]],
    ['sec.quality',    [['anomaly','anomaly'],['quality','quality'],['duplicate','duplicate'],['leaky','leaky'],['similarity','similarity']]],
  ],

  init() {
    Notify.mount();
    document.getElementById('notify-btn').insertAdjacentHTML('afterbegin', Icons.bell(18));
    this.renderSidebar();
    this.initDarkMode();
    this.initShortcuts();
    this.switchTab('viewer');
    this.loadSystemInfo();
  },

  renderSidebar() {
    const t = (k) => I18n.t(k);
    const isDark = document.documentElement.classList.contains('dark');
    const lang = I18n.getLang();

    let html = `
      <div class="sidebar-header">
        <img src="/assets/icon.svg" width="24" height="24" alt="ssook" style="border-radius:4px;">
        <h1>ssook</h1>
      </div>`;

    for (const [secKey, items] of this._nav) {
      html += `<div class="sidebar-section">
        <div class="sidebar-section-title">${t(secKey)}</div>`;
      for (const [tabId, iconName] of items) {
        const active = tabId === this.currentTab ? ' active' : '';
        const iconFn = Icons[iconName] || Icons.logo;
        html += `<div class="nav-item${active}" data-tab="${tabId}">
          ${iconFn(18)}
          <span>${t('nav.' + tabId)}</span>
        </div>`;
      }
      html += '</div>';
    }

    // Footer: dark mode + language
    html += `
      <div style="margin-top:auto;padding-top:0.75rem;border-top:1px solid var(--border-01);display:flex;flex-direction:column;gap:0.5rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;padding:0.375rem 0.75rem;">
          <span class="text-secondary" style="display:flex;align-items:center;gap:0.375rem;">
            ${isDark ? Icons.moon(14) : Icons.sun(14)} ${t('dark_mode')}
          </span>
          <label class="toggle">
            <input type="checkbox" id="dark-toggle" ${isDark ? 'checked' : ''}>
            <span class="toggle-track"></span>
            <span class="toggle-thumb"></span>
          </label>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;padding:0.375rem 0.75rem;">
          <span class="text-secondary" style="display:flex;align-items:center;gap:0.375rem;">
            ${Icons.globe(14)} ${t('language')}
          </span>
          <select id="lang-select" class="form-input input-normal" style="width:auto;padding:0.125rem 0.5rem;font-size:12px;">
            <option value="en" ${lang==='en'?'selected':''}>EN</option>
            <option value="ko" ${lang==='ko'?'selected':''}>KO</option>
          </select>
        </div>
        <button onclick="App.shutdown()" class="btn btn-danger btn-sm" style="margin:0.25rem 0.75rem;font-size:12px;">${t('shutdown')}</button>
      </div>`;

    document.getElementById('sidebar').innerHTML = html;

    // Bind nav clicks
    document.querySelectorAll('.nav-item[data-tab]').forEach(el => {
      el.addEventListener('click', () => App.switchTab(el.dataset.tab));
    });

    // Bind dark mode
    const toggle = document.getElementById('dark-toggle');
    if (toggle) toggle.addEventListener('change', () => {
      document.documentElement.classList.toggle('dark', toggle.checked);
      localStorage.setItem('ssook-dark', toggle.checked);
      this.renderSidebar(); // re-render for icon swap
    });

    // Bind language
    const langSel = document.getElementById('lang-select');
    if (langSel) langSel.addEventListener('change', () => I18n.setLang(langSel.value));
  },

  initDarkMode() {
    const saved = localStorage.getItem('ssook-dark');
    if (saved === 'true' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
    }
  },

  _tabCache: {},

  switchTab(name) {
    const tab = Tabs[name];
    if (!tab) return;

    // 이전 탭 리소스 정리
    const prev = Tabs[this.currentTab];
    if (prev && prev.destroy) prev.destroy();

    // 현재 탭 DOM 캐시 저장
    const body = document.getElementById('page-body');
    if (this.currentTab && body.children.length) {
      this._tabCache[this.currentTab] = body.innerHTML;
    }

    this.currentTab = name;

    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(el => {
      el.classList.toggle('active', el.dataset.tab === name);
    });

    // Update header
    const titleEl = document.getElementById('page-title');
    const titleText = tab.title ? I18n.t('nav.' + name) : name;
    titleEl.innerHTML = titleText;
    document.getElementById('page-actions').innerHTML = `<button class="btn btn-ghost btn-sm" onclick="showHelp('${name}')" style="color:var(--text-02);font-size:12px;">Help</button>`;

    // 캐시된 HTML이 있으면 복원, 없으면 새로 렌더
    if (tab._cachedHTML) {
      body.innerHTML = tab._cachedHTML;
      if (tab.init) tab.init();
    } else if (this._tabCache[name]) {
      body.innerHTML = this._tabCache[name];
      if (tab.init) tab.init();
    } else {
      body.innerHTML = tab.render();
      body.classList.add('animate-fade-in');
      setTimeout(() => body.classList.remove('animate-fade-in'), 200);
      if (tab.init) tab.init();
    }

    // a11y: move focus to the new page title so screen readers announce it.
    try { titleEl.focus({ preventScroll: true }); } catch (e) {}

    // Re-mount header actions (Logs / cheatsheet pills) — switchTab clobbers
    // page-actions on every render, so this is the place to put them back.
    if (this.mountHeaderActions) this.mountHeaderActions();
  },

  // Keyboard shortcuts: Ctrl+1..9 switch tabs by sidebar order;
  // "?" opens the cheatsheet modal (Wave 5);
  // F1 opens contextual help overlay for the current tab.
  initShortcuts() {
    if (this._shortcutsBound) return;
    this._shortcutsBound = true;
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
      if (e.ctrlKey && !e.shiftKey && !e.altKey && /^[1-9]$/.test(e.key)) {
        const idx = +e.key - 1;
        const items = [].concat(...this._nav.map(([, group]) => group));
        if (items[idx]) { e.preventDefault(); this.switchTab(items[idx][0]); }
      } else if (e.key === '?' && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        if (typeof CheatsheetModal !== 'undefined') CheatsheetModal.show();
      } else if (e.key === 'F1') {
        e.preventDefault();
        showHelp(this.currentTab);
      }
    });
  },

  // Header action pills — Wave 5 task queue + log viewer + cheatsheet.
  mountHeaderActions() {
    if (typeof TaskQueuePanel !== 'undefined') {
      try { TaskQueuePanel.mountPill(); } catch (e) {}
    }
    const host = document.getElementById('page-actions');
    if (!host || document.getElementById('ssook-logs-btn')) return;
    const logs = document.createElement('button');
    logs.id = 'ssook-logs-btn';
    logs.className = 'btn btn-ghost btn-sm';
    logs.title = I18n.t('panels.logs_title');
    logs.setAttribute('aria-label', I18n.t('panels.logs_title'));
    logs.style.cssText = 'color:var(--text-02);font-size:12px;';
    logs.textContent = 'Logs';
    logs.onclick = () => LogViewerPanel && LogViewerPanel.toggle();
    host.appendChild(logs);
    const cs = document.createElement('button');
    cs.id = 'ssook-cheatsheet-btn';
    cs.className = 'btn btn-ghost btn-sm';
    cs.title = I18n.t('cheatsheet.title');
    cs.setAttribute('aria-label', I18n.t('cheatsheet.title'));
    cs.style.cssText = 'color:var(--text-02);font-size:12px;';
    cs.textContent = '?';
    cs.onclick = () => CheatsheetModal && CheatsheetModal.show();
    host.appendChild(cs);
  },

  setStatus(text, errorDetail) {
    document.getElementById('status-text').textContent = text;
    if (!text) return;
    const lower = text.toLowerCase();
    if (lower.startsWith('error:') || lower.startsWith('오류:')) {
      Notify.error(text, errorDetail);
    } else if (/\bcomplete$|\bsaved|저장[  ]?완료|완료$/.test(lower)) {
      Notify.success(text);
    }
  },

  // Drop a notification without overwriting the status bar.
  // type: 'info' | 'success' | 'warn' | 'error'.
  // Same (type,msg) within 3s is deduped to avoid flooding on poll loops.
  _toastSeen: {},
  toast(type, msg, detail) {
    const fn = Notify[type] || Notify.info;
    const key = type + '::' + msg;
    const now = Date.now();
    if (this._toastSeen[key] && now - this._toastSeen[key] < 3000) return;
    this._toastSeen[key] = now;
    fn.call(Notify, msg, detail);
  },

  async shutdown() {
    if (!confirm(I18n.t('shutdown_confirm'))) return;
    try { await API.post('/api/shutdown'); } catch(e) {}
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-size:1.5rem;color:var(--text-03);">Server stopped. You can close this tab.</div>';
  },

  async loadSystemInfo() {
    try {
      const info = await API.sysInfo();
      document.getElementById('status-info').textContent =
        `${info.os || ''} | Python ${info.python || ''} | ORT ${info.ort || ''}`;
    } catch(e) {
      console.warn('sysInfo error:', e);
      this.toast('warn', 'System info unavailable', e && e.message);
    }
    // Load default paths from config
    try {
      const c = await API.config();
      if (c.samples_dir && !G.imgDir) setImgDir(c.samples_dir);
      if (c.default_model_path && !G.model) setModel(c.default_model_path);
    } catch(e) {
      console.warn('config load error:', e);
      this.toast('warn', 'Could not load config defaults', e && e.message);
    }
    this.setStatus(I18n.t('ready'));
  }
};

document.addEventListener('DOMContentLoaded', () => {
  App.init();
  // Heartbeat: keep server alive while browser is open
  setInterval(() => fetch('/api/heartbeat', {method:'POST'}).catch(()=>{}), 10000);
});
