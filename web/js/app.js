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
    titleEl.innerHTML = titleText + ' <button class="btn btn-ghost" style="padding:0;font-size:14px;min-width:20px;height:20px;line-height:20px;vertical-align:middle;margin-left:0.25rem;" onclick="showHelp(\'' + name + '\')">❓</button>';

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

  async loadSystemInfo() {
    try {
      const info = await API.sysInfo();
      document.getElementById('status-info').textContent =
        `${info.os || ''} | Python ${info.python || ''} | ORT ${info.ort || ''}`;
    } catch(e) { console.warn('sysInfo error:', e); }
    // Load default paths from config
    try {
      const c = await API.config();
      if (c.samples_dir && !G.imgDir) setImgDir(c.samples_dir);
      if (c.default_model_path && !G.model) setModel(c.default_model_path);
    } catch(e) { console.warn('config load error:', e); }
    this.setStatus(I18n.t('ready'));
  }
};

document.addEventListener('DOMContentLoaded', () => {
  App.init();
  // Heartbeat: keep server alive while browser is open
  setInterval(() => fetch('/api/heartbeat', {method:'POST'}).catch(()=>{}), 10000);
});
