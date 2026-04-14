/* Global state & shared UI helpers */
const G = {
  model: '',      // current model path
  imgDir: '',     // current images/video directory
  lblDir: '',     // current labels directory (auto-set from imgDir)
  models: [],     // list of loaded model paths (for multi-model tabs)
};

/* Auto-set lblDir when imgDir changes */
function setImgDir(dir) {
  G.imgDir = dir;
  if (!G.lblDir || G.lblDir === G.imgDir) G.lblDir = dir;
  // Update all visible imgDir inputs
  document.querySelectorAll('[data-bind="imgDir"]').forEach(el => el.value = dir);
  document.querySelectorAll('[data-bind="lblDir"]').forEach(el => { if (!el.dataset.manual) el.value = dir; });
}
function setLblDir(dir) {
  G.lblDir = dir;
  document.querySelectorAll('[data-bind="lblDir"]').forEach(el => { el.value = dir; el.dataset.manual = '1'; });
}
function setModel(path) {
  G.model = path;
  if (path && !G.models.includes(path)) G.models.push(path);
  document.querySelectorAll('[data-bind="model"]').forEach(el => el.value = path);
  App.setStatus(`Model: ${path.split(/[\\/]/).pop()}`);
}

/* Reusable UI components */
function modelInput(id) {
  return `<div class="form-group">
    <label class="form-label">${t('settings.model')}</label>
    <div style="display:flex;gap:0.5rem;">
      <input type="text" class="form-input input-normal" style="flex:1;" id="${id}" data-bind="model" value="${G.model}" onchange="setModel(this.value)">
      <button class="btn btn-secondary btn-sm" onclick="pickModel('${id}')">
        ${t('browse')}
      </button>
    </div>
  </div>`;
}
function imgDirInput(id) {
  return `<div class="form-group">
    <label class="form-label">${t('explorer.img_dir')}</label>
    <div style="display:flex;gap:0.5rem;">
      <input type="text" class="form-input input-normal" style="flex:1;" id="${id}" data-bind="imgDir" value="${G.imgDir}" onchange="setImgDir(this.value)">
      <button class="btn btn-secondary btn-sm" onclick="pickImgDir('${id}')">
        ${t('browse')}
      </button>
    </div>
  </div>`;
}
function lblDirInput(id) {
  return `<div class="form-group">
    <label class="form-label">${t('explorer.lbl_dir')}</label>
    <div style="display:flex;gap:0.5rem;">
      <input type="text" class="form-input input-normal" style="flex:1;" id="${id}" data-bind="lblDir" value="${G.lblDir}" onchange="setLblDir(this.value)">
      <button class="btn btn-secondary btn-sm" onclick="pickLblDir('${id}')">
        ${t('browse')}
      </button>
    </div>
  </div>`;
}
function outDirInput(id) {
  return `<div class="form-group">
    <label class="form-label">${t('splitter.output')}</label>
    <div style="display:flex;gap:0.5rem;">
      <input type="text" class="form-input input-normal" style="flex:1;" id="${id}" onchange="this.value=this.value.trim()">
      <button class="btn btn-secondary btn-sm" onclick="pickDir('${id}')">
        ${t('browse')}
      </button>
    </div>
  </div>`;
}

/* Multi-model selector: file dialog allows multiple, auto-creates slots */
function multiModelSlots(containerId, listId) {
  return `<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">
    <h3 class="text-heading-h3">${t('bench.models')}</h3>
    <div style="display:flex;gap:0.5rem;">
      <button class="btn btn-secondary btn-sm" onclick="addModelSlot('${containerId}','${listId}')">${t('add_model')}</button>
    </div>
  </div>
  <div id="${containerId}" style="display:flex;flex-direction:column;gap:0.5rem;">
    <div class="text-secondary" style="padding:1rem;text-align:center;" id="${listId}-hint">${t('bench.add_hint')}</div>
  </div>`;
}

/* Picker functions — web-based file browser */
function _showFileBrowser(mode, exts, callback) {
  // mode: "file" | "dir"
  let currentPath = '';
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:9999;display:flex;align-items:center;justify-content:center;';
  const modal = document.createElement('div');
  modal.style.cssText = 'background:var(--background-tint-00);border-radius:12px;width:560px;max-height:70vh;display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,0.3);';
  modal.innerHTML = `
    <div style="padding:1rem 1.25rem;border-bottom:1px solid var(--border-01);display:flex;align-items:center;justify-content:space-between;">
      <strong>${mode === 'dir' ? 'Select Directory' : 'Select File'}</strong>
      <button class="btn btn-ghost btn-sm" id="fb-close">✕</button>
    </div>
    <div style="padding:0.5rem 1.25rem;display:flex;gap:0.5rem;align-items:center;border-bottom:1px solid var(--border-01);">
      <button class="btn btn-ghost btn-sm" id="fb-up">⬆</button>
      <input type="text" class="form-input input-normal" id="fb-path" style="flex:1;font-size:12px;" placeholder="Paste path and press Enter">
    </div>
    <div id="fb-list" style="flex:1;overflow-y:auto;padding:0.5rem;min-height:200px;max-height:50vh;"></div>
    <div style="padding:0.75rem 1.25rem;border-top:1px solid var(--border-01);display:flex;gap:0.5rem;justify-content:flex-end;">
      ${mode === 'dir' ? '<button class="btn btn-primary btn-sm" id="fb-select-dir">Select This Directory</button>' : ''}
      <button class="btn btn-secondary btn-sm" id="fb-cancel">Cancel</button>
    </div>`;
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  const listEl = modal.querySelector('#fb-list');
  const pathEl = modal.querySelector('#fb-path');

  async function navigate(path) {
    listEl.innerHTML = '<div class="text-secondary" style="padding:1rem;text-align:center;">Loading...</div>';
    try {
      const r = await API.post('/api/fs/browse', {path: path || null, exts, mode});
      currentPath = r.current || '';
      pathEl.value = currentPath;
      if (!r.entries || !r.entries.length) {
        listEl.innerHTML = '<div class="text-secondary" style="padding:1rem;text-align:center;">Empty</div>';
        return;
      }
      listEl.innerHTML = '';
      let selectedRow = null;
      r.entries.forEach(e => {
        const row = document.createElement('div');
        row.style.cssText = 'padding:0.35rem 0.5rem;cursor:pointer;border-radius:6px;display:flex;align-items:center;gap:0.5rem;font-size:13px;transition:background 0.1s;';
        row.onmouseenter = () => { if (row !== selectedRow) row.style.background = 'var(--background-tint-01)'; };
        row.onmouseleave = () => { if (row !== selectedRow) row.style.background = ''; };
        if (e.type === 'drive' || e.type === 'dir') {
          row.innerHTML = '<span style="opacity:0.6;">📁</span> ' + e.name;
          row.ondblclick = () => navigate(e.path);
          row.onclick = () => {
            if (selectedRow) { selectedRow.style.background = ''; selectedRow.style.outline = ''; }
            selectedRow = row;
            row.style.background = 'var(--action-link-01, rgba(74,158,255,0.15))';
            row.style.outline = '1px solid var(--action-link-05, #4a9eff)';
            pathEl.value = e.path; currentPath = e.path;
          };
        } else {
          row.innerHTML = '<span style="opacity:0.4;">📄</span> ' + e.name;
          row.onclick = () => { cleanup(); callback(e.path); };
        }
        listEl.appendChild(row);
      });
    } catch(err) {
      listEl.innerHTML = `<div class="text-secondary" style="padding:1rem;text-align:center;">Error: ${err.message}</div>`;
    }
  }

  function cleanup() { overlay.remove(); }

  modal.querySelector('#fb-close').onclick = cleanup;
  modal.querySelector('#fb-cancel').onclick = cleanup;
  overlay.onclick = (e) => { if (e.target === overlay) cleanup(); };
  modal.querySelector('#fb-up').onclick = async () => {
    if (!currentPath) return;
    const r = await API.post('/api/fs/browse', {path: currentPath, exts, mode});
    if (r.parent) navigate(r.parent);
  };
  const selDirBtn = modal.querySelector('#fb-select-dir');
  if (selDirBtn) selDirBtn.onclick = () => { if (currentPath) { cleanup(); callback(currentPath); } };

  pathEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const v = pathEl.value.trim();
      if (v) navigate(v);
    }
  });

  navigate('');
}

async function pickModel(inputId) {
  _showFileBrowser('file', ['.onnx', '.pt'], (path) => {
    setModel(path);
    const el = document.getElementById(inputId);
    if (el) el.value = path;
  });
}
async function pickImgDir(inputId) {
  _showFileBrowser('dir', null, (path) => {
    setImgDir(path);
    const el = document.getElementById(inputId);
    if (el) el.value = path;
  });
}
async function pickLblDir(inputId) {
  _showFileBrowser('dir', null, (path) => {
    setLblDir(path);
    const el = document.getElementById(inputId);
    if (el) el.value = path;
  });
}
async function pickDir(inputId) {
  _showFileBrowser('dir', null, (path) => {
    const el = document.getElementById(inputId);
    if (el) el.value = path;
  });
}
async function pickFile(inputId, filters) {
  // Parse filter string to ext array: "ONNX (*.onnx);;PyTorch (*.pt)" → [".onnx", ".pt"]
  let exts = null;
  if (filters) {
    const m = filters.match(/\*\.\w+/g);
    if (m) exts = m.map(e => e.replace('*', ''));
  }
  _showFileBrowser('file', exts, (path) => {
    const el = document.getElementById(inputId);
    if (el) el.value = path;
  });
}

/* Add model slot to multi-model container */
let _slotN = 0;
async function addModelSlot(containerId, listId) {
  _showFileBrowser('file', ['.onnx'], (path) => {
    _addOneSlot(containerId, listId, path);
  });
}
function _addOneSlot(containerId, listId, path) {
  const c = document.getElementById(containerId);
  const hint = document.getElementById(listId + '-hint');
  if (hint) hint.remove();
  const id = ++_slotN;
  const name = path.split(/[\\/]/).pop();
  const d = document.createElement('div');
  d.className = 'card-flat'; d.id = `ms-${id}`;
  d.style.cssText = 'padding:0.5rem 0.75rem;display:flex;align-items:center;gap:0.5rem;';
  d.innerHTML = `<span class="text-mono" style="flex:1;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${path}">${name}</span>
    <input type="hidden" class="model-slot-path" value="${path}">
    <button class="btn btn-ghost btn-sm" onclick="document.getElementById('ms-${id}').remove()" style="color:var(--action-danger-05);padding:0 0.25rem;">✕</button>`;
  c.appendChild(d);
  if (!G.models.includes(path)) G.models.push(path);
}

/* Collect all model paths from slots */
function getSlotModels(containerId) {
  return [...document.querySelectorAll(`#${containerId} .model-slot-path`)].map(e => e.value);
}

/* ── Detail Modal ───────────────────────────────────── */
function showDetailModal(title, html) {
  let overlay = document.getElementById('detail-modal-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'detail-modal-overlay';
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:9999;display:flex;align-items:center;justify-content:center;';
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    document.body.appendChild(overlay);
  }
  overlay.innerHTML = `<div style="background:#1a1a1a;border-radius:12px;padding:1.5rem;max-width:700px;width:90%;max-height:80vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,0.5);color:#e6e6e6;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
      <h3 class="text-heading-h3">${title}</h3>
      <button class="btn btn-ghost btn-sm" onclick="document.getElementById('detail-modal-overlay').remove()">✕</button>
    </div>
    <div>${html}</div>
  </div>`;
}

/* ── Help Overlay (paginated inline annotations) ────── */
/* _ANNOTATIONS: loaded from help-annotations-main.js & help-annotations-extra.js */
const _ANNOTATIONS = (function() {
  const m = typeof _ANNOTATIONS_MAIN !== 'undefined' ? _ANNOTATIONS_MAIN : {};
  const e = typeof _ANNOTATIONS_EXTRA !== 'undefined' ? _ANNOTATIONS_EXTRA : {};
  return Object.assign({}, m, e);
})();

function showHelp(tabName) {
  const oldBg = document.getElementById('help-bg');
  if (oldBg) { _closeHelp(); return; }

  const pages = _ANNOTATIONS[tabName];
  if (!pages || !pages.length) return;

  /* inject styles once */
  if (!document.getElementById('help-style')) {
    const s = document.createElement('style');
    s.id = 'help-style';
    s.textContent = `
      .help-hl{outline:2px solid #4a9eff!important;outline-offset:2px;border-radius:4px;position:relative;z-index:9999}
      .help-b{position:fixed;z-index:10001;background:#1a1a1a;color:#e6e6e6;border:1px solid #4a9eff;border-radius:8px;padding:6px 10px;font-size:11px;line-height:1.5;white-space:pre-line;max-width:240px;box-shadow:0 4px 12px rgba(0,0,0,0.6);pointer-events:none}
      .help-b::after{content:'';position:absolute;width:8px;height:8px;background:#1a1a1a;border:1px solid #4a9eff;transform:rotate(45deg)}
      .help-b.arrow-left::after{right:-5px;top:12px;border-top:none;border-left:none}
      .help-b.arrow-right::after{left:-5px;top:12px;border-bottom:none;border-right:none}
      .help-b.arrow-top::after{bottom:-5px;left:16px;border-top:none;border-left:none}
      .help-b.arrow-bottom::after{top:-5px;left:16px;border-bottom:none;border-right:none}
      .help-nav{position:fixed;z-index:10002;display:flex;align-items:center;gap:0.75rem;
        background:#1a1a1a;border:1px solid #4a9eff;border-radius:10px;padding:6px 14px;
        box-shadow:0 4px 16px rgba(0,0,0,0.5);left:50%;transform:translateX(-50%);bottom:32px;user-select:none}
      .help-nav button{background:none;border:1px solid #4a9eff;color:#e6e6e6;border-radius:6px;
        width:30px;height:30px;font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s}
      .help-nav button:hover{background:#4a9eff33}
      .help-nav button:disabled{opacity:0.3;cursor:default;background:none}
      .help-nav-title{color:#e6e6e6;font-size:13px;font-weight:600;min-width:120px;text-align:center}
      .help-nav-dots{display:flex;gap:4px}
      .help-nav-dot{width:7px;height:7px;border-radius:50%;background:#555;transition:background .2s}
      .help-nav-dot.active{background:#4a9eff}
    `;
    document.head.appendChild(s);
  }

  /* background overlay */
  const bg = document.createElement('div');
  bg.id = 'help-bg';
  bg.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:9997;background:rgba(0,0,0,0.25);cursor:pointer;';
  bg.onclick = _closeHelp;
  document.body.appendChild(bg);

  /* bubble container */
  const container = document.createElement('div');
  container.id = 'help-bubbles';
  container.style.cssText = 'position:fixed;top:0;left:0;width:0;height:0;z-index:10000;pointer-events:none;';
  document.body.appendChild(container);

  /* navigation bar */
  const nav = document.createElement('div');
  nav.className = 'help-nav'; nav.id = 'help-nav';
  document.body.appendChild(nav);

  let curPage = 0;

  function renderPage(idx) {
    curPage = idx;
    /* clear previous highlights & bubbles */
    document.querySelectorAll('.help-hl').forEach(e => e.classList.remove('help-hl'));
    container.innerHTML = '';

    const page = pages[idx];
    const items = page.items || [];
    const _ht = (v) => typeof v === 'object' ? (v[I18n.getLang()] || v.en || '') : v;

    /* nav bar */
    const dots = pages.map((_, i) => `<span class="help-nav-dot${i===idx?' active':''}"></span>`).join('');
    nav.innerHTML = `<button id="help-prev"${idx===0?' disabled':''}>◀</button>
      <div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
        <span class="help-nav-title">${_ht(page.page)}</span>
        <div class="help-nav-dots">${dots}</div>
        <span style="color:#888;font-size:10px;">${idx+1} / ${pages.length}</span>
      </div>
      <button id="help-next"${idx===pages.length-1?' disabled':''}>▶</button>`;
    document.getElementById('help-prev').onclick = (e) => { e.stopPropagation(); renderPage(idx - 1); };
    document.getElementById('help-next').onclick = (e) => { e.stopPropagation(); renderPage(idx + 1); };

    /* place bubbles */
    const placed = [];
    for (const ann of items) {
      const el = document.querySelector(ann.sel);
      if (!el) continue;
      el.classList.add('help-hl');
      const r = el.getBoundingClientRect();
      const b = document.createElement('div');
      b.className = 'help-b';
      b.textContent = _ht(ann.text);
      container.appendChild(b);
      b.style.left = '-9999px'; b.style.top = '0';
      const bw = b.offsetWidth, bh = b.offsetHeight;
      const vw = window.innerWidth, vh = window.innerHeight;
      const gap = 10;
      const candidates = [
        { x: r.right + gap, y: r.top, arrow: 'arrow-right' },
        { x: r.left - bw - gap, y: r.top, arrow: 'arrow-left' },
        { x: r.left, y: r.bottom + gap, arrow: 'arrow-bottom' },
        { x: r.left, y: r.top - bh - gap, arrow: 'arrow-top' },
      ];
      let best = null;
      for (const c of candidates) {
        if (c.x < 4 || c.x + bw > vw - 4 || c.y < 4 || c.y + bh > vh - 4) continue;
        const overlaps = placed.some(p => c.x < p.x + p.w && c.x + bw > p.x && c.y < p.y + p.h && c.y + bh > p.y);
        if (!overlaps) { best = c; break; }
      }
      if (!best) {
        let ny = r.top;
        for (const p of placed) {
          if (r.right + gap < p.x + p.w && r.right + gap + bw > p.x && ny < p.y + p.h && ny + bh > p.y) ny = p.y + p.h + 4;
        }
        best = { x: Math.min(r.right + gap, vw - bw - 4), y: Math.min(ny, vh - bh - 4), arrow: 'arrow-right' };
      }
      b.style.left = best.x + 'px';
      b.style.top = best.y + 'px';
      b.classList.add(best.arrow);
      placed.push({ x: best.x, y: best.y, w: bw, h: bh });
    }
  }

  renderPage(0);

  /* keyboard navigation */
  function _helpKey(e) {
    if (e.key === 'ArrowRight' && curPage < pages.length - 1) renderPage(curPage + 1);
    else if (e.key === 'ArrowLeft' && curPage > 0) renderPage(curPage - 1);
    else if (e.key === 'Escape') _closeHelp();
  }
  document.addEventListener('keydown', _helpKey);
  bg.dataset.helpKeyHandler = 'true';
  window._helpKeyHandler = _helpKey;
}

function _closeHelp() {
  const bg = document.getElementById('help-bg');
  const bubbles = document.getElementById('help-bubbles');
  const nav = document.getElementById('help-nav');
  if (bg) bg.remove();
  if (bubbles) bubbles.remove();
  if (nav) nav.remove();
  document.querySelectorAll('.help-hl').forEach(e => e.classList.remove('help-hl'));
  if (window._helpKeyHandler) {
    document.removeEventListener('keydown', window._helpKeyHandler);
    window._helpKeyHandler = null;
  }
}


/* ── HuggingFace Hub Browser ────────────────────────── */
function showHFBrowser(targetInputId) {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:9999;display:flex;align-items:center;justify-content:center;';
  const modal = document.createElement('div');
  modal.style.cssText = 'background:var(--background-tint-00);border-radius:12px;width:620px;max-height:75vh;display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,0.3);';
  modal.innerHTML = `
    <div style="padding:1rem 1.25rem;border-bottom:1px solid var(--border-01);display:flex;align-items:center;justify-content:space-between;">
      <strong>🤗 HuggingFace Hub — ONNX Models</strong>
      <button class="btn btn-ghost btn-sm" id="hf-close">✕</button>
    </div>
    <div style="padding:0.75rem 1.25rem;display:flex;gap:0.5rem;border-bottom:1px solid var(--border-01);">
      <input type="text" class="form-input input-normal" id="hf-query" style="flex:1;font-size:12px;" placeholder="Search models (e.g. yolov8, vit, blip)">
      <select class="form-input input-normal" id="hf-task" style="width:160px;font-size:12px;">
        <option value="">All Tasks</option>
        <option value="object-detection">Detection</option>
        <option value="image-classification">Classification</option>
        <option value="image-segmentation">Segmentation</option>
        <option value="zero-shot-image-classification">CLIP/Zero-shot</option>
        <option value="visual-question-answering">VQA</option>
        <option value="image-to-text">Captioning</option>
        <option value="feature-extraction">Embedder</option>
      </select>
      <button class="btn btn-primary btn-sm" id="hf-search-btn">Search</button>
    </div>
    <div id="hf-results" style="flex:1;overflow-y:auto;padding:0.5rem;min-height:200px;max-height:50vh;">
      <div class="text-secondary" style="padding:2rem;text-align:center;">Search for ONNX models on HuggingFace Hub</div>
    </div>
    <div style="padding:0.5rem 1.25rem;border-top:1px solid var(--border-01);display:flex;gap:0.5rem;justify-content:space-between;align-items:center;">
      <button class="btn btn-secondary btn-sm" id="hf-cached-btn">📦 Cached Models</button>
      <button class="btn btn-secondary btn-sm" id="hf-cancel">Cancel</button>
    </div>`;
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  const resultsEl = modal.querySelector('#hf-results');
  const close = () => overlay.remove();
  modal.querySelector('#hf-close').onclick = close;
  modal.querySelector('#hf-cancel').onclick = close;
  overlay.onclick = (e) => { if (e.target === overlay) close(); };

  modal.querySelector('#hf-search-btn').onclick = async () => {
    const q = modal.querySelector('#hf-query').value.trim();
    if (!q) return;
    resultsEl.innerHTML = '<div class="text-secondary" style="padding:2rem;text-align:center;">Searching...</div>';
    try {
      const r = await API.post('/api/hf/search', { query: q, task: modal.querySelector('#hf-task').value });
      if (r.error) { resultsEl.innerHTML = `<div style="padding:1rem;color:var(--action-danger-05);">${r.error}</div>`; return; }
      if (!r.results.length) { resultsEl.innerHTML = '<div class="text-secondary" style="padding:2rem;text-align:center;">No results</div>'; return; }
      resultsEl.innerHTML = '';
      for (const m of r.results) {
        const row = document.createElement('div');
        row.style.cssText = 'padding:0.5rem;border-radius:8px;cursor:pointer;border-bottom:1px solid var(--border-01);';
        row.onmouseenter = () => row.style.background = 'var(--background-tint-01)';
        row.onmouseleave = () => row.style.background = '';
        const onnxBadge = m.has_onnx ? '<span style="background:#10b981;color:#fff;padding:1px 6px;border-radius:4px;font-size:10px;margin-left:4px;">ONNX</span>' : '';
        row.innerHTML = `<div style="font-size:13px;font-weight:500;">${m.repo_id}${onnxBadge}</div>
          <div style="font-size:11px;color:var(--text-02);margin-top:2px;">${m.task || '—'} · ⬇ ${(m.downloads||0).toLocaleString()}${m.ssook_type ? ' · ssook: '+m.ssook_type : ''}</div>`;
        row.onclick = () => _hfShowFiles(m.repo_id, resultsEl, targetInputId, close);
        resultsEl.appendChild(row);
      }
    } catch(e) { resultsEl.innerHTML = `<div style="padding:1rem;color:var(--action-danger-05);">Error: ${e.message}</div>`; }
  };
  modal.querySelector('#hf-query').addEventListener('keydown', (e) => { if (e.key === 'Enter') modal.querySelector('#hf-search-btn').click(); });

  modal.querySelector('#hf-cached-btn').onclick = async () => {
    resultsEl.innerHTML = '<div class="text-secondary" style="padding:2rem;text-align:center;">Loading cached...</div>';
    try {
      const r = await API.get('/api/hf/cached');
      if (!r.models || !r.models.length) { resultsEl.innerHTML = '<div class="text-secondary" style="padding:2rem;text-align:center;">No cached models</div>'; return; }
      resultsEl.innerHTML = '';
      for (const m of r.models) {
        const row = document.createElement('div');
        row.style.cssText = 'padding:0.5rem;border-radius:8px;cursor:pointer;border-bottom:1px solid var(--border-01);display:flex;align-items:center;gap:0.5rem;';
        row.onmouseenter = () => row.style.background = 'var(--background-tint-01)';
        row.onmouseleave = () => row.style.background = '';
        row.innerHTML = `<span style="flex:1;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${m.name}</span><span style="font-size:11px;color:var(--text-02);">${m.size_mb} MB</span>`;
        row.onclick = () => { const el = document.getElementById(targetInputId); if (el) { el.value = m.path; setModel(m.path); } close(); };
        resultsEl.appendChild(row);
      }
    } catch(e) { resultsEl.innerHTML = `<div style="padding:1rem;color:var(--action-danger-05);">Error: ${e.message}</div>`; }
  };
}

async function _hfShowFiles(repoId, container, targetInputId, closeFn) {
  container.innerHTML = `<div class="text-secondary" style="padding:2rem;text-align:center;">Loading ONNX files from ${repoId}...</div>`;
  try {
    const r = await API.post('/api/hf/files', { repo_id: repoId });
    if (r.error) { container.innerHTML = `<div style="padding:1rem;color:var(--action-danger-05);">${r.error}</div>`; return; }
    if (!r.files.length) { container.innerHTML = `<div class="text-secondary" style="padding:2rem;text-align:center;">No .onnx files found in ${repoId}</div>`; return; }
    container.innerHTML = `<div style="padding:0.5rem;font-size:12px;font-weight:600;color:var(--text-03);">${repoId} — Select ONNX file:</div>`;
    for (const f of r.files) {
      const row = document.createElement('div');
      row.style.cssText = 'padding:0.5rem 0.75rem;border-radius:8px;cursor:pointer;display:flex;align-items:center;gap:0.5rem;border-bottom:1px solid var(--border-01);';
      row.onmouseenter = () => row.style.background = 'var(--background-tint-01)';
      row.onmouseleave = () => row.style.background = '';
      row.innerHTML = `<span style="flex:1;font-size:12px;">📄 ${f}</span><button class="btn btn-primary btn-sm" style="font-size:11px;">Download</button>`;
      row.querySelector('button').onclick = async (e) => {
        e.stopPropagation();
        row.querySelector('button').disabled = true;
        row.querySelector('button').textContent = 'Downloading...';
        try {
          const dl = await API.post('/api/hf/download', { repo_id: repoId, filename: f });
          if (dl.error) { App.setStatus('Download error: ' + dl.error); return; }
          const el = document.getElementById(targetInputId);
          if (el) { el.value = dl.path; setModel(dl.path); }
          App.setStatus(`Downloaded: ${f}`);
          closeFn();
        } catch(err) { App.setStatus('Download error: ' + err.message); row.querySelector('button').disabled = false; row.querySelector('button').textContent = 'Download'; }
      };
      container.appendChild(row);
    }
  } catch(e) { container.innerHTML = `<div style="padding:1rem;color:var(--action-danger-05);">Error: ${e.message}</div>`; }
}
