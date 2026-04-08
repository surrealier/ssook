/* tabs-extra.js — Remaining tab renderers using shared helpers */

/* ── Generic functional tab builder ─────────────────── */
function makeTab(opts) {
  return {
    title: true,
    render() {
      let html = '<div style="display:flex;flex-direction:column;gap:1.5rem;">';
      html += `<div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${opts.heading}</h3>`;
      if (opts.needsModel) html += modelInput(opts.id+'-model');
      if (opts.needsImgDir) html += imgDirInput(opts.id+'-img');
      if (opts.needsLblDir) html += lblDirInput(opts.id+'-lbl');
      if (opts.needsOutDir) html += outDirInput(opts.id+'-out');
      if (opts.multiModel) html += multiModelSlots(opts.id+'-slots', opts.id+'-list');
      if (opts.extraHtml) html += opts.extraHtml;
      if (opts.fields) {
        html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">';
        opts.fields.forEach(([label, type, id, val, extra]) => {
          html += `<div class="form-group"><label class="form-label">${label}</label>`;
          if (type === 'select') html += `<select class="form-input input-normal" id="${id}">${val}</select>`;
          else if (type === 'number') html += `<input type="number" class="form-input input-normal" id="${id}" value="${val}" ${extra||''}>`;
          else html += `<input type="text" class="form-input input-normal" id="${id}" value="${val||''}" ${extra||''}>`;
          html += '</div>';
        });
        html += '</div>';
      }
      if (opts.checks) {
        html += '<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.75rem;">';
        opts.checks.forEach(([label, checked]) => {
          html += `<label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" ${checked?'checked':''}> ${label}</label>`;
        });
        html += '</div>';
      }
      const action = opts.action || t('run');
      const onclick = opts.onclick ? ` onclick="${opts.onclick}"` : ` onclick="App.setStatus('Running...')"`;
      html += `<button class="btn btn-primary" style="margin-top:1rem;"${onclick}>${action}</button></div>`;
      if (opts.progress) {
        html += `<div><div class="progress-track"><div class="progress-fill" id="${opts.id}-prog" style="width:0%"></div></div>
          <span class="text-secondary" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>`;
      }
      if (opts.resultCols) {
        html += `<div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${opts.resultTitle || t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr>${opts.resultCols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
          <tbody id="${opts.id}-results"><tr><td colspan="${opts.resultCols.length}" class="text-secondary" style="text-align:center;padding:2rem;">${opts.resultHint||'—'}</td></tr></tbody></table></div>
        </div>`;
      }
      html += '</div>';
      return html;
    }
  };
}

async function _loadModelTypes() {
  try {
    const c = await API.config();
    return c.model_types || {yolo:'YOLO'};
  } catch(e) { return {yolo:'YOLO'}; }
}
function _modelTypeSelect(id) {
  return `<div class="form-group"><label class="form-label">Model Type</label><select class="form-input input-normal" id="${id}" style="width:auto;"></select></div>`;
}
async function _fillModelTypeSelect(id) {
  const types = await _loadModelTypes();
  const el = document.getElementById(id);
  if (el) el.innerHTML = Object.entries(types).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');
}

/* ── Model Compare ──────────────────────────────────── */
Tabs['model-compare'] = {
  title: true,
  _results: null, _idx: 0,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Setup</h3>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
            <div>
              <div class="form-group"><label class="form-label">Model A</label>
                <div style="display:flex;gap:0.5rem;"><input type="text" class="form-input input-normal" style="flex:1;" readonly id="cmp-a" value="${G.model}"><button class="btn btn-secondary btn-sm" onclick="pickModel('cmp-a')">${t('browse')}</button></div>
              </div>
              ${_modelTypeSelect('cmp-type-a')}
            </div>
            <div>
              <div class="form-group"><label class="form-label">Model B</label>
                <div style="display:flex;gap:0.5rem;"><input type="text" class="form-input input-normal" style="flex:1;" readonly id="cmp-b"><button class="btn btn-secondary btn-sm" onclick="pickModel('cmp-b')">${t('browse')}</button></div>
              </div>
              ${_modelTypeSelect('cmp-type-b')}
            </div>
          </div>
          ${imgDirInput('cmp-img')}
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['model-compare'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="cmp-stop" disabled onclick="API.post('/api/force-stop/compare',{});Tabs['model-compare']._polling=false">${t('stop')}</button>
          </div>
          <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="cmp-prog" style="width:0%"></div></div>
            <span class="text-secondary" id="cmp-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1rem;">
          <input type="range" id="cmp-slider" min="0" max="0" value="0" style="width:100%;accent-color:var(--action-link-05);" oninput="Tabs['model-compare']._showAt(+this.value)" disabled>
          <div style="text-align:center;margin-top:0.25rem;" class="text-secondary" id="cmp-counter">0 / 0</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
          <div class="card" style="padding:1rem;min-height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;" id="cmp-panel-a"><span class="text-muted">Model A</span></div>
          <div class="card" style="padding:1rem;min-height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;" id="cmp-panel-b"><span class="text-muted">Model B</span></div>
        </div>
      </div>`;
  },
  async init() { _fillModelTypeSelect('cmp-type-a'); _fillModelTypeSelect('cmp-type-b'); },
  _polling: false,
  async run() {
    const a = document.getElementById('cmp-a').value, b = document.getElementById('cmp-b').value;
    const imgDir = document.getElementById('cmp-img').value || G.imgDir;
    if (!a||!b||!imgDir) { App.setStatus('Select both models and image directory'); return; }
    document.getElementById('cmp-stop').disabled = false;
    document.getElementById('cmp-prog').style.width = '0%';
    try {
      const r = await API.post('/api/analysis/model-compare', {
        model_a: a, model_b: b,
        model_type_a: document.getElementById('cmp-type-a').value,
        model_type_b: document.getElementById('cmp-type-b').value,
        img_dir: imgDir, conf: 0.25
      });
      if (r.error) { App.setStatus('Error: '+r.error); return; }
      this._polling = true; this._poll();
    } catch(e) { App.setStatus('Error: '+e.message, e.stack); }
  },
  async _poll() {
    if (!this._polling) return;
    try {
      const s = await API.get('/api/analysis/model-compare/status');
      const pct = s.total>0 ? Math.round(s.progress/s.total*100) : 0;
      document.getElementById('cmp-prog').style.width = pct+'%';
      document.getElementById('cmp-status').textContent = s.msg||'';
      if (!s.running) {
        this._polling = false;
        document.getElementById('cmp-stop').disabled = true;
        document.getElementById('cmp-prog').style.width = '100%';
        if (s.results) { this._results = s.results; this._idx = 0; const sl = document.getElementById('cmp-slider'); sl.max = s.results.length-1; sl.disabled = false; this._showAt(0); }
        App.setStatus('Compare complete');
      } else setTimeout(()=>this._poll(), 500);
    } catch(e) { setTimeout(()=>this._poll(), 1000); }
  },
  _showAt(i) {
    if (!this._results||!this._results[i]) return;
    const r = this._results[i];
    document.getElementById('cmp-counter').textContent = (i+1)+' / '+this._results.length;
    // 이미지를 개별 API로 로드 (메모리 최적화)
    const pa = document.getElementById('cmp-panel-a');
    const pb = document.getElementById('cmp-panel-b');
    pa.innerHTML = `<span class="text-muted">Loading...</span>`;
    pb.innerHTML = `<span class="text-muted">Loading...</span>`;
    API.get(`/api/analysis/model-compare/image/${i}/a`).then(d => {
      if (d.image) pa.innerHTML = `<img src="data:image/jpeg;base64,${d.image}" style="max-width:100%;max-height:400px;"><div class="text-secondary" style="margin-top:0.5rem;">Boxes: ${r.count_a} | ${r.ms_a}ms</div>`;
      else pa.innerHTML = `<span class="text-muted">Image not available</span>`;
    });
    API.get(`/api/analysis/model-compare/image/${i}/b`).then(d => {
      if (d.image) pb.innerHTML = `<img src="data:image/jpeg;base64,${d.image}" style="max-width:100%;max-height:400px;"><div class="text-secondary" style="margin-top:0.5rem;">Boxes: ${r.count_b} | ${r.ms_b}ms</div>`;
      else pb.innerHTML = `<span class="text-muted">Image not available</span>`;
    });
  }
};

/* ── Error Analyzer ─────────────────────────────────── */
Tabs['error-analyzer'] = {
  title: true,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">FP/FN Analysis</h3>
          ${modelInput('ea-model')}
          ${_modelTypeSelect('ea-type')}
          ${imgDirInput('ea-img')}
          ${lblDirInput('ea-lbl')}
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
            <div class="form-group"><label class="form-label">IoU Threshold</label><input type="number" class="form-input input-normal" id="ea-iou" value="0.5" min="0.1" max="0.9" step="0.05"></div>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['error-analyzer'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="ea-stop" disabled onclick="API.post('/api/force-stop/error_analysis',{});Tabs['error-analyzer']._polling=false">${t('stop')}</button>
          </div>
          <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="ea-prog" style="width:0%"></div></div>
            <span class="text-secondary" id="ea-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Type</th><th>Count</th><th>Small</th><th>Medium</th><th>Large</th><th>Top</th><th>Center</th><th>Bottom</th></tr></thead>
          <tbody id="ea-results"><tr><td colspan="8" class="text-secondary" style="text-align:center;padding:2rem;">Run analysis to see FP/FN breakdown</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async init() { _fillModelTypeSelect('ea-type'); },
  _polling: false,
  async run() {
    const model_path = document.getElementById('ea-model').value || G.model;
    const img_dir = document.getElementById('ea-img').value || G.imgDir;
    const label_dir = document.getElementById('ea-lbl').value || G.lblDir;
    if (!model_path||!img_dir) { App.setStatus('Select model and image directory'); return; }
    document.getElementById('ea-stop').disabled = false;
    document.getElementById('ea-prog').style.width = '0%';
    try {
      const r = await API.post('/api/analysis/error-analysis', {
        model_path, model_type: document.getElementById('ea-type').value,
        img_dir, label_dir, iou_threshold: +document.getElementById('ea-iou').value, conf: 0.25
      });
      if (r.error) { App.setStatus('Error: '+r.error); return; }
      this._polling = true; this._poll();
    } catch(e) { App.setStatus('Error: '+e.message, e.stack); }
  },
  async _poll() {
    if (!this._polling) return;
    try {
      const s = await API.get('/api/analysis/error-analysis/status');
      const pct = s.total>0 ? Math.round(s.progress/s.total*100) : 0;
      document.getElementById('ea-prog').style.width = pct+'%';
      document.getElementById('ea-status').textContent = s.msg||'';
      if (!s.running) {
        this._polling = false;
        document.getElementById('ea-stop').disabled = true;
        document.getElementById('ea-prog').style.width = '100%';
        if (s.results && (s.results.fp || s.results.fn)) {
          const fp = s.results.fp || {};
          const fn = s.results.fn || {};
          document.getElementById('ea-results').innerHTML =
            `<tr><td>FP (False Positive)</td><td>${fp.count||0}</td><td>${fp.small||0}</td><td>${fp.medium||0}</td><td>${fp.large||0}</td><td>${fp.top||0}</td><td>${fp.center||0}</td><td>${fp.bottom||0}</td></tr>` +
            `<tr><td>FN (False Negative)</td><td>${fn.count||0}</td><td>${fn.small||0}</td><td>${fn.medium||0}</td><td>${fn.large||0}</td><td>${fn.top||0}</td><td>${fn.center||0}</td><td>${fn.bottom||0}</td></tr>`;
        }
        App.setStatus('Error analysis complete');
      } else setTimeout(()=>this._poll(), 500);
    } catch(e) { setTimeout(()=>this._poll(), 1000); }
  }
};

/* ── Conf Optimizer ─────────────────────────────────── */
Tabs['conf-optimizer'] = {
  title: true,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Confidence Threshold Optimizer</h3>
          ${modelInput('co-model')}
          ${_modelTypeSelect('co-type')}
          ${imgDirInput('co-img')}
          ${lblDirInput('co-lbl')}
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
            <div class="form-group"><label class="form-label">Step</label><input type="number" class="form-input input-normal" id="co-step" value="0.05" min="0.01" max="0.1" step="0.01"></div>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['conf-optimizer'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="co-stop" disabled onclick="API.post('/api/force-stop/conf_opt',{});Tabs['conf-optimizer']._polling=false">${t('stop')}</button>
          </div>
          <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="co-prog" style="width:0%"></div></div>
            <span class="text-secondary" id="co-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Class</th><th>Best Threshold</th><th>F1</th><th>Precision</th><th>Recall</th><th></th></tr></thead>
          <tbody id="co-results"><tr><td colspan="6" class="text-secondary" style="text-align:center;padding:2rem;">Run optimizer to find best thresholds</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async init() { _fillModelTypeSelect('co-type'); },
  _polling: false,
  async run() {
    const model_path = document.getElementById('co-model').value || G.model;
    const img_dir = document.getElementById('co-img').value || G.imgDir;
    const label_dir = document.getElementById('co-lbl').value || G.lblDir;
    if (!model_path||!img_dir) { App.setStatus('Select model and image directory'); return; }
    document.getElementById('co-stop').disabled = false;
    document.getElementById('co-prog').style.width = '0%';
    try {
      const r = await API.post('/api/analysis/conf-optimizer', {
        model_path, model_type: document.getElementById('co-type').value,
        img_dir, label_dir, step: +document.getElementById('co-step').value, conf_range: [0.05, 0.95]
      });
      if (r.error) { App.setStatus('Error: '+r.error); return; }
      this._polling = true; this._poll();
    } catch(e) { App.setStatus('Error: '+e.message, e.stack); }
  },
  async _poll() {
    if (!this._polling) return;
    try {
      const s = await API.get('/api/analysis/conf-optimizer/status');
      const pct = s.total>0 ? Math.round(s.progress/s.total*100) : 0;
      document.getElementById('co-prog').style.width = pct+'%';
      document.getElementById('co-status').textContent = s.msg||'';
      if (!s.running) {
        this._polling = false;
        document.getElementById('co-stop').disabled = true;
        document.getElementById('co-prog').style.width = '100%';
        if (s.results) {
          this._results = s.results;
          document.getElementById('co-results').innerHTML = s.results.map((r, i) =>
            `<tr><td>${r.class_name||r.class_id}</td><td>${r.best_threshold}</td><td>${r.best_f1?.toFixed(4)}</td><td>${r.precision?.toFixed(4)}</td><td>${r.recall?.toFixed(4)}</td><td><button class="btn btn-ghost btn-sm" onclick="Tabs['conf-optimizer']._showPR(${i})">📈</button></td></tr>`
          ).join('');
        }
        App.setStatus('Conf optimization complete');
      } else setTimeout(()=>this._poll(), 500);
    } catch(e) { setTimeout(()=>this._poll(), 1000); }
  },
  _results: [],
  _showPR(idx) {
    const r = this._results[idx];
    if (!r || !r.pr_curve) return;
    const curve = r.pr_curve;
    const w = 500, h = 320, pad = 50;
    let svg = `<svg width="${w}" height="${h}" style="background:#1e1e1e;border-radius:8px;">`;
    // Axes
    svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h-pad}" stroke="#555" stroke-width="1"/>`;
    svg += `<line x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}" stroke="#555" stroke-width="1"/>`;
    svg += `<text x="${w/2}" y="${h-8}" fill="#aaa" font-size="11" text-anchor="middle">Recall</text>`;
    svg += `<text x="12" y="${h/2}" fill="#aaa" font-size="11" text-anchor="middle" transform="rotate(-90,12,${h/2})">Precision</text>`;
    // Grid lines
    for (let i = 0; i <= 4; i++) {
      const y = pad + (h - 2*pad) * i / 4;
      const x = pad + (w - 2*pad) * i / 4;
      svg += `<line x1="${pad}" y1="${y}" x2="${w-pad}" y2="${y}" stroke="#333" stroke-width="0.5"/>`;
      svg += `<text x="${pad-5}" y="${y+4}" fill="#888" font-size="9" text-anchor="end">${(1 - i/4).toFixed(1)}</text>`;
      svg += `<text x="${x}" y="${h-pad+14}" fill="#888" font-size="9" text-anchor="middle">${(i/4).toFixed(1)}</text>`;
    }
    // PR curve (Precision vs Recall)
    const sx = (v) => pad + v * (w - 2*pad);
    const sy = (v) => pad + (1 - v) * (h - 2*pad);
    let path = '';
    curve.forEach((pt, i) => { path += `${i===0?'M':'L'}${sx(pt.r).toFixed(1)},${sy(pt.p).toFixed(1)} `; });
    svg += `<path d="${path}" fill="none" stroke="#4a9eff" stroke-width="2"/>`;
    // F1 curve (F1 vs Threshold) in green
    let f1path = '';
    curve.forEach((pt, i) => { f1path += `${i===0?'M':'L'}${sx(pt.t).toFixed(1)},${sy(pt.f1).toFixed(1)} `; });
    svg += `<path d="${f1path}" fill="none" stroke="#4ade80" stroke-width="1.5" stroke-dasharray="4,2"/>`;
    // Best point
    svg += `<circle cx="${sx(r.recall)}" cy="${sy(r.precision)}" r="5" fill="#ff6b6b" stroke="#fff" stroke-width="1.5"/>`;
    // Legend
    svg += `<rect x="${w-pad-120}" y="${pad+5}" width="115" height="50" fill="#1e1e1e" stroke="#444" rx="4"/>`;
    svg += `<line x1="${w-pad-110}" y1="${pad+20}" x2="${w-pad-90}" y2="${pad+20}" stroke="#4a9eff" stroke-width="2"/>`;
    svg += `<text x="${w-pad-85}" y="${pad+24}" fill="#ccc" font-size="10">PR Curve</text>`;
    svg += `<line x1="${w-pad-110}" y1="${pad+36}" x2="${w-pad-90}" y2="${pad+36}" stroke="#4ade80" stroke-width="1.5" stroke-dasharray="4,2"/>`;
    svg += `<text x="${w-pad-85}" y="${pad+40}" fill="#ccc" font-size="10">F1 vs Thresh</text>`;
    svg += '</svg>';
    showDetailModal(`PR Curve — ${r.class_name || r.class_id}`,
      `<div style="text-align:center;">${svg}<div class="text-secondary" style="margin-top:0.5rem;">Best: T=${r.best_threshold} F1=${r.best_f1} P=${r.precision} R=${r.recall}</div></div>`);
  }
};

/* ── Embedding Viewer ───────────────────────────────── */
Tabs['embedding-viewer'] = {
  title: true,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Embedding Visualization</h3>
          ${modelInput('ev-model')}
          <div class="text-secondary" style="font-size:10px;margin-top:-0.5rem;margin-bottom:0.5rem;">💡 모델 타입 선택 불필요 — ONNX 세션을 직접 로드하여 임베딩을 추출합니다.</div>
          ${imgDirInput('ev-img')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">Method</label>
            <select class="form-input input-normal" id="ev-method" style="width:auto;"><option>t-SNE</option><option>UMAP</option><option>PCA</option></select>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['embedding-viewer'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="ev-stop" disabled onclick="Tabs['embedding-viewer']._polling=false">${t('stop')}</button>
          </div>
          <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="ev-prog" style="width:0%"></div></div>
            <span class="text-secondary" id="ev-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1.5rem;min-height:400px;display:flex;align-items:center;justify-content:center;" id="ev-plot">
          <span class="text-muted">2D scatter plot will appear here</span>
        </div>
      </div>`;
  },
  _polling: false,
  async run() {
    const model_path = document.getElementById('ev-model').value || G.model;
    const img_dir = document.getElementById('ev-img').value || G.imgDir;
    if (!model_path||!img_dir) { App.setStatus('Select model and image directory'); return; }
    document.getElementById('ev-stop').disabled = false;
    document.getElementById('ev-prog').style.width = '0%';
    document.getElementById('ev-plot').innerHTML = '<span class="text-muted">Computing...</span>';
    try {
      const r = await API.post('/api/analysis/embedding-viewer', {
        model_path, img_dir, method: document.getElementById('ev-method').value
      });
      if (r.error) { App.setStatus('Error: '+r.error); return; }
      this._polling = true; this._poll();
    } catch(e) { App.setStatus('Error: '+e.message, e.stack); }
  },
  async _poll() {
    if (!this._polling) return;
    try {
      const s = await API.get('/api/analysis/embedding-viewer/status');
      const pct = s.total>0 ? Math.round(s.progress/s.total*100) : 0;
      document.getElementById('ev-prog').style.width = pct+'%';
      document.getElementById('ev-status').textContent = s.msg||'';
      if (!s.running) {
        this._polling = false;
        document.getElementById('ev-stop').disabled = true;
        document.getElementById('ev-prog').style.width = '100%';
        if (s.image) document.getElementById('ev-plot').innerHTML = `<img src="data:image/png;base64,${s.image}" style="max-width:100%;max-height:600px;">`;
        App.setStatus('Embedding visualization complete');
      } else setTimeout(()=>this._poll(), 500);
    } catch(e) { setTimeout(()=>this._poll(), 1000); }
  }
};

/* ── Segmentation ───────────────────────────────────── */
Tabs.segmentation = {
  title: true,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Segmentation Evaluation<a href="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx" class="btn btn-ghost btn-sm" style="margin-left:auto;" target="_blank">📥 YOLO11n-seg ONNX</a></h3>
          ${modelInput('seg-model')}
          ${imgDirInput('seg-img')}
          ${lblDirInput('seg-lbl')}
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.segmentation.run()">${t('run')}</button>
          <button class="btn btn-secondary btn-sm" style="margin-top:1rem;margin-left:0.5rem;" onclick="Tabs.segmentation._showDetail()">📋 상세</button>
        </div>
        <div>
          <div class="progress-track"><div class="progress-fill" id="seg-prog" style="width:0%"></div></div>
          <span class="text-secondary" style="margin-top:0.25rem;display:block;" id="seg-msg">${t('ready')}</span>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Class</th><th>IoU</th><th>Dice</th><th>Images</th></tr></thead>
          <tbody id="seg-results"><tr><td colspan="4" class="text-secondary" style="text-align:center;padding:2rem;">Run evaluation to see mIoU/Dice</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async run() {
    const model_path = document.getElementById('seg-model')?.value || G.model;
    const img_dir = document.getElementById('seg-img')?.value || G.imgDir;
    const label_dir = document.getElementById('seg-lbl')?.value || G.lblDir;
    if (!model_path || !img_dir) { App.setStatus('Select model and image directory'); return; }
    try {
      const r = await API.post('/api/segmentation/run', { model_path, img_dir, label_dir });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/segmentation/status');
      const prog = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const progEl = document.getElementById('seg-prog');
      const msgEl = document.getElementById('seg-msg');
      if (progEl) progEl.style.width = prog + '%';
      if (msgEl) msgEl.textContent = s.msg || `${s.progress}/${s.total}`;
      if (s.results && s.results.length) {
        document.getElementById('seg-results').innerHTML = s.results.map(r =>
          `<tr><td>${r.class_name}</td><td>${r.iou}</td><td>${r.dice}</td><td>${r.images}</td></tr>`
        ).join('');
      }
      if (s.running) setTimeout(() => this._poll(), 500);
      else App.setStatus(s.msg || 'Complete');
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  _showDetail() {
    const tbody = document.getElementById('seg-results');
    if (!tbody) return;
    // Fetch detail from last status
    API.get('/api/segmentation/status').then(s => {
      const detail = s.detail || [];
      let html = '<div style="font-size:12px;">';
      // Summary table
      html += '<h4 style="margin-bottom:0.5rem;">📊 Per-Class Summary</h4>';
      html += tbody.parentElement.parentElement.outerHTML;
      // Processing flow
      html += '<h4 style="margin:1rem 0 0.5rem;">🔄 처리 흐름</h4>';
      html += '<div style="display:flex;gap:0.5rem;align-items:center;font-size:11px;color:#aaa;margin-bottom:1rem;">';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">원본 이미지</span>→';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">전처리 (resize)</span>→';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">모델 추론</span>→';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">argmax → 클래스 마스크</span>→';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">GT 비교 (IoU/Dice)</span>';
      html += '</div>';
      // Per-image detail with sample overlays
      if (detail.length) {
        html += '<h4 style="margin:1rem 0 0.5rem;">🖼️ Per-Image Results</h4>';
        // Sample overlays
        const withOverlay = detail.filter(d => d.overlay);
        if (withOverlay.length) {
          html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:0.5rem;margin-bottom:1rem;">';
          for (const d of withOverlay) {
            html += `<div style="text-align:center;"><img src="data:image/jpeg;base64,${d.overlay}" style="max-width:100%;border-radius:4px;"><div style="font-size:10px;margin-top:2px;">${d.file} (IoU: ${d.iou})</div></div>`;
          }
          html += '</div>';
        }
        html += '<table style="width:100%;"><thead><tr><th>File</th><th>Mean IoU</th><th>Classes</th></tr></thead><tbody>';
        for (const d of detail.slice(0, 100)) {
          const color = d.iou < 0.3 ? 'style="background:rgba(255,0,0,0.08);"' : '';
          html += `<tr ${color}><td>${d.file}</td><td>${d.iou}</td><td>${d.classes}</td></tr>`;
        }
        html += '</tbody></table>';
        if (detail.length > 100) html += `<div style="color:#888;margin-top:0.5rem;">... ${detail.length - 100}개 더</div>`;
      }
      html += '</div>';
      showDetailModal('Segmentation Evaluation — 상세 결과', html);
    });
  }
};

/* ── CLIP ───────────────────────────────────────────── */
Tabs.clip = {
  title: true,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">CLIP Zero-Shot<a href="https://huggingface.co/Xenova/clip-vit-base-patch32/tree/main/onnx" target="_blank" class="btn btn-ghost btn-sm" style="margin-left:auto;">📥 CLIP ONNX</a></h3>
          <div class="form-group">
            <label class="form-label">Image Encoder</label>
            <div style="display:flex;gap:0.5rem;">
              <input type="text" class="form-input input-normal" style="flex:1;" readonly id="clip-img-enc">
              <button class="btn btn-secondary btn-sm" onclick="pickFile('clip-img-enc','ONNX (*.onnx)')">${t('browse')}</button>
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">Text Encoder</label>
            <div style="display:flex;gap:0.5rem;">
              <input type="text" class="form-input input-normal" style="flex:1;" readonly id="clip-txt-enc">
              <button class="btn btn-secondary btn-sm" onclick="pickFile('clip-txt-enc','ONNX (*.onnx)')">${t('browse')}</button>
            </div>
          </div>
          ${imgDirInput('clip-img')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">Class Labels (comma-separated)</label>
            <input type="text" class="form-input input-normal" placeholder="cat, dog, bird, car..." id="clip-labels">
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" id="clip-run-btn" onclick="Tabs.clip.run()">${t('run')}</button>
        </div>
        <div>
          <div class="progress-track"><div class="progress-fill" id="clip-prog" style="width:0%"></div></div>
          <span class="text-secondary" style="margin-top:0.25rem;display:block;" id="clip-msg">${t('ready')}</span>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Label</th><th>Total</th><th>Correct</th><th>Accuracy (%)</th></tr></thead>
          <tbody id="clip-results"><tr><td colspan="4" class="text-secondary" style="text-align:center;padding:2rem;">—</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async run() {
    const imgEnc = document.getElementById('clip-img-enc')?.value;
    const txtEnc = document.getElementById('clip-txt-enc')?.value;
    const imgDir = document.getElementById('clip-img')?.value || G.imgDir;
    const labels = document.getElementById('clip-labels')?.value;
    if (!imgEnc || !txtEnc || !imgDir || !labels) {
      App.setStatus('Please fill all fields'); return;
    }
    try {
      const r = await API.post('/api/clip/run', {
        image_encoder: imgEnc, text_encoder: txtEnc, img_dir: imgDir, labels
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/clip/status');
      const prog = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const progEl = document.getElementById('clip-prog');
      const msgEl = document.getElementById('clip-msg');
      if (progEl) progEl.style.width = prog + '%';
      if (msgEl) msgEl.textContent = s.msg || `${s.progress}/${s.total}`;
      if (s.results && s.results.length) {
        const tbody = document.getElementById('clip-results');
        if (tbody) tbody.innerHTML = s.results.map(r =>
          `<tr><td>${r.label}</td><td>${r.total}</td><td>${r.correct}</td><td>${r.accuracy}</td></tr>`
        ).join('');
        this._detail = s.detail || [];
      }
      if (s.running) setTimeout(() => this._poll(), 500);
      else {
        App.setStatus(s.msg || 'Complete');
        if (this._detail && this._detail.length) {
          const btn = document.getElementById('clip-run-btn');
          if (btn) btn.insertAdjacentHTML('afterend',
            ' <button class="btn btn-secondary btn-sm" onclick="Tabs.clip._showDetail()">📋 상세 보기</button>');
        }
      }
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  _detail: [],
  _showDetail() {
    const d = this._detail;
    if (!d.length) return;
    const wrong = d.filter(x => !x.correct);
    let html = `<div style="margin-bottom:1rem;"><b>총 ${d.length}장</b> | 정답 ${d.filter(x=>x.correct).length} | <span style="color:var(--danger);">오답 ${wrong.length}</span></div>`;
    html += '<div style="max-height:400px;overflow-y:auto;"><table style="width:100%;font-size:12px;"><thead><tr><th>File</th><th>GT</th><th>Pred</th><th>Score</th><th>Top-3</th></tr></thead><tbody>';
    for (const r of d.slice(0, 200)) {
      const color = r.correct ? '' : 'style="background:rgba(255,0,0,0.08);"';
      html += `<tr ${color}><td>${r.file}</td><td>${r.gt}</td><td>${r.pred}</td><td>${r.score}</td><td>${r.top3.map(t=>t[0]+':'+t[1]).join(', ')}</td></tr>`;
    }
    html += '</tbody></table></div>';
    if (d.length > 200) html += `<div class="text-secondary" style="margin-top:0.5rem;">... ${d.length - 200}개 더</div>`;
    showDetailModal('CLIP Zero-Shot — 상세 결과', html);
  }
};

/* ── Embedder Eval ──────────────────────────────────── */
Tabs.embedder = {
  title: true,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Embedder Evaluation<a href="https://huggingface.co/immich-app/ViT-B-32__openai/tree/main" target="_blank" class="btn btn-ghost btn-sm" style="margin-left:auto;">📥 Embedder ONNX</a></h3>
          ${modelInput('emb-model')}
          ${imgDirInput('emb-img')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">Top-K</label>
            <input type="number" class="form-input input-normal" id="emb-k" value="5" min="1" max="100">
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.embedder.run()">${t('run')}</button>
          <button class="btn btn-secondary btn-sm" style="margin-top:1rem;margin-left:0.5rem;" onclick="Tabs.embedder._showDetail()">📋 상세</button>
          <button class="btn btn-secondary btn-sm" style="margin-top:1rem;margin-left:0.5rem;" onclick="Tabs.embedder._compareImages()">🔗 이미지 비교</button>
        </div>
        <div>
          <div class="progress-track"><div class="progress-fill" id="emb-prog" style="width:0%"></div></div>
          <span class="text-secondary" style="margin-top:0.25rem;display:block;" id="emb-msg">${t('ready')}</span>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Class</th><th>Retrieval@1 (%)</th><th>Retrieval@K (%)</th><th>Avg Cosine</th></tr></thead>
          <tbody id="emb-results"><tr><td colspan="4" class="text-secondary" style="text-align:center;padding:2rem;">—</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async run() {
    const modelPath = document.getElementById('emb-model')?.value || G.model;
    const imgDir = document.getElementById('emb-img')?.value || G.imgDir;
    const topK = +(document.getElementById('emb-k')?.value || 5);
    if (!modelPath || !imgDir) { App.setStatus('Please select model and image directory'); return; }
    try {
      const r = await API.post('/api/embedder/run', {
        model_path: modelPath, img_dir: imgDir, top_k: topK
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/embedder/status');
      const prog = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const progEl = document.getElementById('emb-prog');
      const msgEl = document.getElementById('emb-msg');
      if (progEl) progEl.style.width = prog + '%';
      if (msgEl) msgEl.textContent = s.msg || `${s.progress}/${s.total}`;
      if (s.results && s.results.length) {
        const tbody = document.getElementById('emb-results');
        if (tbody) tbody.innerHTML = s.results.map(r =>
          `<tr><td>${r.class}</td><td>${r.retrieval_1}</td><td>${r.retrieval_k}</td><td>${r.avg_cosine}</td></tr>`
        ).join('');
      }
      if (s.running) setTimeout(() => this._poll(), 500);
      else App.setStatus(s.msg || 'Complete');
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  _showDetail() {
    const tbody = document.getElementById('emb-results');
    if (!tbody) return;
    API.get('/api/embedder/status').then(s => {
      const detail = s.detail || [];
      let html = '<div style="font-size:12px;">';
      // Summary
      html += '<h4 style="margin-bottom:0.5rem;">📊 Per-Class Summary</h4>';
      html += tbody.parentElement.parentElement.outerHTML;
      // Processing flow
      html += '<h4 style="margin:1rem 0 0.5rem;">🔄 처리 흐름</h4>';
      html += '<div style="display:flex;gap:0.5rem;align-items:center;font-size:11px;color:#aaa;margin-bottom:1rem;">';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">이미지 로드</span>→';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">전처리</span>→';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">모델 → 벡터 추출</span>→';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">L2 정규화</span>→';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">코사인 유사도 계산</span>→';
      html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">Leave-one-out Retrieval</span>';
      html += '</div>';
      // Per-image retrieval detail
      if (detail.length) {
        html += '<h4 style="margin:1rem 0 0.5rem;">🔍 Per-Image Retrieval</h4>';
        html += '<table style="width:100%;"><thead><tr><th>Query</th><th>GT</th><th>Top-1</th><th>Top-1 File</th><th>Sim</th><th>✓</th><th>Top-3</th></tr></thead><tbody>';
        for (const d of detail.slice(0, 100)) {
          const color = d.correct ? '' : 'style="background:rgba(255,0,0,0.08);"';
          const top3 = (d.top_k||[]).map(t => t[0]+':'+t[1]).join(', ');
          html += `<tr ${color}><td>${d.file}</td><td>${d.gt}</td><td>${d.top1}</td><td>${d.top1_file}</td><td>${d.top1_sim}</td><td>${d.correct?'✓':'✗'}</td><td style="font-size:10px;">${top3}</td></tr>`;
        }
        html += '</tbody></table>';
        if (detail.length > 100) html += `<div style="color:#888;margin-top:0.5rem;">... ${detail.length - 100}개 더</div>`;
      }
      html += '</div>';
      showDetailModal('Embedder Evaluation — 상세 결과', html);
    });
  },
  async _compareImages() {
    const modelPath = document.getElementById('emb-model')?.value || G.model;
    if (!modelPath) { App.setStatus('Select model first'); return; }
    try {
      const r = await API.post('/api/fs/select-multi', { filters: 'Images (*.jpg *.jpeg *.png *.bmp)' });
      if (!r.paths || r.paths.length < 2) { App.setStatus('Select at least 2 images'); return; }
      App.setStatus('Computing similarity...');
      const res = await API.post('/api/embedder/compare', { model_path: modelPath, img_paths: r.paths });
      if (res.error) { App.setStatus('Error: ' + res.error); return; }
      const names = res.names;
      const matrix = res.matrix;
      let html = '<div style="overflow-x:auto;"><table style="width:100%;font-size:11px;"><thead><tr><th></th>';
      names.forEach(n => { html += `<th style="max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${n}">${n}</th>`; });
      html += '</tr></thead><tbody>';
      matrix.forEach((row, i) => {
        html += `<tr><td style="font-weight:600;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${names[i]}">${names[i]}</td>`;
        row.forEach((v, j) => {
          const bg = i === j ? '#333' : v > 0.9 ? 'rgba(74,222,128,0.3)' : v > 0.7 ? 'rgba(74,158,255,0.2)' : v < 0.3 ? 'rgba(255,107,107,0.2)' : '';
          html += `<td style="text-align:center;${bg ? 'background:'+bg+';' : ''}">${v.toFixed(3)}</td>`;
        });
        html += '</tr>';
      });
      html += '</tbody></table></div>';
      showDetailModal(`Embedding Similarity — ${names.length} images`, html);
      App.setStatus('Comparison complete');
    } catch(e) { App.setStatus('Error: ' + e.message); }
  }
};

/* ── Converter ──────────────────────────────────────── */
Tabs.converter = {
  title: true,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Format Converter</h3>
          ${lblDirInput('conv-in')}
          ${outDirInput('conv-out')}
          <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:0.5rem;align-items:center;margin-top:1rem;">
            <select class="form-input input-normal" id="conv-from"><option>YOLO</option><option>COCO JSON</option><option>Pascal VOC</option></select>
            <span style="font-size:20px;color:var(--text-02);">→</span>
            <select class="form-input input-normal" id="conv-to"><option>COCO JSON</option><option>YOLO</option><option>Pascal VOC</option></select>
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.converter.run()">${t('run')}</button>
          <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="conv-prog" style="width:0%"></div></div>
            <span class="text-secondary" id="conv-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
      </div>`;
  },
  async run() {
    const input_dir = document.getElementById('conv-in')?.value || G.lblDir;
    const output_dir = document.getElementById('conv-out')?.value;
    if (!input_dir || !output_dir) { App.setStatus('Select directories'); return; }
    try {
      const r = await API.post('/api/data/converter', {
        input_dir, output_dir,
        from_fmt: document.getElementById('conv-from').value,
        to_fmt: document.getElementById('conv-to').value
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/data/converter/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const p = document.getElementById('conv-prog');
      const m = document.getElementById('conv-msg');
      if (p) p.style.width = pct + '%';
      if (m) m.textContent = s.msg || `${s.progress}/${s.total}`;
      if (s.running) setTimeout(() => this._poll(), 500);
      else App.setStatus(s.msg + (s.results ? ` — ${JSON.stringify(s.results)}` : ''));
    } catch(e) {}
  }
};

/* ── Remapper ───────────────────────────────────────── */
Tabs.remapper = {
  title: true,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Class Remapper</h3>
          ${lblDirInput('remap-lbl')}
          ${outDirInput('remap-out')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">Mapping (old:new, comma-separated)</label>
            <input type="text" class="form-input input-normal" placeholder="0:1, 2:0, 3:1" id="remap-map">
          </div>
          <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" checked id="remap-reindex"> Auto-reindex</label>
          <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="remap-recursive"> 하위 폴더 포함 (Recursive)</label>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.remapper.run()">Apply Remap</button>
        </div>
      </div>`;
  },
  async run() {
    const label_dir = document.getElementById('remap-lbl')?.value || G.lblDir;
    const output_dir = document.getElementById('remap-out')?.value;
    if (!label_dir || !output_dir) { App.setStatus('Select directories'); return; }
    const mapStr = document.getElementById('remap-map')?.value || '';
    const mapping = {};
    mapStr.split(',').forEach(p => { const [a,b] = p.trim().split(':'); if (a && b) mapping[a.trim()] = b.trim(); });
    try {
      const r = await API.post('/api/data/remapper', { label_dir, output_dir, mapping, auto_reindex: document.getElementById('remap-reindex')?.checked, recursive: document.getElementById('remap-recursive')?.checked });
      if (r.error) App.setStatus('Error: ' + r.error);
      else App.setStatus(`Remap complete — ${r.files} files, ${r.labels} labels`);
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  }
};

/* ── Merger ──────────────────────────────────────────── */
Tabs.merger = {
  title: true, _n: 1,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Dataset Merger</h3>
          <div id="merger-datasets" style="display:flex;flex-direction:column;gap:0.5rem;">
            ${imgDirInput('merge-d1')}
          </div>
          <button class="btn btn-secondary btn-sm" style="margin-top:0.5rem;" onclick="Tabs.merger.addDataset()">+ Add Dataset</button>
          ${outDirInput('merge-out')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">dHash Threshold</label>
            <input type="number" class="form-input input-normal" value="10" min="0" max="64" id="merge-dhash">
            <div class="text-secondary" style="font-size:10px;margin-top:0.25rem;">dHash는 이미지의 지각적 해시(perceptual hash)입니다. 값이 낮을수록 더 유사한 이미지만 중복으로 판단합니다. 0=완전 동일, 10=기본값, 64=최대 허용.</div>
          </div>
          <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="merge-recursive"> 하위 폴더 포함 (Recursive)</label>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.merger.run()">Merge</button>
        </div>
      </div>`;
  },
  addDataset() {
    this._n++;
    const c = document.getElementById('merger-datasets');
    const d = document.createElement('div');
    d.innerHTML = imgDirInput(`merge-d${this._n}`);
    c.appendChild(d.firstElementChild || d);
  },
  async run() {
    const datasets = [];
    for (let i = 1; i <= this._n; i++) {
      const v = document.getElementById(`merge-d${i}`)?.value;
      if (v) datasets.push(v);
    }
    const output_dir = document.getElementById('merge-out')?.value;
    if (!datasets.length || !output_dir) { App.setStatus('Select datasets and output'); return; }
    try {
      const r = await API.post('/api/data/merger', { datasets, output_dir, dhash_threshold: +(document.getElementById('merge-dhash')?.value || 10), recursive: document.getElementById('merge-recursive')?.checked });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/data/merger/status');
      if (s.running) { setTimeout(() => this._poll(), 500); return; }
      if (s.results) App.setStatus(`Merge complete — ${s.results.copied} copied, ${s.results.duplicates} duplicates skipped`);
      else App.setStatus(s.msg || 'Complete');
    } catch(e) {}
  }
};

/* ── Smart Sampler ──────────────────────────────────── */
Tabs.sampler = {
  title: true,
  render() {
    return `<div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Smart Sampler</h3>
        ${imgDirInput('samp-img')} ${lblDirInput('samp-lbl')} ${outDirInput('samp-out')}
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
          <div class="form-group"><label class="form-label">Strategy</label><select class="form-input input-normal" id="samp-strat"><option>Random</option><option>Balanced</option><option>Stratified</option></select></div>
          <div class="form-group"><label class="form-label">Target Count</label><input type="number" class="form-input input-normal" id="samp-n" value="500" min="1"></div>
          <div class="form-group"><label class="form-label">Seed</label><input type="number" class="form-input input-normal" id="samp-seed" value="42" min="0"></div>
        </div>
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" checked id="samp-lbl-chk"> Include labels</label>
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="samp-recursive"> 하위 폴더 포함 (Recursive)</label>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.sampler.run()">${t('run')}</button>
      </div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('samp-img')?.value || G.imgDir;
    const output_dir = document.getElementById('samp-out')?.value;
    if (!img_dir || !output_dir) { App.setStatus('Select directories'); return; }
    try {
      const r = await API.post('/api/data/sampler', {
        img_dir, label_dir: document.getElementById('samp-lbl')?.value || G.lblDir, output_dir,
        strategy: document.getElementById('samp-strat').value,
        target_count: +document.getElementById('samp-n').value,
        seed: +document.getElementById('samp-seed').value,
        include_labels: document.getElementById('samp-lbl-chk')?.checked,
        recursive: document.getElementById('samp-recursive')?.checked
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/data/sampler/status');
      if (s.running) { setTimeout(() => this._poll(), 500); return; }
      if (s.results) App.setStatus(`Sampled ${s.results.sampled} from ${s.results.total} images`);
      else App.setStatus(s.msg || 'Complete');
    } catch(e) {}
  }
};

/* ── Label Anomaly ──────────────────────────────────── */
Tabs.anomaly = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Label Anomaly Detector</h3>
        ${imgDirInput('anom-img')} ${lblDirInput('anom-lbl')}
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="anom-recursive"> 하위 폴더 포함 (Recursive)</label>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.anomaly.run()">${t('run')}</button>
        <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="anom-prog" style="width:0%"></div></div>
          <span class="text-secondary" id="anom-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      </div>
      <div class="card" style="padding:1.5rem;"><div class="table-container"><table><thead><tr><th>File</th><th>Type</th><th>Details</th><th>Severity</th></tr></thead>
        <tbody id="anom-results"><tr><td colspan="4" class="text-secondary" style="text-align:center;padding:2rem;">Run detector to find anomalies</td></tr></tbody></table></div></div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('anom-img')?.value || G.imgDir;
    const label_dir = document.getElementById('anom-lbl')?.value || G.lblDir;
    if (!img_dir) { App.setStatus('Select image directory'); return; }
    try {
      const r = await API.post('/api/quality/anomaly', { img_dir, label_dir, recursive: document.getElementById('anom-recursive')?.checked });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/quality/anomaly/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const p = document.getElementById('anom-prog'); if (p) p.style.width = pct + '%';
      const m = document.getElementById('anom-msg'); if (m) m.textContent = s.msg || '';
      if (!s.running && s.results) {
        document.getElementById('anom-results').innerHTML = s.results.length
          ? s.results.map(r => `<tr><td>${r.file}</td><td>${r.type}</td><td style="font-size:11px;">${r.details}</td><td>${r.severity}</td></tr>`).join('')
          : '<tr><td colspan="4" class="text-secondary" style="text-align:center;">No anomalies found ✓</td></tr>';
        App.setStatus(s.msg);
      } else if (s.running) setTimeout(() => this._poll(), 500);
    } catch(e) {}
  }
};

/* ── Image Quality ──────────────────────────────────── */
Tabs.quality = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Image Quality Checker</h3>
        ${imgDirInput('qual-img')}
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="qual-recursive"> 하위 폴더 포함 (Recursive)</label>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.quality.run()">${t('run')}</button>
        <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="qual-prog" style="width:0%"></div></div>
          <span class="text-secondary" id="qual-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      </div>
      <div class="card" style="padding:1.5rem;"><div class="table-container"><table><thead><tr><th>File</th><th>Blur</th><th>Brightness</th><th>Entropy</th><th>Aspect</th><th>Issues</th></tr></thead>
        <tbody id="qual-results"><tr><td colspan="6" class="text-secondary" style="text-align:center;padding:2rem;">—</td></tr></tbody></table></div></div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('qual-img')?.value || G.imgDir;
    if (!img_dir) { App.setStatus('Select image directory'); return; }
    try {
      const r = await API.post('/api/quality/image-quality', { img_dir, recursive: document.getElementById('qual-recursive')?.checked });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/quality/image-quality/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const p = document.getElementById('qual-prog'); if (p) p.style.width = pct + '%';
      const m = document.getElementById('qual-msg'); if (m) m.textContent = s.msg || '';
      if (!s.running && s.results) {
        document.getElementById('qual-results').innerHTML = s.results.map(r =>
          `<tr><td>${r.file}</td><td>${r.blur}</td><td>${r.brightness}</td><td>${r.entropy}</td><td>${r.aspect}</td><td>${r.issues}</td></tr>`).join('');
        App.setStatus(s.msg);
      } else if (s.running) setTimeout(() => this._poll(), 500);
    } catch(e) {}
  }
};

/* ── Near Duplicates ────────────────────────────────── */
Tabs.duplicate = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Near-Duplicate Detector</h3>
        ${imgDirInput('dup-img')}
        <div class="form-group" style="margin-top:0.75rem;"><label class="form-label">Hamming Threshold</label><input type="number" class="form-input input-normal" id="dup-thr" value="10" min="0" max="64"></div>
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="dup-recursive"> 하위 폴더 포함 (Recursive)</label>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.duplicate.run()">${t('run')}</button>
        <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="dup-prog" style="width:0%"></div></div>
          <span class="text-secondary" id="dup-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      </div>
      <div class="card" style="padding:1.5rem;"><div class="table-container"><table><thead><tr><th>Group</th><th>Image A</th><th>Image B</th><th>Distance</th></tr></thead>
        <tbody id="dup-results"><tr><td colspan="4" class="text-secondary" style="text-align:center;padding:2rem;">—</td></tr></tbody></table></div></div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('dup-img')?.value || G.imgDir;
    if (!img_dir) { App.setStatus('Select image directory'); return; }
    try {
      const r = await API.post('/api/quality/duplicate', { img_dir, threshold: +document.getElementById('dup-thr').value, recursive: document.getElementById('dup-recursive')?.checked });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/quality/duplicate/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const p = document.getElementById('dup-prog'); if (p) p.style.width = pct + '%';
      const m = document.getElementById('dup-msg'); if (m) m.textContent = s.msg || '';
      if (!s.running && s.results) {
        document.getElementById('dup-results').innerHTML = s.results.length
          ? s.results.map(r => `<tr><td>${r.group}</td><td>${r.image_a}</td><td>${r.image_b}</td><td>${r.distance}</td></tr>`).join('')
          : '<tr><td colspan="4" class="text-secondary" style="text-align:center;">No duplicates found ✓</td></tr>';
        App.setStatus(s.msg);
      } else if (s.running) setTimeout(() => this._poll(), 500);
    } catch(e) {}
  }
};

/* ── Leaky Split ────────────────────────────────────── */
Tabs.leaky = {
  title: true,
  render() {
    const dirInput = (id, label) => `<div class="form-group">
      <label class="form-label">${label}</label>
      <div style="display:flex;gap:0.5rem;">
        <input type="text" class="form-input input-normal" style="flex:1;" readonly id="${id}">
        <button class="btn btn-secondary btn-sm" onclick="pickDir('${id}')">${t('browse')}</button>
      </div></div>`;
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Leaky Split Detector</h3>
          ${dirInput('leak-train', t('splitter.train'))}
          ${dirInput('leak-val', t('splitter.val'))}
          ${dirInput('leak-test', t('splitter.test'))}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">Hamming Threshold</label>
            <input type="number" class="form-input input-normal" value="10" min="0" max="64">
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.leaky.run()">${t('run')}</button>
          <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="leak-prog" style="width:0%"></div></div>
            <span class="text-secondary" id="leak-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <div class="table-container"><table><thead><tr><th>Split Pair</th><th>Duplicates</th><th>Files</th></tr></thead>
          <tbody id="leak-results"><tr><td colspan="3" class="text-secondary" style="text-align:center;padding:2rem;">Run detector to find cross-split duplicates</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async run() {
    const train_dir = document.getElementById('leak-train')?.value;
    const val_dir = document.getElementById('leak-val')?.value;
    const test_dir = document.getElementById('leak-test')?.value;
    if (!train_dir && !val_dir) { App.setStatus('Select at least 2 split directories'); return; }
    try {
      const r = await API.post('/api/quality/leaky', { train_dir, val_dir, test_dir, threshold: 10 });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/quality/leaky/status');
      if (!s.running && s.results) {
        document.getElementById('leak-results').innerHTML = s.results.length
          ? s.results.map(r => `<tr><td>${r.pair}</td><td>${r.duplicates}</td><td style="font-size:11px;">${r.files}</td></tr>`).join('')
          : '<tr><td colspan="3" class="text-secondary" style="text-align:center;">No leaks found ✓</td></tr>';
        App.setStatus(s.msg || 'Complete');
      } else if (s.running) setTimeout(() => this._poll(), 500);
    } catch(e) {}
  }
};

/* ── Similarity Search ──────────────────────────────── */
Tabs.similarity = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Similarity Search</h3>
        ${imgDirInput('sim-img')}
        <div class="form-group" style="margin-top:0.75rem;"><label class="form-label">Query Image</label>
          <div style="display:flex;gap:0.5rem;"><input type="text" class="form-input input-normal" style="flex:1;" readonly id="sim-query"><button class="btn btn-secondary btn-sm" onclick="pickFile('sim-query','Images (*.jpg *.png)')">${t('browse')}</button></div></div>
        <div class="form-group"><label class="form-label">Top-K</label><input type="number" class="form-input input-normal" id="sim-k" value="10" min="1" max="100"></div>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.similarity.run()">Build Index</button>
        <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="sim-prog" style="width:0%"></div></div>
          <span class="text-secondary" id="sim-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      </div>
      <div class="card" style="padding:1.5rem;"><div class="table-container"><table><thead><tr><th>Rank</th><th>Image</th><th>Distance</th></tr></thead>
        <tbody id="sim-results"><tr><td colspan="3" class="text-secondary" style="text-align:center;padding:2rem;">—</td></tr></tbody></table></div></div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('sim-img')?.value || G.imgDir;
    if (!img_dir) { App.setStatus('Select image directory'); return; }
    try {
      const r = await API.post('/api/quality/similarity', { img_dir, query: document.getElementById('sim-query')?.value || '', top_k: +document.getElementById('sim-k').value });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/quality/similarity/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const p = document.getElementById('sim-prog'); if (p) p.style.width = pct + '%';
      const m = document.getElementById('sim-msg'); if (m) m.textContent = s.msg || '';
      if (!s.running && s.results) {
        document.getElementById('sim-results').innerHTML = s.results.map(r =>
          `<tr><td>${r.rank}</td><td>${r.image}</td><td>${r.distance}</td></tr>`).join('');
        App.setStatus(s.msg || 'Complete');
      } else if (s.running) setTimeout(() => this._poll(), 500);
    } catch(e) {}
  }
};

/* ── Batch Inference ────────────────────────────────── */
Tabs.batch = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Batch Inference</h3>
        ${modelInput('bat-model')} ${imgDirInput('bat-img')} ${outDirInput('bat-out')}
        <div class="form-group" style="margin-top:0.75rem;"><label class="form-label">Output Format</label><select class="form-input input-normal" id="bat-fmt"><option>YOLO txt</option><option>JSON</option><option>CSV</option></select></div>
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="bat-vis"> Save visualizations</label>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.batch.run()">Run Batch</button>
        <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="bat-prog" style="width:0%"></div></div>
          <span class="text-secondary" id="bat-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      </div></div>`;
  },
  async run() {
    const model_path = document.getElementById('bat-model')?.value || G.model;
    const img_dir = document.getElementById('bat-img')?.value || G.imgDir;
    const output_dir = document.getElementById('bat-out')?.value;
    if (!model_path || !img_dir || !output_dir) { App.setStatus('Select model, images, and output'); return; }
    try {
      const r = await API.post('/api/batch/run', {
        model_path, img_dir, output_dir,
        output_format: document.getElementById('bat-fmt').value,
        save_vis: document.getElementById('bat-vis')?.checked || false
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/batch/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const p = document.getElementById('bat-prog'); if (p) p.style.width = pct + '%';
      const m = document.getElementById('bat-msg'); if (m) m.textContent = s.msg || '';
      if (s.running) setTimeout(() => this._poll(), 500);
      else App.setStatus(s.msg || 'Complete');
    } catch(e) {}
  }
};

/* ── Augmentation ───────────────────────────────────── */
Tabs.augmentation = {
  title: true,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">Augmentation Preview</h3>
          ${imgDirInput('aug-img')}
          ${lblDirInput('aug-lbl')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">Augmentation Type</label>
            <select class="form-input input-normal" id="aug-type">
              <option>Mosaic 2×2</option><option>Flip</option><option>Rotate</option>
              <option>Brightness</option>
            </select>
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.augmentation.run()">Preview</button>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
          <div class="card" style="padding:1rem;min-height:250px;display:flex;align-items:center;justify-content:center;" id="aug-orig"><span class="text-muted">Original</span></div>
          <div class="card" style="padding:1rem;min-height:250px;display:flex;align-items:center;justify-content:center;" id="aug-result"><span class="text-muted">Augmented</span></div>
        </div>
      </div>`;
  },
  async run() {
    const img_dir = document.getElementById('aug-img')?.value || G.imgDir;
    if (!img_dir) { App.setStatus('Select image directory'); return; }
    App.setStatus('Generating preview...');
    try {
      const r = await API.post('/api/batch/augmentation', { img_dir, aug_type: document.getElementById('aug-type').value });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      document.getElementById('aug-orig').innerHTML = `<div style="text-align:center;"><img src="data:image/jpeg;base64,${r.original}" style="max-width:100%;max-height:300px;"><div class="text-secondary" style="margin-top:0.25rem;">${r.file}</div></div>`;
      document.getElementById('aug-result').innerHTML = `<img src="data:image/jpeg;base64,${r.augmented}" style="max-width:100%;max-height:300px;">`;
      App.setStatus('Preview ready');
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  }
};
