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
        html += `<div><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="${opts.id}-prog" style="width:0%;height:100%;"></div><span id="${opts.id}-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
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
  return `<div class="form-group"><label class="form-label">${t('common.model_type')}</label><select class="form-input input-normal" id="${id}" style="width:auto;"></select></div>`;
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('cmp.setup')}</h3>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
            <div>
              <div class="form-group"><label class="form-label">${t('cmp.model_a')}</label>
                <div style="display:flex;gap:0.5rem;"><input type="text" class="form-input input-normal" style="flex:1;"  id="cmp-a" value="${G.model}"><button class="btn btn-secondary btn-sm" onclick="pickModel('cmp-a')">${t('browse')}</button></div>
              </div>
              ${_modelTypeSelect('cmp-type-a')}
            </div>
            <div>
              <div class="form-group"><label class="form-label">${t('cmp.model_b')}</label>
                <div style="display:flex;gap:0.5rem;"><input type="text" class="form-input input-normal" style="flex:1;"  id="cmp-b"><button class="btn btn-secondary btn-sm" onclick="pickModel('cmp-b')">${t('browse')}</button></div>
              </div>
              ${_modelTypeSelect('cmp-type-b')}
            </div>
          </div>
          ${imgDirInput('cmp-img')}
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['model-compare'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="cmp-stop" disabled onclick="API.post('/api/force-stop/compare',{});Tabs['model-compare']._polling=false">${t('stop')}</button>
          </div>
          <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="cmp-prog" style="width:0%;height:100%;"></div><span id="cmp-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
            <span class="text-secondary" id="cmp-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1rem;">
          <input type="range" id="cmp-slider" min="0" max="0" value="0" style="width:100%;accent-color:var(--action-link-05);" oninput="Tabs['model-compare']._showAt(+this.value)" disabled>
          <div style="text-align:center;margin-top:0.25rem;" class="text-secondary" id="cmp-counter">0 / 0</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
          <div class="card" style="padding:1rem;min-height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;" id="cmp-panel-a"><span class="text-muted">${t('cmp.model_a')}</span></div>
          <div class="card" style="padding:1rem;min-height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;" id="cmp-panel-b"><span class="text-muted">${t('cmp.model_b')}</span></div>
        </div>
      </div>`;
  },
  async init() { _fillModelTypeSelect('cmp-type-a'); _fillModelTypeSelect('cmp-type-b'); },
  _polling: false,
  async run() {
    const a = document.getElementById('cmp-a').value, b = document.getElementById('cmp-b').value;
    const imgDir = document.getElementById('cmp-img').value || G.imgDir;
    if (!a||!b||!imgDir) { App.setStatus(t('cmp.select_both')); return; }
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
      const _p = document.getElementById('cmp-prog'); if (_p) _p.style.width = pct+'%';
      const _pt = document.getElementById('cmp-prog-text'); if (_pt) _pt.textContent = pct+'%';
      document.getElementById('cmp-status').textContent = s.msg||'';
      if (!s.running) {
        this._polling = false;
        document.getElementById('cmp-stop').disabled = true;
        if (_p) _p.style.width = '100%'; if (_pt) _pt.textContent = '100%';
        if (s.results) { this._results = s.results; this._idx = 0; const sl = document.getElementById('cmp-slider'); sl.max = s.results.length-1; sl.disabled = false; this._showAt(0); }
        App.setStatus(t('cmp.complete'));
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
    pa.innerHTML = `<span class="text-muted">${t('cmp.loading')}</span>`;
    pb.innerHTML = `<span class="text-muted">${t('cmp.loading')}</span>`;
    API.get(`/api/analysis/model-compare/image/${i}/a`).then(d => {
      if (d.image) pa.innerHTML = `<img src="data:image/jpeg;base64,${d.image}" style="max-width:100%;max-height:400px;"><div class="text-secondary" style="margin-top:0.5rem;">Boxes: ${r.count_a} | ${r.ms_a}ms</div>`;
      else pa.innerHTML = `<span class="text-muted">${t('cmp.not_available')}</span>`;
    });
    API.get(`/api/analysis/model-compare/image/${i}/b`).then(d => {
      if (d.image) pb.innerHTML = `<img src="data:image/jpeg;base64,${d.image}" style="max-width:100%;max-height:400px;"><div class="text-secondary" style="margin-top:0.5rem;">Boxes: ${r.count_b} | ${r.ms_b}ms</div>`;
      else pb.innerHTML = `<span class="text-muted">${t('cmp.not_available')}</span>`;
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('ea.title')}</h3>
          ${modelInput('ea-model')}
          ${_modelTypeSelect('ea-type')}
          ${imgDirInput('ea-img')}
          ${lblDirInput('ea-lbl')}
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
            <div class="form-group"><label class="form-label">${t('ea.iou_threshold')}</label><input type="number" class="form-input input-normal" id="ea-iou" value="0.5" min="0.1" max="0.9" step="0.05"></div>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['error-analyzer'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="ea-stop" disabled onclick="API.post('/api/force-stop/error_analysis',{});Tabs['error-analyzer']._polling=false">${t('stop')}</button>
          </div>
          <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="ea-prog" style="width:0%;height:100%;"></div><span id="ea-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
            <span class="text-secondary" id="ea-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Type</th><th>Count</th><th>Small</th><th>Medium</th><th>Large</th><th>Top</th><th>Center</th><th>Bottom</th></tr></thead>
          <tbody id="ea-results"><tr><td colspan="8" class="text-secondary" style="text-align:center;padding:2rem;">${t('ea.run_hint')}</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async init() { _fillModelTypeSelect('ea-type'); },
  _polling: false,
  async run() {
    const model_path = document.getElementById('ea-model').value || G.model;
    const img_dir = document.getElementById('ea-img').value || G.imgDir;
    const label_dir = document.getElementById('ea-lbl').value || G.lblDir;
    if (!model_path||!img_dir) { App.setStatus(t('common.select_model_img')); return; }
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
      const _pt_ea_prog = document.getElementById('ea-prog-text'); if (_pt_ea_prog) _pt_ea_prog.textContent = pct+'%';
      document.getElementById('ea-status').textContent = s.msg||'';
      if (!s.running) {
        this._polling = false;
        document.getElementById('ea-stop').disabled = true;
        document.getElementById('ea-prog').style.width = '100%';
        const _pt100_ea_prog = document.getElementById('ea-prog-text'); if (_pt100_ea_prog) _pt100_ea_prog.textContent = '100%';
        if (s.results && (s.results.fp || s.results.fn)) {
          const fp = s.results.fp || {};
          const fn = s.results.fn || {};
          document.getElementById('ea-results').innerHTML =
            `<tr><td>FP (False Positive)</td><td>${fp.count||0}</td><td>${fp.small||0}</td><td>${fp.medium||0}</td><td>${fp.large||0}</td><td>${fp.top||0}</td><td>${fp.center||0}</td><td>${fp.bottom||0}</td></tr>` +
            `<tr><td>FN (False Negative)</td><td>${fn.count||0}</td><td>${fn.small||0}</td><td>${fn.medium||0}</td><td>${fn.large||0}</td><td>${fn.top||0}</td><td>${fn.center||0}</td><td>${fn.bottom||0}</td></tr>`;
        }
        App.setStatus(t('ea.complete'));
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('co.title')}</h3>
          ${modelInput('co-model')}
          ${_modelTypeSelect('co-type')}
          ${imgDirInput('co-img')}
          ${lblDirInput('co-lbl')}
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
            <div class="form-group"><label class="form-label">${t('co.step')}</label><input type="number" class="form-input input-normal" id="co-step" value="0.05" min="0.01" max="0.1" step="0.01"></div>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['conf-optimizer'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="co-stop" disabled onclick="API.post('/api/force-stop/conf_opt',{});Tabs['conf-optimizer']._polling=false">${t('stop')}</button>
          </div>
          <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="co-prog" style="width:0%;height:100%;"></div><span id="co-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
            <span class="text-secondary" id="co-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Class</th><th>Best Threshold</th><th>F1</th><th>Precision</th><th>Recall</th><th></th></tr></thead>
          <tbody id="co-results"><tr><td colspan="6" class="text-secondary" style="text-align:center;padding:2rem;">${t('co.run_hint')}</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async init() { _fillModelTypeSelect('co-type'); },
  _polling: false,
  async run() {
    const model_path = document.getElementById('co-model').value || G.model;
    const img_dir = document.getElementById('co-img').value || G.imgDir;
    const label_dir = document.getElementById('co-lbl').value || G.lblDir;
    if (!model_path||!img_dir) { App.setStatus(t('common.select_model_img')); return; }
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
      const _pt_co_prog = document.getElementById('co-prog-text'); if (_pt_co_prog) _pt_co_prog.textContent = pct+'%';
      document.getElementById('co-status').textContent = s.msg||'';
      if (!s.running) {
        this._polling = false;
        document.getElementById('co-stop').disabled = true;
        document.getElementById('co-prog').style.width = '100%';
        const _pt100_co_prog = document.getElementById('co-prog-text'); if (_pt100_co_prog) _pt100_co_prog.textContent = '100%';
        if (s.results) {
          this._results = s.results;
          document.getElementById('co-results').innerHTML = s.results.map((r, i) =>
            `<tr><td>${r.class_name||r.class_id}</td><td>${r.best_threshold}</td><td>${r.best_f1?.toFixed(4)}</td><td>${r.precision?.toFixed(4)}</td><td>${r.recall?.toFixed(4)}</td><td><button class="btn btn-ghost btn-sm" onclick="Tabs['conf-optimizer']._showPR(${i})">📈</button></td></tr>`
          ).join('');
        }
        App.setStatus(t('co.complete'));
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('ev.title')}</h3>
          ${modelInput('ev-model')}
          <div class="text-secondary" style="font-size:10px;margin-top:0.25rem;margin-bottom:0.5rem;">${t('ev.model_hint')}</div>
          ${imgDirInput('ev-img')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">${t('ev.method')}</label>
            <select class="form-input input-normal" id="ev-method" style="width:auto;"><option>t-SNE</option><option>UMAP</option><option>PCA</option></select>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['embedding-viewer'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="ev-stop" disabled onclick="Tabs['embedding-viewer']._polling=false">${t('stop')}</button>
          </div>
          <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="ev-prog" style="width:0%;height:100%;"></div><span id="ev-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
            <span class="text-secondary" id="ev-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1.5rem;min-height:400px;display:flex;align-items:center;justify-content:center;" id="ev-plot">
          <span class="text-muted">${t('ev.plot_hint')}</span>
        </div>
      </div>`;
  },
  _polling: false,
  async run() {
    const model_path = document.getElementById('ev-model').value || G.model;
    const img_dir = document.getElementById('ev-img').value || G.imgDir;
    if (!model_path||!img_dir) { App.setStatus(t('common.select_model_img')); return; }
    document.getElementById('ev-stop').disabled = false;
    document.getElementById('ev-prog').style.width = '0%';
    document.getElementById('ev-plot').innerHTML = '<span class="text-muted">' + t('ev.computing') + '</span>';
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
      const _pt_ev_prog = document.getElementById('ev-prog-text'); if (_pt_ev_prog) _pt_ev_prog.textContent = pct+'%';
      document.getElementById('ev-status').textContent = s.msg||'';
      if (!s.running) {
        this._polling = false;
        document.getElementById('ev-stop').disabled = true;
        document.getElementById('ev-prog').style.width = '100%';
        const _pt100_ev_prog = document.getElementById('ev-prog-text'); if (_pt100_ev_prog) _pt100_ev_prog.textContent = '100%';
        if (s.image) document.getElementById('ev-plot').innerHTML = `<img src="data:image/png;base64,${s.image}" style="max-width:100%;max-height:600px;">`;
        App.setStatus(t('ev.complete'));
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('seg.title')}<a href="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx" class="btn btn-ghost btn-sm" style="margin-left:auto;" target="_blank">📥 YOLO11n-seg ONNX</a></h3>
          ${modelInput('seg-model')}
          ${imgDirInput('seg-img')}
          ${lblDirInput('seg-lbl')}
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.segmentation.run()">${t('run')}</button>
          <button class="btn btn-secondary btn-sm" style="margin-top:1rem;margin-left:0.5rem;" onclick="Tabs.segmentation._showDetail()">${t('seg.detail')}</button>
        </div>
        <div>
          <div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="seg-prog" style="width:0%;height:100%;"></div><span id="seg-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
          <span class="text-secondary" style="margin-top:0.25rem;display:block;" id="seg-msg">${t('ready')}</span>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Class</th><th>IoU</th><th>Dice</th><th>Images</th></tr></thead>
          <tbody id="seg-results"><tr><td colspan="4" class="text-secondary" style="text-align:center;padding:2rem;">${t('seg.run_hint')}</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async run() {
    const model_path = document.getElementById('seg-model')?.value || G.model;
    const img_dir = document.getElementById('seg-img')?.value || G.imgDir;
    const label_dir = document.getElementById('seg-lbl')?.value || G.lblDir;
    if (!model_path || !img_dir) { App.setStatus(t('common.select_model_img')); return; }
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
      const segPt = document.getElementById('seg-prog-text'); if (segPt) segPt.textContent = prog + '%';
      if (msgEl) msgEl.textContent = s.msg || `${s.progress}/${s.total}`;
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
      html += '<h4 style="margin-bottom:0.5rem;">' + t('seg.per_class') + '</h4>';
      html += tbody.parentElement.parentElement.outerHTML;
      html += '<h4 style="margin:1rem 0 0.5rem;">' + t('seg.flow') + '</h4>';
      html += '<div style="display:flex;gap:0.5rem;align-items:center;font-size:11px;color:#aaa;margin-bottom:1rem;">';
      t('seg.flow_steps').split(',').forEach((step, i) => {
        if (i > 0) html += '→';
        html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">' + step + '</span>';
      });
      html += '</div>';
      if (detail.length) {
        html += '<h4 style="margin:1rem 0 0.5rem;">' + t('seg.per_image') + '</h4>';
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
        if (detail.length > 100) html += `<div style="color:#888;margin-top:0.5rem;">${t('seg.more', {n: detail.length - 100})}</div>`;
      }
      html += '</div>';
      showDetailModal(t('seg.detail_title'), html);
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('clip.title')}<a href="https://huggingface.co/Xenova/clip-vit-base-patch32/tree/main/onnx" target="_blank" class="btn btn-ghost btn-sm" style="margin-left:auto;">📥 CLIP ONNX</a></h3>
          <div class="form-group">
            <label class="form-label">${t('clip.img_enc')}</label>
            <div style="display:flex;gap:0.5rem;">
              <input type="text" class="form-input input-normal" style="flex:1;"  id="clip-img-enc">
              <button class="btn btn-secondary btn-sm" onclick="pickFile('clip-img-enc','ONNX (*.onnx)')">${t('browse')}</button>
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">${t('clip.txt_enc')}</label>
            <div style="display:flex;gap:0.5rem;">
              <input type="text" class="form-input input-normal" style="flex:1;"  id="clip-txt-enc">
              <button class="btn btn-secondary btn-sm" onclick="pickFile('clip-txt-enc','ONNX (*.onnx)')">${t('browse')}</button>
            </div>
          </div>
          ${imgDirInput('clip-img')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">${t('clip.labels')}</label>
            <input type="text" class="form-input input-normal" placeholder="${t('clip.labels_ph')}" id="clip-labels">
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" id="clip-run-btn" onclick="Tabs.clip.run()">${t('run')}</button>
        </div>
        <div>
          <div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="clip-prog" style="width:0%;height:100%;"></div><span id="clip-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
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
      App.setStatus(t('clip.fill_all')); return;
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
      const clipPt = document.getElementById('clip-prog-text'); if (clipPt) clipPt.textContent = prog + '%';
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
            ' <button class="btn btn-secondary btn-sm" onclick="Tabs.clip._showDetail()">' + t('clip.detail_btn') + '</button>');
        }
      }
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  _detail: [],
  _showDetail() {
    const d = this._detail;
    if (!d.length) return;
    const wrong = d.filter(x => !x.correct);
    let html = `<div style="margin-bottom:1rem;"><b>${t('clip.total', {n: d.length})}</b> | ${t('clip.correct')} ${d.filter(x=>x.correct).length} | <span style="color:var(--danger);">${t('clip.wrong')} ${wrong.length}</span></div>`;
    html += '<div style="max-height:400px;overflow-y:auto;"><table style="width:100%;font-size:12px;"><thead><tr><th>File</th><th>GT</th><th>Pred</th><th>Score</th><th>Top-3</th></tr></thead><tbody>';
    for (const r of d.slice(0, 200)) {
      const color = r.correct ? '' : 'style="background:rgba(255,0,0,0.08);"';
      html += `<tr ${color}><td>${r.file}</td><td>${r.gt}</td><td>${r.pred}</td><td>${r.score}</td><td>${r.top3.map(t=>t[0]+':'+t[1]).join(', ')}</td></tr>`;
    }
    html += '</tbody></table></div>';
    if (d.length > 200) html += `<div class="text-secondary" style="margin-top:0.5rem;">${t('seg.more', {n: d.length - 200})}</div>`;
    showDetailModal(t('clip.detail_title'), html);
  }
};

/* ── Embedder Eval ──────────────────────────────────── */
Tabs.embedder = {
  title: true,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('emb.title')}<a href="https://huggingface.co/immich-app/ViT-B-32__openai/tree/main" target="_blank" class="btn btn-ghost btn-sm" style="margin-left:auto;">📥 Embedder ONNX</a></h3>
          ${modelInput('emb-model')}
          ${imgDirInput('emb-img')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">${t('emb.topk')}</label>            <input type="number" class="form-input input-normal" id="emb-k" value="5" min="1" max="100">
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.embedder.run()">${t('run')}</button>
          <button class="btn btn-secondary btn-sm" style="margin-top:1rem;margin-left:0.5rem;" onclick="Tabs.embedder._showDetail()">${t('emb.detail_btn')}</button>
          <button class="btn btn-secondary btn-sm" style="margin-top:1rem;margin-left:0.5rem;" onclick="Tabs.embedder._compareImages()">${t('emb.compare_btn')}</button>
        </div>
        <div>
          <div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="emb-prog" style="width:0%;height:100%;"></div><span id="emb-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
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
    if (!modelPath || !imgDir) { App.setStatus(t('common.select_model_img')); return; }
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
      const embPt = document.getElementById('emb-prog-text'); if (embPt) embPt.textContent = prog + '%';
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
      html += '<h4 style="margin-bottom:0.5rem;">' + t('seg.per_class') + '</h4>';
      html += tbody.parentElement.parentElement.outerHTML;
      html += '<h4 style="margin:1rem 0 0.5rem;">' + t('emb.flow') + '</h4>';
      html += '<div style="display:flex;gap:0.5rem;align-items:center;font-size:11px;color:#aaa;margin-bottom:1rem;">';
      t('emb.flow_steps').split(',').forEach((step, i) => {
        if (i > 0) html += '→';
        html += '<span style="padding:4px 8px;background:#333;border-radius:4px;">' + step + '</span>';
      });
      html += '</div>';
      if (detail.length) {
        html += '<h4 style="margin:1rem 0 0.5rem;">' + t('emb.per_image') + '</h4>';
        html += '<table style="width:100%;"><thead><tr><th>Query</th><th>GT</th><th>Top-1</th><th>Top-1 File</th><th>Sim</th><th>✓</th><th>Top-3</th></tr></thead><tbody>';
        for (const d of detail.slice(0, 100)) {
          const color = d.correct ? '' : 'style="background:rgba(255,0,0,0.08);"';
          const top3 = (d.top_k||[]).map(t => t[0]+':'+t[1]).join(', ');
          html += `<tr ${color}><td>${d.file}</td><td>${d.gt}</td><td>${d.top1}</td><td>${d.top1_file}</td><td>${d.top1_sim}</td><td>${d.correct?'✓':'✗'}</td><td style="font-size:10px;">${top3}</td></tr>`;
        }
        html += '</tbody></table>';
        if (detail.length > 100) html += `<div style="color:#888;margin-top:0.5rem;">${t('seg.more', {n: detail.length - 100})}</div>`;
      }
      html += '</div>';
      showDetailModal(t('emb.detail_title'), html);
    });
  },
  async _compareImages() {
    const modelPath = document.getElementById('emb-model')?.value || G.model;
    if (!modelPath) { App.setStatus(t('emb.select_model')); return; }
    // Collect images via two sequential picks
    const paths = [];
    const pickNext = () => {
      _showFileBrowser('file', ['.jpg','.jpeg','.png','.bmp'], (path) => {
        paths.push(path);
        if (paths.length < 2) {
          App.setStatus(`Selected ${paths.length} image(s), pick more...`);
          pickNext();
        } else {
          this._runCompare(modelPath, paths);
        }
      });
    };
    pickNext();
  },
  async _runCompare(modelPath, paths) {
    try {
      App.setStatus(t('emb.computing'));
      const res = await API.post('/api/embedder/compare', { model_path: modelPath, img_paths: paths });
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
      showDetailModal(t('emb.sim_title', {n: names.length}), html);
      App.setStatus(t('emb.compare_done'));
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('conv.title')}</h3>
          ${lblDirInput('conv-in')}
          ${outDirInput('conv-out')}
          <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:0.5rem;align-items:center;margin-top:1rem;">
            <select class="form-input input-normal" id="conv-from"><option>YOLO</option><option>COCO JSON</option><option>Pascal VOC</option></select>
            <span style="font-size:20px;color:var(--text-02);">→</span>
            <select class="form-input input-normal" id="conv-to"><option>COCO JSON</option><option>YOLO</option><option>Pascal VOC</option></select>
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.converter.run()">${t('run')}</button>
          <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="conv-prog" style="width:0%;height:100%;"></div><span id="conv-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
            <span class="text-secondary" id="conv-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
      </div>`;
  },
  async run() {
    const input_dir = document.getElementById('conv-in')?.value || G.lblDir;
    const output_dir = document.getElementById('conv-out')?.value;
    if (!input_dir || !output_dir) { App.setStatus(t('common.select_dirs')); return; }
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
      const convPt = document.getElementById('conv-prog-text'); if (convPt) convPt.textContent = pct + '%';
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('remap.title')}</h3>
          ${lblDirInput('remap-lbl')}
          ${outDirInput('remap-out')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">${t('remap.mapping')}</label>
            <input type="text" class="form-input input-normal" placeholder="${t('remap.mapping_ph')}" id="remap-map">
          </div>
          <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" checked id="remap-reindex"> ${t('remap.auto_reindex')}</label>
          <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="remap-recursive"> ${t('common.recursive')}</label>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.remapper.run()">${t('remap.apply')}</button>
          <div id="remap-pbar-wrap" style="display:none;margin-top:0.75rem;">
            <div class="progress-track" style="height:20px;position:relative;">
              <div class="progress-fill" id="remap-pbar" style="width:0%;height:100%;"></div>
              <span id="remap-pbar-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span>
            </div>
          </div>
        </div>
      </div>`;
  },
  async run() {
    const label_dir = document.getElementById('remap-lbl')?.value || G.lblDir;
    const output_dir = document.getElementById('remap-out')?.value;
    if (!label_dir || !output_dir) { App.setStatus(t('common.select_dirs')); return; }
    const mapStr = document.getElementById('remap-map')?.value || '';
    const mapping = {};
    mapStr.split(',').forEach(p => { const [a,b] = p.trim().split(':'); if (a && b) mapping[a.trim()] = b.trim(); });
    const w = document.getElementById('remap-pbar-wrap'); if (w) w.style.display = 'block';
    try {
      const r = await API.post('/api/data/remapper', { label_dir, output_dir, mapping, auto_reindex: document.getElementById('remap-reindex')?.checked, recursive: document.getElementById('remap-recursive')?.checked });
      if (r.error) { App.setStatus('Error: ' + r.error); if (w) w.style.display = 'none'; return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/data/remapper/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const p = document.getElementById('remap-pbar'); if (p) p.style.width = pct + '%';
      const pt = document.getElementById('remap-pbar-text'); if (pt) pt.textContent = pct + '%';
      if (s.running) { setTimeout(() => this._poll(), 300); return; }
      if (p) p.style.width = '100%'; if (pt) pt.textContent = '100%';
      if (s.results) App.setStatus(`Remap complete — ${s.results.files} files, ${s.results.labels} labels`);
      else App.setStatus(s.msg || 'Complete');
      setTimeout(() => { const w = document.getElementById('remap-pbar-wrap'); if (w) w.style.display = 'none'; }, 1000);
    } catch(e) { setTimeout(() => this._poll(), 500); }
  }
};

/* ── Merger ──────────────────────────────────────────── */
Tabs.merger = {
  title: true, _n: 1,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('merger.title')}</h3>
          <div id="merger-datasets" style="display:flex;flex-direction:column;gap:0.5rem;">
            ${imgDirInput('merge-d1')}
          </div>
          <button class="btn btn-secondary btn-sm" style="margin-top:0.5rem;" onclick="Tabs.merger.addDataset()">${t('merger.add_dataset')}</button>
          ${outDirInput('merge-out')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">${t('merger.dhash')}</label>
            <input type="number" class="form-input input-normal" value="10" min="0" max="64" id="merge-dhash">
            <div class="text-secondary" style="font-size:10px;margin-top:0.25rem;">${t('merger.dhash_desc')}</div>
          </div>
          <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="merge-recursive"> ${t('common.recursive')}</label>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.merger.run()">${t('merger.run')}</button>
          <div id="merge-pbar-wrap" style="display:none;margin-top:0.75rem;">
            <div class="progress-track" style="height:20px;position:relative;">
              <div class="progress-fill" id="merge-pbar" style="width:0%;height:100%;"></div>
              <span id="merge-pbar-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span>
            </div>
          </div>
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
    if (!datasets.length || !output_dir) { App.setStatus(t('common.select_dirs')); return; }
    const w = document.getElementById('merge-pbar-wrap'); if (w) w.style.display = 'block';
    try {
      const r = await API.post('/api/data/merger', { datasets, output_dir, dhash_threshold: +(document.getElementById('merge-dhash')?.value || 10), recursive: document.getElementById('merge-recursive')?.checked });
      if (r.error) { App.setStatus('Error: ' + r.error); if (w) w.style.display = 'none'; return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/data/merger/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const p = document.getElementById('merge-pbar'); if (p) p.style.width = pct + '%';
      const pt = document.getElementById('merge-pbar-text'); if (pt) pt.textContent = pct + '%';
      if (s.running) { setTimeout(() => this._poll(), 300); return; }
      if (p) p.style.width = '100%'; if (pt) pt.textContent = '100%';
      if (s.results) App.setStatus(`Merge complete — ${s.results.copied} copied, ${s.results.duplicates} duplicates skipped`);
      else App.setStatus(s.msg || 'Complete');
      setTimeout(() => { const w = document.getElementById('merge-pbar-wrap'); if (w) w.style.display = 'none'; }, 1000);
    } catch(e) { setTimeout(() => this._poll(), 500); }
  }
};

/* ── Smart Sampler ──────────────────────────────────── */
Tabs.sampler = {
  title: true,
  render() {
    return `<div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('sampler.title')}</h3>
        ${imgDirInput('samp-img')} ${lblDirInput('samp-lbl')} ${outDirInput('samp-out')}
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
          <div class="form-group"><label class="form-label">${t('sampler.strategy')}</label><select class="form-input input-normal" id="samp-strat"><option>Random</option><option>Balanced</option><option>Stratified</option></select></div>
          <div class="form-group"><label class="form-label">${t('sampler.target')}</label><input type="number" class="form-input input-normal" id="samp-n" value="500" min="1"></div>
          <div class="form-group"><label class="form-label">${t('sampler.seed')}</label><input type="number" class="form-input input-normal" id="samp-seed" value="42" min="0"></div>
        </div>
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" checked id="samp-lbl-chk"> ${t('sampler.include_lbl')}</label>
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="samp-recursive"> ${t('common.recursive')}</label>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.sampler.run()">${t('run')}</button>
        <div id="samp-pbar-wrap" style="display:none;margin-top:0.75rem;">
          <div class="progress-track" style="height:20px;position:relative;">
            <div class="progress-fill" id="samp-pbar" style="width:0%;height:100%;"></div>
            <span id="samp-pbar-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span>
          </div>
        </div>
      </div>
      <div class="card" style="padding:1.5rem;display:none;" id="samp-result-card">
        <div class="table-container"><table><thead><tr><th>Class</th><th>Before</th><th>After</th></tr></thead>
        <tbody id="samp-results"></tbody></table></div>
        <div class="text-secondary" style="margin-top:0.5rem;" id="samp-summary"></div>
      </div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('samp-img')?.value || G.imgDir;
    const output_dir = document.getElementById('samp-out')?.value;
    if (!img_dir || !output_dir) { App.setStatus(t('common.select_dirs')); return; }
    const w = document.getElementById('samp-pbar-wrap'); if (w) w.style.display = 'block';
    const rc = document.getElementById('samp-result-card'); if (rc) rc.style.display = 'none';
    try {
      const r = await API.post('/api/data/sampler', {
        img_dir, label_dir: document.getElementById('samp-lbl')?.value || G.lblDir, output_dir,
        strategy: document.getElementById('samp-strat').value,
        target_count: +document.getElementById('samp-n').value,
        seed: +document.getElementById('samp-seed').value,
        include_labels: document.getElementById('samp-lbl-chk')?.checked,
        recursive: document.getElementById('samp-recursive')?.checked
      });
      if (r.error) { App.setStatus('Error: ' + r.error); if (w) w.style.display = 'none'; return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/data/sampler/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const p = document.getElementById('samp-pbar'); if (p) p.style.width = pct + '%';
      const pt = document.getElementById('samp-pbar-text'); if (pt) pt.textContent = pct + '%';
      if (s.running) { setTimeout(() => this._poll(), 300); return; }
      if (p) p.style.width = '100%'; if (pt) pt.textContent = '100%';
      if (s.results) {
        App.setStatus(`Sampled ${s.results.sampled} from ${s.results.total} images`);
        const before = s.results.before || {}, after = s.results.after || {};
        const classes = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort((a,b)=>a-b);
        const tb = document.getElementById('samp-results');
        if (tb) tb.innerHTML = classes.map(c => `<tr><td>${c}</td><td>${before[c]||0}</td><td>${after[c]||0}</td></tr>`).join('');
        const sm = document.getElementById('samp-summary'); if (sm) sm.textContent = `Sampled ${s.results.sampled} images`;
        const rc = document.getElementById('samp-result-card'); if (rc) rc.style.display = 'block';
      } else App.setStatus(s.msg || 'Complete');
      setTimeout(() => { const w = document.getElementById('samp-pbar-wrap'); if (w) w.style.display = 'none'; }, 1000);
    } catch(e) { setTimeout(() => this._poll(), 500); }
  }
};

/* ── Label Anomaly ──────────────────────────────────── */
Tabs.anomaly = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('anomaly.title')}</h3>
        ${imgDirInput('anom-img')} ${lblDirInput('anom-lbl')}
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="anom-recursive"> ${t('common.recursive')}</label>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.anomaly.run()">${t('run')}</button>
        <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="anom-prog" style="width:0%;height:100%;"></div><span id="anom-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
          <span class="text-secondary" id="anom-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      </div>
      <div class="card" style="padding:1.5rem;"><div class="table-container"><table><thead><tr><th>File</th><th>Type</th><th>Details</th><th>Severity</th></tr></thead>
        <tbody id="anom-results"><tr><td colspan="4" class="text-secondary" style="text-align:center;padding:2rem;">${t('anomaly.run_hint')}</td></tr></tbody></table></div></div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('anom-img')?.value || G.imgDir;
    const label_dir = document.getElementById('anom-lbl')?.value || G.lblDir;
    if (!img_dir) { App.setStatus(t('eval.select_images')); return; }
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
      const pt = document.getElementById('anom-prog-text'); if (pt) pt.textContent = pct + '%';
      const m = document.getElementById('anom-msg'); if (m) m.textContent = s.msg || '';
      if (!s.running && s.results) {
        document.getElementById('anom-results').innerHTML = s.results.length
          ? s.results.map(r => `<tr><td>${r.file}</td><td>${r.type}</td><td style="font-size:11px;">${r.details}</td><td>${r.severity}</td></tr>`).join('')
          : '<tr><td colspan="4" class="text-secondary" style="text-align:center;">' + t('anomaly.none') + '</td></tr>';
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
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('quality.title')}</h3>
        ${imgDirInput('qual-img')}
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="qual-recursive"> ${t('common.recursive')}</label>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.quality.run()">${t('run')}</button>
        <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="qual-prog" style="width:0%;height:100%;"></div><span id="qual-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
          <span class="text-secondary" id="qual-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      </div>
      <div class="card" style="padding:1.5rem;"><div class="table-container"><table><thead><tr><th>File</th><th>Blur</th><th>Brightness</th><th>Entropy</th><th>Aspect</th><th>Issues</th></tr></thead>
        <tbody id="qual-results"><tr><td colspan="6" class="text-secondary" style="text-align:center;padding:2rem;">—</td></tr></tbody></table></div></div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('qual-img')?.value || G.imgDir;
    if (!img_dir) { App.setStatus(t('eval.select_images')); return; }
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
      const pt = document.getElementById('qual-prog-text'); if (pt) pt.textContent = pct + '%';
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
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('dup.title')}</h3>
        ${imgDirInput('dup-img')}
        <div class="form-group" style="margin-top:0.75rem;"><label class="form-label">${t('dup.hamming')}</label><input type="number" class="form-input input-normal" id="dup-thr" value="10" min="0" max="64"></div>
        <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="dup-recursive"> ${t('common.recursive')}</label>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.duplicate.run()">${t('run')}</button>
        <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="dup-prog" style="width:0%;height:100%;"></div><span id="dup-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
          <span class="text-secondary" id="dup-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      </div>
      <div class="card" style="padding:1.5rem;"><div class="table-container"><table><thead><tr><th>Group</th><th>Image A</th><th>Image B</th><th>Distance</th></tr></thead>
        <tbody id="dup-results"><tr><td colspan="4" class="text-secondary" style="text-align:center;padding:2rem;">—</td></tr></tbody></table></div></div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('dup-img')?.value || G.imgDir;
    if (!img_dir) { App.setStatus(t('eval.select_images')); return; }
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
      const pt = document.getElementById('dup-prog-text'); if (pt) pt.textContent = pct + '%';
      const m = document.getElementById('dup-msg'); if (m) m.textContent = s.msg || '';
      if (!s.running && s.results) {
        document.getElementById('dup-results').innerHTML = s.results.length
          ? s.results.map(r => `<tr><td>${r.group}</td><td>${r.image_a}</td><td>${r.image_b}</td><td>${r.distance}</td></tr>`).join('')
          : '<tr><td colspan="4" class="text-secondary" style="text-align:center;">' + t('dup.none') + '</td></tr>';
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
        <input type="text" class="form-input input-normal" style="flex:1;"  id="${id}">
        <button class="btn btn-secondary btn-sm" onclick="pickDir('${id}')">${t('browse')}</button>
      </div></div>`;
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('leaky.title')}</h3>
          ${dirInput('leak-train', t('splitter.train'))}
          ${dirInput('leak-val', t('splitter.val'))}
          ${dirInput('leak-test', t('splitter.test'))}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">${t('leaky.hamming')}</label>
            <input type="number" class="form-input input-normal" value="10" min="0" max="64">
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.leaky.run()">${t('run')}</button>
          <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="leak-prog" style="width:0%;height:100%;"></div><span id="leak-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
            <span class="text-secondary" id="leak-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <div class="table-container"><table><thead><tr><th>Split Pair</th><th>Duplicates</th><th>Files</th></tr></thead>
          <tbody id="leak-results"><tr><td colspan="3" class="text-secondary" style="text-align:center;padding:2rem;">${t('leaky.run_hint')}</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async run() {
    const train_dir = document.getElementById('leak-train')?.value;
    const val_dir = document.getElementById('leak-val')?.value;
    const test_dir = document.getElementById('leak-test')?.value;
    if (!train_dir && !val_dir) { App.setStatus(t('common.select_dirs')); return; }
    try {
      const r = await API.post('/api/quality/leaky', { train_dir, val_dir, test_dir, threshold: 10 });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/quality/leaky/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const lp = document.getElementById('leak-prog'); if (lp) lp.style.width = pct + '%';
      const lpt = document.getElementById('leak-prog-text'); if (lpt) lpt.textContent = pct + '%';
      if (!s.running && s.results) {
        if (lp) lp.style.width = '100%'; if (lpt) lpt.textContent = '100%';
        document.getElementById('leak-results').innerHTML = s.results.length
          ? s.results.map(r => `<tr><td>${r.pair}</td><td>${r.duplicates}</td><td style="font-size:11px;">${r.files}</td></tr>`).join('')
          : '<tr><td colspan="3" class="text-secondary" style="text-align:center;">' + t('leaky.none') + '</td></tr>';
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
        <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('sim.title')}</h3>
        ${imgDirInput('sim-img')}
        <div class="form-group" style="margin-top:0.75rem;"><label class="form-label">${t('sim.query')}</label>
          <div style="display:flex;gap:0.5rem;"><input type="text" class="form-input input-normal" style="flex:1;"  id="sim-query"><button class="btn btn-secondary btn-sm" onclick="pickFile('sim-query','Images (*.jpg *.png)')">${t('browse')}</button></div></div>
        <div class="form-group"><label class="form-label">${t('sim.topk')}</label><input type="number" class="form-input input-normal" id="sim-k" value="10" min="1" max="100"></div>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.similarity.run()">${t('sim.build')}</button>
        <div style="margin-top:0.5rem;"><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="sim-prog" style="width:0%;height:100%;"></div><span id="sim-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
          <span class="text-secondary" id="sim-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      </div>
      <div class="card" style="padding:1.5rem;"><div class="table-container"><table><thead><tr><th>Rank</th><th>Image</th><th>Distance</th></tr></thead>
        <tbody id="sim-results"><tr><td colspan="3" class="text-secondary" style="text-align:center;padding:2rem;">—</td></tr></tbody></table></div></div></div>`;
  },
  async run() {
    const img_dir = document.getElementById('sim-img')?.value || G.imgDir;
    if (!img_dir) { App.setStatus(t('eval.select_images')); return; }
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
      const pt = document.getElementById('sim-prog-text'); if (pt) pt.textContent = pct + '%';
      const m = document.getElementById('sim-msg'); if (m) m.textContent = s.msg || '';
      if (!s.running && s.results) {
        document.getElementById('sim-results').innerHTML = s.results.map(r =>
          `<tr><td>${r.rank}</td><td>${r.image}</td><td>${r.distance}</td></tr>`).join('');
        App.setStatus(s.msg || 'Complete');
      } else if (s.running) setTimeout(() => this._poll(), 500);
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('aug.title')}</h3>
          ${imgDirInput('aug-img')}
          ${lblDirInput('aug-lbl')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">${t('aug.type')}</label>
            <select class="form-input input-normal" id="aug-type">
              <option>Mosaic 2×2</option><option>Flip</option><option>Rotate</option>
              <option>Brightness</option>
            </select>
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.augmentation.run()">${t('aug.preview')}</button>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
          <div class="card" style="padding:1rem;min-height:250px;display:flex;align-items:center;justify-content:center;" id="aug-orig"><span class="text-muted">${t('aug.original')}</span></div>
          <div class="card" style="padding:1rem;min-height:250px;display:flex;align-items:center;justify-content:center;" id="aug-result"><span class="text-muted">${t('aug.augmented')}</span></div>
        </div>
      </div>`;
  },
  async run() {
    const img_dir = document.getElementById('aug-img')?.value || G.imgDir;
    if (!img_dir) { App.setStatus(t('eval.select_images')); return; }
    App.setStatus(t('aug.generating'));
    try {
      const r = await API.post('/api/batch/augmentation', { img_dir, aug_type: document.getElementById('aug-type').value });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      document.getElementById('aug-orig').innerHTML = `<div style="text-align:center;"><img src="data:image/jpeg;base64,${r.original}" style="max-width:100%;max-height:300px;"><div class="text-secondary" style="margin-top:0.25rem;">${r.file}</div></div>`;
      document.getElementById('aug-result').innerHTML = `<img src="data:image/jpeg;base64,${r.augmented}" style="max-width:100%;max-height:300px;">`;
      App.setStatus(t('aug.ready'));
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  }
};


/* ── Phase 1: Pose Estimation Tab ───────────────────── */
Tabs['pose'] = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('pose.title')}</h3>
        ${modelInput('pose-model')}
        ${imgDirInput('pose-img')}
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
          <div class="form-group"><label class="form-label">${t('pose.model_type')}</label>
            <select class="form-input input-normal" id="pose-model-type">
              <option value="pose_yolo">YOLO-Pose (v8/v11)</option>
              <option value="pose_hrnet">HRNet Pose</option>
              <option value="pose_vitpose">ViTPose</option>
            </select></div>
          <div class="form-group"><label class="form-label">${t('settings.conf')}</label>
            <input type="number" class="form-input input-normal" id="pose-conf" value="0.25" min="0.01" max="0.99" step="0.05"></div>
        </div>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.pose.run()">${t('pose.run')}</button>
      </div>
      <div class="card" style="padding:1.5rem;min-height:300px;" id="pose-result">
        <span class="text-secondary">${t('pose.hint')}</span>
      </div>
    </div>`;
  },
  async run() {
    const model_path = document.getElementById('pose-model')?.value || G.model;
    const img_dir = document.getElementById('pose-img')?.value || G.imgDir;
    if (!model_path) { App.setStatus(t('select_model')); return; }
    if (!img_dir) { App.setStatus(t('eval.select_images')); return; }
    App.setStatus(t('pose.running'));
    try {
      const imgs = await API.post('/api/list-files', {dir: img_dir, exts: ['.jpg','.jpeg','.png','.bmp']});
      const image_path = (imgs.files && imgs.files.length) ? imgs.files[0] : img_dir;
      const r = await API.post('/api/infer/pose', {
        model_path, image_path,
        conf: parseFloat(document.getElementById('pose-conf').value) || 0.25,
        model_type: document.getElementById('pose-model-type').value,
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      let html = `<div style="text-align:center;"><img src="data:image/jpeg;base64,${r.image}" style="max-width:100%;max-height:500px;border-radius:8px;"></div>`;
      html += `<div style="margin-top:1rem;" class="text-secondary">${t('pose.persons')}: ${r.num_persons} | ${t('pose.infer_ms')}: ${r.infer_ms}ms</div>`;
      if (r.detections && r.detections.length) {
        html += '<div class="table-container" style="margin-top:0.75rem;"><table><thead><tr><th>#</th><th>Score</th><th>Keypoints (visible)</th></tr></thead><tbody>';
        r.detections.forEach((d, i) => {
          const visible = d.keypoints.filter(k => k[2] > 0.5).length;
          html += `<tr><td>${i+1}</td><td>${d.score}</td><td>${visible}/17</td></tr>`;
        });
        html += '</tbody></table></div>';
      }
      document.getElementById('pose-result').innerHTML = html;
      App.setStatus(t('pose.done'));
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  }
};

/* ── Phase 1: Instance Segmentation Tab ─────────────── */
Tabs['instance-seg'] = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('instseg.title')}</h3>
        ${modelInput('iseg-model')}
        ${imgDirInput('iseg-img')}
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
          <div class="form-group"><label class="form-label">${t('instseg.model_type')}</label>
            <select class="form-input input-normal" id="iseg-model-type">
              <option value="instseg_yolo">YOLO-Seg Instance (v8/v11)</option>
              <option value="instseg_maskrcnn">Mask R-CNN</option>
            </select></div>
          <div class="form-group"><label class="form-label">${t('settings.conf')}</label>
            <input type="number" class="form-input input-normal" id="iseg-conf" value="0.25" min="0.01" max="0.99" step="0.05"></div>
        </div>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs['instance-seg'].run()">${t('instseg.run')}</button>
      </div>
      <div class="card" style="padding:1.5rem;min-height:300px;" id="iseg-result">
        <span class="text-secondary">${t('instseg.hint')}</span>
      </div>
    </div>`;
  },
  async run() {
    const model_path = document.getElementById('iseg-model')?.value || G.model;
    const img_dir = document.getElementById('iseg-img')?.value || G.imgDir;
    if (!model_path) { App.setStatus(t('select_model')); return; }
    if (!img_dir) { App.setStatus(t('eval.select_images')); return; }
    App.setStatus(t('instseg.running'));
    try {
      const imgs = await API.post('/api/list-files', {dir: img_dir, exts: ['.jpg','.jpeg','.png','.bmp']});
      const image_path = (imgs.files && imgs.files.length) ? imgs.files[0] : img_dir;
      const r = await API.post('/api/infer/instance-seg', {
        model_path, image_path,
        conf: parseFloat(document.getElementById('iseg-conf').value) || 0.25,
        model_type: document.getElementById('iseg-model-type').value,
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      let html = `<div style="text-align:center;"><img src="data:image/jpeg;base64,${r.image}" style="max-width:100%;max-height:500px;border-radius:8px;"></div>`;
      html += `<div style="margin-top:1rem;" class="text-secondary">${t('instseg.instances')}: ${r.num_instances} | ${t('instseg.infer_ms')}: ${r.infer_ms}ms</div>`;
      document.getElementById('iseg-result').innerHTML = html;
      App.setStatus(t('instseg.done'));
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  }
};

/* ── Phase 1: Object Tracking Tab ───────────────────── */
Tabs['tracking'] = {
  title: true,
  _trackerId: null,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('tracking.title')}</h3>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;">
          <div class="form-group"><label class="form-label">${t('tracking.tracker_type')}</label>
            <select class="form-input input-normal" id="track-type">
              <option value="bytetrack">ByteTrack</option>
              <option value="sort">SORT</option>
            </select></div>
        </div>
        <p class="text-secondary" style="margin-top:0.75rem;font-size:12px;">${t('tracking.desc')}</p>
        <div style="display:flex;gap:0.5rem;margin-top:1rem;">
          <button class="btn btn-primary" onclick="Tabs.tracking.create()">${t('tracking.create')}</button>
          <button class="btn btn-secondary" onclick="Tabs.tracking.reset()">${t('tracking.reset')}</button>
        </div>
      </div>
      <div class="card" style="padding:1.5rem;" id="track-result">
        <span class="text-secondary">${t('tracking.hint')}</span>
      </div>
    </div>`;
  },
  async create() {
    try {
      const r = await API.post('/api/tracking/create', {tracker_type: document.getElementById('track-type').value});
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._trackerId = r.tracker_id;
      document.getElementById('track-result').innerHTML = `<div class="text-secondary">${t('tracking.created')}: <strong>${r.tracker_id}</strong> (${r.type})</div>`;
      App.setStatus(t('tracking.created_status'));
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async reset() {
    if (!this._trackerId) { App.setStatus(t('tracking.no_tracker')); return; }
    try {
      await API.post('/api/tracking/reset', {tracker_id: this._trackerId});
      App.setStatus(t('tracking.reset_done'));
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  }
};

/* ── Phase 1: ONNX Model Inspector Tab ──────────────── */
Tabs['inspector'] = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('inspector.title')}</h3>
        ${modelInput('insp-model')}
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.inspector.run()">${t('inspector.inspect')}</button>
      </div>
      <div class="card" style="padding:1.5rem;" id="insp-result">
        <span class="text-secondary">${t('inspector.hint')}</span>
      </div>
    </div>`;
  },
  async run() {
    const path = document.getElementById('insp-model')?.value || G.model;
    if (!path) { App.setStatus(t('select_model')); return; }
    App.setStatus(t('inspector.running'));
    try {
      const r = await API.post('/api/inspector/inspect', {path});
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">';
      // Basic info
      html += `<div class="card-flat" style="padding:1rem;">
        <div class="text-label" style="margin-bottom:0.5rem;">${t('inspector.basic')}</div>
        <table style="width:100%;font-size:12px;">
          <tr><td class="text-secondary">File</td><td>${r.file_name}</td></tr>
          <tr><td class="text-secondary">Size</td><td>${r.file_size_mb} MB</td></tr>
          <tr><td class="text-secondary">Opset</td><td>${r.opset_version}</td></tr>
          <tr><td class="text-secondary">IR Version</td><td>${r.ir_version}</td></tr>
          <tr><td class="text-secondary">Producer</td><td>${r.producer || '—'}</td></tr>
          <tr><td class="text-secondary">Nodes</td><td>${r.num_nodes}</td></tr>
          <tr><td class="text-secondary">Parameters</td><td>${r.num_parameters ? r.num_parameters.toLocaleString() : '—'}</td></tr>
        </table></div>`;
      // I/O
      html += `<div class="card-flat" style="padding:1rem;">
        <div class="text-label" style="margin-bottom:0.5rem;">${t('inspector.io')}</div>
        <div style="font-size:12px;"><strong>Inputs:</strong><ul style="margin:0.25rem 0;">`;
      (r.inputs||[]).forEach(i => { html += `<li>${i.name}: [${i.shape.join(', ')}] (${i.dtype})</li>`; });
      html += `</ul><strong>Outputs:</strong><ul style="margin:0.25rem 0;">`;
      (r.outputs||[]).forEach(o => { html += `<li>${o.name}: [${o.shape.join(', ')}] (${o.dtype})</li>`; });
      html += '</ul></div></div>';
      // EP compatibility
      html += `<div class="card-flat" style="padding:1rem;">
        <div class="text-label" style="margin-bottom:0.5rem;">${t('inspector.ep')}</div>
        <div style="font-size:12px;">`;
      (r.compatible_eps||[]).forEach(ep => { html += `<span class="badge" style="margin:2px;">${ep}</span> `; });
      html += '</div></div>';
      // Op counts (top 10)
      html += `<div class="card-flat" style="padding:1rem;">
        <div class="text-label" style="margin-bottom:0.5rem;">${t('inspector.ops')}</div>
        <div style="font-size:12px;">`;
      const ops = Object.entries(r.node_op_counts||{}).sort((a,b)=>b[1]-a[1]).slice(0,10);
      ops.forEach(([op, cnt]) => { html += `<div style="display:flex;justify-content:space-between;"><span>${op}</span><span>${cnt}</span></div>`; });
      html += '</div></div>';
      // Metadata
      if (r.metadata && Object.keys(r.metadata).length) {
        html += `<div class="card-flat" style="padding:1rem;grid-column:span 2;">
          <div class="text-label" style="margin-bottom:0.5rem;">${t('inspector.metadata')}</div>
          <div style="font-size:11px;max-height:200px;overflow-y:auto;word-break:break-all;">`;
        Object.entries(r.metadata).forEach(([k,v]) => {
          const val = String(v).length > 200 ? String(v).substring(0,200) + '…' : v;
          html += `<div><strong>${k}:</strong> ${val}</div>`;
        });
        html += '</div></div>';
      }
      html += '</div>';
      document.getElementById('insp-result').innerHTML = html;
      App.setStatus(t('inspector.done'));
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  }
};

/* ── Phase 1: Model Profiler Tab ────────────────────── */
Tabs['profiler'] = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('profiler.title')}</h3>
        ${modelInput('prof-model')}
        <div class="form-group" style="margin-top:0.75rem;">
          <label class="form-label">${t('profiler.num_runs')}</label>
          <input type="number" class="form-input input-normal" id="prof-runs" value="20" min="5" max="200" style="width:120px;">
        </div>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs.profiler.run()">${t('profiler.run')}</button>
      </div>
      <div id="prof-result">
        <div class="card" style="padding:1.5rem;"><span class="text-secondary">${t('profiler.hint')}</span></div>
      </div>
    </div>`;
  },
  _sevColor(s) { return s==='high'?'#ef4444':s==='medium'?'#f59e0b':'#22c55e'; },
  _fmtNum(n) { return n>=1e9?(n/1e9).toFixed(2)+'G':n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':String(n); },
  async run() {
    const path = document.getElementById('prof-model')?.value || G.model;
    if (!path) { App.setStatus(t('select_model')); return; }
    App.setStatus(t('profiler.running'));
    try {
      const r = await API.post('/api/profiler/run', {path, num_runs: parseInt(document.getElementById('prof-runs').value)||20});
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      const P = this;
      let html = '';

      // Row 1: Latency + Model Info + Memory
      html += '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">';
      html += `<div class="card-flat" style="padding:1rem;">
        <div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.latency')}</div>
        <table style="width:100%;font-size:12px;">
          <tr><td class="text-secondary">Avg</td><td><strong>${r.avg_infer_ms} ms</strong></td></tr>
          <tr><td class="text-secondary">Min / Max</td><td>${r.min_infer_ms} / ${r.max_infer_ms} ms</td></tr>
          <tr><td class="text-secondary">P50 / P95 / P99</td><td>${r.p50_ms} / ${r.p95_ms} / ${r.p99_ms}</td></tr>
          <tr><td class="text-secondary">FPS</td><td><strong>${r.avg_infer_ms > 0 ? (1000/r.avg_infer_ms).toFixed(1) : '—'}</strong></td></tr>
        </table></div>`;
      html += `<div class="card-flat" style="padding:1rem;">
        <div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.model_info')}</div>
        <table style="width:100%;font-size:12px;">
          <tr><td class="text-secondary">Parameters</td><td>${r.num_parameters ? P._fmtNum(r.num_parameters) : '—'}</td></tr>
          <tr><td class="text-secondary">FLOPs</td><td>${r.estimated_flops ? P._fmtNum(r.estimated_flops) : '—'}</td></tr>
          <tr><td class="text-secondary">MACs</td><td>${r.total_macs ? P._fmtNum(r.total_macs) : '—'}</td></tr>
          <tr><td class="text-secondary">${t('profiler.graph_depth')}</td><td>${r.graph_depth || '—'}</td></tr>
        </table></div>`;
      html += `<div class="card-flat" style="padding:1rem;">
        <div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.memory')}</div>
        <table style="width:100%;font-size:12px;">
          <tr><td class="text-secondary">${t('profiler.weights')}</td><td>${r.weight_memory_mb || 0} MB</td></tr>
          <tr><td class="text-secondary">${t('profiler.peak_act')}</td><td>${r.peak_activation_mb || 0} MB</td></tr>
          <tr><td class="text-secondary">${t('profiler.total')}</td><td><strong>${r.total_memory_mb || 0} MB</strong></td></tr>
        </table></div>`;
      html += '</div>';

      // Row 2: I/O info
      if (r.input_info && r.input_info.length) {
        html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem;">';
        html += `<div class="card-flat" style="padding:1rem;"><div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.inputs')}</div>`;
        r.input_info.forEach(i => { html += `<div style="font-size:11px;margin-bottom:0.25rem;"><strong>${i.name}</strong> [${i.shape.join('×')}] <span class="text-secondary">${i.type}</span></div>`; });
        html += '</div>';
        html += `<div class="card-flat" style="padding:1rem;"><div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.outputs')}</div>`;
        (r.output_info||[]).forEach(o => { html += `<div style="font-size:11px;margin-bottom:0.25rem;"><strong>${o.name}</strong> [${o.shape.join('×')}] <span class="text-secondary">${o.type}</span></div>`; });
        html += '</div></div>';
      }

      // Row 3: Op Distribution (bar chart via CSS)
      if (r.op_type_summary && r.op_type_summary.length) {
        html += `<div class="card-flat" style="padding:1rem;margin-top:1rem;">
          <div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.op_dist')}</div>
          <div style="display:flex;flex-direction:column;gap:4px;max-height:300px;overflow-y:auto;">`;
        r.op_type_summary.slice(0, 15).forEach(op => {
          html += `<div style="display:grid;grid-template-columns:100px 1fr 60px 50px;align-items:center;font-size:11px;gap:8px;">
            <span style="font-weight:600;">${op.op_type}</span>
            <div style="background:var(--bg-02);border-radius:4px;height:16px;overflow:hidden;">
              <div style="width:${Math.max(op.time_pct,1)}%;height:100%;background:var(--accent);border-radius:4px;"></div>
            </div>
            <span class="text-secondary">${op.time_pct}%</span>
            <span class="text-secondary">×${op.count}</span>
          </div>`;
        });
        html += '</div></div>';
      }

      // Row 4: Quantization Readiness
      html += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem;">`;
      const qPct = Math.round((r.quantizable_ratio||0)*100);
      const qColor = qPct > 70 ? '#22c55e' : qPct > 40 ? '#f59e0b' : '#ef4444';
      html += `<div class="card-flat" style="padding:1rem;">
        <div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.quant_ready')}</div>
        <div style="display:flex;align-items:center;gap:1rem;">
          <div style="width:60px;height:60px;border-radius:50%;border:4px solid ${qColor};display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:${qColor};">${qPct}%</div>
          <div style="font-size:12px;">
            <div>Est. INT8 Speedup: <strong>${r.estimated_int8_speedup||1}x</strong></div>
            ${r.non_quantizable_ops && r.non_quantizable_ops.length ? `<div class="text-secondary" style="margin-top:0.25rem;">${t('profiler.non_quant')}: ${r.non_quantizable_ops.join(', ')}</div>` : ''}
          </div>
        </div></div>`;

      // Optimization Suggestions
      html += `<div class="card-flat" style="padding:1rem;">
        <div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.opt_suggest')}</div>`;
      if (r.optimization_suggestions && r.optimization_suggestions.length) {
        r.optimization_suggestions.forEach(s => {
          html += `<div style="font-size:11px;padding:0.4rem 0.6rem;margin-bottom:0.25rem;background:var(--bg-02);border-radius:6px;border-left:3px solid var(--accent);">💡 ${s}</div>`;
        });
      } else { html += `<span class="text-secondary" style="font-size:12px;">${t('profiler.no_suggest')}</span>`; }
      html += '</div></div>';

      // Row 5: Bottleneck Diagnosis
      if (r.bottleneck_diagnosis && r.bottleneck_diagnosis.length) {
        html += `<div class="card-flat" style="padding:1rem;margin-top:1rem;">
          <div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.diagnosis')}</div>
          <div style="display:flex;flex-direction:column;gap:0.5rem;max-height:400px;overflow-y:auto;">`;
        r.bottleneck_diagnosis.slice(0, 10).forEach(d => {
          html += `<div style="padding:0.6rem;background:var(--bg-02);border-radius:8px;border-left:4px solid ${P._sevColor(d.severity)};">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:12px;font-weight:600;">${d.op_type} <span class="text-secondary" style="font-weight:400;">${d.layer}</span></span>
              <div style="display:flex;gap:0.5rem;align-items:center;">
                <span style="font-size:10px;padding:2px 6px;border-radius:4px;background:${P._sevColor(d.severity)}20;color:${P._sevColor(d.severity)};font-weight:600;">${d.severity.toUpperCase()}</span>
                <span style="font-size:11px;font-weight:600;">${d.time_us} μs</span>
                <span style="font-size:10px;padding:2px 6px;border-radius:4px;background:var(--bg-03);">${d.category}</span>
              </div>
            </div>
            ${d.suggestion ? `<div style="font-size:11px;color:var(--text-04);margin-top:0.3rem;">→ ${d.suggestion}</div>` : ''}
          </div>`;
        });
        html += '</div></div>';
      }

      // Row 6: Top Bottleneck Layers table
      if (r.top_bottlenecks && r.top_bottlenecks.length) {
        html += `<div class="card-flat" style="padding:1rem;margin-top:1rem;">
          <div class="text-label" style="margin-bottom:0.5rem;">${t('profiler.bottlenecks')}</div>
          <div class="table-container"><table><thead><tr><th>Layer</th><th>Op</th><th>Time (μs)</th></tr></thead><tbody>`;
        r.top_bottlenecks.forEach(l => {
          html += `<tr><td style="font-size:11px;max-width:300px;overflow:hidden;text-overflow:ellipsis;">${l.name}</td><td>${l.op_type}</td><td>${l.duration_us}</td></tr>`;
        });
        html += '</tbody></table></div></div>';
      }

      document.getElementById('prof-result').innerHTML = html;
      App.setStatus(t('profiler.done'));
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  }
};

/* ── Calibration / Quantization Tab ─────────────────── */
Tabs['calibration'] = {
  title: true,
  _timer: null,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('calib.title')}</h3>
        ${modelInput('calib-model')}
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.75rem;">
          <div class="form-group"><label class="form-label">${t('calib.method')}</label>
            <select class="form-input input-normal" id="calib-method" onchange="Tabs['calibration']._onMethod()">
              <option value="dynamic">${t('calib.dynamic')}</option>
              <option value="static">${t('calib.static')}</option>
              <option value="fp16">${t('calib.fp16')}</option>
            </select></div>
          <div class="form-group"><label class="form-label">${t('calib.output')}</label>
            <div style="display:flex;gap:0.5rem;"><input type="text" class="form-input input-normal" style="flex:1;" id="calib-out" placeholder="${t('calib.auto_name')}"><button class="btn btn-secondary btn-sm" onclick="pickDir('calib-out')">${t('browse')}</button></div></div>
        </div>
        <div id="calib-static-opts" style="display:none;margin-top:0.75rem;">
          ${imgDirInput('calib-img')}
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-top:0.75rem;">
            <div class="form-group"><label class="form-label">${t('calib.max_images')}</label>
              <input type="number" class="form-input input-normal" id="calib-max" value="100" min="10" max="1000"></div>
            <div class="form-group"><label class="form-label">${t('calib.weight_type')}</label>
              <select class="form-input input-normal" id="calib-wt"><option value="int8">INT8</option><option value="uint8" selected>UINT8</option></select></div>
            <div class="form-group"><label class="form-label">${t('calib.act_type')}</label>
              <select class="form-input input-normal" id="calib-at"><option value="uint8" selected>UINT8</option><option value="int8">INT8</option></select></div>
            <div class="form-group"><label class="form-label">${t('calib.format')}</label>
              <select class="form-input input-normal" id="calib-fmt"><option value="QDQ" selected>QDQ</option><option value="QOperator">QOperator</option></select></div>
          </div>
          <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" id="calib-perchan" checked> ${t('calib.per_channel')}</label>
        </div>
        <div id="calib-dynamic-opts" style="margin-top:0.75rem;">
          <div class="form-group" style="max-width:200px;"><label class="form-label">${t('calib.weight_type')}</label>
            <select class="form-input input-normal" id="calib-dyn-wt"><option value="uint8" selected>UINT8</option><option value="int8">INT8</option></select></div>
        </div>
        <button class="btn btn-primary" style="margin-top:1rem;" onclick="Tabs['calibration'].run()">${t('calib.run')}</button>
      </div>
      <div><div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="calib-prog" style="width:0%;height:100%;"></div><span id="calib-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
        <span class="text-secondary" id="calib-msg" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
      <div class="card" id="calib-result" style="padding:1.5rem;display:none;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('calib.result')}</h3>
        <div id="calib-result-body"></div>
      </div>
    </div>`;
  },
  _onMethod() {
    const m = document.getElementById('calib-method').value;
    document.getElementById('calib-static-opts').style.display = m === 'static' ? '' : 'none';
    document.getElementById('calib-dynamic-opts').style.display = m === 'dynamic' ? '' : 'none';
  },
  async run() {
    const path = document.getElementById('calib-model')?.value || G.model;
    if (!path) { App.setStatus(t('select_model')); return; }
    const method = document.getElementById('calib-method').value;
    const body = { model_path: path, method, output_path: document.getElementById('calib-out').value || '' };
    if (method === 'static') {
      body.calibration_dir = document.getElementById('calib-img')?.value || '';
      body.max_images = parseInt(document.getElementById('calib-max').value) || 100;
      body.weight_type = document.getElementById('calib-wt').value;
      body.activation_type = document.getElementById('calib-at').value;
      body.quant_format = document.getElementById('calib-fmt').value;
      body.per_channel = document.getElementById('calib-perchan').checked;
      if (!body.calibration_dir) { App.setStatus(t('calib.need_img')); return; }
    } else if (method === 'dynamic') {
      body.weight_type = document.getElementById('calib-dyn-wt').value;
    }
    const r = await API.post('/api/quantize', body);
    if (r.error) { App.setStatus('Error: ' + r.error); return; }
    App.setStatus(t('calib.started'));
    document.getElementById('calib-result').style.display = 'none';
    this._poll();
  },
  _poll() {
    if (this._timer) clearInterval(this._timer);
    this._timer = setInterval(async () => {
      const s = await API.get('/api/quantize/status');
      const prog = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      const el = document.getElementById('calib-prog');
      if (el) el.style.width = (s.running ? Math.max(prog, 5) : (s.msg === 'Complete' ? 100 : 0)) + '%';
      const pt = document.getElementById('calib-prog-text');
      if (pt) pt.textContent = s.running ? prog + '%' : (s.msg === 'Complete' ? '100%' : '');
      const msg = document.getElementById('calib-msg');
      if (msg) msg.textContent = s.msg || '';
      if (!s.running) {
        clearInterval(this._timer); this._timer = null;
        if (s.results && s.results.output_path) {
          const res = s.results;
          let html = `<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
            <div><span class="text-secondary">${t('calib.method')}</span><br><strong>${res.method.toUpperCase()}</strong></div>
            <div><span class="text-secondary">${t('calib.output')}</span><br><strong style="word-break:break-all;font-size:12px;">${res.output_path}</strong></div>
            <div><span class="text-secondary">${t('calib.orig_size')}</span><br><strong>${res.original_size_mb} MB</strong></div>
            <div><span class="text-secondary">${t('calib.new_size')}</span><br><strong>${res.quantized_size_mb} MB</strong></div>
            <div><span class="text-secondary">${t('calib.ratio')}</span><br><strong>${res.compression_ratio}x</strong></div>
            ${res.calibration_images ? `<div><span class="text-secondary">${t('calib.cal_images')}</span><br><strong>${res.calibration_images}</strong></div>` : ''}
          </div>
          <button class="btn btn-secondary" style="margin-top:1rem;" onclick="Tabs['calibration']._openInBenchmark('${res.output_path.replace(/\\/g,'\\\\').replace(/'/g,"\\'")}')">${t('calib.compare_bench')}</button>`;
          document.getElementById('calib-result-body').innerHTML = html;
          document.getElementById('calib-result').style.display = '';
          App.setStatus(t('calib.done'));
        } else if (s.msg && s.msg.startsWith('Error')) {
          App.setStatus(s.msg);
        }
      }
    }, 500);
  },
  _openInBenchmark(outputPath) {
    const modelEl = document.getElementById('calib-model');
    const origPath = modelEl ? modelEl.value || G.model : G.model;
    App.switchTab('benchmark');
    setTimeout(() => {
      const slots = document.querySelectorAll('.bench-model-input');
      if (slots[0]) slots[0].value = origPath;
      if (slots.length > 1) { slots[1].value = outputPath; }
      else {
        const addBtn = document.querySelector('[onclick*="addModelSlot"], .bench-add-slot');
        if (addBtn) { addBtn.click(); setTimeout(() => { const s2 = document.querySelectorAll('.bench-model-input'); if(s2[1]) s2[1].value = outputPath; }, 100); }
      }
    }, 200);
  }
};

/* ── Phase 1: VLM Tab (placeholder) ─────────────────── */
Tabs['vlm'] = {
  title: true,
  render() {
    return `<div style="display:flex;flex-direction:column;gap:1.5rem;">
      <div class="card" style="padding:1.5rem;">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('vlm.title')}</h3>
        ${modelInput('vlm-model')}
        ${imgDirInput('vlm-img')}
        <div class="form-group" style="margin-top:0.75rem;">
          <label class="form-label">${t('vlm.task')}</label>
          <select class="form-input input-normal" id="vlm-task">
            <option value="vqa">Visual Question Answering (VQA)</option>
            <option value="caption">Image Captioning</option>
            <option value="grounding">Visual Grounding</option>
          </select>
        </div>
        <div class="form-group" style="margin-top:0.75rem;">
          <label class="form-label">${t('vlm.prompt')}</label>
          <input type="text" class="form-input input-normal" id="vlm-prompt" placeholder="${t('vlm.prompt_hint')}" value="">
        </div>
        <p class="text-secondary" style="margin-top:0.75rem;font-size:12px;">${t('vlm.desc')}</p>
      </div>
    </div>`;
  }
};
