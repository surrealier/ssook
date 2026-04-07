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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">Setup</h3>
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
            <button class="btn btn-danger btn-sm" id="cmp-stop" disabled onclick="Tabs['model-compare']._polling=false">${t('stop')}</button>
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
    } catch(e) { App.setStatus('Error: '+e.message); }
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
    document.getElementById('cmp-panel-a').innerHTML = `<img src="data:image/jpeg;base64,${r.image_a}" style="max-width:100%;max-height:400px;"><div class="text-secondary" style="margin-top:0.5rem;">Boxes: ${r.boxes_a} | ${r.time_a}ms</div>`;
    document.getElementById('cmp-panel-b').innerHTML = `<img src="data:image/jpeg;base64,${r.image_b}" style="max-width:100%;max-height:400px;"><div class="text-secondary" style="margin-top:0.5rem;">Boxes: ${r.boxes_b} | ${r.time_b}ms</div>`;
  }
};

/* ── Error Analyzer ─────────────────────────────────── */
Tabs['error-analyzer'] = {
  title: true,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">FP/FN Analysis</h3>
          ${modelInput('ea-model')}
          ${_modelTypeSelect('ea-type')}
          ${imgDirInput('ea-img')}
          ${lblDirInput('ea-lbl')}
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
            <div class="form-group"><label class="form-label">IoU Threshold</label><input type="number" class="form-input input-normal" id="ea-iou" value="0.5" min="0.1" max="0.9" step="0.05"></div>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['error-analyzer'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="ea-stop" disabled onclick="Tabs['error-analyzer']._polling=false">${t('stop')}</button>
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
    } catch(e) { App.setStatus('Error: '+e.message); }
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
        if (s.results) {
          document.getElementById('ea-results').innerHTML = s.results.map(r =>
            `<tr><td>${r.type}</td><td>${r.count}</td><td>${r.small}</td><td>${r.medium}</td><td>${r.large}</td><td>${r.top}</td><td>${r.center}</td><td>${r.bottom}</td></tr>`
          ).join('');
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">Confidence Threshold Optimizer</h3>
          ${modelInput('co-model')}
          ${_modelTypeSelect('co-type')}
          ${imgDirInput('co-img')}
          ${lblDirInput('co-lbl')}
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:0.75rem;">
            <div class="form-group"><label class="form-label">Step</label><input type="number" class="form-input input-normal" id="co-step" value="0.05" min="0.01" max="0.1" step="0.01"></div>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" onclick="Tabs['conf-optimizer'].run()">${t('run')}</button>
            <button class="btn btn-danger btn-sm" id="co-stop" disabled onclick="Tabs['conf-optimizer']._polling=false">${t('stop')}</button>
          </div>
          <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" id="co-prog" style="width:0%"></div></div>
            <span class="text-secondary" id="co-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span></div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Class</th><th>Best Threshold</th><th>F1</th><th>Precision</th><th>Recall</th></tr></thead>
          <tbody id="co-results"><tr><td colspan="5" class="text-secondary" style="text-align:center;padding:2rem;">Run optimizer to find best thresholds</td></tr></tbody></table></div>
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
    } catch(e) { App.setStatus('Error: '+e.message); }
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
          document.getElementById('co-results').innerHTML = s.results.map(r =>
            `<tr><td>${r.class_name||r.class_id}</td><td>${r.best_threshold}</td><td>${r.f1?.toFixed(4)}</td><td>${r.precision?.toFixed(4)}</td><td>${r.recall?.toFixed(4)}</td></tr>`
          ).join('');
        }
        App.setStatus('Conf optimization complete');
      } else setTimeout(()=>this._poll(), 500);
    } catch(e) { setTimeout(()=>this._poll(), 1000); }
  }
};

/* ── Embedding Viewer ───────────────────────────────── */
Tabs['embedding-viewer'] = {
  title: true,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">Embedding Visualization</h3>
          ${modelInput('ev-model')}
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
    } catch(e) { App.setStatus('Error: '+e.message); }
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
Tabs.segmentation = makeTab({
  id: 'seg', heading: '<span style="display:flex;align-items:center;width:100%;">Segmentation Evaluation<a href="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx" class="btn btn-ghost btn-sm" style="margin-left:auto;">📥 YOLO11n-seg ONNX</a></span>',
  needsModel: true, needsImgDir: true, needsLblDir: true,
  resultCols: ['Class','IoU','Dice','Images'],
  resultHint: 'Run evaluation to see mIoU/Dice', progress: true,
});

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
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="App.setStatus('Running CLIP...')">${t('run')}</button>
        </div>
      </div>`;
  }
};

/* ── Embedder Eval ──────────────────────────────────── */
Tabs.embedder = makeTab({
  id: 'emb', heading: '<span style="display:flex;align-items:center;width:100%;">Embedder Evaluation<a href="https://huggingface.co/immich-app/ViT-B-32__openai/tree/main" target="_blank" class="btn btn-ghost btn-sm" style="margin-left:auto;">📥 Embedder ONNX</a></span>',
  needsModel: true, needsImgDir: true,
  fields: [['Top-K','number','emb-k','5','min="1" max="100"']],
  resultCols: ['Class','Retrieval@1','Retrieval@K','Avg Cosine'],
  progress: true,
});

/* ── Converter ──────────────────────────────────────── */
Tabs.converter = {
  title: true,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">Format Converter</h3>
          ${lblDirInput('conv-in')}
          ${outDirInput('conv-out')}
          <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:0.5rem;align-items:center;margin-top:1rem;">
            <select class="form-input input-normal"><option>YOLO</option><option>COCO JSON</option><option>Pascal VOC</option></select>
            <span style="font-size:20px;color:var(--text-02);">→</span>
            <select class="form-input input-normal"><option>COCO JSON</option><option>YOLO</option><option>Pascal VOC</option></select>
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="App.setStatus('Converting...')">${t('run')}</button>
          <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" style="width:0%"></div></div></div>
        </div>
      </div>`;
  }
};

/* ── Remapper ───────────────────────────────────────── */
Tabs.remapper = makeTab({
  id: 'remap', heading: 'Class Remapper',
  needsLblDir: true, needsOutDir: true,
  checks: [['Auto-reindex', true]],
  resultCols: ['Original ID','Original Name','→','New ID','New Name'],
  resultHint: 'Load labels to see class mapping', action: 'Apply Remap',
});

/* ── Merger ──────────────────────────────────────────── */
Tabs.merger = {
  title: true, _n: 1,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">Dataset Merger</h3>
          <div id="merger-datasets" style="display:flex;flex-direction:column;gap:0.5rem;">
            ${imgDirInput('merge-d1')}
          </div>
          <button class="btn btn-secondary btn-sm" style="margin-top:0.5rem;" onclick="Tabs.merger.addDataset()">+ Add Dataset</button>
          ${outDirInput('merge-out')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">dHash Threshold</label>
            <input type="number" class="form-input input-normal" value="10" min="0" max="64">
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="App.setStatus('Merging...')">Merge</button>
        </div>
      </div>`;
  },
  addDataset() {
    this._n++;
    const c = document.getElementById('merger-datasets');
    const d = document.createElement('div');
    d.innerHTML = imgDirInput(`merge-d${this._n}`);
    c.appendChild(d.firstElementChild || d);
  }
};

/* ── Smart Sampler ──────────────────────────────────── */
Tabs.sampler = makeTab({
  id: 'samp', heading: 'Smart Sampler',
  needsImgDir: true, needsLblDir: true, needsOutDir: true,
  fields: [
    ['Strategy','select','samp-strat','<option>Random</option><option>Balanced</option><option>Stratified</option>'],
    ['Target Count','number','samp-n','500','min="1"'],
    ['Seed','number','samp-seed','42','min="0"'],
  ],
  checks: [['Include labels', true]], progress: true,
});

/* ── Label Anomaly ──────────────────────────────────── */
Tabs.anomaly = makeTab({
  id: 'anom', heading: 'Label Anomaly Detector',
  needsImgDir: true, needsLblDir: true,
  checks: [['Out-of-bounds', true],['Size outliers', true],['High overlap', true]],
  resultCols: ['File','Type','Details','Severity'],
  resultHint: 'Run detector to find anomalies', progress: true,
});

/* ── Image Quality ──────────────────────────────────── */
Tabs.quality = makeTab({
  id: 'qual', heading: 'Image Quality Checker',
  needsImgDir: true,
  checks: [['Blur', true],['Brightness', true],['Overexposure', true],['Entropy', true],['Aspect ratio', true]],
  resultCols: ['File','Blur','Brightness','Entropy','Aspect','Issues'],
  progress: true,
});

/* ── Near Duplicates ────────────────────────────────── */
Tabs.duplicate = makeTab({
  id: 'dup', heading: 'Near-Duplicate Detector',
  needsImgDir: true,
  fields: [['Hamming Threshold','number','dup-thr','10','min="0" max="64"']],
  resultCols: ['Group','Image A','Image B','Distance'],
  progress: true,
});

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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">Leaky Split Detector</h3>
          ${dirInput('leak-train', t('splitter.train'))}
          ${dirInput('leak-val', t('splitter.val'))}
          ${dirInput('leak-test', t('splitter.test'))}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">Hamming Threshold</label>
            <input type="number" class="form-input input-normal" value="10" min="0" max="64">
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="App.setStatus('Detecting leaks...')">${t('run')}</button>
          <div style="margin-top:0.5rem;"><div class="progress-track"><div class="progress-fill" style="width:0%"></div></div></div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <div class="table-container"><table><thead><tr><th>Split Pair</th><th>Duplicates</th><th>Files</th></tr></thead>
          <tbody><tr><td colspan="3" class="text-secondary" style="text-align:center;padding:2rem;">Run detector to find cross-split duplicates</td></tr></tbody></table></div>
        </div>
      </div>`;
  }
};

/* ── Similarity Search ──────────────────────────────── */
Tabs.similarity = makeTab({
  id: 'sim', heading: 'Similarity Search',
  needsImgDir: true,
  fields: [['Top-K','number','sim-k','10','min="1" max="100"']],
  action: 'Build Index', resultCols: ['Rank','Image','Cosine Similarity'],
  progress: true,
});

/* ── Batch Inference ────────────────────────────────── */
Tabs.batch = makeTab({
  id: 'bat', heading: 'Batch Inference',
  needsModel: true, needsImgDir: true, needsOutDir: true,
  fields: [['Output Format','select','bat-fmt','<option>YOLO txt</option><option>JSON</option><option>CSV</option>']],
  checks: [['Save visualizations', false]],
  progress: true, action: 'Run Batch',
});

/* ── Augmentation ───────────────────────────────────── */
Tabs.augmentation = {
  title: true,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">Augmentation Preview</h3>
          ${imgDirInput('aug-img')}
          ${lblDirInput('aug-lbl')}
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">Augmentation Type</label>
            <select class="form-input input-normal">
              <option>Mosaic 2×2</option><option>Flip</option><option>Rotate</option>
              <option>Brightness</option><option>Albumentations</option>
            </select>
          </div>
          <button class="btn btn-primary" style="margin-top:1rem;" onclick="App.setStatus('Previewing...')">Preview</button>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
          <div class="card" style="padding:1rem;min-height:250px;display:flex;align-items:center;justify-content:center;"><span class="text-muted">Original</span></div>
          <div class="card" style="padding:1rem;min-height:250px;display:flex;align-items:center;justify-content:center;"><span class="text-muted">Augmented</span></div>
        </div>
      </div>`;
  }
};
