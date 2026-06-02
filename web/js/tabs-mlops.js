/* Training & Deployment tabs (MLOps).
 *
 * Long-running-job UI pattern copied from Tabs.benchmark / Tabs.calibration:
 * render() builds a config card with namespaced ids, init() populates async
 * selects, run() POSTs and starts polling, _poll() reads /status and renders
 * progress + results, destroy() stops the poll loop. All server strings are
 * escHtml()'d; all visible text comes from t().
 */

/* ── Training ─────────────────────────────────────────── */
Tabs.training = {
  title: true,
  _polling: false,
  _trainers: [],

  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;max-width:880px;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('training.setup')}</h3>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
            <div class="form-group">
              <label class="form-label">${t('training.trainer')}</label>
              <select class="form-input input-normal" id="train-trainer" onchange="Tabs.training._onTrainerChange()"></select>
            </div>
            <div class="form-group">
              <label class="form-label">${t('training.base_model')}</label>
              <input class="form-input input-normal" id="train-model" placeholder="yolo11n.pt">
            </div>
          </div>
          <div style="margin-top:0.75rem;">
            <label class="form-label">${t('training.images')}</label>
            ${imgDirInput('train-img')}
          </div>
          <div id="train-yolo-fields" style="margin-top:0.75rem;">
            <label class="form-label">${t('training.labels')}</label>
            ${lblDirInput('train-lbl')}
            <div class="form-group" style="margin-top:0.5rem;">
              <label class="form-label">${t('training.names')}</label>
              <input class="form-input input-normal" id="train-names" placeholder="person, car, ...">
            </div>
          </div>
          <div class="text-secondary" id="train-classify-note" style="display:none;font-size:11px;margin-top:0.5rem;">${t('training.note_classify')}</div>

          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:1rem;">
            <div class="form-group"><label class="form-label">${t('training.epochs')}</label><input type="number" class="form-input input-normal" id="train-epochs" value="100" min="1" max="2000"></div>
            <div class="form-group"><label class="form-label">${t('training.imgsz')}</label><input type="number" class="form-input input-normal" id="train-imgsz" value="640" min="32" step="32"></div>
            <div class="form-group"><label class="form-label">${t('training.batch')}</label><input type="number" class="form-input input-normal" id="train-batch" value="16" min="-1"></div>
            <div class="form-group"><label class="form-label">${t('training.device')}</label><input class="form-input input-normal" id="train-device" placeholder="0 / cpu"></div>
          </div>
          <details style="margin-top:0.75rem;">
            <summary class="text-secondary" style="cursor:pointer;font-size:12px;">${t('training.advanced')}</summary>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:0.5rem;">
              <div class="form-group"><label class="form-label">${t('training.lr')}</label><input type="number" class="form-input input-normal" id="train-lr" placeholder="auto" step="0.0001"></div>
              <div class="form-group"><label class="form-label">${t('training.val_ratio')}</label><input type="number" class="form-input input-normal" id="train-val" value="0.2" min="0" max="0.9" step="0.05"></div>
            </div>
          </details>

          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" id="train-run" onclick="Tabs.training.run()">${t('training.run')}</button>
            <button class="btn btn-danger btn-sm" id="train-stop" disabled onclick="Tabs.training.stop()">${t('stop')}</button>
          </div>
          <div style="margin-top:0.75rem;">
            <div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="train-prog" style="width:0%;height:100%;"></div><span id="train-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
            <span class="text-secondary" id="train-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span>
          </div>
        </div>

        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('training.progress')}</h3>
          <div id="train-summary"></div>
          <div class="table-container"><table><thead><tr><th>${t('training.epoch')}</th><th>Metrics</th></tr></thead>
          <tbody id="train-results"><tr><td colspan="2" class="text-secondary" style="text-align:center;padding:2rem;">${t('training.hint')}</td></tr></tbody></table></div>
        </div>
      </div>`;
  },

  async init() {
    try {
      const r = await API.get('/api/training/trainers');
      this._trainers = r.trainers || [];
      const sel = document.getElementById('train-trainer');
      if (sel) {
        sel.innerHTML = this._trainers.map(tr => {
          const dis = tr.available ? '' : ' disabled';
          const note = tr.available ? '' : ` — ${escHtml(tr.unavailable_reason || 'unavailable')}`;
          return `<option value="${tr.name}"${dis}>${escHtml(tr.label)}${note}</option>`;
        }).join('');
        this._onTrainerChange();
      }
    } catch (e) { console.warn('trainers load:', e); }
    // The tab DOM is cached on leave and destroy() stops the poll loop; if a job
    // is still running server-side, resume polling so the UI doesn't freeze.
    try {
      const s = await API.get('/api/training/status');
      if (s && s.running && !this._polling) {
        document.getElementById('train-run').disabled = true;
        document.getElementById('train-run').textContent = t('training.running');
        document.getElementById('train-stop').disabled = false;
        this._polling = true;
        this._poll();
      }
    } catch (e) { /* status optional on init */ }
  },

  _onTrainerChange() {
    const sel = document.getElementById('train-trainer');
    if (!sel) return;
    const tr = this._trainers.find(x => x.name === sel.value);
    if (!tr) return;
    const modelEl = document.getElementById('train-model');
    if (modelEl && tr.default_model) modelEl.value = tr.default_model;
    const isImageFolder = tr.dataset_kind === 'imagefolder';
    const yolo = document.getElementById('train-yolo-fields');
    const note = document.getElementById('train-classify-note');
    if (yolo) yolo.style.display = isImageFolder ? 'none' : '';
    if (note) note.style.display = isImageFolder ? '' : 'none';
  },

  async run() {
    const imagesDir = document.getElementById('train-img').value || G.imgDir;
    if (!imagesDir) { App.setStatus(t('training.no_images')); return; }
    const trainer = document.getElementById('train-trainer').value;
    const payload = {
      trainer,
      model: document.getElementById('train-model').value || '',
      images_dir: imagesDir,
      labels_dir: document.getElementById('train-lbl').value || '',
      names: document.getElementById('train-names').value || null,
      epochs: +document.getElementById('train-epochs').value || 100,
      imgsz: +document.getElementById('train-imgsz').value || 640,
      batch: +document.getElementById('train-batch').value || 16,
      lr0: parseFloat(document.getElementById('train-lr').value) || null,
      device: document.getElementById('train-device').value || '',
      val_ratio: parseFloat(document.getElementById('train-val').value) || 0.2,
    };
    document.getElementById('train-run').disabled = true;
    document.getElementById('train-run').textContent = t('training.running');
    document.getElementById('train-stop').disabled = false;
    document.getElementById('train-prog').style.width = '0%';
    document.getElementById('train-results').innerHTML = '';
    document.getElementById('train-summary').innerHTML = '';
    App.setStatus(t('training.running'));
    try {
      const r = await API.post('/api/training/run', payload);
      if (r.error) { App.setStatus('Error: ' + r.error); this._resetButtons(); return; }
      this._polling = true;
      this._poll();
    } catch (e) {
      App.setStatus(`Error: ${e.message}`, e.stack);
      this._resetButtons();
    }
  },

  async _poll() {
    if (!this._polling) return;
    try {
      const s = await API.get('/api/training/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      document.getElementById('train-prog').style.width = pct + '%';
      const pt = document.getElementById('train-prog-text'); if (pt) pt.textContent = pct + '%';
      document.getElementById('train-status').textContent = s.msg || '';
      const epochs = (s.results && s.results.epochs) || [];
      if (epochs.length) this._renderEpochs(epochs);
      if (s.running) {
        setTimeout(() => this._poll(), 1000);
      } else {
        this._polling = false;
        this._resetButtons();
        if (s.msg && s.msg.startsWith('Error')) { App.setStatus(s.msg); return; }
        this._renderSummary(s.results && s.results.summary);
        App.setStatus(t('training.complete'));
      }
    } catch (e) {
      console.warn('poll error:', e);
      document.getElementById('train-status').textContent = 'Poll error: ' + e.message;
      setTimeout(() => this._poll(), 2000);
    }
  },

  _renderEpochs(epochs) {
    // Union of metric keys (besides 'epoch') -> dynamic columns.
    const keys = [];
    for (const row of epochs) for (const k of Object.keys(row)) if (k !== 'epoch' && !keys.includes(k)) keys.push(k);
    const head = `<tr><th>${t('training.epoch')}</th>${keys.map(k => `<th>${escHtml(k)}</th>`).join('')}</tr>`;
    const body = epochs.map(r => `<tr><td>${r.epoch}</td>${keys.map(k => `<td>${r[k] != null ? r[k] : '—'}</td>`).join('')}</tr>`).join('');
    const table = document.querySelector('#train-results').closest('table');
    if (table) { table.querySelector('thead').innerHTML = head; document.getElementById('train-results').innerHTML = body; }
  },

  _renderSummary(summary) {
    if (!summary) return;
    const best = summary.best_model || '';
    const metrics = summary.metrics || {};
    const metricHtml = Object.keys(metrics).map(k => `<div><span class="text-secondary">${escHtml(k)}</span><br><strong>${metrics[k]}</strong></div>`).join('');
    let html = `<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.75rem;margin-bottom:1rem;">
      <div><span class="text-secondary">${t('training.best_model')}</span><br><strong style="word-break:break-all;font-size:12px;">${escHtml(best)}</strong></div>
      <div><span class="text-secondary">${t('training.epochs')}</span><br><strong>${summary.epochs_completed || '—'}</strong></div>
      ${metricHtml}
    </div>`;
    if (best) {
      html += `<button class="btn btn-secondary" onclick="Tabs.deployment._loadModel('${best.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}')">${t('training.deploy_now')}</button>`;
    }
    document.getElementById('train-summary').innerHTML = html;
  },

  _resetButtons() {
    const run = document.getElementById('train-run');
    if (run) { run.disabled = false; run.textContent = t('training.run'); }
    const stop = document.getElementById('train-stop');
    if (stop) stop.disabled = true;
  },

  stop() {
    // Request cancellation but keep polling — the server worker owns `running`
    // and stops at the next epoch boundary; the poll loop resets the buttons
    // and renders the final state once it actually ends.
    API.get('/api/training/stop').catch(() => {});
    App.setStatus(t('stopping'));
    const stopBtn = document.getElementById('train-stop');
    if (stopBtn) stopBtn.disabled = true;
  },

  destroy() {
    this._polling = false;
  },
};


/* ── Deployment / Export ──────────────────────────────── */
Tabs.deployment = {
  title: true,
  _polling: false,
  _targets: [],

  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;max-width:880px;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('deploy.setup')}</h3>
          <div class="form-group">
            <label class="form-label">${t('deploy.model')}</label>
            ${modelInput('deploy-model')}
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.75rem;">
            <div class="form-group">
              <label class="form-label">${t('deploy.target')}</label>
              <select class="form-input input-normal" id="deploy-target"></select>
            </div>
            <div class="form-group">
              <label class="form-label">${t('deploy.precision')}</label>
              <select class="form-input input-normal" id="deploy-precision">
                <option value="fp32">FP32</option>
                <option value="fp16">FP16</option>
                <option value="int8">INT8</option>
              </select>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:0.75rem;">
            <div class="form-group"><label class="form-label">${t('deploy.opset')}</label><input type="number" class="form-input input-normal" id="deploy-opset" placeholder="auto" min="9" max="22"></div>
            <div class="form-group"><label class="form-label">${t('deploy.batch')}</label><input type="number" class="form-input input-normal" id="deploy-batch" value="1" min="1"></div>
            <div class="form-group"><label class="form-label">${t('deploy.input_size')}</label><input class="form-input input-normal" id="deploy-input" placeholder="640 or 640x480"></div>
            <div class="form-group"><label class="form-label">${t('deploy.workspace')}</label><input type="number" class="form-input input-normal" id="deploy-workspace" value="4" min="1" step="1"></div>
          </div>
          <div style="display:flex;gap:1.25rem;margin-top:0.75rem;align-items:center;flex-wrap:wrap;">
            <label style="font-size:12px;display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" id="deploy-dynamic"> ${t('deploy.dynamic')}</label>
            <label style="font-size:12px;display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" id="deploy-simplify"> ${t('deploy.simplify')}</label>
          </div>
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">${t('deploy.calibration')}</label>
            <input type="text" class="form-input input-normal" id="deploy-calib" placeholder="(optional) image dir for INT8 calibration">
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" id="deploy-run" onclick="Tabs.deployment.run()">${t('deploy.run')}</button>
            <button class="btn btn-danger btn-sm" id="deploy-stop" disabled onclick="Tabs.deployment.stop()">${t('stop')}</button>
          </div>
          <div style="margin-top:0.75rem;">
            <div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="deploy-prog" style="width:0%;height:100%;"></div><span id="deploy-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
            <span class="text-secondary" id="deploy-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span>
          </div>
        </div>

        <div class="card" style="padding:1.5rem;display:none;" id="deploy-result">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('deploy.result')}</h3>
          <div id="deploy-result-body"></div>
        </div>
      </div>`;
  },

  async init() {
    try {
      const r = await API.get('/api/deploy/targets');
      this._targets = r.targets || [];
      const sel = document.getElementById('deploy-target');
      if (sel) {
        sel.innerHTML = this._targets.map(tg => {
          const dis = tg.available ? '' : ' disabled';
          const note = tg.available ? '' : ` ${t('deploy.unavailable')}`;
          return `<option value="${tg.name}"${dis}>${escHtml(tg.label)}${note}</option>`;
        }).join('');
      }
    } catch (e) { console.warn('targets load:', e); }
    // Resume polling if an export is still running (cached tab DOM on return).
    try {
      const s = await API.get('/api/deploy/status');
      if (s && s.running && !this._polling) {
        document.getElementById('deploy-run').disabled = true;
        document.getElementById('deploy-run').textContent = t('deploy.running');
        document.getElementById('deploy-stop').disabled = false;
        this._polling = true;
        this._poll();
      }
    } catch (e) { /* status optional on init */ }
  },

  _loadModel(path) {
    // Called from the Training tab's "Deploy this model" button.
    App.switchTab('deployment');
    setTimeout(() => {
      const el = document.getElementById('deploy-model');
      if (el) el.value = path;
    }, 200);
  },

  _parseInputSize(raw) {
    if (!raw) return null;
    if (/[x,]/i.test(raw)) return raw;          // "640x480" — server parses
    const n = +raw; return isNaN(n) ? null : n;
  },

  async run() {
    const modelPath = document.getElementById('deploy-model').value || G.model;
    if (!modelPath) { App.setStatus(t('deploy.no_model')); return; }
    const payload = {
      model_path: modelPath,
      target: document.getElementById('deploy-target').value,
      precision: document.getElementById('deploy-precision').value,
      opset: +document.getElementById('deploy-opset').value || null,
      batch: +document.getElementById('deploy-batch').value || null,
      input_size: this._parseInputSize(document.getElementById('deploy-input').value),
      dynamic: document.getElementById('deploy-dynamic').checked,
      simplify: document.getElementById('deploy-simplify').checked,
      workspace_gb: +document.getElementById('deploy-workspace').value || 4,
      calibration_dir: document.getElementById('deploy-calib').value || '',
    };
    document.getElementById('deploy-run').disabled = true;
    document.getElementById('deploy-run').textContent = t('deploy.running');
    document.getElementById('deploy-stop').disabled = false;
    document.getElementById('deploy-prog').style.width = '0%';
    document.getElementById('deploy-result').style.display = 'none';
    App.setStatus(t('deploy.running'));
    try {
      const r = await API.post('/api/deploy/run', payload);
      if (r.error) { App.setStatus('Error: ' + r.error); this._resetButtons(); return; }
      this._polling = true;
      this._poll();
    } catch (e) {
      App.setStatus(`Error: ${e.message}`, e.stack);
      this._resetButtons();
    }
  },

  async _poll() {
    if (!this._polling) return;
    try {
      const s = await API.get('/api/deploy/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      document.getElementById('deploy-prog').style.width = (s.running ? Math.max(pct, 5) : 100) + '%';
      const pt = document.getElementById('deploy-prog-text'); if (pt) pt.textContent = (s.running ? Math.max(pct, 5) : 100) + '%';
      document.getElementById('deploy-status').textContent = s.msg || '';
      if (s.running) {
        setTimeout(() => this._poll(), 700);
      } else {
        this._polling = false;
        this._resetButtons();
        if (s.msg && s.msg.startsWith('Error')) {
          const err = (s.results && s.results.error) || s.msg;
          this._renderError(err);
          App.setStatus(s.msg);
          return;
        }
        this._renderResult(s.results || {});
        App.setStatus(t('deploy.complete'));
      }
    } catch (e) {
      console.warn('poll error:', e);
      document.getElementById('deploy-status').textContent = 'Poll error: ' + e.message;
      setTimeout(() => this._poll(), 1500);
    }
  },

  _renderError(err) {
    document.getElementById('deploy-result-body').innerHTML =
      `<div style="color:var(--action-danger-05);">${escHtml(String(err))}</div>`;
    document.getElementById('deploy-result').style.display = '';
  },

  _renderResult(res) {
    if (res.error) { this._renderError(res.error); return; }
    const out = res.output_path || '';
    const rows = [
      [t('deploy.output'), out, true],
      [t('deploy.size'), res.output_size_mb != null ? res.output_size_mb + ' MB' : '—'],
      [t('deploy.provider'), res.provider || '—'],
      [t('deploy.precision'), res.precision || 'fp32'],
      [t('deploy.validated'), res.validated === true ? '✓' : (res.validated === false ? '✗' : '—')],
      [t('deploy.manifest'), res.manifest_path || '—', true],
    ];
    let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;">';
    for (const [label, val, wrap] of rows) {
      html += `<div><span class="text-secondary">${label}</span><br><strong style="${wrap ? 'word-break:break-all;font-size:12px;' : ''}">${escHtml(String(val))}</strong></div>`;
    }
    html += '</div>';
    if (res.validation_error) {
      html += `<div style="color:var(--action-danger-05);margin-top:0.75rem;font-size:12px;">${escHtml(res.validation_error)}</div>`;
    }
    if (out && /\.onnx$/i.test(out)) {
      html += `<button class="btn btn-secondary" style="margin-top:1rem;" onclick="Tabs.deployment._openInBenchmark('${out.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}')">${t('deploy.open_bench')}</button>`;
    }
    document.getElementById('deploy-result-body').innerHTML = html;
    document.getElementById('deploy-result').style.display = '';
  },

  _openInBenchmark(outputPath) {
    App.switchTab('benchmark');
    setTimeout(() => {
      const slots = document.querySelectorAll('.bench-model-input');
      if (slots[0]) slots[0].value = outputPath;
    }, 250);
  },

  _resetButtons() {
    const run = document.getElementById('deploy-run');
    if (run) { run.disabled = false; run.textContent = t('deploy.run'); }
    const stop = document.getElementById('deploy-stop');
    if (stop) stop.disabled = true;
  },

  stop() {
    // Export is single-shot; request cancellation but keep polling. The worker
    // owns `running`, so the poll loop resets the buttons when it ends.
    API.get('/api/deploy/stop').catch(() => {});
    App.setStatus(t('stopping'));
    const stopBtn = document.getElementById('deploy-stop');
    if (stopBtn) stopBtn.disabled = true;
  },

  destroy() {
    this._polling = false;
  },
};
