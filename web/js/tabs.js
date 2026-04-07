/* Tab content renderers — uses I18n.t(), G.*, shared helpers */
var Tabs = {};
const t = (k, p) => I18n.t(k, p);

/* ── Viewer ─────────────────────────────────────────── */
Tabs.viewer = {
  title: true,
  render() {
    const speeds = ['x1','x2','x3','x4','x8'];
    const speedOpts = speeds.map(s => `<option${s==='x1'?' selected':''}>${s}</option>`).join('');
    return `
      <div style="display:flex;gap:0.75rem;height:100%;">
        <!-- Left: file browser -->
        <div style="width:220px;display:flex;flex-direction:column;gap:0.75rem;overflow-y:auto;flex-shrink:0;">
          <div class="card-flat" style="padding:0.75rem;">
            <div style="display:flex;gap:0.25rem;align-items:center;margin-bottom:0.4rem;">
              <label class="form-label" style="font-size:10px;margin:0;white-space:nowrap;">${t('settings.model_type')}</label>
              <select class="form-input input-normal" id="v-model-type" style="flex:1;height:30px;font-size:10px;min-width:0;overflow:hidden;text-overflow:ellipsis;padding:0 6px;" onchange="Tabs.viewer._onModelTypeChange()"></select>
            </div>
            <div style="display:flex;gap:0.25rem;align-items:center;margin-bottom:0.4rem;">
              <label class="form-label" style="font-size:10px;margin:0;white-space:nowrap;">배치</label>
              <input type="number" class="form-input input-normal" value="1" min="1" max="16" id="v-batch-size" style="flex:1;height:26px;font-size:10px;" onchange="Tabs.viewer._onBatchChange()">
            </div>
            <div style="margin-bottom:0;">
              <div style="display:flex;align-items:center;gap:0.25rem;">
                <label class="form-label" style="font-size:10px;margin:0;white-space:nowrap;">${t('settings.conf')}</label>
                <span style="font-size:10px;margin-left:auto;" id="v-conf-value">0.25</span>
              </div>
              <input type="range" min="1" max="99" step="1" value="25" id="v-conf-slider" style="width:100%;accent-color:var(--action-link-05);" oninput="Tabs.viewer._onConfChange()">
            </div>
          </div>
          <div class="card-flat" style="padding:0.75rem;">
            <div class="text-label" style="margin-bottom:0.5rem;">${t('settings.model')}</div>
            <div style="display:flex;gap:0.25rem;margin-bottom:0.5rem;">
              <button class="btn btn-secondary btn-sm" style="flex:1;" onclick="Tabs.viewer.browseModel()">${t('browse')}</button>
              <button class="btn btn-ghost btn-sm" onclick="Tabs.viewer.refreshModels()" title="Refresh">↻</button>
            </div>
            <div id="v-model-list" style="flex:1;overflow-y:auto;font-size:12px;" class="text-secondary">Loading...</div>
          </div>
          <div class="card-flat" style="padding:0.75rem;flex:1;display:flex;flex-direction:column;">
            <div class="text-label" style="margin-bottom:0.5rem;">Video / Image</div>
            <div style="display:flex;gap:0.25rem;margin-bottom:0.5rem;">
              <button class="btn btn-secondary btn-sm" style="flex:1;" onclick="Tabs.viewer.browseVideo()">${t('browse')}</button>
              <button class="btn btn-ghost btn-sm" onclick="Tabs.viewer.refreshVideos()" title="Refresh">↻</button>
            </div>
            <div id="v-video-list" style="flex:1;overflow-y:auto;font-size:12px;" class="text-secondary">Loading...</div>
          </div>
        </div>
        <!-- Center: canvas + controls -->
        <div style="flex:1;display:flex;flex-direction:column;gap:0.5rem;min-width:0;">
          <div class="card" style="flex:1;display:flex;align-items:center;justify-content:center;min-height:360px;overflow:hidden;">
            <div id="viewer-canvas" style="color:var(--text-02);text-align:center;">${t('viewer.open_hint')}</div>
          </div>
          <!-- Seek slider -->
          <input type="range" id="v-seek" min="0" max="0" value="0" style="width:100%;accent-color:var(--action-link-05);margin:0;" disabled>
          <!-- Control bar -->
          <div style="display:flex;gap:0.35rem;align-items:center;flex-wrap:wrap;">
            <button class="btn btn-primary btn-sm" id="btn-play" onclick="Tabs.viewer.togglePlay()" disabled>▶ ${t('viewer.play')}</button>
            <button class="btn btn-secondary btn-sm" id="btn-stop" onclick="Tabs.viewer.stop()" disabled>⏹</button>
            <span style="width:1px;height:20px;background:var(--border-default);margin:0 0.25rem;"></span>
            <button class="btn btn-secondary btn-sm" id="btn-snapshot" onclick="Tabs.viewer.snapshot()" disabled style="display:flex;align-items:center;gap:2px;">${Icons._svg('<rect x="2" y="4" width="20" height="16" rx="2"/><circle cx="12" cy="13" r="4"/><path d="M17 4l-2-2H9L7 4"/>',14)}</button>
            <span style="width:1px;height:20px;background:var(--border-default);margin:0 0.25rem;"></span>
            <label class="text-secondary" style="font-size:12px;">${t('viewer.frame_skip')}:</label>
            <select class="form-input" id="v-speed" style="width:76px;height:30px;font-size:10px;min-width:0;overflow:hidden;text-overflow:ellipsis;padding:0 6px;" onchange="Tabs.viewer.setSpeed(this.value)">${speedOpts}</select>
            <div style="flex:1;"></div>
            <span class="text-secondary" style="font-size:12px;" id="v-fps">FPS: —</span>
            <span class="text-secondary" style="font-size:12px;margin-left:0.5rem;" id="v-frame-counter">0 / 0</span>
          </div>
        </div>
        <!-- Right: info panels -->
        <div style="width:210px;display:flex;flex-direction:column;gap:0.5rem;overflow-y:auto;flex-shrink:0;font-size:11px;">
          <div class="card-flat" style="padding:0.6rem;">
            <div class="text-label" style="margin-bottom:0.35rem;font-size:11px;">${t('viewer.model_info')}</div>
            <div id="v-model-info" class="text-secondary">—</div>
          </div>
          <div class="card-flat" style="padding:0.6rem;">
            <div class="text-label" style="margin-bottom:0.35rem;font-size:11px;">${t('viewer.video_info')}</div>
            <div id="v-video-info" class="text-secondary">—</div>
          </div>
          <div class="card-flat" style="padding:0.6rem;">
            <div class="text-label" style="margin-bottom:0.35rem;font-size:11px;">${t('viewer.infer_stats')}</div>
            <div id="v-infer-stats" class="text-secondary">—</div>
          </div>
          <div class="card-flat" style="padding:0.6rem;">
            <div class="text-label" style="margin-bottom:0.35rem;font-size:11px;">${t('viewer.det_results')}</div>
            <div id="viewer-results" class="text-secondary">—</div>
          </div>
          <div class="card-flat" style="padding:0.6rem;">
            <div class="text-label" style="margin-bottom:0.35rem;font-size:11px;">${t('viewer.hw_stats')}</div>
            <div id="v-hw-stats" class="text-secondary">—</div>
          </div>
          <div class="card-flat" style="padding:0.6rem;">
            <div class="text-label" style="margin-bottom:0.35rem;font-size:11px;">${t('viewer.sys_info')}</div>
            <div id="v-sys-info" class="text-secondary">—</div>
          </div>
        </div>
      </div>`;
  },
  async init() {
    this.refreshModels();
    this.refreshVideos();
    // Load config into viewer controls
    try {
      const c = await API.config();
      if (!c.error) {
        const mt = document.getElementById('v-model-type');
        if (mt && c.model_types) {
          mt.innerHTML = '';
          for (const [key, label] of Object.entries(c.model_types)) mt.add(new Option(label, key));
          mt.value = c.model_type || 'yolo';
        }
        document.getElementById('v-batch-size').value = c.batch_size || 1;
        const cs = document.getElementById('v-conf-slider');
        cs.value = Math.round((c.conf_threshold || 0.25) * 100);
        document.getElementById('v-conf-value').textContent = (c.conf_threshold || 0.25).toFixed(2);
      }
    } catch(e) {}
    // System info
    try {
      const info = await API.sysInfo();
      const el = document.getElementById('v-sys-info');
      if (el) el.innerHTML = `OS: ${info.os||'—'}<br>Python: ${info.python||'—'}<br>ORT: ${info.ort||'—'}<br>Torch: ${info.torch||'—'}<br>CUDA: ${info.cuda||'—'}<br>GPU: ${info.gpu_name||'—'}`;
    } catch(e) {}
    if (G.model) this._showModelInfo(G.model);
    // Seek slider
    const seek = document.getElementById('v-seek');
    if (seek) seek.oninput = () => this._onSeek(+seek.value);
    // Start HW polling
    this._hwInterval = setInterval(() => this._pollHW(), 2000);
    this._pollHW();
    // Keyboard shortcuts
    this._keyHandler = (e) => this._onKey(e);
    document.addEventListener('keydown', this._keyHandler);
  },
  _hwInterval: null,
  async _pollHW() {
    try {
      const h = await API.hwStats();
      const el = document.getElementById('v-hw-stats');
      if (el) el.innerHTML = `CPU: ${h.cpu}%<br>RAM: ${h.ram_mb} MB<br>GPU: ${h.gpu_util||0}%<br>VRAM: ${h.gpu_mem_used||0}/${h.gpu_mem_total||0} MB<br>Temp: ${h.gpu_temp||'—'}°C`;
    } catch(e) {}
  },
  _onKey(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === ' ') { e.preventDefault(); this._streamSessionId ? (this._paused ? this.play() : this.pause()) : this.play(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); this.stepBack(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); this.stepFwd(); }
    else if (e.key === 's' || e.key === 'S') this.snapshot();
    else if (e.key === '+' || e.key === '=') this._changeSpeed(1);
    else if (e.key === '-') this._changeSpeed(-1);
  },
  _getConf() { return +(document.getElementById('v-conf-slider')?.value || 25) / 100; },
  async _onModelTypeChange() {
    const v = document.getElementById('v-model-type').value;
    await API.post('/api/config', { model_type: v });
  },
  async _onBatchChange() {
    await API.post('/api/config', { batch_size: +document.getElementById('v-batch-size').value });
  },
  async _onConfChange() {
    const v = this._getConf();
    document.getElementById('v-conf-value').textContent = v.toFixed(2);
    await API.post('/api/config', { conf_threshold: v });
    // 이미지가 로드되어 있으면 재추론
    if (G.videoPath && G.model) {
      const ext = G.videoPath.split('.').pop().toLowerCase();
      if (['jpg','jpeg','png','bmp'].includes(ext)) this._inferImage();
    }
  },
  _changeSpeed(dir) {
    const speeds = [1,2,3,4,8];
    const sel = document.getElementById('v-speed');
    const cur = parseInt(sel.value.replace('x',''));
    const idx = speeds.indexOf(cur);
    const next = speeds[Math.max(0, Math.min(speeds.length-1, idx+dir))];
    sel.value = 'x'+next;
    this.setSpeed('x'+next);
  },
  async refreshModels() {
    try {
      const r = await API.listDir({ path: 'Models', exts: ['.onnx','.pt'] });
      const el = document.getElementById('v-model-list');
      if (!r.files || !r.files.length) { el.textContent = 'No models in Models/'; return; }
      el.innerHTML = r.files.map(f =>
        `<div class="nav-item" style="padding:0.25rem 0.5rem;cursor:pointer;font-size:12px;border-radius:4px;" onclick="Tabs.viewer.selectModel('${f.path.replace(/\\/g,'\\\\').replace(/'/g,"\\'")}')" title="${f.path}">${f.name}</div>`
      ).join('');
    } catch(e) { document.getElementById('v-model-list').textContent = 'Models/ not found'; }
  },
  async refreshVideos() {
    try {
      const r = await API.listDir({ path: 'Videos', exts: ['.mp4','.avi','.mov','.mkv','.jpg','.jpeg','.png','.bmp'] });
      const el = document.getElementById('v-video-list');
      if (!r.files || !r.files.length) { el.textContent = 'No files in Videos/'; return; }
      el.innerHTML = r.files.map(f =>
        `<div class="nav-item" style="padding:0.25rem 0.5rem;cursor:pointer;font-size:12px;border-radius:4px;" onclick="Tabs.viewer.selectVideo('${f.path.replace(/\\/g,'\\\\').replace(/'/g,"\\'")}')" title="${f.path}">${f.name}</div>`
      ).join('');
    } catch(e) { document.getElementById('v-video-list').textContent = 'Videos/ not found'; }
  },
  async selectModel(path) {
    setModel(path);
    await this._showModelInfo(path);
    // 이미지가 이미 로드되어 있으면 자동 추론
    if (G.videoPath) {
      const ext = G.videoPath.split('.').pop().toLowerCase();
      if (['jpg','jpeg','png','bmp'].includes(ext)) this._inferImage();
    }
  },
  async _showModelInfo(path) {
    try {
      const info = await API.post('/api/model/load', { path });
      const el = document.getElementById('v-model-info');
      if (el && info && !info.error) {
        el.innerHTML = `File: ${info.name||'—'}<br>Input: ${info.input_shape||'—'}<br>Output: ${info.output_shape||'—'}<br>Layout: ${(info.layout||'—').toUpperCase()}<br>Task: ${info.task||'—'}<br>Classes: ${info.num_classes||0}`;
      }
      document.getElementById('btn-play').disabled = !G.model;
    } catch(e) { document.getElementById('v-model-info').textContent = `Error: ${e.message}`; }
  },
  async selectVideo(path) {
    // 기존 스트림 세션 정리
    if (this._streamSessionId) {
      try { await API.post('/api/viewer/stop/' + this._streamSessionId, {}); } catch(e) {}
      this._streamSessionId = null;
      this._paused = false;
    }
    G.videoPath = path;
    const name = path.split(/[\\/]/).pop();
    App.setStatus(`Video: ${name}`);
    document.getElementById('btn-play').disabled = !G.model;
    // Fetch video info
    try {
      const vi = await API.videoInfo(path);
      const el = document.getElementById('v-video-info');
      if (el && !vi.error) {
        el.innerHTML = `File: ${name}<br>Resolution: ${vi.width} × ${vi.height}<br>FPS: ${vi.fps}<br>Frames: ${vi.total_frames?.toLocaleString()}<br>Duration: ${vi.duration}`;
        const seek = document.getElementById('v-seek');
        if (seek) { seek.max = vi.total_frames - 1; seek.disabled = false; }
        // Show first frame (#1)
        if (vi.first_frame) {
          document.getElementById('viewer-canvas').innerHTML = `<img src="data:image/jpeg;base64,${vi.first_frame}" style="max-width:100%;max-height:100%;">`;
        }
      }
    } catch(e) { document.getElementById('v-video-info').innerHTML = `File: ${name}`; }
    // 이미지 파일이면 자동 추론
    const ext = path.split('.').pop().toLowerCase();
    if (['jpg','jpeg','png','bmp'].includes(ext) && G.model) this._inferImage();
  },
  async browseModel() {
    try {
      const r = await API.selectFile({ filters: 'Models (*.onnx *.pt)' });
      if (r.path) this.selectModel(r.path);
    } catch(e) {}
  },
  async browseVideo() {
    try {
      const r = await API.selectFile({ filters: 'Media (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp)' });
      if (r.path) this.selectVideo(r.path);
    } catch(e) {}
  },
  togglePlay() {
    if (!G.model) { App.setStatus('Select a model first'); return; }
    if (!G.videoPath) { App.setStatus('Select a video/image first'); return; }
    // If playing, pause
    if (this._streamSessionId && !this._paused) {
      this._togglePause(); return;
    }
    // If paused, resume
    if (this._streamSessionId && this._paused) {
      this._togglePause(); return;
    }
    App.setStatus('Starting inference...');
    const ext = G.videoPath.split('.').pop().toLowerCase();
    if (['jpg','jpeg','png','bmp'].includes(ext)) this._inferImage();
    else this._startStream();
  },
  async _inferImage() {
    try {
      const conf = this._getConf();
      const r = await API.post('/api/infer/image', {
        model_path: G.model, image_path: G.videoPath, conf
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      document.getElementById('viewer-canvas').innerHTML = `<img src="data:image/jpeg;base64,${r.image}" style="max-width:100%;max-height:100%;">`;
      document.getElementById('viewer-results').textContent = `${r.detections} detections`;
      document.getElementById('v-infer-stats').innerHTML = `Infer: ${r.infer_ms} ms`;
      App.setStatus(`Inference done: ${r.detections} detections, ${r.infer_ms}ms`);
    } catch(e) { App.setStatus('Error: ' + e.message); }
  },
  _streamSessionId: null,
  _paused: false,
  async _startStream() {
    try {
      const conf = this._getConf();
      const r = await API.post('/api/viewer/start', {
        model_path: G.model, video_path: G.videoPath, conf
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._streamSessionId = r.session_id;
      this._paused = false;
      document.getElementById('viewer-canvas').innerHTML = `<img src="/api/viewer/stream/${r.session_id}" style="max-width:100%;max-height:100%;">`;
      this._setControls(true);
      App.setStatus(`Playing: ${r.total_frames} frames @ ${r.fps?.toFixed(1)} FPS`);
      this._pollStatus();
    } catch(e) { App.setStatus('Error: ' + e.message); }
  },
  _setControls(playing) {
    const btn = document.getElementById('btn-play');
    btn.disabled = false;
    btn.innerHTML = playing ? '⏸ ' + t('viewer.pause') : '▶ ' + t('viewer.play');
    document.getElementById('btn-stop').disabled = !this._streamSessionId;
    document.getElementById('btn-snapshot').disabled = !this._streamSessionId;
  },
  async _pollStatus() {
    if (!this._streamSessionId) return;
    try {
      const s = await API.get('/api/viewer/status/' + this._streamSessionId);
      document.getElementById('viewer-results').textContent = `${s.detections} detections`;
      document.getElementById('v-infer-stats').innerHTML = `Infer: ${s.infer_ms} ms`;
      document.getElementById('v-frame-counter').textContent = `${s.frame_idx} / ${s.total}`;
      const seek = document.getElementById('v-seek');
      if (seek && !seek.matches(':active')) seek.value = s.frame_idx;
      if (s.playing && !s.paused) setTimeout(() => this._pollStatus(), 300);
      else if (!s.playing) { App.setStatus('Playback finished'); this._resetAll(); }
    } catch(e) {}
  },
  _resetAll() {
    this._streamSessionId = null;
    this._paused = false;
    const btn = document.getElementById('btn-play');
    btn.disabled = !G.model || !G.videoPath;
    btn.innerHTML = '▶ ' + t('viewer.play');
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('btn-snapshot').disabled = true;
  },
  async _togglePause() {
    if (!this._streamSessionId) return;
    const r = await API.post('/api/viewer/pause/' + this._streamSessionId, {});
    this._paused = r.paused;
    const btn = document.getElementById('btn-play');
    btn.innerHTML = this._paused ? '▶ ' + t('viewer.play') : '⏸ ' + t('viewer.pause');
    App.setStatus(this._paused ? 'Paused' : 'Playing');
    if (!this._paused) this._pollStatus();
  },
  pause() { this._togglePause(); },
  async stop() {
    if (!this._streamSessionId) return;
    await API.post('/api/viewer/stop/' + this._streamSessionId, {});
    document.getElementById('viewer-canvas').innerHTML = t('viewer.open_hint');
    this._resetAll();
    App.setStatus(t('ready'));
  },
  async stepFwd() {
    if (!this._streamSessionId) return;
    await API.post('/api/viewer/step/' + this._streamSessionId, { delta: 1 });
  },
  async stepBack() {
    if (!this._streamSessionId) return;
    await API.post('/api/viewer/step/' + this._streamSessionId, { delta: -1 });
  },
  async _onSeek(frame) {
    if (!this._streamSessionId) return;
    await API.post('/api/viewer/seek/' + this._streamSessionId, { frame });
  },
  async setSpeed(val) {
    const speed = parseInt(val.replace('x','')) || 1;
    if (!this._streamSessionId) return;
    await API.post('/api/viewer/speed/' + this._streamSessionId, { speed });
  },
  async snapshot() {
    if (!this._streamSessionId) return;
    const r = await API.post('/api/viewer/snapshot/' + this._streamSessionId, {});
    if (r.ok) App.setStatus(`Snapshot saved: ${r.path}`);
    else App.setStatus('Snapshot failed: ' + (r.error||''));
  },
};

/* ── Settings ───────────────────────────────────────── */
Tabs.settings = {
  title: true,
  render() {
    return `
      <div style="display:flex;gap:1.5rem;align-items:flex-start;">
        <div style="max-width:480px;flex:1;display:flex;flex-direction:column;gap:1.5rem;">
          <div class="card" style="padding:1.5rem;">
            <h3 class="text-heading-h3" style="margin-bottom:1rem;">모델 타입 관리</h3>
            <button class="btn btn-secondary btn-sm" onclick="Tabs.settings.openCustomTypeDialog()">모델 타입 추가…</button>
          </div>
          <div class="card" style="padding:1.5rem;">
            <h3 class="text-heading-h3" style="margin-bottom:1rem;">테스트 모델 다운로드</h3>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
              <a href="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx" class="btn btn-secondary btn-sm">📥 Detection (YOLO11n)</a>
              <a href="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.onnx" class="btn btn-secondary btn-sm">📥 Classification (YOLO11n-cls)</a>
              <a href="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx" class="btn btn-secondary btn-sm">📥 Segmentation (YOLO11n-seg)</a>
              <a href="https://huggingface.co/Xenova/clip-vit-base-patch32/tree/main/onnx" target="_blank" class="btn btn-secondary btn-sm">📥 CLIP (ViT-B/32 ONNX)</a>
              <a href="https://huggingface.co/immich-app/ViT-B-32__openai/tree/main" target="_blank" class="btn btn-secondary btn-sm">📥 Embedder (ViT-B/32 ONNX)</a>
            </div>
          </div>
          <div class="card" style="padding:1.5rem;">
            <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('settings.display')}</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
              <div class="form-group"><label class="form-label">${t('settings.box_thick')}</label><input type="number" class="form-input input-normal" value="2" min="1" max="10" id="box-thickness"></div>
              <div class="form-group"><label class="form-label">${t('settings.label_size')}</label><input type="number" class="form-input input-normal" value="0.55" min="0.1" max="2.0" step="0.05" id="label-size"></div>
            </div>
            <div style="display:flex;gap:1.5rem;margin-top:0.75rem;">
              <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" checked id="show-labels"> ${t('settings.show_labels')}</label>
              <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" checked id="show-conf"> ${t('settings.show_conf')}</label>
              <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" checked id="show-label-bg"> 라벨 배경</label>
            </div>
          </div>
          <div style="display:flex;gap:0.5rem;">
            <button class="btn btn-primary" onclick="Tabs.settings.save()">${t('save')}</button>
            <button class="btn btn-secondary" onclick="Tabs.settings.loadConfig()">${t('reset')}</button>
          </div>
        </div>
        <div style="flex:1;max-width:480px;">
          <div class="card" style="padding:1.5rem;">
            <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('settings.class_table')}</h3>
            <div id="class-table-container" class="text-secondary" style="font-size:12px;">Load a model to see class settings</div>
          </div>
        </div>
      </div>`;
  },
  async init() {
    this.loadConfig();
  },
  async loadConfig() {
    try {
      const c = await API.config();
      if (c.error) return;
      document.getElementById('box-thickness').value = c.box_thickness || 2;
      document.getElementById('label-size').value = c.label_size || 0.55;
      document.getElementById('show-labels').checked = c.show_labels !== false;
      document.getElementById('show-conf').checked = c.show_confidence !== false;
      document.getElementById('show-label-bg').checked = c.show_label_bg !== false;
    } catch(e) {}
    // Load class table if model is loaded
    try {
      const mi = await API.modelInfo();
      const c2 = await API.config();
      if (mi.loaded && mi.info && mi.info.names) this._buildClassTable(mi.info.names, c2.class_styles || {});
    } catch(e) {}
  },
  _buildClassTable(names, classStyles) {
    const container = document.getElementById('class-table-container');
    if (!names || !Object.keys(names).length) { container.textContent = 'No classes'; return; }
    const styles = classStyles || {};
    let rows = '';
    for (const [id, name] of Object.entries(names)) {
      const s = styles[String(id)] || {};
      const enabled = s.enabled !== false;
      const color = s.color ? `#${(s.color[2]||0).toString(16).padStart(2,'0')}${(s.color[1]||0).toString(16).padStart(2,'0')}${(s.color[0]||0).toString(16).padStart(2,'0')}` : '#00ff00';
      const thick = s.thickness || 0;
      rows += `<tr>
        <td style="padding:4px 6px;">${id}: ${name}</td>
        <td style="padding:4px 6px;text-align:center;"><input type="checkbox" ${enabled?'checked':''} data-cls="${id}" class="cls-enabled" onchange="Tabs.settings._onClassChange(${id})"></td>
        <td style="padding:4px 6px;text-align:center;"><input type="color" value="${color}" data-cls="${id}" class="cls-color" style="width:32px;height:22px;border:none;cursor:pointer;" onchange="Tabs.settings._onClassChange(${id})"></td>
        <td style="padding:4px 6px;"><input type="number" value="${thick}" min="0" max="10" data-cls="${id}" class="cls-thick" style="width:50px;font-size:11px;" title="0=default" onchange="Tabs.settings._onClassChange(${id})"></td>
      </tr>`;
    }
    container.innerHTML = `<div style="max-height:400px;overflow-y:auto;">
      <table style="width:100%;font-size:12px;"><thead><tr>
        <th style="text-align:left;padding:4px 6px;">Class</th>
        <th style="padding:4px 6px;">${t('settings.enabled')}</th>
        <th style="padding:4px 6px;">${t('settings.color')}</th>
        <th style="padding:4px 6px;">${t('settings.thickness')}</th>
      </tr></thead><tbody>${rows}</tbody></table></div>`;
  },
  async _onClassChange(clsId) {
    const en = document.querySelector(`.cls-enabled[data-cls="${clsId}"]`);
    const co = document.querySelector(`.cls-color[data-cls="${clsId}"]`);
    const th = document.querySelector(`.cls-thick[data-cls="${clsId}"]`);
    if (!en || !co) return;
    const hex = co.value;
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    try {
      await API.post('/api/config/class-style', {
        class_id: clsId, enabled: en.checked,
        color: [b, g, r], thickness: +(th?.value||0),
      });
    } catch(e) {}
  },
  async save() {
    try {
      await API.post('/api/config', {
        box_thickness: +document.getElementById('box-thickness').value,
        label_size: +document.getElementById('label-size').value,
        show_labels: document.getElementById('show-labels').checked,
        show_confidence: document.getElementById('show-conf').checked,
        show_label_bg: document.getElementById('show-label-bg').checked,
      });
      App.setStatus(t('settings.saved'));
    } catch(e) { App.setStatus(`Error: ${e.message}`); }
  },

  /* ── Custom Model Type Dialog (#4) ─────────────────── */
  _cmtModelPath: '',
  _cmtTestImgPath: '',
  _cmtInferredOutputs: [],  // [{index, name, shape:[...]}, ...]

  openCustomTypeDialog() {
    const overlay = document.createElement('div');
    overlay.id = 'cmt-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:1000;display:flex;align-items:center;justify-content:center;';
    overlay.innerHTML = `
      <div style="background:var(--background-neutral-01);border-radius:12px;padding:1.5rem;width:860px;max-height:90vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">모델 타입 추가 — Output Shape 매핑</h3>
        <div class="form-group">
          <label class="form-label">타입 이름</label>
          <input type="text" class="form-input input-normal" id="cmt-name" placeholder="예: my_custom_detr">
        </div>
        <div class="form-group" style="margin-top:0.75rem;">
          <label class="form-label">ONNX 모델</label>
          <div style="display:flex;gap:0.5rem;">
            <span id="cmt-model-label" class="text-secondary" style="flex:1;line-height:36px;">선택 안 됨</span>
            <button class="btn btn-secondary btn-sm" onclick="Tabs.settings._cmtBrowseModel()">모델 선택…</button>
          </div>
        </div>
        <div id="cmt-shape-info" class="text-secondary" style="margin:0.75rem 0;font-size:12px;font-family:monospace;">모델을 로드하면 실제 추론 Output Shape이 표시됩니다.</div>
        <div class="form-group">
          <label class="form-label">사용할 출력 텐서</label>
          <select class="form-input input-normal" id="cmt-oi" style="width:auto;" onchange="Tabs.settings._cmtOnOutputSelected()"></select>
        </div>
        <div id="cmt-dim-mapping" style="margin-top:0.75rem;"></div>
        <div style="display:flex;gap:1rem;align-items:center;margin-top:0.75rem;">
          <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;"><input type="checkbox" id="cmt-nms" checked> NMS 적용</label>
          <label class="form-label" style="margin:0;">Confidence:</label>
          <input type="number" class="form-input input-normal" id="cmt-conf" value="0.25" min="0.01" max="1.0" step="0.05" style="width:80px;height:28px;">
        </div>
        <div class="form-group" style="margin-top:0.75rem;">
          <label class="form-label">클래스 이름</label>
          <input type="text" class="form-input input-normal" id="cmt-class-names" placeholder="0:person, 1:car, 2:bike  (비우면 자동)">
        </div>
        <div style="border-top:1px solid var(--border-default);margin-top:1rem;padding-top:1rem;">
          <div class="form-group">
            <label class="form-label">테스트 이미지</label>
            <div style="display:flex;gap:0.5rem;">
              <span id="cmt-test-label" class="text-secondary" style="flex:1;line-height:36px;">선택 안 됨</span>
              <button class="btn btn-secondary btn-sm" onclick="Tabs.settings._cmtBrowseTestImg()">이미지 선택…</button>
              <button class="btn btn-primary btn-sm" onclick="Tabs.settings._cmtRunTest()">추론 실행</button>
            </div>
          </div>
          <div id="cmt-test-result" style="margin-top:0.5rem;min-height:180px;display:flex;align-items:center;justify-content:center;background:var(--background-neutral-02);border-radius:8px;">
            <span class="text-secondary">추론 결과가 여기에 표시됩니다.</span>
          </div>
        </div>
        <div style="display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem;">
          <button class="btn btn-primary" onclick="Tabs.settings._cmtSave()">저장</button>
          <button class="btn btn-secondary" onclick="document.getElementById('cmt-overlay').remove()">취소</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    this._cmtModelPath = '';
    this._cmtTestImgPath = '';
    this._cmtInferredOutputs = [];
  },

  async _cmtBrowseModel() {
    try {
      const r = await API.selectFile({ filters: 'ONNX (*.onnx)' });
      if (!r.path) return;
      this._cmtModelPath = r.path;
      document.getElementById('cmt-model-label').textContent = r.path.split(/[\\/]/).pop();
      document.getElementById('cmt-shape-info').textContent = '추론 중...';
      // 실제 추론으로 output shape 획득 (#1c)
      const res = await API.post('/api/model/infer-shapes', { path: r.path });
      if (res.error) { document.getElementById('cmt-shape-info').textContent = 'Error: ' + res.error; return; }
      this._cmtInferredOutputs = res.outputs;
      let info = '입력: ' + JSON.stringify(res.input_shape);
      res.outputs.forEach(o => { info += '\n출력[' + o.index + '] ' + o.name + ': ' + JSON.stringify(o.shape); });
      document.getElementById('cmt-shape-info').innerText = info;
      // 출력 텐서 선택 드롭다운
      const sel = document.getElementById('cmt-oi');
      sel.innerHTML = res.outputs.map(o =>
        '<option value="' + o.index + '">출력[' + o.index + '] ' + o.name + ' — ' + JSON.stringify(o.shape) + '</option>'
      ).join('');
      this._cmtOnOutputSelected();
    } catch(e) { document.getElementById('cmt-shape-info').textContent = 'Error: ' + e.message; }
  },

  _cmtOnOutputSelected() {
    const idx = +(document.getElementById('cmt-oi').value || 0);
    const out = this._cmtInferredOutputs.find(o => o.index === idx);
    if (!out) return;
    const shape = out.shape; // e.g. [1, 300, 6]
    this._cmtBuildDimMapping(shape);
  },

  _cmtBuildDimMapping(shape) {
    const container = document.getElementById('cmt-dim-mapping');
    // 각 차원별로: 크기 표시 + 의미 선택 드롭다운
    // 차원 역할: batch, num_detections, attributes(좌표+conf)
    const DIM_ROLES = ['batch', 'num_detections', 'attributes'];
    const ATTR_CHOICES = ['x1','y1','x2','y2','x_center','y_center','width','height','objectness','confidence','class_id'];
    for (let i = 0; i < 200; i++) ATTR_CHOICES.push('conf_class' + i);
    const attrOpts = ATTR_CHOICES.map(a => '<option>' + a + '</option>').join('');
    const dimOpts = DIM_ROLES.map(r => '<option>' + r + '</option>').join('');

    let html = '<div class="text-label" style="margin-bottom:0.5rem;">각 차원의 의미를 지정하세요</div>';
    html += '<div style="display:flex;flex-direction:column;gap:0.75rem;">';
    shape.forEach((size, i) => {
      const defaultRole = i === 0 ? 'batch' : (i === shape.length - 1 ? 'attributes' : 'num_detections');
      html += '<div class="card-flat" style="padding:0.75rem;">';
      html += '<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">';
      html += '<strong>dim[' + i + '] = ' + size + '</strong>';
      html += '<select class="form-input cmt-dim-role" data-dim="' + i + '" data-size="' + size + '" style="width:auto;height:28px;font-size:11px;padding:2px 4px;" onchange="Tabs.settings._cmtOnDimRoleChange(' + i + ')">' + dimOpts + '</select>';
      html += '</div>';
      html += '<div id="cmt-dim-detail-' + i + '"></div>';
      html += '</div>';
    });
    html += '</div>';
    container.innerHTML = html;

    // 기본값 설정
    document.querySelectorAll('.cmt-dim-role').forEach(sel => {
      const i = +sel.dataset.dim;
      sel.value = i === 0 ? 'batch' : (i === shape.length - 1 ? 'attributes' : 'num_detections');
      this._cmtOnDimRoleChange(i);
    });
  },

  _cmtOnDimRoleChange(dimIdx) {
    const sel = document.querySelector('.cmt-dim-role[data-dim="' + dimIdx + '"]');
    const role = sel.value;
    const size = +sel.dataset.size;
    const detail = document.getElementById('cmt-dim-detail-' + dimIdx);

    if (role === 'attributes') {
      // 각 슬롯에 의미 매핑
      const ATTR_CHOICES = ['(없음)','x1','y1','x2','y2','x_center','y_center','width','height','objectness','confidence','class_id'];
      for (let c = 0; c < 200; c++) ATTR_CHOICES.push('conf_class' + c);
      const opts = ATTR_CHOICES.map(a => '<option>' + a + '</option>').join('');
      let html = '<div style="max-height:180px;overflow-y:auto;display:flex;flex-direction:column;gap:1px;font-size:12px;">';
      for (let j = 0; j < Math.min(size, 200); j++) {
        const def = j < 4 ? ['x_center','y_center','width','height'][j] : 'conf_class' + (j - 4);
        html += '<div style="display:flex;align-items:center;gap:4px;">';
        html += '<span style="min-width:36px;text-align:right;color:var(--text-03);">[' + j + ']</span>';
        html += '<select class="form-input cmt-attr-sel" data-dim="' + dimIdx + '" data-idx="' + j + '" style="height:28px;font-size:11px;flex:1;padding:2px 4px;line-height:1;">' + opts + '</select>';
        html += '</div>';
      }
      html += '</div>';
      detail.innerHTML = html;
      // 기본값
      detail.querySelectorAll('.cmt-attr-sel').forEach(s => {
        const j = +s.dataset.idx;
        s.value = j < 4 ? ['x_center','y_center','width','height'][j] : 'conf_class' + (j - 4);
      });
    } else {
      detail.innerHTML = '<span class="text-secondary" style="font-size:11px;">' +
        (role === 'batch' ? '배치 차원 (무시됨)' : '탐지 개수 차원') + '</span>';
    }
  },

  async _cmtBrowseTestImg() {
    try {
      const r = await API.selectFile({ filters: 'Images (*.jpg *.jpeg *.png *.bmp)' });
      if (r.path) {
        this._cmtTestImgPath = r.path;
        document.getElementById('cmt-test-label').textContent = r.path.split(/[\\/]/).pop();
      }
    } catch(e) {}
  },

  _cmtCollectAttrRoles() {
    // attributes 역할인 차원에서 attr_roles 수집
    const roles = [...document.querySelectorAll('.cmt-dim-role')];
    const dimRoles = roles.map(s => s.value);
    const attrDimIdx = dimRoles.indexOf('attributes');
    if (attrDimIdx < 0) return { dimRoles, attrRoles: [], hasObjectness: false };
    const attrSels = [...document.querySelectorAll('.cmt-attr-sel[data-dim="' + attrDimIdx + '"]')];
    const attrRoles = attrSels.map(s => s.value).filter(v => v !== '(없음)');
    const hasObjectness = attrRoles.includes('objectness');
    return { dimRoles, attrRoles: attrSels.map(s => s.value === '(없음)' ? '' : s.value), hasObjectness };
  },

  _cmtParseClassNames() {
    const txt = (document.getElementById('cmt-class-names')?.value || '').trim();
    if (!txt) return null;
    const cn = {};
    for (const part of txt.split(',')) {
      const p = part.trim();
      const idx = p.indexOf(':');
      if (idx > 0) cn[p.slice(0, idx).trim()] = p.slice(idx + 1).trim();
    }
    return Object.keys(cn).length ? cn : null;
  },

  async _cmtRunTest() {
    if (!this._cmtModelPath) { App.setStatus('모델을 먼저 선택하세요.'); return; }
    const name = document.getElementById('cmt-name').value.trim() || 'test';
    const { dimRoles, attrRoles, hasObjectness } = this._cmtCollectAttrRoles();
    const oi = +(document.getElementById('cmt-oi').value || 0);
    const nms = document.getElementById('cmt-nms').checked;
    const conf = parseFloat(document.getElementById('cmt-conf').value || '0.25');
    const classNames = this._cmtParseClassNames();
    try {
      if (this._cmtTestImgPath) {
        await API.post('/api/config/custom-model-type', {
          name, model_path: this._cmtModelPath, output_index: oi,
          attr_roles: attrRoles, dim_roles: dimRoles,
          has_objectness: hasObjectness, nms, conf_threshold: conf,
          class_names: classNames,
        });
        await API.post('/api/config', { model_type: 'custom:' + name, conf_threshold: conf });
        const r = await API.post('/api/infer/image', {
          model_path: this._cmtModelPath, image_path: this._cmtTestImgPath, conf
        });
        if (r.error) {
          document.getElementById('cmt-test-result').innerHTML = '<span style="color:var(--action-danger-05);">Error: ' + r.error + '</span>';
        } else {
          document.getElementById('cmt-test-result').innerHTML =
            '<div style="text-align:center;"><img src="data:image/jpeg;base64,' + r.image + '" style="max-width:100%;max-height:250px;cursor:pointer;" ondblclick="Tabs.settings._cmtZoomImg(this.src)" title="더블클릭으로 확대"><br><span class="text-secondary">탐지: ' + r.detections + '개</span></div>';
        }
      } else {
        const r = await API.post('/api/config/custom-model-type/test', {
          name, model_path: this._cmtModelPath, output_index: oi,
          attr_roles: attrRoles, dim_roles: dimRoles,
          has_objectness: hasObjectness, nms, conf_threshold: conf,
        });
        document.getElementById('cmt-test-result').innerHTML = r.error
          ? '<span style="color:var(--action-danger-05);">Error: ' + r.error + '</span>'
          : '<span class="text-secondary">더미 테스트: ' + r.detections + '개 탐지</span>';
      }
    } catch(e) { document.getElementById('cmt-test-result').innerHTML = '<span style="color:var(--action-danger-05);">' + e.message + '</span>'; }
  },

  _cmtZoomImg(src) {
    const ov = document.createElement('div');
    ov.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:9999;display:flex;align-items:center;justify-content:center;cursor:pointer;';
    ov.innerHTML = '<img src="' + src + '" style="max-width:95vw;max-height:95vh;object-fit:contain;">';
    ov.onclick = () => ov.remove();
    document.body.appendChild(ov);
  },

  async _cmtSave() {
    const name = document.getElementById('cmt-name').value.trim();
    if (!name) { App.setStatus('타입 이름을 입력하세요.'); return; }
    if (!this._cmtModelPath) { App.setStatus('모델을 먼저 선택하세요.'); return; }
    const { dimRoles, attrRoles, hasObjectness } = this._cmtCollectAttrRoles();
    const oi = +(document.getElementById('cmt-oi').value || 0);
    const nms = document.getElementById('cmt-nms').checked;
    const conf = parseFloat(document.getElementById('cmt-conf').value || '0.25');
    const classNames = this._cmtParseClassNames();
    try {
      const r = await API.post('/api/config/custom-model-type', {
        name, model_path: this._cmtModelPath, output_index: oi,
        attr_roles: attrRoles, dim_roles: dimRoles,
        has_objectness: hasObjectness, nms, conf_threshold: conf,
        class_names: classNames,
      });
      if (r.ok) {
        App.setStatus("'" + name + "' 모델 타입 저장 완료");
        const sel = document.getElementById('v-model-type');
        if (sel && ![...sel.options].some(o => o.value === 'custom:' + name)) {
          sel.add(new Option(name, 'custom:' + name));
        }
        document.getElementById('cmt-overlay').remove();
      } else {
        App.setStatus('Error: ' + (r.error || ''));
      }
    } catch(e) { App.setStatus('Error: ' + e.message); }
  },
};

/* ── Benchmark ──────────────────────────────────────── */
Tabs.benchmark = {
  title: true,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          ${multiModelSlots('bench-slots','bench-list')}
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('bench.config')}</h3>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
            <div class="form-group"><label class="form-label">${t('bench.iters')}</label><input type="number" class="form-input input-normal" value="500" min="10" max="5000" step="100" id="bench-iters"></div>
            <div class="form-group"><label class="form-label">${t('bench.input_size')}</label><select class="form-input input-normal" id="bench-size"><option>640</option><option>320</option><option>1280</option></select></div>
            <div class="form-group"><label class="form-label">${t('bench.warmup')}</label><span class="form-input" style="background:var(--background-neutral-02);color:var(--text-03);">${t('bench.fixed')}</span></div>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" id="bench-run" onclick="Tabs.benchmark.run()">${t('bench.run')}</button>
            <button class="btn btn-danger btn-sm" id="bench-stop" disabled onclick="Tabs.benchmark.stop()">${t('stop')}</button>
          </div>
          <div style="margin-top:0.75rem;">
            <div class="progress-track"><div class="progress-fill" id="bench-progress" style="width:0%"></div></div>
            <span class="text-secondary" id="bench-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span>
          </div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
            <h3 class="text-heading-h3">${t('bench.results')}</h3>
            <div style="display:flex;gap:0.5rem;">
              <button class="btn btn-secondary btn-sm" onclick="Tabs.benchmark.exportResults()">${t('export')}</button>
              <button class="btn btn-ghost btn-sm" onclick="document.getElementById('bench-results').innerHTML='<tr><td colspan=13 class=text-secondary style=text-align:center;padding:2rem>${t('bench.run_hint')}</td></tr>'">${t('reset')}</button>
            </div>
          </div>
          <div class="table-container"><table><thead><tr><th>Model</th><th>Provider</th><th>FPS</th><th>Avg(ms)</th><th>Pre(ms)</th><th>Infer(ms)</th><th>Post(ms)</th><th title="P50 (중앙값): 전체 측정값의 50%가 이 값 이하인 지점. 일반적인 응답 시간.">P50(ms)</th><th title="P95: 전체 측정값의 95%가 이 값 이하. 대부분의 요청이 이 시간 내에 완료.">P95(ms)</th><th title="P99: 전체 측정값의 99%가 이 값 이하. 최악에 가까운 응답 시간 (tail latency).">P99(ms)</th><th>CPU%</th><th>RAM(MB)</th><th>GPU%</th></tr></thead>
          <tbody id="bench-results"><tr><td colspan="13" class="text-secondary" style="text-align:center;padding:2rem;">${t('bench.run_hint')}</td></tr></tbody></table></div>
        </div>
      </div>`;
  },
  async run() {
    const models = getSlotModels('bench-slots');
    if (!models.length) { App.setStatus(t('bench.no_models')); return; }
    document.getElementById('bench-run').disabled = true;
    document.getElementById('bench-stop').disabled = false;
    document.getElementById('bench-status').textContent = t('bench.running');
    document.getElementById('bench-progress').style.width = '0%';
    App.setStatus(t('bench.running'));
    try {
      const r = await API.post('/api/benchmark/run', {
        models, iterations: +document.getElementById('bench-iters').value,
        input_size: +document.getElementById('bench-size').value
      });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      // Poll for results
      this._polling = true;
      this._poll();
    } catch(e) { App.setStatus(`Error: ${e.message}`); document.getElementById('bench-run').disabled = false; }
  },
  _polling: false,
  async _poll() {
    if (!this._polling) return;
    try {
      const s = await API.get('/api/benchmark/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      document.getElementById('bench-progress').style.width = pct + '%';
      document.getElementById('bench-status').textContent = s.msg || '';
      if (s.results && s.results.length) {
        const tb = document.getElementById('bench-results');
        tb.innerHTML = s.results.map((x, i) => {
          if (x.error) return `<tr><td colspan="13" style="color:var(--action-danger-05);">${x.error}</td></tr>`;
          const hl = i === 0 ? ' style="background:var(--status-success-01);font-weight:600;"' : '';
          const gpu = x.gpu_pct != null ? `${x.gpu_pct}%` : 'N/A';
          return `<tr${hl}><td>${x.name||'—'}</td><td>${x.provider||'—'}</td><td>${x.fps||'—'}</td><td>${x.avg||'—'}</td><td>${x.pre_ms||'—'}</td><td>${x.infer_ms||'—'}</td><td>${x.post_ms||'—'}</td><td>${x.p50||'—'}</td><td>${x.p95||'—'}</td><td>${x.p99||'—'}</td><td>${x.cpu_pct||'—'}</td><td>${x.ram_mb||'—'}</td><td>${gpu}</td></tr>`;
        }).join('');
      }
      if (s.running) {
        setTimeout(() => this._poll(), 500);
      } else {
        this._polling = false;
        document.getElementById('bench-run').disabled = false;
        document.getElementById('bench-stop').disabled = true;
        document.getElementById('bench-progress').style.width = '100%';
        App.setStatus(t('bench.complete'));
      }
    } catch(e) { setTimeout(() => this._poll(), 1000); }
  },
  stop() {
    this._polling = false;
    App.setStatus('Stopped');
    document.getElementById('bench-run').disabled = false;
    document.getElementById('bench-stop').disabled = true;
  },
  exportResults() {
    window.open('/api/benchmark/export-csv', '_blank');
  },
};


/* ── Evaluation ─────────────────────────────────────── */
Tabs.evaluation = {
  title: true,
  _cachedHTML: null,
  render() {
    if (this._cachedHTML) return this._cachedHTML;
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('eval.setup')}</h3>
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">
            <h3 class="text-heading-h3">${t('bench.models')}</h3>
            <button class="btn btn-secondary btn-sm" onclick="Tabs.evaluation._addModel()">${t('add_model')}</button>
          </div>
          <div id="eval-model-slots" style="display:flex;flex-direction:column;gap:0.5rem;">
            <div class="text-secondary" style="padding:1rem;text-align:center;" id="eval-slots-hint">${t('bench.add_hint')}</div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.75rem;">
            <div class="form-group">
              <label class="form-label">Confidence</label>
              <input type="number" class="form-input input-normal" value="0.25" min="0.01" max="1.0" step="0.05" id="eval-conf" style="width:100px;">
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.75rem;">
            ${imgDirInput('eval-img')}
            ${lblDirInput('eval-lbl')}
          </div>
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">GT 클래스 매핑 (id: name, 한 줄에 하나)</label>
            <textarea class="form-input" id="eval-classmap" rows="4" style="font-size:12px;font-family:monospace;" placeholder="0: person&#10;1: car&#10;2: bicycle"></textarea>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" id="eval-run-btn" onclick="Tabs.evaluation.run()">${t('eval.run')}</button>
            <button class="btn btn-danger btn-sm" id="eval-stop-btn" disabled onclick="Tabs.evaluation.stop()">${t('stop')}</button>
            <div style="flex:1;"></div>
            <button class="btn btn-secondary btn-sm" onclick="Tabs.evaluation.exportCSV()">CSV 내보내기</button>
          </div>
          <div style="margin-top:0.5rem;">
            <div class="progress-track"><div class="progress-fill" id="eval-prog" style="width:0%"></div></div>
            <span class="text-secondary" id="eval-status" style="margin-top:0.25rem;display:block;">${t('ready')}</span>
          </div>
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('eval.results')}</h3>
          <div class="table-container"><table><thead><tr><th>Model</th><th>mAP@50</th><th>mAP@50:95</th><th>Precision</th><th>Recall</th><th>F1</th><th></th></tr></thead>
          <tbody id="eval-results"><tr><td colspan="7" class="text-secondary" style="text-align:center;padding:2rem;">${t('eval.run_hint')}</td></tr></tbody></table></div>
        </div>
        <div id="eval-detail-container"></div>
      </div>`;
  },
  async init() {
    // 캐시된 결과가 있으면 복원
    if (this._lastResults && this._lastResults.length) {
      this._renderResults(this._lastResults);
    }
    // 모델 타입 목록 로드 (슬롯용)
    try {
      const c = await API.config();
      this._modelTypes = c.model_types || {};
    } catch(e) { this._modelTypes = {yolo:'YOLO'}; }
  },
  _modelTypes: {},
  _evalSlotN: 0,
  async _addModel() {
    try {
      const r = await API.post('/api/fs/select-multi', { filters: 'ONNX (*.onnx)' });
      if (!r.paths || !r.paths.length) return;
      for (const path of r.paths) {
        this._addSlot(path);
      }
    } catch(e) {}
  },
  _addSlot(path) {
    const c = document.getElementById('eval-model-slots');
    const hint = document.getElementById('eval-slots-hint');
    if (hint) hint.remove();
    const id = ++this._evalSlotN;
    const name = path.split(/[\\/]/).pop();
    const typeOpts = Object.entries(this._modelTypes).map(([k,v]) =>
      `<option value="${k}">${v}</option>`).join('');
    const d = document.createElement('div');
    d.className = 'card-flat'; d.id = `eval-ms-${id}`;
    d.style.cssText = 'padding:0.5rem 0.75rem;display:flex;align-items:center;gap:0.5rem;';
    d.innerHTML = `<span class="text-mono" style="flex:1;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${path}">${name}</span>
      <input type="hidden" class="eval-slot-path" value="${path}">
      <select class="form-input eval-slot-type" style="width:auto;height:28px;font-size:11px;padding:0 6px;">${typeOpts}</select>
      <button class="btn btn-ghost btn-sm" onclick="document.getElementById('eval-ms-${id}').remove()" style="color:var(--action-danger-05);padding:0 0.25rem;">✕</button>`;
    c.appendChild(d);
  },
  _polling: false,
  _lastResults: [],
  async run() {
    const slots = document.querySelectorAll('#eval-model-slots .eval-slot-path');
    const types = document.querySelectorAll('#eval-model-slots .eval-slot-type');
    const models = [];
    slots.forEach((el, i) => {
      models.push({ path: el.value, model_type: types[i]?.value || 'yolo' });
    });
    const imgDir = document.getElementById('eval-img').value || G.imgDir;
    const lblDir = document.getElementById('eval-lbl').value || G.lblDir;
    const conf = parseFloat(document.getElementById('eval-conf')?.value || '0.25');
    if (!models.length) { App.setStatus('Add at least one model'); return; }
    if (!imgDir) { App.setStatus('Select images directory'); return; }

    // GT 클래스 스캔 + 모델 클래스 로드 → 매핑 다이얼로그
    App.setStatus('Loading class info...');
    try {
      const gtRes = await API.post('/api/gt/classes', { label_dir: lblDir || imgDir });
      const gtClasses = gtRes.classes || [];
      const classmapText = document.getElementById('eval-classmap')?.value || '';
      const classmapNames = {};
      classmapText.split('\n').forEach(line => {
        const m = line.match(/^\s*(\d+)\s*:\s*(.+)/);
        if (m) classmapNames[parseInt(m[1])] = m[2].trim();
      });

      const modelInfos = [];
      for (const m of models) {
        const r = await API.post('/api/model/classes', { path: m.path, model_type: m.model_type });
        const name = m.path.split(/[\\/]/).pop();
        modelInfos.push({ name, names: r.names || {} });
      }

      // 매핑 다이얼로그 표시
      const result = await this._showMappingDialog(gtClasses, classmapNames, modelInfos);
      if (!result) return; // 취소

      document.getElementById('eval-run-btn').disabled = true;
      document.getElementById('eval-stop-btn').disabled = false;
      document.getElementById('eval-status').textContent = 'Running...';
      document.getElementById('eval-prog').style.width = '0%';

      const r = await API.post('/api/evaluation/run-async', {
        models, img_dir: imgDir, label_dir: lblDir, conf,
        per_model_mappings: result.mappings,
        mapped_only: result.mapped_only,
      });
      if (r.error) { App.setStatus('Error: ' + r.error); document.getElementById('eval-run-btn').disabled = false; return; }
      this._polling = true;
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message); document.getElementById('eval-run-btn').disabled = false; }
  },
  _savedMappings: {},
  _savedMappedOnly: true,
  _showMappingDialog(gtClasses, classmapNames, modelInfos) {
    return new Promise((resolve) => {
      const overlay = document.createElement('div');
      overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:1000;display:flex;align-items:center;justify-content:center;';
      const dlg = document.createElement('div');
      dlg.style.cssText = 'background:var(--background-neutral-01);border-radius:12px;padding:1.5rem;width:820px;max-height:90vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,0.3);';

      const gtItems = gtClasses.map(id => ({ id, name: classmapNames[id] || `class_${id}` }));
      const self = this;
      // Per-model connections: { modelName: { modelClsId: gtClsId } }
      const connections = {};
      modelInfos.forEach(mi => {
        connections[mi.name] = { ...(self._savedMappings[mi.name] || {}) };
        // Auto-map by matching ID if no saved mapping
        if (!self._savedMappings[mi.name]) {
          const modelCls = Object.keys(mi.names).map(Number);
          modelCls.forEach(mid => { if (gtClasses.includes(mid)) connections[mi.name][mid] = mid; });
        }
      });

      let curTab = 0;

      // Build tabs
      let tabsHtml = '';
      modelInfos.forEach((mi, idx) => {
        tabsHtml += `<span class="mapping-tab" data-idx="${idx}" style="cursor:pointer;padding:0.25rem 0.75rem;${idx===0?'font-weight:bold;border-bottom:2px solid var(--action-link-05);':''}">${mi.name}</span>`;
      });

      // Copy-from dropdown
      let copyOpts = '<option value="">— 다른 모델에서 복사 —</option>';
      modelInfos.forEach((mi, idx) => { copyOpts += `<option value="${idx}">${mi.name}</option>`; });

      dlg.innerHTML = `
        <h3 style="margin-bottom:0.75rem;">클래스 매핑 — 선분으로 연결하세요</h3>
        <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;">
          <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;"><input type="checkbox" id="mapping-mapped-only" ${this._savedMappedOnly?'checked':''}> 매핑된 클래스만 평가</label>
          <div style="margin-left:auto;display:flex;gap:0.5rem;">
            <select class="form-input input-normal" id="mapping-copy-from" style="height:28px;font-size:11px;width:auto;">${copyOpts}</select>
            <button class="btn btn-ghost btn-sm" id="mapping-copy-btn">복사</button>
            <button class="btn btn-ghost btn-sm" id="mapping-clear-btn" style="color:var(--action-danger-05);">초기화</button>
          </div>
        </div>
        <div style="display:flex;gap:0.5rem;border-bottom:1px solid var(--border-default);margin-bottom:0.75rem;padding-bottom:0.5rem;">${tabsHtml}</div>
        <div id="mapping-canvas-area" style="position:relative;"></div>
        <div style="font-size:11px;color:var(--text-03);margin-top:0.5rem;">💡 왼쪽 모델 클래스를 클릭한 뒤 오른쪽 GT 클래스를 클릭하면 연결됩니다. 연결선을 클릭하면 삭제됩니다.</div>
        <div style="display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem;">
          <button class="btn btn-secondary" id="mapping-cancel">취소</button>
          <button class="btn btn-primary" id="mapping-ok">확인</button>
        </div>`;

      overlay.appendChild(dlg);
      document.body.appendChild(overlay);

      const COLORS = ['#4fc3f7','#81c784','#ffb74d','#e57373','#ba68c8','#4dd0e1','#aed581','#ff8a65','#f06292','#7986cb'];

      function renderPanel(tabIdx) {
        curTab = tabIdx;
        const mi = modelInfos[tabIdx];
        const modelCls = Object.entries(mi.names).map(([k,v]) => ({ id: parseInt(k), name: v }));
        const conn = connections[mi.name];
        const area = dlg.querySelector('#mapping-canvas-area');

        const ROW_H = 32, PAD = 8, COL_W = 220, GAP = 280;
        const maxRows = Math.max(modelCls.length, gtItems.length);
        const canvasH = maxRows * ROW_H + PAD * 2;
        const canvasW = COL_W * 2 + GAP;

        let html = `<svg id="mapping-svg" width="${canvasW}" height="${canvasH}" style="position:absolute;top:0;left:0;pointer-events:none;z-index:1;"></svg>`;
        html += `<div style="display:flex;justify-content:space-between;position:relative;z-index:2;">`;

        // Left column: model classes
        html += `<div style="width:${COL_W}px;">`;
        html += `<div style="font-size:11px;font-weight:600;color:var(--text-02);margin-bottom:4px;">모델 클래스</div>`;
        modelCls.forEach((mc, i) => {
          const mapped = conn[mc.id] !== undefined;
          const color = mapped ? COLORS[mc.id % COLORS.length] : 'var(--border-default)';
          html += `<div class="mapping-node mapping-left" data-mid="${mc.id}" style="height:${ROW_H-4}px;line-height:${ROW_H-4}px;padding:0 8px;margin:2px 0;border-radius:6px;cursor:pointer;font-size:12px;border:2px solid ${color};background:${mapped?color+'22':'var(--background-neutral-02)'};display:flex;align-items:center;justify-content:space-between;transition:all 0.15s;">
            <span>${mc.id}: ${mc.name}</span>
            <span style="width:10px;height:10px;border-radius:50%;background:${color};flex-shrink:0;"></span>
          </div>`;
        });
        html += `</div>`;

        // Right column: GT classes
        html += `<div style="width:${COL_W}px;">`;
        html += `<div style="font-size:11px;font-weight:600;color:var(--text-02);margin-bottom:4px;">GT 클래스</div>`;
        gtItems.forEach((g, i) => {
          const mapped = Object.values(conn).includes(g.id);
          const color = mapped ? COLORS[g.id % COLORS.length] : 'var(--border-default)';
          html += `<div class="mapping-node mapping-right" data-gid="${g.id}" style="height:${ROW_H-4}px;line-height:${ROW_H-4}px;padding:0 8px;margin:2px 0;border-radius:6px;cursor:pointer;font-size:12px;border:2px solid ${color};background:${mapped?color+'22':'var(--background-neutral-02)'};display:flex;align-items:center;gap:4px;transition:all 0.15s;">
            <span style="width:10px;height:10px;border-radius:50%;background:${color};flex-shrink:0;"></span>
            <span>${g.id}: ${g.name}</span>
          </div>`;
        });
        html += `</div></div>`;

        area.innerHTML = html;
        area.style.minHeight = canvasH + 'px';

        drawLines();
        bindEvents();
      }

      let pendingLeft = null; // model class id being connected

      function drawLines() {
        const svg = dlg.querySelector('#mapping-svg');
        if (!svg) return;
        const mi = modelInfos[curTab];
        const conn = connections[mi.name];
        const area = dlg.querySelector('#mapping-canvas-area');
        const areaRect = area.getBoundingClientRect();
        let lines = '';
        for (const [midStr, gid] of Object.entries(conn)) {
          const mid = parseInt(midStr);
          const leftEl = area.querySelector(`.mapping-left[data-mid="${mid}"]`);
          const rightEl = area.querySelector(`.mapping-right[data-gid="${gid}"]`);
          if (!leftEl || !rightEl) continue;
          const lr = leftEl.getBoundingClientRect();
          const rr = rightEl.getBoundingClientRect();
          const x1 = lr.right - areaRect.left;
          const y1 = lr.top + lr.height/2 - areaRect.top;
          const x2 = rr.left - areaRect.left;
          const y2 = rr.top + rr.height/2 - areaRect.top;
          const color = COLORS[mid % COLORS.length];
          // Invisible thick line for click target
          lines += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="transparent" stroke-width="14" style="pointer-events:stroke;cursor:pointer;" data-mid="${mid}" class="mapping-line-hit"/>`;
          // Visible line
          lines += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2.5" stroke-linecap="round" style="pointer-events:none;" data-mid="${mid}" class="mapping-line-vis"/>`;
        }
        svg.innerHTML = lines;
        svg.style.pointerEvents = 'auto';
        // Click on line to remove
        svg.querySelectorAll('.mapping-line-hit').forEach(line => {
          line.addEventListener('click', (e) => {
            e.stopPropagation();
            const mid = parseInt(line.dataset.mid);
            delete connections[modelInfos[curTab].name][mid];
            renderPanel(curTab);
          });
          line.addEventListener('mouseenter', () => {
            const vis = svg.querySelector(`.mapping-line-vis[data-mid="${line.dataset.mid}"]`);
            if (vis) { vis.setAttribute('stroke-width', '4'); vis.setAttribute('stroke-dasharray', '6,3'); }
          });
          line.addEventListener('mouseleave', () => {
            const vis = svg.querySelector(`.mapping-line-vis[data-mid="${line.dataset.mid}"]`);
            if (vis) { vis.setAttribute('stroke-width', '2.5'); vis.removeAttribute('stroke-dasharray'); }
          });
        });
      }

      function bindEvents() {
        const area = dlg.querySelector('#mapping-canvas-area');
        area.querySelectorAll('.mapping-left').forEach(el => {
          el.addEventListener('click', () => {
            const mid = parseInt(el.dataset.mid);
            // If already connected, disconnect
            if (connections[modelInfos[curTab].name][mid] !== undefined && pendingLeft === null) {
              delete connections[modelInfos[curTab].name][mid];
              renderPanel(curTab);
              return;
            }
            // Start connection
            pendingLeft = mid;
            area.querySelectorAll('.mapping-left').forEach(n => n.style.opacity = '0.4');
            el.style.opacity = '1';
            el.style.boxShadow = '0 0 0 3px var(--action-link-05)';
          });
        });
        area.querySelectorAll('.mapping-right').forEach(el => {
          el.addEventListener('click', () => {
            if (pendingLeft === null) return;
            const gid = parseInt(el.dataset.gid);
            connections[modelInfos[curTab].name][pendingLeft] = gid;
            pendingLeft = null;
            renderPanel(curTab);
          });
        });
      }

      // Tab switching
      dlg.querySelectorAll('.mapping-tab').forEach(tab => {
        tab.onclick = () => {
          pendingLeft = null;
          dlg.querySelectorAll('.mapping-tab').forEach(t => t.style.cssText = 'cursor:pointer;padding:0.25rem 0.75rem;');
          tab.style.cssText = 'cursor:pointer;padding:0.25rem 0.75rem;font-weight:bold;border-bottom:2px solid var(--action-link-05);';
          renderPanel(parseInt(tab.dataset.idx));
        };
      });

      // Copy from another model
      dlg.querySelector('#mapping-copy-btn').onclick = () => {
        const sel = dlg.querySelector('#mapping-copy-from');
        const srcIdx = parseInt(sel.value);
        if (isNaN(srcIdx)) return;
        const srcName = modelInfos[srcIdx].name;
        const curName = modelInfos[curTab].name;
        if (srcName === curName) return;
        connections[curName] = { ...connections[srcName] };
        renderPanel(curTab);
        App.setStatus(`'${srcName}'의 매핑을 복사했습니다.`);
      };

      // Clear
      dlg.querySelector('#mapping-clear-btn').onclick = () => {
        connections[modelInfos[curTab].name] = {};
        renderPanel(curTab);
      };

      // Cancel / OK
      dlg.querySelector('#mapping-cancel').onclick = () => { overlay.remove(); resolve(null); };
      dlg.querySelector('#mapping-ok').onclick = () => {
        const mappings = {};
        for (const [name, conn] of Object.entries(connections)) {
          if (Object.keys(conn).length) mappings[name] = { ...conn };
        }
        const mapped_only = dlg.querySelector('#mapping-mapped-only').checked;
        self._savedMappings = mappings;
        self._savedMappedOnly = mapped_only;
        overlay.remove();
        resolve({ mappings, mapped_only });
      };
      overlay.onclick = (e) => { if (e.target === overlay) { overlay.remove(); resolve(null); } };

      // Initial render
      renderPanel(0);
    });
  },
  async _poll() {
    if (!this._polling) return;
    try {
      const s = await API.get('/api/evaluation/status');
      const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
      document.getElementById('eval-prog').style.width = pct + '%';
      document.getElementById('eval-status').textContent = s.msg || '';
      if (s.results && s.results.length) {
        this._lastResults = s.results;
        this._renderResults(s.results);
      }
      if (s.running) { setTimeout(() => this._poll(), 500); }
      else {
        this._polling = false;
        document.getElementById('eval-run-btn').disabled = false;
        document.getElementById('eval-stop-btn').disabled = true;
        document.getElementById('eval-prog').style.width = '100%';
        // 캐시 저장
        this._cachedHTML = document.getElementById('page-body').innerHTML;
        App.setStatus('Evaluation complete');
      }
    } catch(e) { setTimeout(() => this._poll(), 1000); }
  },
  _renderResults(results) {
    const tb = document.getElementById('eval-results');
    if (!tb) return;
    tb.innerHTML = results.map((x, i) => {
      if (x.error) return '<tr><td>' + (x.name||'') + '</td><td colspan="6" style="color:var(--action-danger-05);">' + x.error + '</td></tr>';
      const detBtn = x.detail ? '<button class="btn btn-ghost btn-sm" onclick="Tabs.evaluation.showDetail('+i+')">상세</button>' : '';
      return '<tr><td>'+(x.name||'')+'</td><td>'+(x.map50?.toFixed(4)||0)+'%</td><td>'+(x.map5095?.toFixed(4)||0)+'%</td><td>'+(x.precision?.toFixed(4)||0)+'%</td><td>'+(x.recall?.toFixed(4)||0)+'%</td><td>'+(x.f1?.toFixed(4)||0)+'%</td><td>'+detBtn+'</td></tr>';
    }).join('');
  },
  stop() {
    this._polling = false;
    document.getElementById('eval-run-btn').disabled = false;
    document.getElementById('eval-stop-btn').disabled = true;
  },
  showDetail(idx) {
    const r = this._lastResults[idx];
    if (!r || !r.detail) { App.setStatus('상세 데이터 없음 (서버에서 detail 포함 필요)'); return; }
    const c = document.getElementById('eval-detail-container');
    const keys = Object.keys(r.detail).filter(k => k !== '__overall__').sort((a,b) => +a - +b);
    let rows = keys.map(cid => {
      const v = r.detail[cid];
      return '<tr><td>'+cid+'</td><td>'+(v.ap*100).toFixed(4)+'%</td><td>'+(v.precision*100).toFixed(4)+'%</td><td>'+(v.recall*100).toFixed(4)+'%</td><td>'+(v.f1*100).toFixed(4)+'%</td><td>'+v.tp+'</td><td>'+v.fp+'</td><td>'+v.fn+'</td></tr>';
    }).join('');
    c.innerHTML = '<div class="card" style="padding:1.5rem;"><h3 class="text-heading-h3" style="margin-bottom:1rem;">클래스별 상세 — '+r.name+'</h3><div class="table-container"><table><thead><tr><th>Class</th><th>AP@50</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr></thead><tbody>'+rows+'</tbody></table></div></div>';
  },
  exportCSV() { window.open('/api/evaluation/export-csv', '_blank'); },
};

/* ── Analysis ───────────────────────────────────────── */
Tabs.analysis = {
  title: true,
  render() {
    return `
      <div style="display:flex;gap:1rem;height:100%;">
        <div style="flex:1;display:flex;flex-direction:column;gap:1rem;">
          <div class="card" style="padding:1rem;">
            ${modelInput('ana-model')}
            <div class="form-group"><label class="form-label">Model Type</label><select class="form-input input-normal" id="ana-type" style="width:auto;"></select></div>
            <div class="form-group">
              <label class="form-label">Image</label>
              <div style="display:flex;gap:0.5rem;">
                <input type="text" class="form-input input-normal" style="flex:1;" readonly id="ana-img">
                <button class="btn btn-secondary btn-sm" onclick="pickFile('ana-img','Images (*.jpg *.jpeg *.png *.bmp)')">${t('browse')}</button>
              </div>
            </div>
            <button class="btn btn-primary" style="margin-top:0.5rem;" onclick="Tabs.analysis.run()">${t('run')}</button>
          </div>
          <div class="card" style="flex:1;padding:1rem;display:flex;flex-wrap:wrap;gap:0.5rem;align-items:flex-start;justify-content:center;min-height:300px;overflow-y:auto;" id="ana-panels">
            <span class="text-muted">${t('viewer.open_hint')}</span>
          </div>
        </div>
        <div style="width:260px;display:flex;flex-direction:column;gap:0.75rem;">
          <div class="card-flat" style="padding:1rem;"><div class="text-label" style="margin-bottom:0.5rem;">Inference Stats</div><div class="text-secondary" id="ana-stats">—</div></div>
          <div class="card-flat" style="padding:1rem;"><div class="text-label" style="margin-bottom:0.5rem;">Detections</div><div class="text-secondary" id="ana-dets">—</div></div>
        </div>
      </div>`;
  },
  async init() {
    try {
      const c = await API.config();
      const el = document.getElementById('ana-type');
      if (el && c.model_types) el.innerHTML = Object.entries(c.model_types).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');
    } catch(e) {}
  },
  async run() {
    const model_path = document.getElementById('ana-model').value || G.model;
    const image_path = document.getElementById('ana-img').value;
    if (!model_path||!image_path) { App.setStatus('Select model and image'); return; }
    App.setStatus('Running inference analysis...');
    try {
      const r = await API.post('/api/analysis/inference-analysis', {
        model_path, model_type: document.getElementById('ana-type').value, image_path, conf: 0.25
      });
      if (r.error) { App.setStatus('Error: '+r.error); return; }
      let html = '';
      if (r.original) html += `<div style="text-align:center;"><div class="text-label" style="margin-bottom:0.25rem;">Original</div><img src="data:image/jpeg;base64,${r.original}" style="max-width:100%;max-height:250px;"></div>`;
      if (r.letterbox) html += `<div style="text-align:center;"><div class="text-label" style="margin-bottom:0.25rem;">Letterbox</div><img src="data:image/jpeg;base64,${r.letterbox}" style="max-width:100%;max-height:250px;"></div>`;
      if (r.tensor_heatmap) html += `<div style="text-align:center;"><div class="text-label" style="margin-bottom:0.25rem;">Tensor Heatmap</div><img src="data:image/jpeg;base64,${r.tensor_heatmap}" style="max-width:100%;max-height:250px;"></div>`;
      if (r.result) html += `<div style="text-align:center;"><div class="text-label" style="margin-bottom:0.25rem;">Detections</div><img src="data:image/jpeg;base64,${r.result}" style="max-width:100%;max-height:250px;"></div>`;
      document.getElementById('ana-panels').innerHTML = html || '<span class="text-muted">No results</span>';
      document.getElementById('ana-stats').innerHTML = `Pre: ${r.pre_ms||'—'}ms<br>Infer: ${r.infer_ms||'—'}ms<br>Post: ${r.post_ms||'—'}ms`;
      document.getElementById('ana-dets').textContent = (r.detections||0)+' detections';
      App.setStatus('Inference analysis complete');
    } catch(e) { App.setStatus('Error: '+e.message); }
  }
};

/* ── Explorer ───────────────────────────────────────── */
Tabs.explorer = {
  title: true,
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <div style="display:grid;grid-template-columns:1fr 1fr auto;gap:1rem;align-items:end;">
            ${imgDirInput('exp-img')}
            ${lblDirInput('exp-lbl')}
            <button class="btn btn-primary" style="height:36px;" onclick="Tabs.explorer.load()">${t('explorer.load')}</button>
          </div>
        </div>
        <div style="display:flex;gap:1rem;">
          <div style="width:200px;">
            <div class="card-flat" style="padding:1rem;">
              <div class="text-label" style="margin-bottom:0.5rem;">${t('explorer.filter')}</div>
              <div class="form-group" style="margin-bottom:0.5rem;">
                <label class="form-label">Class</label>
                <select class="form-input input-normal" id="exp-class-filter"><option value="">All</option></select>
              </div>
              <div class="form-group">
                <label class="form-label">Min boxes</label>
                <input type="number" class="form-input input-normal" value="0" min="0" id="exp-min-boxes">
              </div>
            </div>
            <div class="card-flat" style="padding:1rem;margin-top:0.75rem;">
              <div class="text-label" style="margin-bottom:0.5rem;">${t('explorer.stats')}</div>
              <div id="exp-stats" class="text-secondary">—</div>
            </div>
          </div>
          <div style="flex:1;">
            <div id="exp-gallery" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:0.5rem;">
              <div class="text-secondary" style="grid-column:1/-1;text-align:center;padding:3rem;">${t('explorer.load_hint')}</div>
            </div>
          </div>
        </div>
      </div>`;
  },
  async load() { App.setStatus('Loading dataset...'); }
};

/* ── Splitter ───────────────────────────────────────── */
Tabs.splitter = {
  title: true,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('splitter.input')}</h3>
          ${imgDirInput('split-img')}
          ${lblDirInput('split-lbl')}
          ${outDirInput('split-out')}
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('splitter.ratio')}</h3>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
            <div class="form-group"><label class="form-label">${t('splitter.train')}</label><input type="number" class="form-input input-normal" value="0.7" min="0" max="1" step="0.05"></div>
            <div class="form-group"><label class="form-label">${t('splitter.val')}</label><input type="number" class="form-input input-normal" value="0.2" min="0" max="1" step="0.05"></div>
            <div class="form-group"><label class="form-label">${t('splitter.test')}</label><input type="number" class="form-input input-normal" value="0.1" min="0" max="1" step="0.05"></div>
          </div>
          <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.75rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" checked> ${t('splitter.stratified')}</label>
        </div>
        <button class="btn btn-primary" onclick="App.setStatus('Splitting...')">${t('splitter.run')}</button>
      </div>`;
  }
};
