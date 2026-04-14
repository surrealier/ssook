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
              <label class="form-label" style="font-size:10px;margin:0;white-space:nowrap;">${t('viewer.batch')}</label>
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
              <input type="text" class="form-input input-normal" style="flex:1;font-size:11px;height:28px;" id="v-model-path" placeholder="Model path" value="${G.model}" onchange="Tabs.viewer.selectModel(this.value)">
              <button class="btn btn-secondary btn-sm" onclick="Tabs.viewer.browseModel()">${t('browse')}</button>
              <button class="btn btn-ghost btn-sm" onclick="Tabs.viewer.refreshModels()" title="Refresh">↻</button>
            </div>
            <div id="v-model-list" style="flex:1;overflow-y:auto;font-size:12px;" class="text-secondary">${t('viewer.loading')}</div>
          </div>
          <div class="card-flat" style="padding:0.75rem;flex:1;display:flex;flex-direction:column;">
            <div class="text-label" style="margin-bottom:0.5rem;">${t('viewer.video_image')}</div>
            <div style="display:flex;gap:0.25rem;margin-bottom:0.5rem;">
              <input type="text" class="form-input input-normal" style="flex:1;font-size:11px;height:28px;" id="v-video-path" placeholder="Video/Image path" value="${G.videoPath||''}" onchange="Tabs.viewer.selectVideo(this.value)">
              <button class="btn btn-secondary btn-sm" onclick="Tabs.viewer.browseVideo()">${t('browse')}</button>
              <button class="btn btn-secondary btn-sm" onclick="Tabs.viewer.browseImageFolder()" title="Open image folder">📁</button>
              <button class="btn btn-ghost btn-sm" onclick="Tabs.viewer.refreshVideos()" title="Refresh">↻</button>
            </div>
            <div id="v-img-nav-bar" style="display:none;align-items:center;gap:0.25rem;margin-bottom:0.5rem;">
              <button class="btn btn-ghost btn-sm" onclick="Tabs.viewer._navImage(-1)">◀</button>
              <span id="v-img-nav" style="flex:1;text-align:center;font-size:11px;" class="text-secondary"></span>
              <button class="btn btn-ghost btn-sm" onclick="Tabs.viewer._navImage(1)">▶</button>
            </div>
            <div id="v-video-list" style="flex:1;overflow-y:auto;font-size:12px;" class="text-secondary">${t('viewer.loading')}</div>
          </div>
          <!-- CLIP/VLM text input panel (hidden by default) -->
          <div id="v-text-panel" class="card-flat" style="padding:0.75rem;display:none;">
            <div class="text-label" style="margin-bottom:0.35rem;font-size:11px;">Text Input</div>
            <div id="v-clip-inputs" style="display:none;">
              <input type="text" class="form-input input-normal" style="font-size:11px;height:28px;margin-bottom:0.35rem;" id="v-clip-labels" placeholder="Labels (comma separated): dog, cat, bird">
              <div class="form-group" style="margin-bottom:0.35rem;">
                <label class="form-label" style="font-size:10px;">Text Encoder</label>
                <div style="display:flex;gap:0.25rem;">
                  <input type="text" class="form-input input-normal" style="flex:1;font-size:11px;height:28px;" id="v-clip-txt-enc" placeholder="Text encoder .onnx">
                  <button class="btn btn-secondary btn-sm" onclick="pickFile('v-clip-txt-enc','ONNX (*.onnx)')" style="font-size:10px;">📂</button>
                </div>
              </div>
            </div>
            <div id="v-vlm-inputs" style="display:none;">
              <textarea class="form-input input-normal" style="font-size:11px;height:60px;resize:vertical;" id="v-vlm-prompt" placeholder="Enter prompt / question..."></textarea>
            </div>
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
            <label style="font-size:11px;display:flex;align-items:center;gap:3px;cursor:pointer;" title="${t('viewer.save_crops_tip')}">
              <input type="checkbox" id="v-save-crops"> ${t('viewer.save_crops')}
            </label>
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
          <div class="card-flat" style="padding:0.6rem;">
            <div class="text-label" style="margin-bottom:0.35rem;font-size:11px;">Execution Provider</div>
            <div id="v-ep-info" class="text-secondary">—</div>
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
          const groups = {
            'Detection': [], 'Classification': [], 'Segmentation': [],
            'Pose': [], 'Instance Seg': [], 'Tracking': [],
            'CLIP': [], 'Embedder': [], 'VLM': [], 'Custom': []
          };
          for (const [key, label] of Object.entries(c.model_types)) {
            if (key.startsWith('cls_')) groups['Classification'].push([key, label]);
            else if (key.startsWith('seg_')) groups['Segmentation'].push([key, label]);
            else if (key.startsWith('clip_')) groups['CLIP'].push([key, label]);
            else if (key.startsWith('emb_')) groups['Embedder'].push([key, label]);
            else if (key.startsWith('pose_')) groups['Pose'].push([key, label]);
            else if (key.startsWith('instseg_')) groups['Instance Seg'].push([key, label]);
            else if (key.startsWith('track_')) groups['Tracking'].push([key, label]);
            else if (key.startsWith('vlm_')) groups['VLM'].push([key, label]);
            else if (key.startsWith('custom:')) groups['Custom'].push([key, label]);
            else groups['Detection'].push([key, label]);
          }
          for (const [gname, items] of Object.entries(groups)) {
            if (!items.length) continue;
            const og = document.createElement('optgroup');
            og.label = gname;
            for (const [k, l] of items) og.appendChild(new Option(l, k));
            mt.appendChild(og);
          }
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
    // EP status
    try {
      const ep = await API.epStatus();
      const el = document.getElementById('v-ep-info');
      if (el) {
        const sel = ep.selected || 'auto';
        const prov = ep.provider || 'CPUExecutionProvider';
        const avail = (ep.available_eps || []).join(', ') || '—';
        const bundled = (ep.bundled_eps || []).join(', ') || '—';
        el.innerHTML = `Active: <b>${sel.toUpperCase()}</b><br>Provider: ${prov}<br>Available: ${avail}<br>Bundled: ${bundled}`;
        if (ep.fallback) {
          el.innerHTML += `<br><span style="color:var(--color-warning,#e6a700);">Fallback</span>`;
        }
      }
      // Fallback notification
      if (ep.fallback && ep.skipped && ep.skipped.length > 0) {
        let detail = '';
        for (const s of ep.skipped) {
          detail += `[${s.ep.toUpperCase()}] ${s.reason}\n  -> ${s.fix}\n\n`;
        }
        const msg = `EP fallback to ${(ep.selected||'cpu').toUpperCase()}: ${ep.fallback_reason||''}`;
        if (typeof Notify !== 'undefined' && Notify.warn) {
          Notify.warn(msg, detail.trim());
        } else {
          App.setStatus(msg, detail.trim());
        }
      }
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
  destroy() {
    // 탭 전환 시 리소스 정리 — HW 폴링 중지, MJPEG 스트림 해제, 키보드 핸들러 제거
    if (this._hwInterval) { clearInterval(this._hwInterval); this._hwInterval = null; }
    if (this._keyHandler) { document.removeEventListener('keydown', this._keyHandler); this._keyHandler = null; }
    // MJPEG img src 제거 — 브라우저가 연결을 끊도록
    const canvas = document.getElementById('viewer-canvas');
    if (canvas) {
      const img = canvas.querySelector('img');
      if (img) img.src = '';
    }
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
    else if (e.key === 'ArrowLeft') { e.preventDefault(); this._imgList ? this._navImage(-1) : this.stepBack(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); this._imgList ? this._navImage(1) : this.stepFwd(); }
    else if (e.key === 's' || e.key === 'S') this.snapshot();
    else if (e.key === '+' || e.key === '=') this._changeSpeed(1);
    else if (e.key === '-') this._changeSpeed(-1);
  },
  _getConf() { return +(document.getElementById('v-conf-slider')?.value || 25) / 100; },
  async _onModelTypeChange() {
    const v = document.getElementById('v-model-type').value;
    await API.post('/api/config', { model_type: v });
    // Show/hide text input panels based on model type
    const textPanel = document.getElementById('v-text-panel');
    const clipInputs = document.getElementById('v-clip-inputs');
    const vlmInputs = document.getElementById('v-vlm-inputs');
    if (v.startsWith('clip_') || v.startsWith('emb_')) {
      textPanel.style.display = ''; clipInputs.style.display = ''; vlmInputs.style.display = 'none';
    } else if (v.startsWith('vlm_')) {
      textPanel.style.display = ''; clipInputs.style.display = 'none'; vlmInputs.style.display = '';
    } else {
      textPanel.style.display = 'none';
    }
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
      if (!r.files || !r.files.length) { el.textContent = t('viewer.no_models'); return; }
      el.innerHTML = r.files.map(f =>
        `<div class="nav-item" style="padding:0.25rem 0.5rem;cursor:pointer;font-size:12px;border-radius:4px;" onclick="Tabs.viewer.selectModel('${f.path.replace(/\\/g,'\\\\').replace(/'/g,"\\'")}')" title="${f.path}">${f.name}</div>`
      ).join('');
    } catch(e) { document.getElementById('v-model-list').textContent = t('viewer.models_not_found'); }
  },
  async refreshVideos() {
    try {
      const r = await API.listDir({ path: 'Videos', exts: ['.mp4','.avi','.mov','.mkv','.jpg','.jpeg','.png','.bmp'] });
      const el = document.getElementById('v-video-list');
      if (!r.files || !r.files.length) { el.textContent = t('viewer.no_files'); return; }
      el.innerHTML = r.files.map(f =>
        `<div class="nav-item" style="padding:0.25rem 0.5rem;cursor:pointer;font-size:12px;border-radius:4px;" onclick="Tabs.viewer.selectVideo('${f.path.replace(/\\/g,'\\\\').replace(/'/g,"\\'")}')" title="${f.path}">${f.name}</div>`
      ).join('');
    } catch(e) { document.getElementById('v-video-list').textContent = t('viewer.videos_not_found'); }
  },
  async selectModel(path) {
    setModel(path);
    const mp = document.getElementById('v-model-path');
    if (mp) mp.value = path;
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
    const vp = document.getElementById('v-video-path');
    if (vp) vp.value = path;
    // 단일 파일 직접 선택 시 폴더 네비게이션 해제
    if (!this._imgList || !this._imgList.includes(path)) {
      this._imgList = null; this._imgIdx = 0;
      const navBar = document.getElementById('v-img-nav-bar');
      if (navBar) navBar.style.display = 'none';
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
    _showFileBrowser('file', ['.onnx', '.pt'], (path) => this.selectModel(path));
  },
  async browseVideo() {
    _showFileBrowser('file', ['.mp4','.avi','.mov','.mkv','.jpg','.jpeg','.png','.bmp'], (path) => this.selectVideo(path));
  },
  async browseImageFolder() {
    _showFileBrowser('dir', null, async (dirPath) => {
      try {
        const r = await API.post('/api/fs/list', { path: dirPath, exts: ['.jpg','.jpeg','.png','.bmp'] });
        if (!r.files || !r.files.length) { App.setStatus('No images found in folder'); return; }
        this._imgList = r.files.map(f => f.path);
        this._imgIdx = 0;
        this.selectVideo(this._imgList[0]);
        const navBar = document.getElementById('v-img-nav-bar');
        if (navBar) navBar.style.display = 'flex';
        this._updateImgNav();
      } catch(e) { App.setStatus('Error: ' + e.message); }
    });
  },
  _imgList: null,
  _imgIdx: 0,
  _navImage(delta) {
    if (!this._imgList || !this._imgList.length) return;
    this._imgIdx = Math.max(0, Math.min(this._imgList.length - 1, this._imgIdx + delta));
    this.selectVideo(this._imgList[this._imgIdx]);
    this._updateImgNav();
  },
  _updateImgNav() {
    const el = document.getElementById('v-img-nav');
    if (el && this._imgList) el.textContent = `${this._imgIdx + 1} / ${this._imgList.length}`;
  },
  togglePlay() {
    if (!G.model) { App.setStatus(t('viewer.select_model_first')); return; }
    if (!G.videoPath) { App.setStatus(t('viewer.select_video_first')); return; }
    // If playing, pause
    if (this._streamSessionId && !this._paused) {
      this._togglePause(); return;
    }
    // If paused, resume
    if (this._streamSessionId && this._paused) {
      this._togglePause(); return;
    }
    App.setStatus(t('viewer.starting'));
    const ext = G.videoPath.split('.').pop().toLowerCase();
    if (['jpg','jpeg','png','bmp'].includes(ext)) this._inferImage();
    else this._startStream();
  },
  async _inferImage() {
    try {
      const conf = this._getConf();
      const body = { model_path: G.model, image_path: G.videoPath, conf };
      // CLIP/Embedder: text labels + text encoder
      const clipLabels = document.getElementById('v-clip-labels')?.value;
      if (clipLabels) body.clip_labels = clipLabels;
      const clipTxtEnc = document.getElementById('v-clip-txt-enc')?.value;
      if (clipTxtEnc) body.clip_text_encoder = clipTxtEnc;
      // VLM: prompt
      const vlmPrompt = document.getElementById('v-vlm-prompt')?.value;
      if (vlmPrompt) body.vlm_prompt = vlmPrompt;
      const r = await API.post('/api/infer/image', body);
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      document.getElementById('viewer-canvas').innerHTML = `<img src="data:image/jpeg;base64,${r.image}" style="max-width:100%;max-height:100%;">`;
      const resEl = document.getElementById('viewer-results');
      if (r.classification) resEl.innerHTML = `<b>${r.classification}</b>` + (r.top_k ? '<br>' + r.top_k.map(t=>`${t.class}: ${t.score}`).join('<br>') : '');
      else if (r.segmentation) resEl.textContent = r.segmentation;
      else if (r.embedding) resEl.textContent = r.embedding;
      else if (r.pose) resEl.textContent = r.pose;
      else if (r.instance_seg) resEl.textContent = r.instance_seg;
      else if (r.clip_result) resEl.innerHTML = r.clip_result.map(c=>`${c.label}: ${(c.score*100).toFixed(1)}%`).join('<br>');
      else if (r.vlm_result) resEl.textContent = r.vlm_result;
      else resEl.textContent = `${r.detections} detections`;
      document.getElementById('v-infer-stats').innerHTML = `Infer: ${r.infer_ms} ms`;
      const status = r.classification ? `Classification: ${r.classification}` : r.segmentation ? `Segmentation: ${r.segmentation}` : r.pose ? `Pose: ${r.pose}` : r.instance_seg ? `Instance Seg: ${r.instance_seg}` : r.clip_result ? `CLIP: ${r.clip_result[0]?.label}` : r.vlm_result ? `VLM: ${r.vlm_result.substring(0,50)}` : `Inference done: ${r.detections} detections, ${r.infer_ms}ms`;
      App.setStatus(status);
      if (document.getElementById('v-save-crops')?.checked && r.detections > 0 && !r.classification && !r.segmentation && !r.embedding && !r.pose && !r.instance_seg) {
        const cr = await API.post('/api/infer/save-crops', { model_path: G.model, image_path: G.videoPath, conf });
        if (cr.ok) App.setStatus(t('viewer.crops_saved', {count: cr.count, path: cr.path}));
      }
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
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
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
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
      else if (!s.playing) { App.setStatus(t('viewer.playback_done')); this._resetAll(); }
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
    App.setStatus(this._paused ? t('viewer.paused') : t('viewer.playing'));
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
    if (r.ok) App.setStatus(t('viewer.snapshot_saved', {path: r.path}));
    else App.setStatus(t('viewer.snapshot_failed') + ': ' + (r.error||''));
    if (document.getElementById('v-save-crops')?.checked) {
      const cr = await API.post('/api/viewer/save-crops/' + this._streamSessionId, {});
      if (cr.ok) App.setStatus(t('viewer.crops_saved', {count: cr.count, path: cr.path}));
    }
  },
};

/* ── Settings ───────────────────────────────────────── */
Tabs.settings = {
  title: true,
  render() {
    return `
      <div style="display:flex;gap:1.5rem;align-items:flex-start;">
        <div style="max-width:480px;flex:1;display:flex;flex-direction:column;gap:1.5rem;">
          <div class="card" style="padding:1.5rem;" id="settings-model-section">
            <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('settings.model_type_mgmt')}</h3>
            <button class="btn btn-secondary btn-sm" onclick="Tabs.settings.openCustomTypeDialog()">${t('settings.add_model_type')}</button>
          </div>
          <div class="card" style="padding:1.5rem;">
            <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('settings.test_model_dl')}</h3>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
              <a href="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx" target="_blank" rel="noopener" class="btn btn-secondary btn-sm">📥 Detection (YOLO11n)</a>
              <a href="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.onnx" target="_blank" rel="noopener" class="btn btn-secondary btn-sm">📥 Classification (YOLO11n-cls)</a>
              <a href="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx" target="_blank" rel="noopener" class="btn btn-secondary btn-sm">📥 Segmentation (YOLO11n-seg)</a>
              <a href="https://huggingface.co/Xenova/clip-vit-base-patch32/tree/main/onnx" target="_blank" class="btn btn-secondary btn-sm">📥 CLIP (ViT-B/32 ONNX)</a>
              <a href="https://huggingface.co/immich-app/ViT-B-32__openai/tree/main" target="_blank" class="btn btn-secondary btn-sm">📥 Embedder (ViT-B/32 ONNX)</a>
            </div>
            <h4 class="text-heading-h3" style="margin:1.25rem 0 0.75rem;font-size:14px;">${t('settings.test_data_dl')}</h4>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
              <a href="https://ultralytics.com/assets/coco128.zip" target="_blank" rel="noopener" class="btn btn-secondary btn-sm">📥 COCO128 Dataset (YOLO format)</a>
            </div>
            <h4 class="text-heading-h3" style="margin:1.25rem 0 0.75rem;font-size:14px;">${t('settings.builtin_samples')}</h4>
            <div class="text-secondary" style="font-size:12px;margin-bottom:0.5rem;">${t('settings.builtin_samples_desc')}</div>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
              <a href="/assets/samples/bus.jpg" download class="btn btn-secondary btn-sm">🖼️ bus.jpg</a>
              <a href="/assets/samples/zidane.jpg" download class="btn btn-secondary btn-sm">🖼️ zidane.jpg</a>
              <a href="/assets/samples/people.mp4" download class="btn btn-secondary btn-sm">🎬 people.mp4</a>
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
              <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;color:var(--text-04);"><input type="checkbox" checked id="show-label-bg"> ${t('settings.label_bg')}</label>
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
            <div id="class-table-container" class="text-secondary" style="font-size:12px;">${t('settings.load_model_hint')}</div>
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
    if (!names || !Object.keys(names).length) { container.textContent = t('settings.no_classes'); return; }
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
    container.innerHTML = `<div style="display:flex;gap:0.5rem;margin-bottom:0.5rem;">
      <button class="btn btn-secondary btn-sm" onclick="Tabs.settings._toggleAllClasses(true)">${t('settings.select_all')}</button>
      <button class="btn btn-secondary btn-sm" onclick="Tabs.settings._toggleAllClasses(false)">${t('settings.deselect_all')}</button>
    </div>
    <div style="max-height:400px;overflow-y:auto;">
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
  async _toggleAllClasses(enabled) {
    const checkboxes = document.querySelectorAll('.cls-enabled');
    for (const cb of checkboxes) {
      cb.checked = enabled;
      const clsId = +cb.dataset.cls;
      const co = document.querySelector(`.cls-color[data-cls="${clsId}"]`);
      const th = document.querySelector(`.cls-thick[data-cls="${clsId}"]`);
      const hex = co?.value || '#00ff00';
      const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
      try {
        await API.post('/api/config/class-style', {
          class_id: clsId, enabled,
          color: [b, g, r], thickness: +(th?.value||0),
        });
      } catch(e) {}
    }
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
        <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('cmt.title')}</h3>
        <div class="form-group">
          <label class="form-label">${t('cmt.type_name')}</label>
          <input type="text" class="form-input input-normal" id="cmt-name" placeholder="${t('cmt.type_name_ph')}">
        </div>
        <div class="form-group" style="margin-top:0.75rem;">
          <label class="form-label">${t('cmt.onnx_model')}</label>
          <div style="display:flex;gap:0.5rem;">
            <input type="text" class="form-input input-normal" style="flex:1;" id="cmt-model-path" placeholder="ONNX model path" onchange="Tabs.settings._cmtLoadModel(this.value)">
            <button class="btn btn-secondary btn-sm" onclick="Tabs.settings._cmtBrowseModel()">${t('cmt.select_model')}</button>
          </div>
        </div>
        <div id="cmt-shape-info" class="text-secondary" style="margin:0.75rem 0;font-size:12px;font-family:monospace;">${t('cmt.shape_hint')}</div>
        <div class="form-group">
          <label class="form-label">${t('cmt.output_tensor')}</label>
          <select class="form-input input-normal" id="cmt-oi" style="width:auto;" onchange="Tabs.settings._cmtOnOutputSelected()"></select>
        </div>
        <div id="cmt-dim-mapping" style="margin-top:0.75rem;"></div>
        <div style="display:flex;gap:1rem;align-items:center;margin-top:0.75rem;">
          <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;"><input type="checkbox" id="cmt-nms" checked> ${t('cmt.apply_nms')}</label>
          <label class="form-label" style="margin:0;">Confidence:</label>
          <input type="number" class="form-input input-normal" id="cmt-conf" value="0.25" min="0.01" max="1.0" step="0.05" style="width:80px;height:28px;">
        </div>
        <div class="form-group" style="margin-top:0.75rem;">
          <label class="form-label">${t('cmt.class_names')}</label>
          <input type="text" class="form-input input-normal" id="cmt-class-names" placeholder="${t('cmt.class_names_ph')}">
        </div>
        <div style="border-top:1px solid var(--border-default);margin-top:1rem;padding-top:1rem;">
          <div class="form-group">
            <label class="form-label">${t('cmt.test_image')}</label>
            <div style="display:flex;gap:0.5rem;">
              <input type="text" class="form-input input-normal" style="flex:1;" id="cmt-test-path" placeholder="Image path" onchange="Tabs.settings._cmtTestPath=this.value;document.getElementById('cmt-test-label').textContent=this.value.split(/[\\\\/]/).pop()">
              <button class="btn btn-secondary btn-sm" onclick="Tabs.settings._cmtBrowseTestImg()">${t('cmt.select_image')}</button>
              <button class="btn btn-primary btn-sm" onclick="Tabs.settings._cmtRunTest()">${t('cmt.run_infer')}</button>
            </div>
          </div>
          <div id="cmt-test-result" style="margin-top:0.5rem;min-height:180px;display:flex;align-items:center;justify-content:center;background:var(--background-neutral-02);border-radius:8px;">
            <span class="text-secondary">${t('cmt.result_hint')}</span>
          </div>
        </div>
        <div style="display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem;">
          <button class="btn btn-primary" onclick="Tabs.settings._cmtSave()">${t('save')}</button>
          <button class="btn btn-secondary" onclick="document.getElementById('cmt-overlay').remove()">${t('cancel')}</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    this._cmtModelPath = '';
    this._cmtTestImgPath = '';
    this._cmtInferredOutputs = [];
  },

  async _cmtLoadModel(path) {
    if (!path) return;
    this._cmtModelPath = path;
    document.getElementById('cmt-shape-info').textContent = t('cmt.inferring');
    try {
      const res = await API.post('/api/model/infer-shapes', { path });
      if (res.error) { document.getElementById('cmt-shape-info').textContent = 'Error: ' + res.error; return; }
      this._cmtInferredOutputs = res.outputs;
      let info = t('cmt.input') + ': ' + JSON.stringify(res.input_shape);
      res.outputs.forEach(o => { info += '\n' + t('cmt.output') + '[' + o.index + '] ' + o.name + ': ' + JSON.stringify(o.shape); });
      document.getElementById('cmt-shape-info').innerText = info;
      const sel = document.getElementById('cmt-oi');
      sel.innerHTML = res.outputs.map(o =>
        '<option value="' + o.index + '">' + t('cmt.output') + '[' + o.index + '] ' + o.name + ' — ' + JSON.stringify(o.shape) + '</option>'
      ).join('');
      this._cmtOnOutputSelected();
    } catch(e) { document.getElementById('cmt-shape-info').textContent = 'Error: ' + e.message; }
  },

  async _cmtBrowseModel() {
    _showFileBrowser('file', ['.onnx'], async (path) => {
      try {
      this._cmtModelPath = path;
      const pathEl = document.getElementById('cmt-model-path');
      if (pathEl) pathEl.value = path;
      document.getElementById('cmt-shape-info').textContent = t('cmt.inferring');
      // 실제 추론으로 output shape 획득 (#1c)
      const res = await API.post('/api/model/infer-shapes', { path });
      if (res.error) { document.getElementById('cmt-shape-info').textContent = 'Error: ' + res.error; return; }
      this._cmtInferredOutputs = res.outputs;
      let info = t('cmt.input') + ': ' + JSON.stringify(res.input_shape);
      res.outputs.forEach(o => { info += '\n' + t('cmt.output') + '[' + o.index + '] ' + o.name + ': ' + JSON.stringify(o.shape); });
      document.getElementById('cmt-shape-info').innerText = info;
      // 출력 텐서 선택 드롭다운
      const sel = document.getElementById('cmt-oi');
      sel.innerHTML = res.outputs.map(o =>
        '<option value="' + o.index + '">' + t('cmt.output') + '[' + o.index + '] ' + o.name + ' — ' + JSON.stringify(o.shape) + '</option>'
      ).join('');
      this._cmtOnOutputSelected();
      } catch(e) { document.getElementById('cmt-shape-info').textContent = 'Error: ' + e.message; }
    });
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

    let html = '<div class="text-label" style="margin-bottom:0.5rem;">' + t('cmt.dim_meaning') + '</div>';
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
      const ATTR_CHOICES = [t('cmt.none'),'x1','y1','x2','y2','x_center','y_center','width','height','objectness','confidence','class_id'];
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
        (role === 'batch' ? t('cmt.batch_dim') : t('cmt.det_count_dim')) + '</span>';
    }
  },

  async _cmtBrowseTestImg() {
    _showFileBrowser('file', ['.jpg','.jpeg','.png','.bmp'], (path) => {
      this._cmtTestImgPath = path;
      const pathEl = document.getElementById('cmt-test-path');
      if (pathEl) pathEl.value = path;
    });
  },

  _cmtCollectAttrRoles() {
    // attributes 역할인 차원에서 attr_roles 수집
    const roles = [...document.querySelectorAll('.cmt-dim-role')];
    const dimRoles = roles.map(s => s.value);
    const attrDimIdx = dimRoles.indexOf('attributes');
    if (attrDimIdx < 0) return { dimRoles, attrRoles: [], hasObjectness: false };
    const attrSels = [...document.querySelectorAll('.cmt-attr-sel[data-dim="' + attrDimIdx + '"]')];
    const attrRoles = attrSels.map(s => s.value).filter(v => v !== t('cmt.none'));
    const hasObjectness = attrRoles.includes('objectness');
    return { dimRoles, attrRoles: attrSels.map(s => s.value === t('cmt.none') ? '' : s.value), hasObjectness };
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
    if (!this._cmtModelPath) { App.setStatus(t('cmt.select_model_first')); return; }
    const name = document.getElementById('cmt-name').value.trim() || 'test';
    const { dimRoles, attrRoles, hasObjectness } = this._cmtCollectAttrRoles();
    const oi = +(document.getElementById('cmt-oi').value || 0);
    const nms = document.getElementById('cmt-nms').checked;
    const conf = parseFloat(document.getElementById('cmt-conf').value || '0.25');
    const classNames = this._cmtParseClassNames();
    try {
      const testImg = this._cmtTestImgPath || (document.getElementById('cmt-test-path')?.value || '');
      if (testImg) {
        await API.post('/api/config/custom-model-type', {
          name, model_path: this._cmtModelPath, output_index: oi,
          attr_roles: attrRoles, dim_roles: dimRoles,
          has_objectness: hasObjectness, nms, conf_threshold: conf,
          class_names: classNames,
        });
        await API.post('/api/config', { model_type: 'custom:' + name, conf_threshold: conf });
        const r = await API.post('/api/infer/image', {
          model_path: this._cmtModelPath, image_path: testImg, conf
        });
        if (r.error) {
          document.getElementById('cmt-test-result').innerHTML = '<span style="color:var(--action-danger-05);">Error: ' + r.error + '</span>';
        } else {
          document.getElementById('cmt-test-result').innerHTML =
            '<div style="text-align:center;"><img src="data:image/jpeg;base64,' + r.image + '" style="max-width:100%;max-height:250px;cursor:pointer;" ondblclick="Tabs.settings._cmtZoomImg(this.src)" title="' + t('cmt.dblclick_zoom') + '"><br><span class="text-secondary">' + t('cmt.detections', {n: r.detections}) + '</span></div>';
        }
      } else {
        const r = await API.post('/api/config/custom-model-type/test', {
          name, model_path: this._cmtModelPath, output_index: oi,
          attr_roles: attrRoles, dim_roles: dimRoles,
          has_objectness: hasObjectness, nms, conf_threshold: conf,
        });
        document.getElementById('cmt-test-result').innerHTML = r.error
          ? '<span style="color:var(--action-danger-05);">Error: ' + r.error + '</span>'
          : '<span class="text-secondary">' + t('cmt.dummy_test', {n: r.detections}) + '</span>';
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
    if (!name) { App.setStatus(t('cmt.enter_name')); return; }
    if (!this._cmtModelPath) { App.setStatus(t('cmt.select_model_first')); return; }
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
        App.setStatus(t('cmt.saved', {name}));
        const sel = document.getElementById('v-model-type');
        if (sel && ![...sel.options].some(o => o.value === 'custom:' + name)) {
          sel.add(new Option(name, 'custom:' + name));
        }
        document.getElementById('cmt-overlay').remove();
      } else {
        App.setStatus('Error: ' + (r.error || ''));
      }
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('bench.config')}</h3>
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
            <div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="bench-progress" style="width:0%;height:100%;"></div><span id="bench-progress-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
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
          <div class="table-container"><table><thead><tr><th>Model</th><th>Provider</th><th>FPS</th><th>Avg(ms)</th><th>Pre(ms)</th><th>Infer(ms)</th><th>Post(ms)</th><th title="${t('bench.p50_tip')}">P50(ms)</th><th title="${t('bench.p95_tip')}">P95(ms)</th><th title="${t('bench.p99_tip')}">P99(ms)</th><th>CPU%</th><th>RAM(MB)</th><th>GPU%</th></tr></thead>
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
    const _bpt0 = document.getElementById('bench-progress-text'); if (_bpt0) _bpt0.textContent = '0%';
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
      const _bpt = document.getElementById('bench-progress-text'); if (_bpt) _bpt.textContent = pct + '%';
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
        const _bpt2 = document.getElementById('bench-progress-text'); if (_bpt2) _bpt2.textContent = '100%';
        App.setStatus(t('bench.complete'));
      }
    } catch(e) { setTimeout(() => this._poll(), 1000); }
  },
  stop() {
    this._polling = false;
    App.setStatus(t('stopped'));
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
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('eval.setup')}</h3>
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">
            <h3 class="text-heading-h3">${t('bench.models')}</h3>
            <button class="btn btn-secondary btn-sm" onclick="Tabs.evaluation._addModel()">${t('add_model')}</button>
          </div>
          <div id="eval-model-slots" style="display:flex;flex-direction:column;gap:0.5rem;">
            <div class="text-secondary" style="padding:1rem;text-align:center;" id="eval-slots-hint">${t('bench.add_hint')}</div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.75rem;">
            <div class="form-group">
              <label class="form-label">${t('common.confidence')}</label>
              <input type="number" class="form-input input-normal" value="0.25" min="0.01" max="1.0" step="0.05" id="eval-conf" style="width:100px;">
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.75rem;">
            ${imgDirInput('eval-img')}
            ${lblDirInput('eval-lbl')}
          </div>
          <div class="form-group" style="margin-top:0.75rem;">
            <label class="form-label">${t('eval.gt_classmap')}</label>
            <textarea class="form-input" id="eval-classmap" rows="4" style="font-size:12px;font-family:monospace;" placeholder="0: person&#10;1: car&#10;2: bicycle"></textarea>
          </div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;">
            <button class="btn btn-primary" id="eval-run-btn" onclick="Tabs.evaluation.run()">${t('eval.run')}</button>
            <button class="btn btn-danger btn-sm" id="eval-stop-btn" disabled onclick="Tabs.evaluation.stop()">${t('stop')}</button>
            <div style="flex:1;"></div>
            <button class="btn btn-secondary btn-sm" onclick="Tabs.evaluation.exportCSV()">${t('eval.csv_export')}</button>
          </div>
          <div style="margin-top:0.5rem;">
            <div class="progress-track" style="height:20px;position:relative;"><div class="progress-fill" id="eval-prog" style="width:0%;height:100%;"></div><span id="eval-prog-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span></div>
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
    _showFileBrowser('file', ['.onnx'], (path) => {
      this._addSlot(path);
    });
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
    if (!models.length) { App.setStatus(t('eval.add_one_model')); return; }
    if (!imgDir) { App.setStatus(t('eval.select_images')); return; }

    // GT 클래스 스캔 + 모델 클래스 로드 → 매핑 다이얼로그
    App.setStatus(t('eval.loading_class'));
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
      document.getElementById('eval-status').textContent = t('eval.running');
      document.getElementById('eval-prog').style.width = '0%';
      const _ept0 = document.getElementById('eval-prog-text'); if (_ept0) _ept0.textContent = '0%';

      const r = await API.post('/api/evaluation/run-async', {
        models, img_dir: imgDir, label_dir: lblDir, conf,
        per_model_mappings: result.mappings,
        mapped_only: result.mapped_only,
      });
      if (r.error) { App.setStatus('Error: ' + r.error); document.getElementById('eval-run-btn').disabled = false; return; }
      this._polling = true;
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); document.getElementById('eval-run-btn').disabled = false; }
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
      let copyOpts = '<option value="">' + t('mapping.copy_from') + '</option>';
      modelInfos.forEach((mi, idx) => { copyOpts += `<option value="${idx}">${mi.name}</option>`; });

      dlg.innerHTML = `
        <h3 style="margin-bottom:0.75rem;">${t('mapping.title')}</h3>
        <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;">
          <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;"><input type="checkbox" id="mapping-mapped-only" ${this._savedMappedOnly?'checked':''}> ${t('mapping.mapped_only')}</label>
          <div style="margin-left:auto;display:flex;gap:0.5rem;">
            <select class="form-input input-normal" id="mapping-copy-from" style="height:28px;font-size:11px;width:auto;">${copyOpts}</select>
            <button class="btn btn-ghost btn-sm" id="mapping-copy-btn">${t('mapping.copy')}</button>
            <button class="btn btn-ghost btn-sm" id="mapping-clear-btn" style="color:var(--action-danger-05);">${t('mapping.clear')}</button>
          </div>
        </div>
        <div style="display:flex;gap:0.5rem;border-bottom:1px solid var(--border-default);margin-bottom:0.75rem;padding-bottom:0.5rem;">${tabsHtml}</div>
        <div id="mapping-canvas-area" style="position:relative;"></div>
        <div style="font-size:11px;color:var(--text-03);margin-top:0.5rem;">${t('mapping.hint')}</div>
        <div style="display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem;">
          <button class="btn btn-secondary" id="mapping-cancel">${t('cancel')}</button>
          <button class="btn btn-primary" id="mapping-ok">${t('confirm')}</button>
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
        html += `<div style="font-size:11px;font-weight:600;color:var(--text-02);margin-bottom:4px;">${t('mapping.model_cls')}</div>`;
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
        html += `<div style="font-size:11px;font-weight:600;color:var(--text-02);margin-bottom:4px;">${t('mapping.gt_cls')}</div>`;
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
        App.setStatus(t('mapping.copied', {name: srcName}));
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
      const _ept = document.getElementById('eval-prog-text'); if (_ept) _ept.textContent = pct + '%';
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
        const _ept2 = document.getElementById('eval-prog-text'); if (_ept2) _ept2.textContent = '100%';
        // 캐시 저장
        this._cachedHTML = document.getElementById('page-body').innerHTML;
        App.setStatus(t('eval.complete'));
      }
    } catch(e) { setTimeout(() => this._poll(), 1000); }
  },
  _renderResults(results) {
    const tb = document.getElementById('eval-results');
    if (!tb) return;
    tb.innerHTML = results.map((x, i) => {
      if (x.error) return '<tr><td>' + (x.name||'') + '</td><td colspan="6" style="color:var(--action-danger-05);">' + x.error + '</td></tr>';
      const detBtn = x.detail ? '<button class="btn btn-ghost btn-sm" onclick="Tabs.evaluation.showDetail('+i+')">' + t('detail') + '</button>' : '';
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
    if (!r || !r.detail) { App.setStatus(t('eval.no_detail')); return; }
    const c = document.getElementById('eval-detail-container');
    const keys = Object.keys(r.detail).filter(k => k !== '__overall__').sort((a,b) => +a - +b);
    let rows = keys.map(cid => {
      const v = r.detail[cid];
      return '<tr><td>'+cid+'</td><td>'+(v.ap*100).toFixed(4)+'%</td><td>'+(v.precision*100).toFixed(4)+'%</td><td>'+(v.recall*100).toFixed(4)+'%</td><td>'+(v.f1*100).toFixed(4)+'%</td><td>'+v.tp+'</td><td>'+v.fp+'</td><td>'+v.fn+'</td></tr>';
    }).join('');
    c.innerHTML = '<div class="card" style="padding:1.5rem;"><h3 class="text-heading-h3" style="margin-bottom:1rem;">' + t('eval.per_class', {name: r.name}) + '</h3><div class="table-container"><table><thead><tr><th>Class</th><th>AP@50</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr></thead><tbody>'+rows+'</tbody></table></div></div>';
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
            <div class="form-group"><label class="form-label">${t('common.model_type')}</label><select class="form-input input-normal" id="ana-type" style="width:auto;"></select></div>
            <div class="form-group">
              <label class="form-label">Image</label>
              <div style="display:flex;gap:0.5rem;">
                <input type="text" class="form-input input-normal" style="flex:1;"  id="ana-img">
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
    if (!model_path||!image_path) { App.setStatus(t('common.select_model_img')); return; }
    App.setStatus(t('viewer.starting'));
    try {
      const r = await API.post('/api/analysis/inference-analysis', {
        model_path, model_type: document.getElementById('ana-type').value, image_path, conf: 0.25
      });
      if (r.error) { App.setStatus('Error: '+r.error); return; }
      let html = '';
      if (r.original_image) html += `<div style="text-align:center;"><div class="text-label" style="margin-bottom:0.25rem;">Original</div><img src="data:image/jpeg;base64,${r.original_image}" style="max-width:100%;max-height:250px;"></div>`;
      if (r.letterbox_image) html += `<div style="text-align:center;"><div class="text-label" style="margin-bottom:0.25rem;">Letterbox</div><img src="data:image/jpeg;base64,${r.letterbox_image}" style="max-width:100%;max-height:250px;"></div>`;
      if (r.detection_image) html += `<div style="text-align:center;"><div class="text-label" style="margin-bottom:0.25rem;">Detections</div><img src="data:image/jpeg;base64,${r.detection_image}" style="max-width:100%;max-height:250px;"></div>`;
      document.getElementById('ana-panels').innerHTML = html || '<span class="text-muted">' + t('common.no_results') + '</span>';
      const tm = r.timing || {};
      document.getElementById('ana-stats').innerHTML = `Pre: ${tm.pre_ms||'—'}ms<br>Infer: ${tm.infer_ms||'—'}ms<br>Post: ${tm.post_ms||'—'}ms<br>Total: ${tm.total_ms||'—'}ms`;
      const dets = r.detections || [];
      document.getElementById('ana-dets').innerHTML = Array.isArray(dets)
        ? dets.map(d => `${d.class_name}: ${d.confidence}`).join('<br>') || 'No detections'
        : dets + ' detections';
      App.setStatus('Inference analysis complete');
    } catch(e) { App.setStatus('Error: '+e.message); }
  }
};

/* ── Explorer ───────────────────────────────────────── */
Tabs.explorer = {
  title: true,
  _data: null,  // cached loaded data
  _viewMode: 'list',  // list | chart_image | chart_box | size_dist | aspect_dist
  render() {
    return `
      <div style="display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <div style="display:grid;grid-template-columns:1fr 1fr auto auto;gap:1rem;align-items:end;">
            ${imgDirInput('exp-img')}
            ${lblDirInput('exp-lbl')}
            <button class="btn btn-primary" style="height:36px;" onclick="Tabs.explorer.load()">${t('explorer.load')}</button>
          </div>
          <div id="exp-pbar-wrap" style="display:none;margin-top:0.75rem;">
            <div class="progress-track" style="height:20px;position:relative;">
              <div class="progress-fill" id="exp-pbar" style="width:0%;height:100%;"></div>
              <span id="exp-pbar-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span>
            </div>
          </div>
        </div>
        <div style="display:flex;gap:1rem;">
          <div style="width:220px;">
            <div class="card-flat" style="padding:1rem;">
              <div class="text-label" style="margin-bottom:0.5rem;">${t('explorer.view_mode')}</div>
              <select class="form-input input-normal" id="exp-view-mode" onchange="Tabs.explorer._onViewChange()" style="margin-bottom:0.75rem;">
                <option value="list">${t('explorer.view_list')}</option>
                <option value="chart_box">${t('explorer.view_chart_box')}</option>
                <option value="chart_image">${t('explorer.view_chart_image')}</option>
                <option value="size_dist">${t('explorer.view_size_dist')}</option>
                <option value="aspect_dist">${t('explorer.view_aspect_dist')}</option>
                <option value="box_aspect_dist">${t('explorer.view_box_aspect_dist')}</option>
              </select>
              <div class="text-label" style="margin-bottom:0.5rem;">${t('explorer.filter')}</div>
              <div class="form-group" style="margin-bottom:0.5rem;">
                <label class="form-label">Class</label>
                <div id="exp-class-filter" style="max-height:150px;overflow-y:auto;border:1px solid var(--border-02);border-radius:4px;padding:0.25rem;"></div>
              </div>
              <div class="form-group">
                <label class="form-label">Boxes</label>
                <div style="display:flex;gap:0.25rem;">
                  <select class="form-input input-normal" id="exp-box-op" style="width:55px;min-width:55px;flex-shrink:0;">
                    <option value=">=">&gt;=</option>
                    <option value="=">=</option>
                    <option value="<=">&lt;=</option>
                  </select>
                  <input type="number" class="form-input input-normal" value="0" min="0" id="exp-box-val" style="width:60px;min-width:0;">
                </div>
              </div>
              <button class="btn btn-secondary btn-sm" style="width:100%;margin-top:0.5rem;" onclick="Tabs.explorer._applyFilter()">${t('explorer.filter')}</button>
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
  async load() {
    const img_dir = document.getElementById('exp-img')?.value || G.imgDir;
    const label_dir = document.getElementById('exp-lbl')?.value || G.lblDir;
    if (!img_dir) { App.setStatus(t('eval.select_images')); return; }
    App.setStatus(t('viewer.loading'));
    const pbarWrap = document.getElementById('exp-pbar-wrap');
    if (pbarWrap) pbarWrap.style.display = 'block';
    try {
      const r = await API.post('/api/data/explorer', { img_dir, label_dir });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._pollLoad();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _pollLoad() {
    try {
      const s = await API.get('/api/data/explorer/status');
      const pbar = document.getElementById('exp-pbar');
      const pbarText = document.getElementById('exp-pbar-text');
      if (s.running) {
        const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
        if (pbar) pbar.style.width = pct + '%';
        if (pbarText) pbarText.textContent = pct + '%';
        setTimeout(() => this._pollLoad(), 300);
        return;
      }
      if (pbar) pbar.style.width = '100%';
      if (pbarText) pbarText.textContent = '100%';
      if (s.files) {
        this._data = s;
        this._buildClassFilter(s.class_counts || {});
        this._renderView();
        App.setStatus(`Loaded ${s.total} images`);
      } else {
        App.setStatus(s.msg || 'Error');
      }
      setTimeout(() => {
        const w = document.getElementById('exp-pbar-wrap');
        if (w) w.style.display = 'none';
      }, 1000);
    } catch(e) {}
  },
  _buildClassFilter(class_counts) {
    const container = document.getElementById('exp-class-filter');
    if (!container) return;
    const classes = Object.keys(class_counts).sort((a, b) => Number(a) - Number(b));
    container.innerHTML = `<label style="display:flex;align-items:center;gap:4px;font-size:11px;padding:2px 0;cursor:pointer;">
        <input type="checkbox" checked onchange="Tabs.explorer._toggleAll(this.checked)"> <b>All</b>
      </label>` +
      classes.map(c => `<label style="display:flex;align-items:center;gap:4px;font-size:11px;padding:2px 0;cursor:pointer;">
        <input type="checkbox" checked class="exp-cls-cb" value="${c}"> ${c} (${class_counts[c]})
      </label>`).join('');
  },
  _toggleAll(checked) {
    document.querySelectorAll('.exp-cls-cb').forEach(cb => cb.checked = checked);
  },
  _getSelectedClasses() {
    const cbs = document.querySelectorAll('.exp-cls-cb');
    if (!cbs.length) return null;  // no filter
    const selected = [...cbs].filter(cb => cb.checked).map(cb => Number(cb.value));
    return selected;
  },
  _applyFilter() {
    this._renderView();
  },
  _onViewChange() {
    this._viewMode = document.getElementById('exp-view-mode')?.value || 'list';
    this._renderView();
  },
  _filterFiles() {
    if (!this._data) return [];
    const selectedClasses = this._getSelectedClasses();
    const op = document.getElementById('exp-box-op')?.value || '>=';
    const val = parseInt(document.getElementById('exp-box-val')?.value || '0', 10);
    return this._data.files.filter(f => {
      // class filter
      if (selectedClasses && selectedClasses.length > 0) {
        if (f.classes.length === 0 && !selectedClasses.includes(-1)) return false;
        if (f.classes.length > 0 && !f.classes.some(c => selectedClasses.includes(c))) return false;
      }
      // box filter
      if (op === '>=' && f.boxes < val) return false;
      if (op === '=' && f.boxes !== val) return false;
      if (op === '<=' && f.boxes > val) return false;
      return true;
    });
  },
  _renderView() {
    const gallery = document.getElementById('exp-gallery');
    if (!gallery || !this._data) return;
    const filtered = this._filterFiles();
    const mode = this._viewMode;
    // Update stats
    const stats = document.getElementById('exp-stats');
    if (stats) {
      stats.innerHTML = `Total: ${this._data.total}<br>Shown: ${filtered.length}<br>Classes: ${Object.keys(this._data.class_counts || {}).length}`;
    }
    if (mode === 'list') {
      gallery.style.display = 'grid';
      gallery.innerHTML = filtered.map((f, i) =>
        `<div class="card-flat" style="padding:0.5rem;font-size:11px;text-align:center;cursor:pointer;" ondblclick="Tabs.explorer._preview(${i})">
          <div style="font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${f.name}">${f.name}</div>
          <div class="text-secondary">${f.boxes} boxes</div>
        </div>`).join('') || '<div class="text-secondary" style="grid-column:1/-1;text-align:center;padding:2rem;">' + t('common.no_results') + '</div>';
      this._filteredFiles = filtered;
    } else if (mode === 'chart_box' || mode === 'chart_image') {
      gallery.style.display = 'block';
      const counts = mode === 'chart_box' ? (this._data.class_counts || {}) : (this._data.img_class_counts || {});
      this._renderBarChart(gallery, counts, mode === 'chart_box' ? 'Box Count per Class' : 'Image Count per Class');
    } else if (mode === 'size_dist') {
      gallery.style.display = 'block';
      this._renderSizeChart(gallery);
    } else if (mode === 'aspect_dist') {
      gallery.style.display = 'block';
      this._renderAspectChart(gallery, this._data.aspect_ratios || [], 'Image Aspect Ratio Distribution (W/H)', 'Total images');
    } else if (mode === 'box_aspect_dist') {
      gallery.style.display = 'block';
      this._renderAspectChart(gallery, this._data.box_aspect_ratios || [], 'Box Aspect Ratio Distribution (W/H)', 'Total boxes');
    }
  },
  _renderBarChart(container, counts, title) {
    const entries = Object.entries(counts).sort((a, b) => Number(a[0]) - Number(b[0]));
    if (!entries.length) { container.innerHTML = '<div class="text-secondary" style="text-align:center;padding:2rem;">No data</div>'; return; }
    const maxVal = Math.max(...entries.map(e => e[1]));
    const total = entries.reduce((s, e) => s + e[1], 0);
    const colors = ['#4a9eff','#4ade80','#f59e0b','#ef4444','#a78bfa','#f472b6','#22d3ee','#fb923c','#84cc16','#e879f9'];
    container.innerHTML = `<div class="card-flat" style="padding:1.25rem;">
      <div class="text-label" style="margin-bottom:0.5rem;">${title}</div>
      <div class="text-secondary" style="font-size:11px;margin-bottom:1rem;">Total: ${total} | Classes: ${entries.length}</div>
      ${entries.map(([cls, cnt], i) => {
        const pct = maxVal > 0 ? (cnt / maxVal * 100) : 0;
        const color = colors[i % colors.length];
        const ratio = total > 0 ? (cnt / total * 100).toFixed(1) : 0;
        return `<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:6px;">
          <span style="min-width:60px;font-size:12px;text-align:right;font-weight:500;">Class ${cls}</span>
          <div style="flex:1;background:var(--background-neutral-03);border-radius:4px;height:22px;overflow:hidden;position:relative;">
            <div style="background:${color};height:100%;width:${pct}%;min-width:2px;border-radius:4px;transition:width 0.3s;"></div>
            <span style="position:absolute;right:6px;top:0;line-height:22px;font-size:10px;color:#ccc;">${ratio}%</span>
          </div>
          <span style="min-width:45px;font-size:12px;font-weight:500;">${cnt}</span>
        </div>`;
      }).join('')}</div>`;
  },
  _renderSizeChart(container) {
    const sizes = this._data.box_sizes || [];
    if (!sizes.length) { container.innerHTML = '<div class="text-secondary" style="text-align:center;padding:2rem;">No box data</div>'; return; }
    let s = 0, m = 0, l = 0;
    sizes.forEach(b => {
      const area = b.w * b.h;
      if (area < 0.01) s++;
      else if (area < 0.09) m++;
      else l++;
    });
    const cats = [['Small (area<1%)', s, '#4a9eff'], ['Medium (1%~9%)', m, '#4ade80'], ['Large (area>9%)', l, '#f59e0b']];
    const maxVal = Math.max(s, m, l, 1);
    const total = s + m + l;
    container.innerHTML = `<div class="card-flat" style="padding:1.25rem;">
      <div class="text-label" style="margin-bottom:0.5rem;">Box Size Distribution (normalized area)</div>
      <div class="text-secondary" style="font-size:11px;margin-bottom:1rem;">Total boxes: ${total}</div>
      ${cats.map(([label, cnt, color]) => {
        const pct = cnt / maxVal * 100;
        const ratio = total > 0 ? (cnt / total * 100).toFixed(1) : 0;
        return `<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:6px;">
          <span style="min-width:130px;font-size:12px;text-align:right;font-weight:500;">${label}</span>
          <div style="flex:1;background:var(--background-neutral-03);border-radius:4px;height:22px;overflow:hidden;position:relative;">
            <div style="background:${color};height:100%;width:${pct}%;min-width:2px;border-radius:4px;"></div>
            <span style="position:absolute;right:6px;top:0;line-height:22px;font-size:10px;color:#ccc;">${ratio}%</span>
          </div>
          <span style="min-width:45px;font-size:12px;font-weight:500;">${cnt}</span>
        </div>`;
      }).join('')}</div>`;
  },
  _renderAspectChart(container, ratios, title, totalLabel) {
    if (!ratios.length) { container.innerHTML = '<div class="text-secondary" style="text-align:center;padding:2rem;">No data</div>'; return; }
    const buckets = {'Portrait (<0.5)': 0, 'Tall (0.5~0.8)': 0, 'Square (0.8~1.2)': 0, 'Wide (1.2~1.8)': 0, 'Ultra-wide (>1.8)': 0};
    const bcolors = ['#a78bfa','#4a9eff','#4ade80','#f59e0b','#ef4444'];
    ratios.forEach(r => {
      if (r < 0.5) buckets['Portrait (<0.5)']++;
      else if (r < 0.8) buckets['Tall (0.5~0.8)']++;
      else if (r < 1.2) buckets['Square (0.8~1.2)']++;
      else if (r < 1.8) buckets['Wide (1.2~1.8)']++;
      else buckets['Ultra-wide (>1.8)']++;
    });
    const maxVal = Math.max(...Object.values(buckets), 1);
    const total = ratios.length;
    container.innerHTML = `<div class="card-flat" style="padding:1.25rem;">
      <div class="text-label" style="margin-bottom:0.5rem;">${title}</div>
      <div class="text-secondary" style="font-size:11px;margin-bottom:1rem;">${totalLabel}: ${total}</div>
      ${Object.entries(buckets).map(([label, cnt], i) => {
        const pct = cnt / maxVal * 100;
        const ratio = total > 0 ? (cnt / total * 100).toFixed(1) : 0;
        return `<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:6px;">
          <span style="min-width:130px;font-size:12px;text-align:right;font-weight:500;">${label}</span>
          <div style="flex:1;background:var(--background-neutral-03);border-radius:4px;height:22px;overflow:hidden;position:relative;">
            <div style="background:${bcolors[i]};height:100%;width:${pct}%;min-width:2px;border-radius:4px;"></div>
            <span style="position:absolute;right:6px;top:0;line-height:22px;font-size:10px;color:#ccc;">${ratio}%</span>
          </div>
          <span style="min-width:45px;font-size:12px;font-weight:500;">${cnt}</span>
        </div>`;
      }).join('')}</div>`;
  },
  _filteredFiles: [],
  async _preview(idx) {
    const f = this._filteredFiles[idx];
    if (!f) return;
    const label_dir = document.getElementById('exp-lbl')?.value || G.lblDir;
    try {
      const r = await API.post('/api/data/explorer/preview', { img_path: f.path, label_dir });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      showDetailModal(f.name, `<div style="text-align:center;">
        <img src="data:image/jpeg;base64,${r.image}" style="max-width:100%;max-height:70vh;">
        <div class="text-secondary" style="margin-top:0.5rem;">${r.width}×${r.height} — ${r.box_count} boxes</div>
      </div>`);
    } catch(e) { App.setStatus('Error: ' + e.message); }
  }
};

/* ── Splitter ───────────────────────────────────────── */
Tabs.splitter = {
  title: true,
  render() {
    return `
      <div style="max-width:640px;display:flex;flex-direction:column;gap:1.5rem;">
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;display:flex;align-items:center;">${t('splitter.input')}</h3>
          ${imgDirInput('split-img')}
          ${lblDirInput('split-lbl')}
          ${outDirInput('split-out')}
        </div>
        <div class="card" style="padding:1.5rem;">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('splitter.strategy')}</h3>
          <select class="form-input input-normal" id="split-strategy" style="margin-bottom:1rem;">
            <option value="random">${t('splitter.strategy_random')}</option>
            <option value="stratified">${t('splitter.strategy_stratified')}</option>
          </select>
          <div id="split-strategy-desc" class="text-secondary" style="font-size:11px;margin-bottom:1rem;">${t('splitter.strategy_random_desc')}</div>
        </div>
        <div class="card" style="padding:1.5rem;" id="split-ratio-section">
          <h3 class="text-heading-h3" style="margin-bottom:1rem;">${t('splitter.ratio')}</h3>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
            <div class="form-group"><label class="form-label">${t('splitter.train')}</label><input type="number" class="form-input input-normal" value="0.7" min="0" max="1" step="0.05" id="split-train"></div>
            <div class="form-group"><label class="form-label">${t('splitter.val')}</label><input type="number" class="form-input input-normal" value="0.2" min="0" max="1" step="0.05" id="split-val"></div>
            <div class="form-group"><label class="form-label">${t('splitter.test')}</label><input type="number" class="form-input input-normal" value="0.1" min="0" max="1" step="0.05" id="split-test"></div>
          </div>
          <div class="text-secondary" style="font-size:11px;margin-top:0.5rem;">${t('splitter.ratio_hint')}</div>
        </div>
        <div id="split-pbar-wrap" style="display:none;">
          <div class="progress-track" style="height:20px;position:relative;">
            <div class="progress-fill" id="split-pbar" style="width:0%;height:100%;"></div>
            <span id="split-pbar-text" style="position:absolute;top:0;left:50%;transform:translateX(-50%);font-size:11px;line-height:20px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,0.8);">0%</span>
          </div>
        </div>
        <button class="btn btn-primary" onclick="Tabs.splitter.run()">${t('splitter.run')}</button>
        <div id="split-result" class="text-secondary"></div>
      </div>`;
  },
  async init() {
    // Strategy description update
    setTimeout(() => {
      const sel = document.getElementById('split-strategy');
      if (sel) sel.onchange = () => {
        const desc = document.getElementById('split-strategy-desc');
        if (desc) desc.textContent = t('splitter.strategy_' + sel.value + '_desc');
      };
    }, 100);
  },
  async run() {
    const img_dir = document.getElementById('split-img')?.value || G.imgDir;
    const label_dir = document.getElementById('split-lbl')?.value || G.lblDir;
    const output_dir = document.getElementById('split-out')?.value;
    if (!img_dir || !output_dir) { App.setStatus(t('common.select_dirs')); return; }
    const train = parseFloat(document.getElementById('split-train')?.value || '0.7');
    const val = parseFloat(document.getElementById('split-val')?.value || '0.2');
    const test = parseFloat(document.getElementById('split-test')?.value || '0.1');
    const strategy = document.getElementById('split-strategy')?.value || 'random';
    const pbarWrap = document.getElementById('split-pbar-wrap');
    if (pbarWrap) pbarWrap.style.display = 'block';
    App.setStatus('Splitting...');
    try {
      const r = await API.post('/api/data/splitter', { img_dir, label_dir, output_dir, train, val, test, strategy });
      if (r.error) { App.setStatus('Error: ' + r.error); return; }
      this._poll();
    } catch(e) { App.setStatus('Error: ' + e.message, e.stack); }
  },
  async _poll() {
    try {
      const s = await API.get('/api/data/splitter/status');
      const pbar = document.getElementById('split-pbar');
      const pbarText = document.getElementById('split-pbar-text');
      if (s.running) {
        const pct = s.total > 0 ? Math.round(s.progress / s.total * 100) : 0;
        if (pbar) pbar.style.width = pct + '%';
        if (pbarText) pbarText.textContent = pct + '%';
        setTimeout(() => this._poll(), 500);
        return;
      }
      if (pbar) pbar.style.width = '100%';
      if (pbarText) pbarText.textContent = '100%';
      if (s.results) {
        const res = document.getElementById('split-result');
        if (res) res.innerHTML = `✅ Train: ${s.results.train||0}, Val: ${s.results.val||0}, Test: ${s.results.test||0}`;
        App.setStatus(`Split complete — Train: ${s.results.train||0}, Val: ${s.results.val||0}, Test: ${s.results.test||0}`);
      } else {
        App.setStatus(s.msg || 'Complete');
      }
      setTimeout(() => {
        const w = document.getElementById('split-pbar-wrap');
        if (w) w.style.display = 'none';
      }, 2000);
    } catch(e) {}
  }
};
