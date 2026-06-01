/* API client for ssook backend */
const API = {
  base: '',

  // Last 422/Pydantic detail payload — surfaced for Form.showBackendErrors.
  lastValidationError: null,

  async _handle(r) {
    if (r.ok) return r.json();
    // 422 = Pydantic validation; envelope from server.errors includes
    // a `detail` array. Keep the structured info so the caller can map
    // it to DOM fields via Form.showBackendErrors().
    if (r.status === 422) {
      try {
        const body = await r.json();
        API.lastValidationError = body;
        if (typeof Form !== 'undefined' && body.detail) {
          Form.showBackendErrors(body.detail);
        }
        const msg = (body.detail && body.detail[0] && body.detail[0].msg) || 'Validation error';
        const e = new Error(`422: ${msg}`);
        e.detail = body.detail;
        e.traceId = body.trace_id;
        throw e;
      } catch (parseErr) {
        if (parseErr.detail) throw parseErr;
      }
    }
    // 400 with our envelope shape — same path safety / SsookError surface.
    if (r.status === 400 || r.status === 413 || r.status === 500) {
      try {
        const body = await r.json();
        const e = new Error(body.error || `${r.status} ${r.statusText}`);
        e.code = body.code;
        e.traceId = body.trace_id;
        e.detail = body.detail;
        throw e;
      } catch (parseErr) {
        if (parseErr.code) throw parseErr;
      }
    }
    throw new Error(`${r.status} ${r.statusText}`);
  },

  async get(path) {
    const r = await fetch(this.base + path);
    return this._handle(r);
  },

  async post(path, body) {
    API.lastValidationError = null;
    const r = await fetch(this.base + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return this._handle(r);
  },

  async postForm(path, formData) {
    const r = await fetch(this.base + path, { method: 'POST', body: formData });
    return this._handle(r);
  },

  /* Convenience wrappers */
  config:     ()       => API.get('/api/config'),
  saveConfig: (cfg)    => API.post('/api/config', cfg),
  loadModel:  (path)   => API.post('/api/model/load', { path }),
  modelInfo:  ()       => API.get('/api/model/info'),
  benchmark:  (params) => API.post('/api/benchmark/run', params),
  benchmarkStatus: ()  => API.get('/api/benchmark/status'),
  sysInfo:    ()       => API.get('/api/system/info'),
  epStatus:   ()       => API.get('/api/system/ep'),
  hwStats:    ()       => API.get('/api/system/hw'),
  videoInfo:  (path)   => API.post('/api/video/info', { path }),
  listDir:    (params) => API.post('/api/fs/list', params),
  selectFile: (opts)   => API.post('/api/fs/select', opts || {}),
  selectDir:  ()       => API.post('/api/fs/select-dir'),
  evaluate:   (params) => API.post('/api/evaluation/run', params),

  // VLM + classifier + pairing convenience wrappers.
  classifyModel:   (path) => API.post('/api/model/classify', { path }),
  findPartner:     (path) => API.post('/api/model/find-partner', { path }),
  vlmBatch:        (params) => API.post('/api/vlm/batch', params),
  vlmStatus:       () => API.get('/api/vlm/status'),
  vlmBackends:     () => API.get('/api/vlm/backends'),

  // Cooperative cancel for long-running tasks. Data/analysis Stop buttons call this
  // with the task id (e.g. 'compare', 'merger') matching the backend TaskState key.
  forceStop:       (task) => API.post('/api/force-stop/' + encodeURIComponent(task), {}),

  // Class catalog
  classCatalogs:   () => API.get('/api/classes/catalog'),
  classCatalog:    (name) => API.get('/api/classes/catalog/' + encodeURIComponent(name)),
  suggestCatalog:  (n) => API.post('/api/classes/suggest', { num_classes: n }),

  // Tasks + logs (Wave 5)
  tasks:           () => API.get('/api/tasks'),
  logsTail:        (lines) => API.get('/api/logs/tail?lines=' + (lines || 200)),
};
