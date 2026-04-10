/* API client for ssook backend */
const API = {
  base: '',

  async get(path) {
    const r = await fetch(this.base + path);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  },

  async post(path, body) {
    const r = await fetch(this.base + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  },

  async postForm(path, formData) {
    const r = await fetch(this.base + path, { method: 'POST', body: formData });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
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
};
