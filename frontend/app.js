const state = {
  category: null,
  defect: null,
  inferenceResult: null,
  activeTab: 'original',
  currentImagePath: null,
  trainedCategories: new Set(),
  trainingPollers: {},
};

// ── Init ──────────────────────────────────────────────────────────────────

async function init() {
  try {
    const [categories, trainedModels] = await Promise.all([
      apiFetch('/api/categories'),
      apiFetch('/api/models'),
    ]);
    state.trainedCategories = new Set(trainedModels);
    renderCategories(categories);
  } catch (e) {
    document.getElementById('category-list').innerHTML =
      `<div class="empty-state"><p style="color:var(--anomaly)">API unreachable: ${e.message}</p></div>`;
  }
}

// ── API helpers ───────────────────────────────────────────────────────────

async function apiFetch(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

// ── Render categories ─────────────────────────────────────────────────────

function renderCategories(categories) {
  const el = document.getElementById('category-list');
  if (!categories.length) {
    el.innerHTML = '<div class="empty-state"><p>No categories found</p></div>';
    return;
  }
  el.innerHTML = categories.map(cat => {
    const trained = state.trainedCategories.has(cat);
    return `<div class="list-item" id="cat-${cat}" onclick="selectCategory('${cat}')">
      <span>${cat}</span>
      ${trained ? '<span class="badge" style="color:var(--good)">trained</span>' : '<span class="badge">untrained</span>'}
    </div>`;
  }).join('');
}

async function selectCategory(cat) {
  state.category = cat;
  state.defect = null;

  document.querySelectorAll('#category-list .list-item').forEach(el => el.classList.remove('active'));
  const el = document.getElementById(`cat-${cat}`);
  if (el) el.classList.add('active');

  updateTrainButton();
  clearImageArea('Select a defect type to browse images');

  const defectList = document.getElementById('defect-list');
  defectList.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';

  try {
    const defects = await apiFetch(`/api/defect-types?category=${encodeURIComponent(cat)}`);
    renderDefects(defects);
  } catch (e) {
    defectList.innerHTML = `<div class="empty-state"><p style="color:var(--anomaly)">${e.message}</p></div>`;
  }
}

// ── Render defect types ───────────────────────────────────────────────────

function renderDefects(defects) {
  const el = document.getElementById('defect-list');
  if (!defects.length) {
    el.innerHTML = '<div class="empty-state"><p>No defect types found</p></div>';
    return;
  }
  el.innerHTML = defects.map(d => {
    const isGood = d === 'good';
    return `<div class="list-item" id="def-${d}" onclick="selectDefect('${d}')">
      <div style="display:flex;align-items:center;gap:6px;">
        <span class="dot ${isGood ? 'good' : 'anomaly'}"></span>
        <span>${d}</span>
      </div>
    </div>`;
  }).join('');
}

async function selectDefect(defect) {
  state.defect = defect;

  document.querySelectorAll('#defect-list .list-item').forEach(el => el.classList.remove('active'));
  const el = document.getElementById(`def-${defect}`);
  if (el) el.classList.add('active');

  updateBreadcrumb();
  const imageArea = document.getElementById('image-area');
  imageArea.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';

  try {
    const images = await apiFetch(
      `/api/images?category=${encodeURIComponent(state.category)}&defect=${encodeURIComponent(defect)}`
    );
    renderImageGrid(images);
  } catch (e) {
    imageArea.innerHTML = `<div class="empty-state"><p style="color:var(--anomaly)">${e.message}</p></div>`;
  }
}

// ── Render image grid ─────────────────────────────────────────────────────

function renderImageGrid(images) {
  const area = document.getElementById('image-area');
  if (!images.length) {
    area.innerHTML = '<div class="empty-state"><p>No images found</p></div>';
    return;
  }
  area.innerHTML = `<div class="image-grid">${images.map(img =>
    `<div class="image-card" onclick="openImage('${escapeAttr(img.path)}', '${escapeAttr(img.filename)}')">
      <img src="/api/image?path=${encodeURIComponent(img.path)}" alt="${escapeAttr(img.filename)}" loading="lazy" />
      ${img.is_anomaly ? '<div class="anomaly-dot"></div>' : ''}
      <div class="card-label">${img.filename}</div>
    </div>`
  ).join('')}</div>`;
}

// ── Modal / Inference ─────────────────────────────────────────────────────

async function openImage(imagePath, filename) {
  if (!state.category) return;

  state.currentImagePath = imagePath;
  state.inferenceResult = null;
  state.activeTab = 'original';

  document.getElementById('modal-title').textContent =
    `${state.category} / ${state.defect} / ${filename}`;
  document.getElementById('modal-meta').innerHTML = '';
  setModalBody('<div class="modal-loading"><div class="spinner"></div><span>Running inference…</span></div>');

  ['original', 'heatmap', 'overlay'].forEach(t => {
    document.getElementById(`tab-${t}`).classList.toggle('active', t === 'original');
  });

  document.getElementById('modal-backdrop').classList.add('open');

  if (!state.trainedCategories.has(state.category)) {
    setModalBody(`<div class="modal-loading">
      <span style="color:var(--text-muted)">Model not trained for <b>${state.category}</b>.</span>
      <button class="btn" style="margin-top:12px;width:auto;padding:8px 20px;" onclick="trainSelected()">Train Now</button>
    </div>`);
    document.getElementById('modal-meta').innerHTML = '';
    return;
  }

  try {
    const result = await apiFetch('/api/inference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category: state.category, image_path: imagePath }),
    });
    state.inferenceResult = result;
    renderModalResult(result, imagePath);
  } catch (e) {
    setModalBody(`<div class="modal-loading"><span style="color:var(--anomaly)">Error: ${e.message}</span></div>`);
  }
}

function renderModalResult(result, imagePath) {
  const meta = document.getElementById('modal-meta');
  const badge = result.is_anomaly
    ? `<span class="score-badge anomaly">Anomaly</span>`
    : `<span class="score-badge good">Normal</span>`;
  meta.innerHTML = `${badge} <span>Score: <b>${result.score.toFixed(3)}</b></span> <span style="color:var(--text-muted);font-size:11px;">(p95-normalized, threshold ${result.threshold})</span> <span style="color:var(--text-muted);">${result.inference_time_ms.toFixed(0)} ms</span>`;
  showTab('original', imagePath);
}

function switchTab(tab) {
  state.activeTab = tab;
  ['original', 'heatmap', 'overlay'].forEach(t =>
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab)
  );
  if (!state.inferenceResult) return;
  showTab(tab, state.currentImagePath);
}

function showTab(tab, imagePath) {
  const result = state.inferenceResult;
  if (tab === 'original') {
    setModalBody(`<img src="/api/image?path=${encodeURIComponent(imagePath)}" alt="original" />`);
  } else if (tab === 'heatmap' && result) {
    setModalBody(`<img src="data:image/png;base64,${result.heatmap_b64}" alt="heatmap" />`);
  } else if (tab === 'overlay' && result) {
    setModalBody(`<img src="data:image/png;base64,${result.overlay_b64}" alt="overlay" />`);
  }
}

function setModalBody(html) {
  document.getElementById('modal-body').innerHTML = html;
}

function closeModal(event) {
  if (event && event.target !== document.getElementById('modal-backdrop')) return;
  document.getElementById('modal-backdrop').classList.remove('open');
  state.inferenceResult = null;
  state.currentImagePath = null;
}

// ── Training ──────────────────────────────────────────────────────────────

function updateTrainButton() {
  const btn = document.getElementById('train-btn');
  const status = document.getElementById('train-status');
  if (!state.category) {
    btn.disabled = true;
    status.textContent = '';
    return;
  }
  const isTraining = state.trainingPollers[state.category] !== undefined;
  const isTrained = state.trainedCategories.has(state.category);

  btn.disabled = isTraining;
  btn.textContent = isTraining ? 'Training…' : isTrained ? 'Retrain' : 'Train Selected';
  status.textContent = isTrained && !isTraining ? '✓ Model ready' : '';
}

async function trainSelected() {
  if (!state.category) return;
  const cat = state.category;

  document.getElementById('train-btn').disabled = true;
  document.getElementById('train-status').textContent = 'Starting…';

  try {
    await apiFetch(`/api/train?categories=${encodeURIComponent(cat)}`);
    startTrainingPoller(cat);
  } catch (e) {
    showToast(`Training error: ${e.message}`, true);
    updateTrainButton();
  }
}

function startTrainingPoller(cat) {
  if (state.trainingPollers[cat]) return;

  document.getElementById('train-status').textContent = 'Training in progress…';

  const poll = setInterval(async () => {
    try {
      const res = await apiFetch(`/api/train/status?category=${encodeURIComponent(cat)}`);
      if (res.status === 'done') {
        clearInterval(poll);
        delete state.trainingPollers[cat];
        state.trainedCategories.add(cat);
        updateTrainButton();
        refreshCategoryBadge(cat, true);
        showToast(`✓ Training complete for "${cat}"`);
      } else if (res.status && res.status.startsWith('error:')) {
        clearInterval(poll);
        delete state.trainingPollers[cat];
        showToast(`Training failed for "${cat}": ${res.status.slice(6)}`, true);
        updateTrainButton();
      }
    } catch (_) {}
  }, 3000);

  state.trainingPollers[cat] = poll;
}

function refreshCategoryBadge(cat, trained) {
  const el = document.getElementById(`cat-${cat}`);
  if (!el) return;
  const badge = el.querySelector('.badge');
  if (badge) {
    badge.style.color = trained ? 'var(--good)' : '';
    badge.textContent = trained ? 'trained' : 'untrained';
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function updateBreadcrumb() {
  const el = document.getElementById('breadcrumb');
  el.innerHTML = state.category && state.defect
    ? `<span class="crumb">${state.category}</span> / <span class="crumb">${state.defect}</span>`
    : '<span>Select a defect type</span>';
}

function clearImageArea(msg = '') {
  document.getElementById('image-area').innerHTML =
    `<div class="empty-state"><p>${msg}</p></div>`;
  updateBreadcrumb();
}

function escapeAttr(str) {
  return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

let _toastTimer;
function showToast(msg, isError = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.toggle('error', isError);
  el.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), 3500);
}

// Keyboard: Escape closes modal
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal({target: document.getElementById('modal-backdrop')});
});

init();
