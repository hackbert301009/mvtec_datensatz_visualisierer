'use strict';

// ── State ─────────────────────────────────────────────────────────────────

const state = {
  dataset: 'mvtec_ad',
  category: null,
  defect: null,
  inferenceResult: null,
  activeTab: 'original',
  currentImagePath: null,
  trainedCategories: new Set(),
  trainingPollers: {},
  datasets: [],
  defectTypes: [],   // [{id, label, is_anomaly}]
};

// ── Category icons ────────────────────────────────────────────────────────

const ICONS = {
  // AD1
  bottle: '🍶', cable: '🔌', capsule: '💊', carpet: '🟫',
  grid: '▦', hazelnut: '🌰', leather: '🟤', metal_nut: '🔩',
  pill: '🩺', screw: '🔧', tile: '🔷', toothbrush: '🪥',
  transistor: '⚡', wood: '🪵', zipper: '🪡',
  // AD2
  can: '🥫', fabric: '🧵', fruit_jelly: '🍮', rice: '🍚',
  sheet_metal: '⚙️', vial: '🧪', wallplugs: '🔌', walnuts: '🌰',
  // AD3
  bagel: '🥯', cable_gland: '🔩', carrot: '🥕', cookie: '🍪',
  dowel: '🪛', foam: '🟦', peach: '🍑', potato: '🥔',
  rope: '🪢', tire: '🔘',
};

const DS_META = {
  mvtec_ad:  { icon: '🔬', badge: null,   label: 'MVTec AD'    },
  mvtec_ad2: { icon: '⚗️',  badge: null,   label: 'MVTec AD 2'  },
  mvtec_ad3: { icon: '📐', badge: '3D',   label: 'MVTec 3D AD' },
};

function icon(name) {
  return ICONS[name] || '📦';
}

// ── Init ──────────────────────────────────────────────────────────────────

async function init() {
  try {
    const datasets = await apiFetch('/api/datasets');
    state.datasets = datasets;
    renderDatasetTabs(datasets);

    // Pick first available dataset
    const first = datasets.find(d => d.available) || datasets[0];
    if (first) await switchDataset(first.id, false);
  } catch (e) {
    document.getElementById('category-list').innerHTML =
      `<div class="empty-state"><p style="color:var(--anomaly)">API unreachable: ${e.message}</p></div>`;
  }
}

// ── Dataset tabs ──────────────────────────────────────────────────────────

function renderDatasetTabs(datasets) {
  const el = document.getElementById('ds-tabs');
  el.innerHTML = datasets.map(ds => {
    const meta = DS_META[ds.id] || { icon: '📁', badge: null };
    const badge3d = meta.badge ? `<span class="ds-tab-3d">${meta.badge}</span>` : '';
    return `
    <button
      class="ds-tab ${ds.id === state.dataset ? 'active' : ''} ${!ds.available ? 'unavailable' : ''}"
      id="dstab-${ds.id}"
      onclick="switchDataset('${ds.id}')"
      ${!ds.available ? 'title="Dataset wird noch extrahiert…"' : ''}
    >
      <span class="ds-tab-icon">${meta.icon}</span>
      <span class="ds-tab-name">${ds.name}${badge3d}</span>
      <span class="ds-tab-count">${ds.category_count} cats</span>
    </button>`;
  }).join('');
}

async function switchDataset(dsId, animate = true) {
  if (dsId === state.dataset && state.category !== null) return;

  state.dataset = dsId;
  state.category = null;
  state.defect = null;
  state.trainedCategories = new Set();

  // Update active tab
  document.querySelectorAll('.ds-tab').forEach(el => {
    el.classList.toggle('active', el.id === `dstab-${dsId}`);
  });

  clearDefectList();
  clearImageArea();
  updateBreadcrumb();

  const catList = document.getElementById('category-list');
  catList.innerHTML = `<div class="skeleton-wrap">
    ${Array(7).fill('<div class="skel-item"></div>').join('')}
  </div>`;

  try {
    const [categories, trainedModels] = await Promise.all([
      apiFetch(`/api/categories?dataset=${dsId}`),
      apiFetch(`/api/models?dataset=${dsId}`),
    ]);
    state.trainedCategories = new Set(trainedModels);
    renderCategories(categories);
    updateHeaderStats(categories.length, trainedModels.length);
  } catch (e) {
    catList.innerHTML = `<div class="empty-state"><p style="color:var(--anomaly)">${e.message}</p></div>`;
  }
}

// ── Render categories ─────────────────────────────────────────────────────

function renderCategories(categories) {
  const el = document.getElementById('category-list');
  document.getElementById('cat-count').textContent = categories.length;

  if (!categories.length) {
    el.innerHTML = '<div class="empty-state"><p>No categories found</p></div>';
    return;
  }

  el.innerHTML = categories.map(cat => {
    const trained = state.trainedCategories.has(cat);
    const training = state.trainingPollers[cat] !== undefined;
    const dotClass = training ? 'training' : trained ? 'yes' : '';
    return `<div class="cat-item" id="cat-${cat}" onclick="selectCategory('${cat}')">
      <span class="cat-icon">${icon(cat)}</span>
      <span class="cat-name">${cat}</span>
      <span class="trained-dot ${dotClass}" title="${training ? 'Training…' : trained ? 'Model ready' : 'Not trained'}"></span>
    </div>`;
  }).join('');
}

async function selectCategory(cat) {
  if (state.category === cat) return;
  state.category = cat;
  state.defect = null;

  document.querySelectorAll('.cat-item').forEach(el => el.classList.remove('active'));
  document.getElementById(`cat-${cat}`)?.classList.add('active');

  updateTrainButton();
  clearImageArea();
  updateBreadcrumb();

  const defectList = document.getElementById('defect-list');
  defectList.innerHTML = `<div class="skeleton-wrap">${Array(4).fill('<div class="skel-item"></div>').join('')}</div>`;
  document.getElementById('defect-count').textContent = '';

  try {
    const defects = await apiFetch(`/api/defect-types?category=${enc(cat)}&dataset=${state.dataset}`);
    state.defectTypes = defects;
    renderDefects(defects);
  } catch (e) {
    defectList.innerHTML = `<div class="empty-state"><p style="color:var(--anomaly)">${e.message}</p></div>`;
  }
}

// ── Render defect types ───────────────────────────────────────────────────

function renderDefects(defects) {
  const el = document.getElementById('defect-list');
  document.getElementById('defect-count').textContent = defects.length;

  if (!defects.length) {
    el.innerHTML = '<div class="empty-state"><p>No defect types found</p></div>';
    return;
  }

  el.innerHTML = defects.map(d => {
    const dotClass = d.is_anomaly === true ? 'anomaly' : d.is_anomaly === false ? 'good' : 'unknown';
    return `<div class="defect-item" id="def-${d.id}" onclick="selectDefect('${d.id}')">
      <span class="defect-dot ${dotClass}"></span>
      <span class="defect-name">${d.label}</span>
    </div>`;
  }).join('');
}

async function selectDefect(defectId) {
  state.defect = defectId;

  document.querySelectorAll('.defect-item').forEach(el => el.classList.remove('active'));
  document.getElementById(`def-${defectId}`)?.classList.add('active');

  updateBreadcrumb();
  document.getElementById('image-area').innerHTML =
    `<div class="empty-state"><div class="scan-loader"><div class="scan-bar"></div></div></div>`;

  try {
    const images = await apiFetch(
      `/api/images?category=${enc(state.category)}&defect=${enc(defectId)}&dataset=${state.dataset}`
    );
    renderImageGrid(images);
  } catch (e) {
    document.getElementById('image-area').innerHTML =
      `<div class="empty-state"><p style="color:var(--anomaly)">${e.message}</p></div>`;
  }
}

// ── Image grid ────────────────────────────────────────────────────────────

function renderImageGrid(images) {
  const area = document.getElementById('image-area');
  if (!images.length) {
    area.innerHTML = '<div class="empty-state"><p>No images found</p></div>';
    return;
  }
  const is3d = state.dataset === 'mvtec_ad3';
  area.innerHTML = `<div class="image-grid">${images.map(img => `
    <div class="image-card" onclick="openImage('${escAttr(img.path)}', '${escAttr(img.filename)}')">
      <img src="/api/image?path=${encodeURIComponent(img.path)}" alt="${escAttr(img.filename)}" loading="lazy"/>
      ${img.is_anomaly ? '<div class="card-anomaly-badge">⚠</div>' : ''}
      ${is3d ? '<div class="card-3d-badge">3D</div>' : ''}
      <div class="card-overlay">
        <span class="card-filename">${img.filename}</span>
      </div>
    </div>`
  ).join('')}</div>`;
}

// ── Modal / Inference ─────────────────────────────────────────────────────

async function openImage(imagePath, filename) {
  state.currentImagePath = imagePath;
  state.inferenceResult = null;
  state.activeTab = 'original';

  // Set title
  const defect = state.defectTypes.find(d => d.id === state.defect);
  const defectLabel = defect ? defect.label : state.defect;
  document.getElementById('modal-title').textContent =
    `${state.category}  ·  ${defectLabel}`;
  document.getElementById('modal-path').textContent = filename;
  document.getElementById('modal-meta').innerHTML = '';
  document.getElementById('modal-score-row').style.display = 'none';

  // Reset tabs
  ['original','heatmap','overlay'].forEach(t =>
    document.getElementById(`tab-${t}`).classList.toggle('active', t === 'original')
  );

  setModalBody(`<div class="scan-loader"><div class="scan-bar"></div><span>Running inference…</span></div>`);
  document.getElementById('modal-backdrop').classList.add('open');

  if (!state.trainedCategories.has(state.category)) {
    setModalBody(`<div class="scan-loader">
      <span style="color:var(--text-muted);text-align:center">
        Model not trained for <b>${state.category}</b>.<br>
        <button class="btn-train" style="margin-top:12px;width:auto;padding:8px 18px" onclick="trainSelected()">
          Train now
        </button>
      </span>
    </div>`);
    return;
  }

  try {
    const result = await apiFetch('/api/inference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: state.dataset, category: state.category, image_path: imagePath }),
    });
    state.inferenceResult = result;
    renderModalResult(result, imagePath);
  } catch (e) {
    setModalBody(`<div class="scan-loader"><span style="color:var(--anomaly)">Error: ${e.message}</span></div>`);
  }
}

function renderModalResult(result, imagePath) {
  const isAnom = result.is_anomaly;
  const badge = isAnom
    ? `<span class="score-badge anomaly">⚠ Anomaly</span>`
    : `<span class="score-badge good">✓ Normal</span>`;

  document.getElementById('modal-meta').innerHTML =
    `${badge}
     <span class="meta-score">Score: <b>${result.score.toFixed(3)}</b></span>
     <span class="meta-time">${result.inference_time_ms.toFixed(0)} ms</span>`;

  showTab('original', imagePath);

  // Animate score bar
  const maxScore = 5.0;
  const pct = Math.min(result.score / maxScore * 100, 100).toFixed(1);
  const thresholdPct = (result.threshold / maxScore * 100).toFixed(1);
  const color = isAnom ? 'var(--anomaly)' : 'var(--good)';

  const scoreRow = document.getElementById('modal-score-row');
  scoreRow.style.display = '';
  document.getElementById('score-thresh-lbl').textContent = result.threshold;
  document.getElementById('score-needle').style.left = `${thresholdPct}%`;

  const fill = document.getElementById('score-fill');
  fill.style.background = color;
  fill.style.width = '0%';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { fill.style.width = `${pct}%`; });
  });

  document.getElementById('score-val').innerHTML =
    `<span style="color:${color}">${result.score.toFixed(4)}</span>
     <span style="color:var(--text-muted);font-weight:400"> / ${maxScore} (threshold: ${result.threshold})</span>`;
}

function switchTab(tab) {
  state.activeTab = tab;
  ['original','heatmap','overlay'].forEach(t =>
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab)
  );
  if (!state.inferenceResult) return;
  showTab(tab, state.currentImagePath);
}

function showTab(tab, imagePath) {
  const r = state.inferenceResult;
  if (tab === 'original') {
    setModalBody(`<img src="/api/image?path=${encodeURIComponent(imagePath)}" alt="original"/>`);
  } else if (tab === 'heatmap' && r) {
    setModalBody(`<img src="data:image/png;base64,${r.heatmap_b64}" alt="heatmap"/>`);
  } else if (tab === 'overlay' && r) {
    setModalBody(`<img src="data:image/png;base64,${r.overlay_b64}" alt="overlay"/>`);
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
  document.getElementById('modal-score-row').style.display = 'none';
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
  btn.innerHTML = isTraining
    ? `<span class="skel-item" style="height:14px;width:80px;display:inline-block;border-radius:4px"></span>`
    : `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>${isTrained ? 'Retrain' : 'Train Selected'}`;
  status.innerHTML = isTrained && !isTraining
    ? `<span style="color:var(--good)">✓ Model ready</span>`
    : isTraining ? 'Training in progress…' : '';
}

async function trainSelected() {
  if (!state.category) return;
  const cat = state.category;

  document.getElementById('train-btn').disabled = true;
  document.getElementById('train-status').textContent = 'Starting…';

  try {
    await apiFetch(`/api/train?categories=${enc(cat)}&dataset=${state.dataset}`);
    startTrainingPoller(cat);
  } catch (e) {
    showToast(`Training error: ${e.message}`, 'error');
    updateTrainButton();
  }
}

function startTrainingPoller(cat) {
  if (state.trainingPollers[cat]) return;
  document.getElementById('train-status').textContent = 'Training…';
  refreshCategoryDot(cat, 'training');

  const poll = setInterval(async () => {
    try {
      const res = await apiFetch(`/api/train/status?category=${enc(cat)}&dataset=${state.dataset}`);
      if (res.status === 'done') {
        clearInterval(poll);
        delete state.trainingPollers[cat];
        state.trainedCategories.add(cat);
        updateTrainButton();
        refreshCategoryDot(cat, 'yes');
        showToast(`Training complete for "${cat}"`, 'success');
      } else if (res.status?.startsWith('error:')) {
        clearInterval(poll);
        delete state.trainingPollers[cat];
        refreshCategoryDot(cat, '');
        showToast(`Training failed: ${res.status.slice(6)}`, 'error');
        updateTrainButton();
      }
    } catch (_) {}
  }, 3000);

  state.trainingPollers[cat] = poll;
}

function refreshCategoryDot(cat, cls) {
  const el = document.getElementById(`cat-${cat}`);
  if (!el) return;
  const dot = el.querySelector('.trained-dot');
  if (dot) dot.className = `trained-dot ${cls}`;
}

// ── Helpers ───────────────────────────────────────────────────────────────

function updateBreadcrumb() {
  const el = document.getElementById('breadcrumb');
  if (!state.category) {
    el.innerHTML = '<span class="bc-hint">Choose a category and defect type to browse images</span>';
    return;
  }
  const defect = state.defectTypes.find(d => d.id === state.defect);
  const defectLabel = defect ? defect.label : state.defect;
  const is3d = state.dataset === 'mvtec_ad3';
  const badge3d = is3d ? '<span class="bc-3d-badge">📐 3D · RGB</span>' : '';

  el.innerHTML = state.defect
    ? `<span class="bc-crumb">${icon(state.category)} ${state.category}</span>
       <span class="bc-sep">/</span>
       <span class="bc-crumb">${defectLabel}</span>${badge3d}`
    : `<span class="bc-crumb">${icon(state.category)} ${state.category}</span>${badge3d}`;
}

function clearDefectList() {
  document.getElementById('defect-list').innerHTML =
    `<div class="empty-state"><div class="empty-arrow">←</div><p>Select a category</p></div>`;
  document.getElementById('defect-count').textContent = '';
  state.defectTypes = [];
}

function clearImageArea() {
  document.getElementById('image-area').innerHTML =
    `<div class="empty-state hero-empty">
       <div class="hero-icon">🔬</div>
       <p class="hero-title">Images will appear here</p>
       <p class="hero-hint">Click a defect type in the middle panel</p>
     </div>`;
}

function updateHeaderStats(catCount, modelCount) {
  document.getElementById('stat-cats').textContent = catCount;
  document.getElementById('stat-models').textContent = modelCount;
}

function enc(s) { return encodeURIComponent(s); }

function escAttr(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/'/g, '&#39;')
    .replace(/"/g, '&quot;');
}

async function apiFetch(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

let _toastTimer;
function showToast(msg, type = '') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = type ? `show ${type}` : 'show';
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.classList.remove('show'); }, 3800);
}

// Keyboard: Escape closes modal
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeModal({ target: document.getElementById('modal-backdrop') });
  }
});

init();
