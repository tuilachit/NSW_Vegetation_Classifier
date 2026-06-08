/*NSW Vegetation Classifier — Frontend  */

'use strict';

const API_BASE = window.VEGEMAP_API_BASE || 'http://127.0.0.1:8000';

/* Final output class definitions: 9 vegetation groups + Other */
const FORMATIONS = [
  { id:  1, name: 'Alpine',                         color: '#b0d8e8' },
  { id:  2, name: 'Arid Shrublands',                color: '#c97a3e' },
  { id:  3, name: 'Dry Sclerophyll Forests',        color: '#4a7c32' },
  { id:  4, name: 'Wetlands',                       color: '#2a9d8f' },
  { id:  5, name: 'Grass / Grassy Woodlands',       color: '#a8c046' },
  { id:  6, name: 'Heathlands',                     color: '#8b5cf6' },
  { id:  7, name: 'Rainforests',                    color: '#1a5c2e' },
  { id:  8, name: 'Semi-arid Woodlands',            color: '#d4845a' },
  { id:  9, name: 'Wet Sclerophyll Forests',        color: '#52b788' },
  { id: 17, name: 'Other / non-target land cover',  color: '#4b5563' },
];

/*  State  */
const state = {
  step: 1,
  completed: new Set(),
  inputMode: 'raw-workflow',
  aerial: { file: null, bounds: null, width: 0, height: 0, crs: null, bandCount: 0, modelReady: false },
  lidar:  { file: null, ready: false },
  settings: { resolution: '0.6', soilSource: 'auto' },
  soilReady: false,
  processingStart: null,
  modelResult: null,
  inferenceError: null,
};

/*  Helpers  */
function $(id) { return document.getElementById(id); }

function formatBytes(b) {
  if (b < 1024)        return b + ' B';
  if (b < 1024 ** 2)   return (b / 1024).toFixed(1) + ' KB';
  if (b < 1024 ** 3)   return (b / 1024 ** 2).toFixed(1) + ' MB';
  return (b / 1024 ** 3).toFixed(2) + ' GB';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/* Web-Mercator → lat/lng */
function mercToLatLng(x, y) {
  const lng = (x / 20037508.34) * 180;
  const lat = (Math.atan(Math.exp((y / 20037508.34) * Math.PI)) * 360 / Math.PI) - 90;
  return { lat: +lat.toFixed(6), lng: +lng.toFixed(6) };
}

function setBadge(container, type, text) {
  container.innerHTML = `<span class="badge badge-${type}">${text}</span>`;
}

function setValItem(id, state) { // state: 'ok' | 'error' | 'running'
  const el = $(id);
  if (!el) return;
  const dot = el.querySelector('.val-dot');
  dot.className = 'val-dot ' + state;
}

function goToStep(n) {
  document.querySelectorAll('.step-panel').forEach(p => p.classList.add('hidden'));
  $(`panel-${n}`).classList.remove('hidden');

  document.querySelectorAll('.step-btn').forEach(btn => {
    const s = +btn.dataset.step;
    btn.classList.remove('active', 'done');
    if (s === n) btn.classList.add('active');
    else if (state.completed.has(s)) btn.classList.add('done');
  });

  state.step = n;
}

function enableStepBtn(n) {
  const btn = document.querySelector(`.step-btn[data-step="${n}"]`);
  if (btn) btn.disabled = false;
}

function initStep1() {
  const drop  = $('aerial-drop');
  const input = $('aerial-input');
  const browseBtn = $('aerial-browse');

  browseBtn.addEventListener('click', () => input.click());

  input.addEventListener('change', () => {
    if (input.files[0]) handleAerialFile(input.files[0]);
  });

  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) handleAerialFile(f);
  });

  $('step1-next').addEventListener('click', () => {
    state.completed.add(1);
    if (state.aerial.modelReady) {
      state.completed.add(2);
      state.completed.add(3);
      state.completed.add(4);
      enableStepBtn(2);
      enableStepBtn(3);
      enableStepBtn(4);
      enableStepBtn(5);
      populateBBox();
      goToStep(5);
      startProcessing();
      return;
    }
    enableStepBtn(2);
    populateBBox();
    goToStep(2);
  });
}

async function handleAerialFile(file) {
  state.aerial.file = file;
  state.inputMode = 'raw-workflow';
  state.aerial.modelReady = false;
  state.aerial.bandCount = 0;
  state.modelResult = null;
  state.inferenceError = null;
  updateInputModeBanner('pending');

  // Show filename card
  $('aerial-card').classList.remove('hidden');
  $('aerial-filename').textContent = file.name;
  $('aerial-filesize').textContent  = formatBytes(file.size);
  setBadge($('aerial-badge'), 'pending', 'Validating…');

  // Show validation list
  $('aerial-validation').classList.remove('hidden');
  ['val-format','val-crs','val-georef','val-bands'].forEach(id => setValItem(id, 'running'));

  // Validate format
  const ext = file.name.split('.').pop().toLowerCase();
  const supported = ['tif','tiff'];
  await sleep(300);
  if (!supported.includes(ext)) {
    setValItem('val-format', 'error');
    setValItem('val-crs', 'error');
    setValItem('val-georef', 'error');
    setValItem('val-bands', 'error');
    setBadge($('aerial-badge'), 'error', 'GeoTIFF required');
    updateInputModeBanner('pending');
    return;
  }
  setValItem('val-format', 'ok');

  await parseGeoTIFF(file);
}

async function parseGeoTIFF(file) {
  try {
    const buf  = await file.arrayBuffer();
    const tiff = await GeoTIFF.fromArrayBuffer(buf);
    const img  = await tiff.getImage();

    // Bounding box
    const bbox = img.getBoundingBox(); // [west, south, east, north] in image CRS
    const fileDir = img.fileDirectory;

    // Detect CRS from GeoKeys
    let crsCode = null;
    if (img.geoKeys) {
      crsCode = img.geoKeys.ProjectedCSTypeGeoKey || img.geoKeys.GeographicTypeGeoKey || null;
    }

    // Convert bbox to lat/lng
    let north, south, east, west;
    if (crsCode === 4326 || (!crsCode && Math.abs(bbox[0]) < 360)) {
      // Already geographic
      west  = +bbox[0].toFixed(6);
      south = +bbox[1].toFixed(6);
      east  = +bbox[2].toFixed(6);
      north = +bbox[3].toFixed(6);
    } else {
      // Assume Web Mercator
      const sw = mercToLatLng(bbox[0], bbox[1]);
      const ne = mercToLatLng(bbox[2], bbox[3]);
      west = sw.lng; south = sw.lat; east = ne.lng; north = ne.lat;
    }
    state.aerial.bounds = { north, south, east, west };
    state.aerial.width  = img.getWidth();
    state.aerial.height = img.getHeight();
    state.aerial.crs    = crsCode;

    const isExpectedCrs = crsCode === 3857;
    await sleep(250);
    setValItem('val-crs', isExpectedCrs ? 'ok' : 'error');
    $('preview-crs').textContent = crsCode ? `EPSG:${crsCode}` : 'CRS missing';

    await sleep(200);
    setValItem('val-georef', bbox[0] !== 0 ? 'ok' : 'error');

    // Sample bands count
    const bandCount = fileDir.SamplesPerPixel || img.getSamplesPerPixel?.() || 3;
    state.aerial.bandCount = bandCount;
    await sleep(200);
    setValItem('val-bands', bandCount >= 3 ? 'ok' : 'error');

    // GSD estimate
    const gsdX = img.getResolution?.()?.[0] || null;
    const gsdLabel = gsdX ? Math.abs(gsdX).toFixed(2) + ' m/px' : '~0.6 m/px';

    if (!isExpectedCrs) {
      setBadge($('aerial-badge'), 'error', 'EPSG:3857 required');
      $('step1-next').disabled = true;
      updateInputModeBanner('pending');
      await renderAerialPreview(img);
      return;
    }

    const isSupportedTileSize = state.aerial.width === 512 && state.aerial.height === 512;
    const isModelReady = bandCount === 11 && isSupportedTileSize;
    const isRawReady = bandCount >= 3 && isSupportedTileSize;
    state.inputMode = isModelReady ? 'model-ready' : 'raw-workflow';
    state.aerial.modelReady = isModelReady;

    showAerialStats(state.aerial.width, state.aerial.height, gsdLabel, `EPSG:${crsCode || '?'}`, bandCount);
    if (!isRawReady) {
      setBadge($('aerial-badge'), 'error', '512×512 tile required');
      $('step1-next').disabled = true;
      updateInputModeBanner('pending');
      await renderAerialPreview(img);
      return;
    }

    setBadge($('aerial-badge'), isModelReady ? 'success' : 'pending', isModelReady ? 'Model-ready' : 'Needs LiDAR zip');
    updateInputModeBanner(isModelReady ? 'model-ready' : 'raw');
    $('step1-next').innerHTML = isModelReady
      ? `Run model inference <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>`
      : `Continue to LiDAR Order <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>`;
    $('step1-next').disabled = false;

    // Render preview
    await renderAerialPreview(img);

  } catch (err) {
    console.warn('GeoTIFF parse error:', err);
    setValItem('val-crs',    'error');
    setValItem('val-georef', 'error');
    setValItem('val-bands',  'error');
    $('preview-crs').textContent = 'Unreadable GeoTIFF';
    setBadge($('aerial-badge'), 'error', 'Cannot parse GeoTIFF');
    $('step1-next').disabled = true;
    state.aerial.bounds = null;
    state.aerial.width = 0; state.aerial.height = 0; state.aerial.bandCount = 0;
    state.aerial.modelReady = false;
    updateInputModeBanner('pending');
  }
}

async function renderAerialPreview(img) {
  try {
    const srcW = img.getWidth();
    const srcH = img.getHeight();
    const maxSide = 512;
    const scale = Math.min(1, maxSide / Math.max(srcW, srcH));
    const W = Math.max(1, Math.round(srcW * scale));
    const H = Math.max(1, Math.round(srcH * scale));
    const data = await img.readRasters({ interleave: true, width: W, height: H, samples: [0, 1, 2] });
    const canvas = $('aerial-canvas');
    canvas.width = W; canvas.height = H;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.objectFit = 'contain';
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(W, H);

    const stretches = [];
    for (let c = 0; c < 3; c++) {
      const values = [];
      for (let i = c; i < data.length; i += 3) {
        const v = data[i];
        if (Number.isFinite(v)) values.push(v);
      }
      values.sort((a, b) => a - b);
      const lo = values[Math.floor(values.length * 0.02)] ?? 0;
      const hiRaw = values[Math.floor(values.length * 0.98)] ?? 255;
      stretches.push({ lo, hi: hiRaw > lo ? hiRaw : lo + 1 });
    }

    for (let i = 0; i < W * H; i++) {
      for (let c = 0; c < 3; c++) {
        const raw = data[i * 3 + c] ?? 0;
        const { lo, hi } = stretches[c];
        imgData.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(((raw - lo) / (hi - lo)) * 255)));
      }
      imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    $('aerial-empty').classList.add('hidden');
    canvas.classList.remove('hidden');
  } catch (e) {
    console.warn('Preview render failed:', e);
  }
}

function showAerialStats(w, h, gsd, crsLabel, bandCount = 3) {
  $('pval-dims').textContent  = `${w.toLocaleString()} × ${h.toLocaleString()} px`;
  $('pval-gsd').textContent   = gsd;
  $('pval-bands').textContent = bandCount === 11 ? '11 (model stack)' : `${bandCount} band${bandCount === 1 ? '' : 's'}`;
  ['pstat-dims','pstat-gsd','pstat-bands'].forEach(id => $( id).style.display = '');
  if (crsLabel) $('preview-crs').textContent = crsLabel;
}

function updateInputModeBanner(mode) {
  const banner = $('input-mode-banner');
  if (!banner) return;
  banner.classList.remove('hidden', 'mode-success', 'mode-warning');

  if (mode === 'pending') {
    banner.classList.add('hidden');
    return;
  }

  if (mode === 'model-ready') {
    banner.classList.add('mode-success');
    banner.innerHTML = `<strong>Model-ready input detected.</strong> This 512 x 512, 11-band GeoTIFF already contains RGB, SoilCode, CHM, canopy/strata and TWI channels. The app can send it directly to the TensorFlow backend.`;
    return;
  }

  banner.classList.add('mode-warning');
  banner.innerHTML = `<strong>Raw workflow mode.</strong> This file is not the 11-band model stack. Real inference also needs the matching ELVIS zip and a local backend configured with the ASC soil shapefile.`;
}

function initStep2() {
  $('step2-back').addEventListener('click', () => goToStep(1));
  $('step2-next').addEventListener('click', () => {
    state.completed.add(2);
    enableStepBtn(3);
    goToStep(3);
  });
  $('copy-coords-btn').addEventListener('click', copyCoords);
  $('open-elvis-btn').addEventListener('click', openElvisPortal);
}

function populateBBox() {
  const b = state.aerial.bounds;
  $('coord-north').textContent = formatCoordinate(b?.north);
  $('coord-south').textContent = formatCoordinate(b?.south);
  $('coord-east').textContent  = formatCoordinate(b?.east);
  $('coord-west').textContent  = formatCoordinate(b?.west);
}

function formatCoordinate(value) {
  return Number.isFinite(value) ? `${value.toFixed(6)}°` : '—';
}

function copyCoords() {
  const text = getCurrentBBoxText();
  const visibleCopy = $('coords-copy-text');
  visibleCopy.value = text;
  visibleCopy.classList.add('active');
  visibleCopy.focus();
  visibleCopy.select();
  copyTextToClipboard(text)
    .then(() => showButtonMessage($('copy-coords-btn'), 'Copied!', 'var(--accent-green)'))
    .catch(() => {
      visibleCopy.focus();
      visibleCopy.select();
      showButtonMessage($('copy-coords-btn'), 'Copy manually', 'var(--accent-amber)');
    });
}

function getCurrentBBoxText() {
  const fromDom = {
    north: $('coord-north')?.textContent?.replace('°', '').trim(),
    south: $('coord-south')?.textContent?.replace('°', '').trim(),
    east: $('coord-east')?.textContent?.replace('°', '').trim(),
    west: $('coord-west')?.textContent?.replace('°', '').trim(),
  };
  if (fromDom.north && fromDom.south && fromDom.east && fromDom.west && fromDom.north !== '—') {
    return `North: ${fromDom.north}\nSouth: ${fromDom.south}\nEast:  ${fromDom.east}\nWest:  ${fromDom.west}`;
  }

  const b = state.aerial.bounds || { north: 0, south: 0, east: 0, west: 0 };
  return `North: ${b.north.toFixed(6)}\nSouth: ${b.south.toFixed(6)}\nEast:  ${b.east.toFixed(6)}\nWest:  ${b.west.toFixed(6)}`;
}

function copyTextToClipboard(text) {
  return new Promise((resolve, reject) => {
    const textarea = $('coords-copy-text') || document.createElement('textarea');
    textarea.value = text;
    if (!textarea.parentNode) document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    try {
      const ok = document.execCommand('copy');
      if (ok) {
        resolve();
        return;
      }
    } catch (err) {
      console.warn('execCommand copy failed:', err);
    }

    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(resolve).catch(reject);
      return;
    }
    reject(new Error('copy command failed'));
  });
}

function showButtonMessage(btn, message, colour) {
  const orig = btn.innerHTML;
  btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><polyline points="3 8 6.5 11.5 13 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg> ${message}`;
  btn.style.color = colour;
  setTimeout(() => { btn.innerHTML = orig; btn.style.color = ''; }, 2000);
}

function openElvisPortal(event) {
  event.preventDefault();
  const url = 'https://elevation.fsdf.org.au';
  const opened = window.open(url, '_blank');
  if (opened) {
    opened.opener = null;
  } else {
    window.location.href = url;
  }
}

function initStep3() {
  const drop  = $('lidar-drop');
  const input = $('lidar-input');
  const browseBtn = $('lidar-browse');

  browseBtn.addEventListener('click', () => input.click());
  input.addEventListener('change', () => { if (input.files[0]) handleLidarFile(input.files[0]); });

  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) handleLidarFile(f);
  });

  $('step3-back').addEventListener('click', () => goToStep(2));
  $('step3-next').addEventListener('click', () => {
    state.completed.add(3);
    enableStepBtn(4);
    goToStep(4);
    startSoilFetch();
  });
}

async function handleLidarFile(file) {
  if (!file.name.toLowerCase().endsWith('.zip')) {
    alert('Please upload a .zip archive from your ELVIS order.');
    return;
  }
  state.lidar.file = file;

  $('lidar-card').classList.remove('hidden');
  $('lidar-filename').textContent = file.name;
  $('lidar-filesize').textContent  = formatBytes(file.size);
  setBadge($('lidar-badge'), 'pending', 'Checking…');
  $('lidar-stages').classList.remove('hidden');

  // The heavy LAS/LAZ raster generation runs on the backend during inference.
  // These quick checks keep the UI responsive before the final backend call.
  const stages = ['ps-extract', 'ps-verify', 'ps-chm', 'ps-strata', 'ps-final'];
  const delays  = [500, 400, 500, 400, 300];

  for (let i = 0; i < stages.length; i++) {
    const el = $(stages[i]);
    el.classList.add('running');
    const spinner = document.createElement('div');
    spinner.className = 'ps-spinner';
    el.querySelector('.ps-dot')?.replaceWith(spinner);
    await sleep(delays[i]);
    spinner.replaceWith(createDoneDot());
    el.classList.remove('running');
    el.classList.add('done');
  }

  setBadge($('lidar-badge'), 'success', 'Ready for backend');
  state.lidar.ready = true;
  $('step3-next').disabled = false;
}

function createDoneDot() {
  const d = document.createElement('div');
  d.className = 'ps-dot';
  d.style.background = 'var(--accent-green)';
  return d;
}

function initStep4() {
  $('step4-back').addEventListener('click', () => goToStep(3));
  $('step4-next').addEventListener('click', () => {
    const res = document.querySelector('input[name="resolution"]:checked');
    state.settings.resolution = res ? res.value : '1.0';
    state.settings.soilSource = $('soil-source').value;
    state.completed.add(4);
    enableStepBtn(5);
    goToStep(5);
    startProcessing();
  });
}

async function startSoilFetch() {
  const spinner = $('soil-spinner');
  const text    = $('soil-status-text');
  const clSoil  = $('cl-soil');

  if (state.aerial.modelReady) {
    spinner.style.display = 'none';
    text.textContent = 'SoilCode already included in uploaded 11-band model stack.';
    text.style.color = 'var(--accent-green)';
    markChecklistReady(clSoil);
    state.soilReady = true;
    return;
  }

  spinner.style.display = '';
  text.textContent = 'Checking backend ASC soil shapefile...';
  text.style.color = 'var(--text-muted)';
  markChecklistPending(clSoil);
  state.soilReady = false;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${API_BASE}/health`, { cache: 'no-store', signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`Backend health returned ${res.status}`);
    const health = await res.json();

    spinner.style.display = 'none';
    if (health.soil_assets_ready) {
      text.textContent = 'ASC soil shapefile ready on backend. SoilCode will be rasterised during raw inference.';
      text.style.color = 'var(--accent-green)';
      markChecklistReady(clSoil);
      state.soilReady = true;
      return;
    }

    text.textContent = 'ASC soil shapefile is missing on backend. Raw inference will fail until soil assets are configured.';
    text.style.color = 'var(--accent-red)';
    markChecklistPending(clSoil);
  } catch (err) {
    console.warn('Backend soil readiness check failed:', err);
    spinner.style.display = 'none';
    text.textContent = 'Could not check backend soil assets. Confirm the backend is running before raw inference.';
    text.style.color = 'var(--accent-red)';
    markChecklistPending(clSoil);
  }
}

function markChecklistPending(item) {
  if (!item) return;
  item.classList.remove('ready');
  item.classList.add('pending');
  const dot = item.querySelector('.cl-dot');
  if (dot) dot.className = 'cl-dot cl-amber';
  const tick = item.querySelector('.cl-tick');
  if (tick) {
    const spinner = document.createElement('div');
    spinner.className = 'mini-spinner';
    tick.replaceWith(spinner);
  }
}

function markChecklistReady(item) {
  if (!item) return;
  item.classList.remove('pending');
  item.classList.add('ready');
  const spinnerEl = item.querySelector('.mini-spinner');
  if (spinnerEl) {
    const tick = document.createElementNS('http://www.w3.org/2000/svg','svg');
    tick.setAttribute('class','cl-tick');
    tick.setAttribute('width','14'); tick.setAttribute('height','14');
    tick.setAttribute('viewBox','0 0 16 16');
    tick.setAttribute('fill','none');
    tick.innerHTML = `<polyline points="3 8 6.5 11.5 13 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
    spinnerEl.replaceWith(tick);
  }
  const dot = item.querySelector('.cl-dot');
  if (dot) dot.className = 'cl-dot cl-green';
}

function initStep5() {
  $('step5-back').addEventListener('click', () => goToStep(4));
  $('restart-btn').addEventListener('click', () => {
    Object.assign(state, {
      step: 1, completed: new Set(),
      inputMode: 'raw-workflow',
      aerial: { file: null, bounds: null, width: 0, height: 0, crs: null, bandCount: 0, modelReady: false },
      lidar: { file: null, ready: false },
      settings: { resolution: '0.6', soilSource: 'auto' },
      soilReady: false,
      modelResult: null,
      inferenceError: null,
    });
    document.querySelectorAll('.step-btn').forEach((b,i) => {
      b.classList.remove('active','done');
      if (i > 0) b.disabled = true;
    });
    $('processing-view').classList.remove('hidden');
    $('results-view').classList.add('hidden');
    $('results-heading').textContent   = 'Running Classifier';
    $('results-subheading').textContent = 'Processing your multi-channel input stack through the two-stage vegetation model.';
    $('results-badge').className = 'step-badge badge-purple';
    $('results-badge').textContent = 'Processing';
    $('step5-back').style.display = 'none';
    resetProcessingUI();
    goToStep(1);
    // Reset upload zones
    resetUploadZone('aerial'); resetUploadZone('lidar');
    $('aerial-canvas').classList.add('hidden');
    $('aerial-empty').classList.remove('hidden');
    $('step1-next').innerHTML = `Continue to LiDAR Order <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>`;
    $('step1-next').disabled = true;
    updateInputModeBanner('pending');
  });

  $('map-zoom-in').addEventListener('click', () => zoomMap(1.2));
  $('map-zoom-out').addEventListener('click', () => zoomMap(0.83));

  $('dl-geotiff').addEventListener('click', downloadClassificationRaster);
  $('dl-report').addEventListener('click',  downloadReport);
}

function resetUploadZone(type) {
  $(`${type}-card`)?.classList.add('hidden');
  $(`${type}-validation`)?.classList.add('hidden');
  $(`${type}-stages`)?.classList.add('hidden');
}

function resetProcessingUI() {
  const items = ['pl-preprocess','pl-reproject','pl-inference','pl-postprocess','pl-export'];
  items.forEach((id, i) => {
    const el = $(id);
    el.className = 'pipeline-item' + (i > 0 ? ' pl-pending' : '');
    const icon = el.querySelector('.pl-icon');
    icon.className = 'pl-icon' + (i === 0 ? ' pl-running' : '');
    const state = el.querySelector('.pl-state');
    state.textContent = i === 0 ? 'Running' : 'Pending';
  });
  $('progress-fill').style.width = '0%';
  $('progress-label').textContent = 'Preprocessing…';
  $('progress-pct').textContent   = '0%';
  $('progress-eta').textContent   = 'Estimated time: calculating…';
}

const PIPELINE_STAGES = [
  { id: 'pl-preprocess', label: 'Preprocessing…',        pct: 12, eta: '~45 sec' },
  { id: 'pl-reproject',  label: 'Reprojecting…',          pct: 28, eta: '~38 sec' },
  { id: 'pl-inference',  label: 'Running model inference…',pct: 72, eta: '~35 sec' },
  { id: 'pl-postprocess',label: 'Post-processing…',        pct: 90, eta: '~8 sec'  },
  { id: 'pl-export',     label: 'Exporting outputs…',     pct: 100,eta: '~2 sec'  },
];

const STAGE_DURATIONS = [2800, 2200, 5000, 2000, 1400]; // ms

function startProcessing() {
  state.processingStart = Date.now();
  state.modelResult = null;
  animatePipeline(0);
}

async function runBackendInferenceIfAvailable() {
  if (!state.aerial.file) return null;

  const form = new FormData();
  const endpoint = state.aerial.modelReady ? 'predict-tile' : 'predict-raw';
  if (state.aerial.modelReady) {
    form.append('file', state.aerial.file);
  } else {
    if (!state.lidar.file) {
      state.inferenceError = 'Raw inference requires both the RGB aerial GeoTIFF and the matching ELVIS zip.';
      state.modelResult = null;
      return null;
    }
    form.append('aerial', state.aerial.file);
    form.append('lidar_zip', state.lidar.file);
  }

  try {
    const response = await fetch(`${API_BASE}/${endpoint}`, {
      method: 'POST',
      body: form,
    });
    if (!response.ok) {
      const message = await response.text();
      throw new Error(message);
    }
    state.modelResult = await response.json();
    return state.modelResult;
  } catch (err) {
    console.warn('Backend inference unavailable.', err);
    state.inferenceError = err.message || String(err);
    state.modelResult = null;
    return null;
  }
}

async function animatePipeline(idx) {
  if (idx >= PIPELINE_STAGES.length) {
    await sleep(400);
    showResults();
    return;
  }

  const stage = PIPELINE_STAGES[idx];
  const el    = $(stage.id);

  // Mark current as running
  el.className = 'pipeline-item pl-running';
  const icon  = el.querySelector('.pl-icon');
  icon.className = 'pl-icon pl-running';
  // Swap icon content with spinner
  const spinner = document.createElement('div');
  spinner.className = 'pl-spinner';
  icon.innerHTML = '';
  icon.appendChild(spinner);
  el.querySelector('.pl-state').textContent = 'Running';

  // Animate progress bar toward this stage's pct
  const prevPct = idx > 0 ? PIPELINE_STAGES[idx-1].pct : 0;
  animateProgress(prevPct, stage.pct, STAGE_DURATIONS[idx], stage.label, stage.eta);

  if (stage.id === 'pl-inference') {
    await runBackendInferenceIfAvailable();
  }

  await sleep(STAGE_DURATIONS[idx]);

  // Mark done
  el.className = 'pipeline-item pl-done';
  icon.className = 'pl-icon pl-done';
  icon.innerHTML = `<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><polyline points="3 8 6.5 11.5 13 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  el.querySelector('.pl-state').textContent = 'Done';

  // Activate next stage
  if (idx + 1 < PIPELINE_STAGES.length) {
    $(PIPELINE_STAGES[idx+1].id).classList.remove('pl-pending');
  }

  animatePipeline(idx + 1);
}

function animateProgress(from, to, duration, label, eta) {
  $('progress-label').textContent = label;
  $('progress-eta').textContent   = `Estimated time remaining: ${eta}`;
  const start = performance.now();
  function frame(now) {
    const t = Math.min((now - start) / duration, 1);
    const current = from + (to - from) * easeOut(t);
    $('progress-fill').style.width = current.toFixed(1) + '%';
    $('progress-pct').textContent  = Math.round(current) + '%';
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
function easeOut(t) { return 1 - (1 - t) ** 3; }

let mapScale = 1;

function showResults() {
  const elapsed = ((Date.now() - state.processingStart) / 1000).toFixed(1);
  $('processing-view').classList.add('hidden');
  $('results-view').classList.remove('hidden');
  $('results-heading').textContent    = 'Classification Results';
  $('results-subheading').textContent = state.modelResult
    ? 'Two-stage model inference complete. Download the outputs below.'
    : (state.aerial.modelReady
      ? 'Model-ready tile uploaded, but backend inference did not complete. Check that the backend is running and the model weights are installed.'
      : 'Raw inference did not complete. Check that the backend is running, the ASC soil shapefile is configured, and the ELVIS zip overlaps the aerial tile.');
  $('results-badge').className        = state.modelResult ? 'step-badge badge-blue' : 'step-badge badge-red';
  $('results-badge').textContent      = state.modelResult ? 'Complete' : 'Failed';
  $('step5-back').style.display = '';
  state.completed.add(5);
  updateExportAvailability();

  const res = +state.settings.resolution;
  const areaPx = (state.aerial.width || 512) * (state.aerial.height || 512);
  const areaHa = ((areaPx * res * res) / 10000).toFixed(1);
  const detected = state.modelResult
    ? state.modelResult.classes.filter(c => c.class_value !== 0).length
    : 0;

  $('rstat-area').textContent    = state.modelResult && parseFloat(areaHa) > 0 ? areaHa : '0.0';
  $('rstat-classes').textContent = String(detected);
  $('rstat-conf').textContent    = state.modelResult ? 'Ready' : 'Failed';
  $('rstat-time').textContent    = elapsed;

  renderResultsMap();
  buildLegend();
}

function updateExportAvailability() {
  const realInference = Boolean(state.modelResult);
  const geotiff = $('dl-geotiff');

  if (geotiff) {
    geotiff.disabled = !realInference;
    geotiff.title = realInference ? 'Download backend prediction GeoTIFF' : 'Real GeoTIFF export requires backend inference.';
  }
}

function renderResultsMap() {
  const canvas = $('results-canvas');
  const wrap = canvas.parentElement;
  const W = Math.max(wrap.clientWidth  || 600, 300);
  const H = Math.max(wrap.clientHeight || 400, 200);
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');

  if (state.modelResult?.preview_png_base64) {
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, W, H);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, W, H);
    };
    img.src = `data:image/png;base64,${state.modelResult.preview_png_base64}`;
    return;
  }

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b1020';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#94a3b8';
  ctx.font = '600 15px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('No model output available', W / 2, H / 2 - 8);
  ctx.font = '12px Inter, sans-serif';
  ctx.fillText('Check backend health, model assets, soil assets, and input overlap.', W / 2, H / 2 + 16);
}

function buildLegend() {
  const container = $('legend-items');
  container.innerHTML = '';
  const usedIds = state.modelResult
    ? state.modelResult.classes.map(c => c.class_value).filter(v => v !== 0)
    : [];
  FORMATIONS.filter(f => usedIds.includes(f.id)).forEach(f => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-swatch" style="background:${f.color}"></div><span>${f.name}</span>`;
    container.appendChild(item);
  });
}

function zoomMap(factor) {
  mapScale *= factor;
  mapScale = Math.min(4, Math.max(0.5, mapScale));
  $('results-canvas').style.transform = `scale(${mapScale})`;
  $('results-canvas').style.transformOrigin = 'center center';
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function base64ToBlob(base64, mimeType) {
  const bytes = atob(base64);
  const chunks = [];
  for (let i = 0; i < bytes.length; i += 8192) {
    const slice = bytes.slice(i, i + 8192);
    const arr = new Uint8Array(slice.length);
    for (let j = 0; j < slice.length; j++) arr[j] = slice.charCodeAt(j);
    chunks.push(arr);
  }
  return new Blob(chunks, { type: mimeType });
}

function withBtnLoadingState(btn, label, task) {
  const orig = btn.innerHTML;
  btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 16 16" fill="none" style="animation:spin .7s linear infinite"><circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="2" stroke-dasharray="28" stroke-dashoffset="10"/></svg> ${label}`;
  btn.disabled = true;
  Promise.resolve().then(task).finally(() => {
    btn.innerHTML = orig;
    btn.disabled  = false;
  });
}

function downloadClassificationRaster() {
  if (!state.modelResult?.prediction_geotiff_base64) {
    alert('Real GeoTIFF export requires a successful backend inference result.');
    return;
  }
  withBtnLoadingState($('dl-geotiff'), 'Exporting…', () => {
    const blob = base64ToBlob(state.modelResult.prediction_geotiff_base64, 'image/tiff');
    triggerDownload(blob, 'vegetation_classification.tif');
    return Promise.resolve();
  });
}

function downloadReport() {
  withBtnLoadingState($('dl-report'), 'Building…', () => {
    const area    = $('rstat-area').textContent;
    const classes = $('rstat-classes').textContent;
    const modelStatus = $('rstat-conf').textContent;
    const time    = $('rstat-time').textContent;
    const res     = state.settings.resolution;
    const b       = state.aerial.bounds || { north: '—', south: '—', east: '—', west: '—' };
    const date    = new Date().toLocaleDateString('en-AU', { day: 'numeric', month: 'long', year: 'numeric' });
    const usedIds = state.modelResult
      ? state.modelResult.classes.map(c => c.class_value).filter(v => v !== 0)
      : [];

    const legendRows = FORMATIONS
      .filter(f => usedIds.includes(f.id))
      .map(f => {
        const row = state.modelResult?.classes.find(c => c.class_value === f.id);
        const classArea = row ? ((Number(row.fraction) * Number(area || 0))).toFixed(2) : '—';
        const pixels = row ? Number(row.pixels).toLocaleString() : '—';
        return `<tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${f.color};margin-right:8px;vertical-align:middle"></span>${f.name}</td><td>${classArea}</td><td>${pixels}</td></tr>`;
      })
      .join('');

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>NSW Vegetation Classification Report</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 860px; margin: 40px auto; color: #1a2332; font-size: 14px; line-height: 1.6; }
  h1   { font-size: 22px; color: #0f172a; margin-bottom: 4px; }
  h2   { font-size: 15px; color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 28px; }
  .meta { color: #64748b; font-size: 13px; margin-bottom: 24px; }
  .stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin: 20px 0; }
  .stat  { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; }
  .stat-val  { font-size: 26px; font-weight: 700; color: #0f172a; }
  .stat-lbl  { font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: #94a3b8; margin-bottom: 4px; }
  .stat-unit { font-size: 11px; color: #94a3b8; }
  table  { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid #e2e8f0; font-size: 13px; }
  th     { background: #f1f5f9; font-weight: 600; color: #475569; }
  .bbox  { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
  .bbox-item { background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:10px 14px; }
  .bbox-dir  { font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:#94a3b8; }
  .bbox-coord{ font-family:monospace; font-size:13px; font-weight:600; color:#1e293b; }
  .footer    { margin-top:40px; padding-top:14px; border-top:1px solid #e2e8f0; font-size:11px; color:#94a3b8; }
  @media print { body { margin: 20px; } }
</style>
</head>
<body>
<h1>NSW Vegetation Classification Report</h1>
<p class="meta">Generated ${date} &nbsp;·&nbsp; Project 41 — UTS 42028 Deep Learning &nbsp;·&nbsp; Two-stage U-Net + ResNet50 pipeline</p>

<h2>Summary Statistics</h2>
<div class="stats">
  <div class="stat"><div class="stat-lbl">Area Classified</div><div class="stat-val">${area}</div><div class="stat-unit">hectares</div></div>
  <div class="stat"><div class="stat-lbl">Classes Detected</div><div class="stat-val">${classes}</div><div class="stat-unit">of 9 groups + Other</div></div>
  <div class="stat"><div class="stat-lbl">Model Output</div><div class="stat-val">${modelStatus}</div><div class="stat-unit">backend status</div></div>
  <div class="stat"><div class="stat-lbl">Processing Time</div><div class="stat-val">${time}s</div><div class="stat-unit">wall-clock time</div></div>
</div>

<h2>Model Configuration</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Architecture</td><td>Binary Vegetation/Other gate + 9-group U-Net with ResNet-50 encoder</td></tr>
  <tr><td>Input channels</td><td>11 (RGB · Soil · CHM · Canopy Cover · Strata × 4 · TWI)</td></tr>
  <tr><td>Tile size</td><td>512 × 512 px</td></tr>
  <tr><td>Output resolution</td><td>${res} m/px</td></tr>
  <tr><td>Label schema</td><td>0 = no data, 1–9 = grouped vegetation, 17 = Other / non-target</td></tr>
  <tr><td>Model result</td><td>${state.modelResult ? 'Backend TensorFlow inference' : 'No output - preprocessing/inference did not complete'}</td></tr>
</table>

<h2>Study Area</h2>
<div class="bbox">
  <div class="bbox-item"><div class="bbox-dir">North</div><div class="bbox-coord">${typeof b.north === 'number' ? b.north.toFixed(6) + '°' : b.north}</div></div>
  <div class="bbox-item"><div class="bbox-dir">South</div><div class="bbox-coord">${typeof b.south === 'number' ? b.south.toFixed(6) + '°' : b.south}</div></div>
  <div class="bbox-item"><div class="bbox-dir">East</div><div class="bbox-coord">${typeof b.east  === 'number' ? b.east.toFixed(6)  + '°' : b.east}</div></div>
  <div class="bbox-item"><div class="bbox-dir">West</div><div class="bbox-coord">${typeof b.west  === 'number' ? b.west.toFixed(6)  + '°' : b.west}</div></div>
</div>

<h2>Per-Formation Results</h2>
<table>
  <tr><th>Vegetation Formation</th><th>Area (ha)</th><th>Pixels</th></tr>
  ${legendRows}
</table>

</body>
</html>`;

    const blob = new Blob([html], { type: 'text/html' });
    triggerDownload(blob, 'vegetation_classification_report.html');
    return Promise.resolve();
  });
}

function initStepper() {
  document.querySelectorAll('.step-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const n = +btn.dataset.step;
      if (!btn.disabled) goToStep(n);
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initStepper();
  initStep1();
  initStep2();
  initStep3();
  initStep4();
  initStep5();
  goToStep(1);
  if (new URLSearchParams(window.location.search).has('smoketest')) {
    runSmokeTest();
  }
});

function runSmokeTest() {
  const requiredIds = [
    'aerial-input',
    'copy-coords-btn',
    'open-elvis-btn',
    'soil-source',
    'dl-geotiff',
    'dl-report',
    'results-canvas',
  ];
  const missing = requiredIds.filter(id => !$(id));
  const result = {
    ok: missing.length === 0,
    missing,
    apiBase: API_BASE,
    timestamp: new Date().toISOString(),
  };
  document.body.dataset.smoketest = result.ok ? 'pass' : 'fail';
  window.__VEGEMAP_SMOKETEST__ = result;
  console.info('VegeMap smoke test:', result);
}
