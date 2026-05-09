import { sharpnessScoreFromImageData } from './blur_score.js';

const video = document.getElementById('video');
const shot = document.getElementById('shot');
const shotCtx = shot.getContext('2d', { willReadFrequently: true });
const SHARPNESS_MAX_WIDTH = 640;
const BURST_FRAME_COUNT = 3;
const BURST_FRAME_DELAY_MS = 45;
const FIRST_LOCALIZE_DELAY_MS = 150;
const ENCODE_TIMEOUT_MS = 2500;
const LOCALIZE_TIMEOUT_MS = 30000;
const scoreCanvas = document.createElement('canvas');
const scoreCtx = scoreCanvas.getContext('2d', { willReadFrequently: true });
const mapStage = document.getElementById('mapStage');
const mapImage = document.getElementById('mapImage');
const mapOverlay = document.getElementById('mapOverlay');
const mapCtx = mapOverlay.getContext('2d');
const statusText = document.getElementById('statusText');
const engineText = document.getElementById('engineText');
const durationText = document.getElementById('durationText');
const busyText = document.getElementById('busyText');
const confText = document.getElementById('confText');
const successText = document.getElementById('successText');
const streamResolutionText = document.getElementById('streamResolutionText');
const captureResolutionText = document.getElementById('captureResolutionText');
const sharpnessText = document.getElementById('sharpnessText');
const burstScoresText = document.getElementById('burstScoresText');
const selectedFrameText = document.getElementById('selectedFrameText');
const resultBox = document.getElementById('resultBox');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const intervalMsInput = document.getElementById('intervalMs');
const captureWidthSelect = document.getElementById('captureWidth');
const jpegQualityInput = document.getElementById('jpegQuality');
const qualityPreset = document.getElementById('qualityPreset');

let stream = null;
let timer = null;
let floorplanImg = null;
let floorplanNaturalWidth = 0;
let floorplanNaturalHeight = 0;
let lastResult = null;
let isLocalizing = false;
let keepRunning = false;
let sessionId = 0;
let lastCaptureInfo = null;
let lastBurstScores = [];
let lastSelectedFrameIndex = null;
let lastSharpnessScore = null;

function setStatus(text, state = 'ok') {
  statusText.textContent = text;
  statusText.className = state;
}

function setControls(running) {
  startBtn.disabled = running;
  stopBtn.disabled = !running;
}

function summarizeResult(result) {
  if (!result) return '等待定位。';
  if (!result.success) {
    return `未定位\n原因：${result.failure_reason || '-'}\n最佳候选：${result.best_candidate || '-'}\n匹配数：${result.num_matches_best ?? '-'}`;
  }
  const heading = result.heading_deg == null ? '-' : `${Number(result.heading_deg).toFixed(1)}°`;
  const confidence = result.confidence == null ? '-' : Number(result.confidence).toFixed(3);
  return [
    `x：${Number(result.x).toFixed(1)}`,
    `y：${Number(result.y).toFixed(1)}`,
    `heading：${heading}`,
    `confidence：${confidence}`,
    `candidate：${result.best_candidate || '-'}`,
  ].join('\n');
}

function updateRuntimeStatus(status) {
  engineText.textContent = status && status.is_localizing ? '定位中' : '空闲';
  durationText.textContent = status && status.last_duration_ms != null ? `${status.last_duration_ms.toFixed(0)} ms` : '-';
  busyText.textContent = status && status.busy_dropped_count != null ? String(status.busy_dropped_count) : '0';
}

async function fetchStatus() {
  const res = await fetch('/api/status');
  return await res.json();
}

async function loadFloorplan() {
  try {
    const res = await fetch('/api/floorplan');
    if (!res.ok) return false;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const img = await new Promise((resolve, reject) => {
      const nextImg = new Image();
      nextImg.onload = () => resolve(nextImg);
      nextImg.onerror = () => reject(new Error('floorplan load failed'));
      nextImg.src = url;
    });
    floorplanImg = img;
    floorplanNaturalWidth = floorplanImg.naturalWidth || floorplanImg.width;
    floorplanNaturalHeight = floorplanImg.naturalHeight || floorplanImg.height;
    mapImage.src = url;
    drawMap();
    return true;
  } catch {
    return false;
  }
}

function fitOverlayToRenderedFloorplan() {
  const rect = mapImage.getBoundingClientRect();
  const cssWidth = Math.max(320, Math.round(rect.width || mapStage.clientWidth || 320));
  const cssHeight = Math.max(240, Math.round(rect.height || (cssWidth * 0.75)));
  mapOverlay.width = cssWidth;
  mapOverlay.height = cssHeight;
}

function drawMap() {
  fitOverlayToRenderedFloorplan();
  mapCtx.clearRect(0, 0, mapOverlay.width, mapOverlay.height);
  if (!floorplanImg || !lastResult || lastResult.x == null || lastResult.y == null) return;
  const sx = mapOverlay.width / floorplanNaturalWidth;
  const sy = mapOverlay.height / floorplanNaturalHeight;
  const x = lastResult.x * sx;
  const y = lastResult.y * sy;
  mapCtx.strokeStyle = '#d92d20';
  mapCtx.lineWidth = 4;
  mapCtx.beginPath();
  mapCtx.arc(x, y, 10, 0, Math.PI * 2);
  mapCtx.stroke();
  if (lastResult.heading_deg != null) {
    const a = lastResult.heading_deg * Math.PI / 180;
    const x2 = x + 56 * Math.cos(a);
    const y2 = y - 56 * Math.sin(a);
    mapCtx.strokeStyle = '#1f6feb';
    mapCtx.beginPath();
    mapCtx.moveTo(x, y);
    mapCtx.lineTo(x2, y2);
    mapCtx.stroke();
  }
}

function applyQualityPreset() {
  const p = qualityPreset ? qualityPreset.value : 'balanced';
  if (p === 'fast') {
    intervalMsInput.value = '700';
    captureWidthSelect.value = '1280';
    jpegQualityInput.value = '0.84';
  } else if (p === 'quality') {
    intervalMsInput.value = '1200';
    captureWidthSelect.value = 'auto-max';
    jpegQualityInput.value = '0.92';
  } else {
    intervalMsInput.value = '900';
    captureWidthSelect.value = '1920';
    jpegQualityInput.value = '0.90';
  }
}

function readIntervalMs() {
  const v = Number(intervalMsInput && intervalMsInput.value ? intervalMsInput.value : 900);
  if (!Number.isFinite(v)) return 900;
  return Math.max(300, Math.min(5000, Math.round(v)));
}

function readCaptureWidthMode() {
  return captureWidthSelect && captureWidthSelect.value ? String(captureWidthSelect.value) : '1920';
}

function readCaptureWidth() {
  const mode = readCaptureWidthMode();
  if (mode === 'auto-max') return 4096;
  const v = Number(mode);
  if (!Number.isFinite(v)) return 1920;
  return Math.max(1280, Math.min(4096, Math.round(v)));
}

function readJpegQuality() {
  const v = Number(jpegQualityInput && jpegQualityInput.value ? jpegQualityInput.value : 0.9);
  if (!Number.isFinite(v)) return 0.9;
  return Math.max(0.6, Math.min(0.98, v));
}

function updateSharpnessUi() {
  sharpnessText.textContent = Number.isFinite(lastSharpnessScore) ? lastSharpnessScore.toFixed(1) : '-';
  burstScoresText.textContent = lastBurstScores.length ? lastBurstScores.map(v => v.toFixed(1)).join(', ') : '-';
  selectedFrameText.textContent = Number.isInteger(lastSelectedFrameIndex) ? String(lastSelectedFrameIndex + 1) : '-';
}

function updateCaptureInfoUi() {
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;
  streamResolutionText.textContent = (vw > 0 && vh > 0) ? `${vw} x ${vh}` : '-';
  captureResolutionText.textContent = lastCaptureInfo ? `${lastCaptureInfo.width} x ${lastCaptureInfo.height}` : '-';
  updateSharpnessUi();
}

function buildVideoConstraints() {
  const targetW = readCaptureWidth();
  const targetH = Math.round(targetW * 9 / 16);
  return {
    facingMode: { ideal: 'environment' },
    width: { ideal: targetW },
    height: { ideal: targetH },
    aspectRatio: { ideal: 16 / 9 },
  };
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function waitForNextPaint() {
  return new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
}

async function waitForVideoReady(timeoutMs = 5000) {
  const startedAt = performance.now();
  while (performance.now() - startedAt < timeoutMs) {
    if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) return;
    await sleep(50);
  }
  throw new Error('camera stream is not ready');
}

function currentFrameGeometry(maxWidth) {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const landscape = vw >= vh;
  const srcW = landscape ? vw : vh;
  const srcH = landscape ? vh : vw;
  const targetW = Math.min(maxWidth, srcW);
  const targetH = Math.round(srcH * targetW / srcW);
  return { landscape, targetW, targetH };
}

function drawVideoToCanvas(ctx, canvas, width, height, landscape) {
  canvas.width = width;
  canvas.height = height;
  ctx.save();
  if (landscape) {
    ctx.drawImage(video, 0, 0, width, height);
  } else {
    ctx.translate(width / 2, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.drawImage(video, -height / 2, -width / 2, height, width);
  }
  ctx.restore();
}

function scoreCurrentFrame() {
  const geom = currentFrameGeometry(SHARPNESS_MAX_WIDTH);
  drawVideoToCanvas(scoreCtx, scoreCanvas, geom.targetW, geom.targetH, geom.landscape);
  const imageData = scoreCtx.getImageData(0, 0, geom.targetW, geom.targetH);
  return sharpnessScoreFromImageData(imageData);
}

function canvasToDataUrl(canvas, quality, timeoutMs = ENCODE_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    const timerId = setTimeout(() => reject(new Error('image encoding timed out')), timeoutMs);
    canvas.toBlob((blob) => {
      if (!blob) {
        clearTimeout(timerId);
        reject(new Error('image encoding failed'));
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        clearTimeout(timerId);
        resolve(reader.result);
      };
      reader.onerror = () => {
        clearTimeout(timerId);
        reject(reader.error || new Error('image read failed'));
      };
      reader.readAsDataURL(blob);
    }, 'image/jpeg', quality);
  });
}

function captureFrameToCanvas() {
  const geom = currentFrameGeometry(readCaptureWidth());
  drawVideoToCanvas(shotCtx, shot, geom.targetW, geom.targetH, geom.landscape);
  return { width: geom.targetW, height: geom.targetH };
}

async function captureBestDataUrl() {
  const scores = [];
  let best = null;
  for (let i = 0; i < BURST_FRAME_COUNT; i += 1) {
    const sharpness = scoreCurrentFrame();
    scores.push(sharpness);
    if (!best || sharpness > best.sharpness) {
      const frame = captureFrameToCanvas();
      best = { index: i, sharpness, width: frame.width, height: frame.height };
    }
    if (i < BURST_FRAME_COUNT - 1) await sleep(BURST_FRAME_DELAY_MS);
  }
  const dataUrl = await canvasToDataUrl(shot, readJpegQuality());
  lastBurstScores = scores;
  lastSelectedFrameIndex = best.index;
  lastSharpnessScore = best.sharpness;
  lastCaptureInfo = { width: best.width, height: best.height };
  updateCaptureInfoUi();
  return dataUrl;
}

async function localizeOnce() {
  if (!stream || isLocalizing) return;
  const runId = sessionId;
  isLocalizing = true;
  try {
    setStatus('采集中', 'warn');
    await waitForNextPaint();
    const image = await captureBestDataUrl();
    setStatus('定位中', 'warn');
    const controller = new AbortController();
    const timerId = setTimeout(() => controller.abort(), LOCALIZE_TIMEOUT_MS);
    const res = await fetch('/api/localize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image }),
      signal: controller.signal,
    });
    clearTimeout(timerId);
    if (runId !== sessionId) return;
    const data = await res.json();
    if (!data.ok) {
      updateRuntimeStatus(data.status);
      const busy = data.error === 'busy';
      setStatus(busy ? '忙碌' : '请求失败', busy ? 'warn' : 'bad');
      resultBox.textContent = busy ? '当前帧已跳过。' : '定位请求未完成。';
      return;
    }
    lastResult = data.result;
    const status = await fetchStatus();
    if (runId !== sessionId) return;
    updateRuntimeStatus(status);
    resultBox.textContent = summarizeResult(lastResult);
    setStatus(lastResult.success ? '已定位' : '未定位', lastResult.success ? 'ok' : 'warn');
    confText.textContent = (lastResult.confidence != null) ? Number(lastResult.confidence).toFixed(3) : '-';
    successText.textContent = lastResult.success ? '是' : '否';
    drawMap();
  } catch (e) {
    if (runId !== sessionId) return;
    const timedOut = e && e.name === 'AbortError';
    setStatus(timedOut ? '请求超时' : '请求失败', 'bad');
    resultBox.textContent = timedOut ? '定位耗时超过等待上限。' : '当前定位请求未完成。';
  } finally {
    isLocalizing = false;
  }
}

function scheduleNext() {
  if (!keepRunning) return;
  if (timer) clearTimeout(timer);
  timer = setTimeout(async () => {
    await localizeOnce();
    scheduleNext();
  }, readIntervalMs());
}

async function start() {
  try {
    if (keepRunning || stream) return;
    setControls(true);
    setStatus('准备中', 'warn');
    const status = await fetchStatus();
    if (!status.ready) {
      setStatus('数据未就绪', 'bad');
      resultBox.textContent = '定位数据未就绪。';
      setControls(false);
      return;
    }

    if (!window.isSecureContext) {
      setStatus('需要安全上下文', 'bad');
      resultBox.textContent = '当前页面无法直接访问摄像头。';
      setControls(false);
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('摄像头不可用', 'bad');
      resultBox.textContent = '当前环境无法访问摄像头。';
      setControls(false);
      return;
    }

    await loadFloorplan();
    stream = await navigator.mediaDevices.getUserMedia({ video: buildVideoConstraints(), audio: false });
    video.srcObject = stream;
    video.muted = true;
    await video.play().catch(() => {});
    await waitForVideoReady();
    updateCaptureInfoUi();
    keepRunning = true;
    sessionId += 1;
    setStatus('运行中', 'ok');
    if (timer) clearTimeout(timer);
    timer = setTimeout(async () => {
      await localizeOnce();
      scheduleNext();
    }, FIRST_LOCALIZE_DELAY_MS);
  } catch {
    keepRunning = false;
    setControls(false);
    setStatus('启动失败', 'bad');
    resultBox.textContent = '无法启动摄像头或定位会话。';
  }
}

function stop() {
  keepRunning = false;
  sessionId += 1;
  isLocalizing = false;
  if (timer) clearTimeout(timer);
  timer = null;
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }
  lastCaptureInfo = null;
  lastBurstScores = [];
  lastSelectedFrameIndex = null;
  lastSharpnessScore = null;
  updateCaptureInfoUi();
  setControls(false);
  setStatus('已停止', 'warn');
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
if (qualityPreset) {
  qualityPreset.addEventListener('change', () => {
    applyQualityPreset();
  });
}

setControls(false);
loadFloorplan();
drawMap();
window.addEventListener('resize', () => {
  setTimeout(drawMap, 0);
});
applyQualityPreset();

fetchStatus().then(s => {
  updateRuntimeStatus(s);
  setStatus(s.ready ? '就绪' : '数据未就绪', s.ready ? 'ok' : 'bad');
});
setInterval(async () => {
  try {
    const s = await fetchStatus();
    updateRuntimeStatus(s);
  } catch {}
}, 1500);
