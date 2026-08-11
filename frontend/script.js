const apiUrlInput = document.getElementById('apiUrl');
const statusDot = document.getElementById('statusDot');

/* ---------------- tabs ---------------- */

const tabs = document.querySelectorAll('.tab');
const drawPanel = document.getElementById('drawPanel');
const uploadPanel = document.getElementById('uploadPanel');

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const target = tab.dataset.tab;
    drawPanel.classList.toggle('active', target === 'draw');
    uploadPanel.classList.toggle('active', target === 'upload');
    hideError();
  });
});

/* ---------------- drawing canvas ---------------- */

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

const CANVAS_BG = '#fbfaf6';
const INK_COLOR = '#141826';

let currentTool = 'pencil';
const pencilTool = document.getElementById('pencilTool');
const eraserTool = document.getElementById('eraserTool');
const brushSize = document.getElementById('brushSize');

function setTool(tool) {
  currentTool = tool;
  pencilTool.classList.toggle('active', tool === 'pencil');
  eraserTool.classList.toggle('active', tool === 'eraser');
  canvas.classList.toggle('eraser-mode', tool === 'eraser');
}
pencilTool.addEventListener('click', () => setTool('pencil'));
eraserTool.addEventListener('click', () => setTool('eraser'));

function applyStrokeStyle() {
  ctx.lineWidth = currentTool === 'eraser'
    ? Number(brushSize.value) * 1.8   // eraser feels more natural a bit bigger
    : Number(brushSize.value);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = currentTool === 'eraser' ? CANVAS_BG : INK_COLOR;
}

function resetCanvas() {
  ctx.fillStyle = CANVAS_BG;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  applyStrokeStyle();
}
resetCanvas();

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
}

function startDraw(e) {
  drawing = true;
  applyStrokeStyle();
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  e.preventDefault();
}
function moveDraw(e) {
  if (!drawing) return;
  const p = getPos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  e.preventDefault();
}
function endDraw() { drawing = false; }

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', moveDraw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseleave', endDraw);
canvas.addEventListener('touchstart', startDraw, { passive: false });
canvas.addEventListener('touchmove', moveDraw, { passive: false });
canvas.addEventListener('touchend', endDraw);

document.getElementById('clearBtn').addEventListener('click', () => {
  resetCanvas();
  hideResult();
  hideError();
});

document.getElementById('predictDrawBtn').addEventListener('click', () => {
  canvas.toBlob(blob => predict(blob), 'image/png');
});

/* ---------------- upload ---------------- */

const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const uploadEmpty = document.getElementById('uploadEmpty');
const predictUploadBtn = document.getElementById('predictUploadBtn');
let uploadedFile = null;

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  uploadedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewImg.hidden = false;
  uploadEmpty.style.display = 'none';
  predictUploadBtn.disabled = false;
  hideResult();
  hideError();
});

document.getElementById('clearUploadBtn').addEventListener('click', () => {
  uploadedFile = null;
  fileInput.value = '';
  previewImg.hidden = true;
  uploadEmpty.style.display = 'flex';
  predictUploadBtn.disabled = true;
  hideResult();
  hideError();
});

predictUploadBtn.addEventListener('click', () => {
  if (uploadedFile) predict(uploadedFile);
});

/* ---------------- prediction ---------------- */

const specimenEmpty = document.getElementById('specimenEmpty');
const specimenBody = document.getElementById('specimenBody');
const glyphLg = document.getElementById('glyphLg');
const glyphMd = document.getElementById('glyphMd');
const glyphSm = document.getElementById('glyphSm');
const classCode = document.getElementById('classCode');
const confValue = document.getElementById('confValue');
const proofList = document.getElementById('proofList');
const errorMsg = document.getElementById('errorMsg');

function hideResult() {
  specimenBody.classList.remove('active');
  specimenEmpty.style.display = 'block';
}
function showResult() {
  specimenBody.classList.add('active');
  specimenEmpty.style.display = 'none';
}
function hideError() {
  errorMsg.classList.remove('active');
  errorMsg.textContent = '';
}
function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add('active');
  statusDot.className = 'readout-dot err';
}

async function predict(blobOrFile) {
  hideError();
  const formData = new FormData();
  formData.append('file', blobOrFile, 'input.png');

  try {
    const resp = await fetch(apiUrlInput.value, { method: 'POST', body: formData });
    if (!resp.ok) throw new Error(`server responded ${resp.status}`);
    const data = await resp.json();
    statusDot.className = 'readout-dot ok';
    renderResult(data);
  } catch (err) {
    showError(`Could not reach the API — ${err.message}. Check the endpoint above and confirm FastAPI is running with CORS enabled.`);
  }
}

function renderResult(data) {
  const label = data.predicted_class;
  glyphLg.textContent = label;
  glyphMd.textContent = label;
  glyphSm.textContent = label;
  classCode.textContent = label;
  confValue.textContent = `${(data.confidence * 100).toFixed(1)}%`;

  const top3 = data.top3 && data.top3.length
    ? data.top3
    : [{ class_name: data.predicted_class, confidence: data.confidence }];

  proofList.innerHTML = '';
  top3.forEach((item, i) => {
    const pct = (item.confidence * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = `proof-row rank-${i}`;
    row.innerHTML = `
      <div class="rank">${String(i + 1).padStart(2, '0')}</div>
      <div class="name-bar">
        <span class="name">${item.class_name}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      </div>
      <div class="pct">${pct}%</div>
    `;
    proofList.appendChild(row);
  });

  showResult();
}