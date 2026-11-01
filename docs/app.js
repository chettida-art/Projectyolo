// ทายรูปด้วย YOLO Polygon (Segmentation) ในเบราว์เซอร์ ด้วย onnxruntime-web
// ใช้กับ GitHub Pages ได้ เพราะไม่ต้องมี server Python
// โมเดล: docs/model/best.onnx (สร้างด้วย webapp/export_web.py)

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const IOU_THRES = 0.7;   // ค่าเดียวกับ ultralytics
const MAX_DET = 300;
const MAX_DRAW_SIDE = 1600; // ย่อรูปที่ใหญ่มากก่อนวาด
// สีเส้น polygon ของแต่ละ class (ตรงกับเว็บ Flask)
const PALETTE = ["rgb(230,40,40)", "rgb(20,120,230)", "rgb(160,40,200)", "rgb(255,140,30)",
                 "rgb(40,170,60)", "rgb(0,90,180)", "rgb(200,30,90)"];
const colorOf = c => PALETTE[c % PALETTE.length];

const $ = id => document.getElementById(id);
const drop = $("drop"), fileInput = $("file"), btn = $("btn");
let currentFile = null, session = null, names = [], imgsz = 640, stride = 32;

// ---------- โหลดโมเดล ----------
(async () => {
  try {
    const meta = await (await fetch("model/meta.json")).json();
    names = meta.names; imgsz = meta.imgsz; stride = meta.stride || 32;
    $("classes").innerHTML = names.map(n => `<span class="chip">${n}</span>`).join("");
    session = await ort.InferenceSession.create("model/best.onnx", { executionProviders: ["wasm"] });
    $("status").textContent = "โมเดลพร้อมใช้งาน";
    btn.disabled = !currentFile;
  } catch (e) {
    $("status").textContent = "";
    $("err").textContent = "โหลดโมเดลไม่สำเร็จ: " + e.message;
  }
})();

// ---------- เลือกรูป / ลากวาง / Ctrl+V ----------
$("conf").oninput = e => $("confVal").textContent = e.target.value;
drop.onclick = () => fileInput.click();
fileInput.onchange = () => setFile(fileInput.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); setFile(e.dataTransfer.files[0]); };

// วางรูปที่ copy มา (Ctrl+V) ได้ทั้งหน้า แล้วทายให้ทันที
document.addEventListener("paste", e => {
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/"));
  if (!item) {
    $("err").textContent = "คลิปบอร์ดไม่มีรูปภาพ (ลองคลิกขวาที่รูปแล้วเลือก Copy image)";
    return;
  }
  e.preventDefault();
  const blob = item.getAsFile();
  const ext = blob.type.split("/")[1] || "png";
  if (setFile(new File([blob], `pasted.${ext}`, { type: blob.type }))) predict();
});

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return false;
  currentFile = f;
  $("preview").innerHTML = `<img src="${URL.createObjectURL(f)}"><span>${f.name}</span>`;
  btn.disabled = !session;
  $("err").textContent = "";
  return true;
}

btn.onclick = () => predict();

async function predict() {
  if (!currentFile) return;
  if (!session) { $("err").textContent = "รอโหลดโมเดลสักครู่"; return; }
  btn.disabled = true; btn.textContent = "กำลังทาย...";
  $("err").textContent = "";
  try {
    const img = await loadImage(currentFile);
    const t0 = performance.now();
    const objects = await runModel(img, parseFloat($("conf").value));
    $("status").textContent = `ใช้เวลา ${((performance.now() - t0) / 1000).toFixed(2)} วินาที`;
    render({ image: drawPolygons(img, objects), detections: toDetections(objects) });
  } catch (e) {
    console.error(e);
    $("err").textContent = "ไฟล์นี้ไม่ใช่รูปภาพที่รองรับ หรือเกิดข้อผิดพลาด: " + e.message;
  } finally {
    btn.disabled = false; btn.textContent = "Predict";
  }
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("เปิดรูปไม่ได้"));
    img.src = URL.createObjectURL(file);
  });
}

// ---------- รันโมเดล ----------
async function runModel(img, conf) {
  const W = img.naturalWidth, H = img.naturalHeight;

  // letterbox แบบเดียวกับ ultralytics: ย่อให้ด้านยาวเท่า imgsz แล้วเติมขอบสีเทา (114)
  // ให้ด้านกว้าง/สูงหาร stride ลงตัว (รูปสี่เหลี่ยมผืนผ้า เช่น 640x384) ผลจึงตรงกับ Python
  const r = Math.min(imgsz / W, imgsz / H);
  const nw = Math.round(W * r), nh = Math.round(H * r);
  const dw = ((imgsz - nw) % stride) / 2, dh = ((imgsz - nh) % stride) / 2;
  const padX = Math.round(dw - 0.1), padY = Math.round(dh - 0.1);
  const inW = nw + padX + Math.round(dw + 0.1), inH = nh + padY + Math.round(dh + 0.1);
  const cv = document.createElement("canvas");
  cv.width = inW; cv.height = inH;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, inW, inH);
  ctx.drawImage(img, padX, padY, nw, nh);
  const px = ctx.getImageData(0, 0, inW, inH).data;
  const area = inW * inH, input = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    input[i] = px[i * 4] / 255;
    input[area + i] = px[i * 4 + 1] / 255;
    input[2 * area + i] = px[i * 4 + 2] / 255;
  }

  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, inH, inW]) });
  const pred = out[session.outputNames[0]];   // [1, 4 + nc + 32, N]
  const proto = out[session.outputNames[1]];  // [1, 32, mh, mw]
  const keep = decodeWithNms(pred, names.length, proto.dims[1], conf);

  const lb = { r, padX, padY, nw, nh, inW, inH };
  return keep.map(k => ({
    cls_id: k.cls,
    label: names[k.cls],
    confidence: k.score,
    box: k.box.map((v, j) => (v - (j % 2 ? padY : padX)) / r),
    polygons: maskToPolygons(k, proto, lb),
  }));
}

// แปลงผลโมเดล [1, 4 + nc + 32, N] (cx cy w h) แล้วทำ NMS แยกตาม class
function decodeWithNms(pred, nc, nm, conf) {
  const N = pred.dims[2], p = pred.data, cands = [];
  for (let i = 0; i < N; i++) {
    let best = -1, score = conf;
    for (let c = 0; c < nc; c++) {
      const s = p[(4 + c) * N + i];
      if (s >= score) { score = s; best = c; }
    }
    if (best < 0) continue;
    const cx = p[i], cy = p[N + i], w = p[2 * N + i], h = p[3 * N + i];
    const coef = new Float32Array(nm);
    for (let k = 0; k < nm; k++) coef[k] = p[(4 + nc + k) * N + i];
    cands.push({ cls: best, score, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], coef });
  }
  cands.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cands) {
    if (keep.length >= MAX_DET) break;
    if (keep.every(k => k.cls !== c.cls || iou(k.box, c.box) <= IOU_THRES)) keep.push(c);
  }
  return keep;
}

function iou(a, b) {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
}

// ---------- mask -> polygon ----------
// ทำแบบเดียวกับเว็บ Flask: หาขอบของแต่ละก้อนแยกกัน ไม่ลากเส้นเชื่อมข้ามรูป
// ทิ้งเศษชิ้นเล็ก แล้วลดจำนวนจุดให้เส้นเรียบ
function maskToPolygons(det, proto, lb) {
  const nm = proto.dims[1], mh = proto.dims[2], mw = proto.dims[3], pd = proto.data;
  const sx = mw / lb.inW, sy = mh / lb.inH;

  // กรอบของวัตถุ (ในพิกัดรูปที่เข้าโมเดล) ตัดไม่ให้เกินส่วนที่เป็นรูปจริง
  const x0 = Math.max(Math.floor(det.box[0]), lb.padX), y0 = Math.max(Math.floor(det.box[1]), lb.padY);
  const x1 = Math.min(Math.ceil(det.box[2]), lb.padX + lb.nw), y1 = Math.min(Math.ceil(det.box[3]), lb.padY + lb.nh);
  const w = x1 - x0, h = y1 - y0;
  if (w < 2 || h < 2) return [];

  // คำนวณ mask ความละเอียดต่ำ (เฉพาะช่วงที่ครอบกรอบ) = sigmoid(coef · proto)
  const gx0 = Math.max(0, Math.floor(x0 * sx) - 1), gx1 = Math.min(mw - 1, Math.ceil(x1 * sx) + 1);
  const gy0 = Math.max(0, Math.floor(y0 * sy) - 1), gy1 = Math.min(mh - 1, Math.ceil(y1 * sy) + 1);
  const gw = gx1 - gx0 + 1, gh = gy1 - gy0 + 1, low = new Float32Array(gw * gh);
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      const off = (gy + gy0) * mw + gx + gx0;
      let s = 0;
      for (let k = 0; k < nm; k++) s += det.coef[k] * pd[k * mh * mw + off];
      low[gy * gw + gx] = 1 / (1 + Math.exp(-s));
    }
  }

  // ขยาย mask แบบ bilinear มาที่ขนาดรูปที่เข้าโมเดล แล้วตัดที่ 0.5
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(Math.max((y + y0 + 0.5) * sy - 0.5 - gy0, 0), gh - 1);
    const ya = Math.floor(fy), yb = Math.min(ya + 1, gh - 1), ty = fy - ya;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(Math.max((x + x0 + 0.5) * sx - 0.5 - gx0, 0), gw - 1);
      const xa = Math.floor(fx), xb = Math.min(xa + 1, gw - 1), tx = fx - xa;
      const v = (low[ya * gw + xa] * (1 - tx) + low[ya * gw + xb] * tx) * (1 - ty)
              + (low[yb * gw + xa] * (1 - tx) + low[yb * gw + xb] * tx) * ty;
      mask[y * w + x] = v > 0.5 ? 1 : 0;
    }
  }

  // แยกก้อน (connected components แบบ 8 ทิศ)
  const labels = new Int32Array(w * h), sizes = [0], starts = [0];
  const stack = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (!mask[i] || labels[i]) continue;
    const id = sizes.length;
    let top = 0, n = 0;
    stack[top++] = i; labels[i] = id;
    while (top) {
      const j = stack[--top], jx = j % w, jy = (j - jx) / w;
      n++;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = jx + dx, ny = jy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q = ny * w + nx;
        if (mask[q] && !labels[q]) { labels[q] = id; stack[top++] = q; }
      }
    }
    sizes.push(n); starts.push(i); // i คือพิกเซลแรกของก้อนตามลำดับสแกน (บนสุด-ซ้ายสุด)
  }
  if (sizes.length === 1) return [];

  // ทิ้งเศษชิ้นเล็ก (เล็กกว่า 5% ของชิ้นใหญ่สุด)
  const largest = Math.max(...sizes);
  const ids = sizes.map((s, id) => id).filter(id => id && sizes[id] >= Math.max(0.05 * largest, 30))
                   .sort((a, b) => sizes[b] - sizes[a]);

  const polygons = [];
  for (const id of ids) {
    const contour = traceContour(labels, w, h, id, starts[id]);
    if (contour.length < 3) continue;
    let perim = 0;
    for (let i = 0; i < contour.length; i++) {
      const a = contour[i], b = contour[(i + 1) % contour.length];
      perim += Math.hypot(a[0] - b[0], a[1] - b[1]);
    }
    const simple = simplify(contour, Math.max(0.003 * perim, 1));
    if (simple.length < 3) continue;
    // แปลงกลับเป็นพิกัดของรูปต้นฉบับ
    polygons.push(simple.map(([x, y]) => [(x + x0 - lb.padX) / lb.r, (y + y0 - lb.padY) / lb.r]));
  }
  return polygons;
}

// เดินตามขอบด้านนอกของก้อน (Moore-neighbor tracing)
const DIRS = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]];
function traceContour(labels, w, h, id, start) {
  const fg = (x, y) => x >= 0 && y >= 0 && x < w && y < h && labels[y * w + x] === id;
  const sx = start % w, sy = (start - sx) / w;
  const pts = [[sx, sy]];
  let cx = sx, cy = sy, bx = sx - 1, by = sy;
  let second = null;
  const limit = 4 * w * h + 8;
  for (let step = 0; step < limit; step++) {
    const bi = DIRS.findIndex(([dx, dy]) => dx === bx - cx && dy === by - cy);
    let moved = false;
    for (let k = 1; k <= 8; k++) {
      const d = (bi + k) % 8, nx = cx + DIRS[d][0], ny = cy + DIRS[d][1];
      if (fg(nx, ny)) {
        const pd = (d + 7) % 8;
        bx = cx + DIRS[pd][0]; by = cy + DIRS[pd][1];
        const px = cx, py = cy;
        cx = nx; cy = ny; moved = true;
        if (!second) second = [cx, cy];
        else if (px === sx && py === sy && cx === second[0] && cy === second[1]) { pts.pop(); return pts; }
        pts.push([cx, cy]);
        break;
      }
    }
    if (!moved) break; // จุดเดี่ยว
  }
  return pts;
}

// ลดจำนวนจุดแบบ Douglas-Peucker (ปิดรูป)
function simplify(pts, eps) {
  if (pts.length < 4) return pts;
  // แบ่งรูปปิดเป็น 2 เส้นที่จุดที่ไกลจากจุดแรกที่สุด
  let far = 0, fd = -1;
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - pts[0][0], pts[i][1] - pts[0][1]);
    if (d > fd) { fd = d; far = i; }
  }
  const a = rdp(pts.slice(0, far + 1), eps), b = rdp(pts.slice(far).concat([pts[0]]), eps);
  return a.slice(0, -1).concat(b.slice(0, -1));
}

function rdp(pts, eps) {
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [s, e] = stack.pop();
    const [ax, ay] = pts[s], [bx, by] = pts[e];
    const len = Math.hypot(bx - ax, by - ay);
    let idx = -1, dmax = eps;
    for (let i = s + 1; i < e; i++) {
      const [x, y] = pts[i];
      const d = len ? Math.abs((bx - ax) * (ay - y) - (ax - x) * (by - ay)) / len : Math.hypot(x - ax, y - ay);
      if (d > dmax) { dmax = d; idx = i; }
    }
    if (idx >= 0) { keep[idx] = 1; stack.push([s, idx], [idx, e]); }
  }
  return pts.filter((_, i) => keep[i]);
}

// ---------- วาดผล ----------
function drawPolygons(img, objects) {
  const W = img.naturalWidth, H = img.naturalHeight;
  const s = Math.min(1, MAX_DRAW_SIDE / Math.max(W, H));
  const cw = Math.round(W * s), ch = Math.round(H * s);
  const canvas = document.createElement("canvas");
  canvas.width = cw; canvas.height = ch;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, cw, ch);

  const thickness = Math.max(2, Math.round(Math.min(cw, ch) / 250));
  const fontPx = Math.max(13, Math.round(Math.min(cw, ch) / 900 * 22));
  const objs = objects.filter(o => o.polygons.length).sort((a, b) => a.confidence - b.confidence);
  const path = poly => { ctx.beginPath(); poly.forEach(([x, y], i) => i ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)); ctx.closePath(); };

  // ระบายสีโปร่งใสเล็กน้อยภายใน polygon (วาดทึบบนอีก layer แล้วค่อยผสม 25%)
  const fill = document.createElement("canvas");
  fill.width = cw; fill.height = ch;
  const fctx = fill.getContext("2d");
  for (const o of objs) {
    fctx.fillStyle = colorOf(o.cls_id);
    for (const poly of o.polygons) {
      fctx.beginPath(); poly.forEach(([x, y], i) => i ? fctx.lineTo(x * s, y * s) : fctx.moveTo(x * s, y * s)); fctx.closePath(); fctx.fill();
    }
  }
  ctx.globalAlpha = 0.25; ctx.drawImage(fill, 0, 0); ctx.globalAlpha = 1;

  // เส้นขอบ + จุดยอด
  ctx.lineWidth = thickness; ctx.lineJoin = "round";
  for (const o of objs) {
    ctx.strokeStyle = ctx.fillStyle = colorOf(o.cls_id);
    for (const poly of o.polygons) {
      path(poly); ctx.stroke();
      for (const [x, y] of poly) { ctx.beginPath(); ctx.arc(x * s, y * s, thickness, 0, Math.PI * 2); ctx.fill(); }
    }
  }

  // ป้ายชื่อวางไว้ที่จุดบนสุดของส่วนที่ใหญ่ที่สุด ถ้าทับกับป้ายอื่นให้เลื่อนลง
  ctx.font = `600 ${fontPx}px Sarabun, sans-serif`;
  ctx.textBaseline = "bottom";
  const placed = [];
  for (const o of [...objs].sort((a, b) => b.confidence - a.confidence)) {
    const label = `${o.label} ${o.confidence.toFixed(2)}`;
    const top = o.polygons[0].reduce((m, p) => p[1] < m[1] ? p : m);
    const w = Math.ceil(ctx.measureText(label).width) + 10, h = fontPx + 8;
    let x = Math.round(Math.min(Math.max(top[0] * s - w / 2, 0), cw - w));
    let y = Math.round(Math.max(top[1] * s - 4, h));
    while (placed.some(([px, py, pw, ph]) => x < px + pw && px < x + w && y - h < py && py - ph < y)) y += h + 2;
    y = Math.min(y, ch);
    placed.push([x, y, w, h]);
    ctx.fillStyle = colorOf(o.cls_id);
    ctx.fillRect(x, y - h, w, h);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 5, y - 3);
  }
  return canvas.toDataURL("image/jpeg", 0.9);
}

function toDetections(objects) {
  return objects.map(o => ({
    label: o.label,
    confidence: o.confidence,
    polygon_points: o.polygons.reduce((n, p) => n + p.length, 0),
  })).sort((a, b) => b.confidence - a.confidence);
}

function render(data) {
  const d = data.detections;
  let html = `<img src="${data.image}" alt="ผลลัพธ์">`;
  if (!d.length) {
    html += `<p class="top">ไม่พบวัตถุที่รู้จัก (ลองลด Confidence ลง)</p>`;
  } else {
    html += `<p class="top">โมเดลทายว่าเป็น: <b>${d[0].label}</b> (${(d[0].confidence * 100).toFixed(1)}%) — พบทั้งหมด ${d.length} วัตถุ</p>`;
    html += `<table><tr><th>#</th><th>Class</th><th>Confidence</th><th>จุด Polygon</th></tr>`;
    d.forEach((x, i) => {
      const p = (x.confidence * 100).toFixed(1);
      html += `<tr><td>${i + 1}</td><td>${x.label}</td>
        <td><div style="display:flex;gap:8px;align-items:center"><div class="bar" style="flex:1"><span style="width:${p}%"></span></div>${p}%</div></td>
        <td>${x.polygon_points}</td></tr>`;
    });
    html += `</table>`;
  }
  $("result").innerHTML = html;
}
