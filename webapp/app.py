"""Web app สำหรับทดลองทายรูปด้วยโมเดล YOLO Segmentation (Polygon weight)

รัน:  C:\\ml_focus\\python.exe webapp\\app.py
แล้วเปิด http://127.0.0.1:5000
"""
import base64
import io
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS = BASE_DIR.parent / "runs" / "segment" / "stage1" / "weights" / "best.pt"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # จำกัดไฟล์ 20MB

# โหลด Polygon weight ตอนเริ่ม server และโหลดใหม่เองเมื่อไฟล์ best.pt ถูกเทรนทับ
model = YOLO(str(WEIGHTS))
model_mtime = WEIGHTS.stat().st_mtime


def refresh_model():
    """ถ้า best.pt เปลี่ยน (เทรนรอบใหม่) ให้โหลด weight ตัวใหม่โดยไม่ต้องรีสตาร์ท server"""
    global model, model_mtime
    mtime = WEIGHTS.stat().st_mtime
    if mtime == model_mtime:
        return
    try:
        model = YOLO(str(WEIGHTS))
        model_mtime = mtime
        print(f"Reloaded new weights: {WEIGHTS}", flush=True)
    except Exception as e:  # ไฟล์อาจกำลังถูกเขียนอยู่ ใช้ตัวเดิมไปก่อน
        print(f"Could not reload weights yet: {e}", flush=True)

# สีเส้น polygon ของแต่ละ class (BGR) เลือกให้เข้มพอที่ตัวหนังสือสีขาวบนป้ายอ่านออก
PALETTE = [(40, 40, 230), (230, 120, 20), (200, 40, 160), (30, 140, 255),
           (60, 170, 40), (180, 90, 0), (90, 30, 200)]


def colors(cls_id):
    return PALETTE[cls_id % len(PALETTE)]


def mask_to_polygons(mask):
    """แปลง mask ของวัตถุ 1 ชิ้นเป็นรายการ polygon แยกตามส่วน

    masks.xy ของ ultralytics เอาทุกส่วนของ mask มาต่อกันเป็นเส้นเดียว
    ถ้าวัตถุถูกบังจนแยกเป็นหลายก้อน จะเกิดเส้นตรงลากข้ามรูปมั่วๆ
    จึงหา contour เองแล้วเก็บแต่ละส่วนแยกกัน
    """
    mask = (mask > 0.5).astype(np.uint8)
    # ปิดรูเล็กๆ และลบจุด noise ก่อนหาขอบ
    k = max(3, (min(mask.shape) // 200) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    largest = max(cv2.contourArea(c) for c in contours)
    polygons = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        # ทิ้งชิ้นเล็กที่เป็นเศษ (เล็กกว่า 5% ของชิ้นใหญ่สุด)
        if cv2.contourArea(c) < max(0.05 * largest, 30):
            continue
        poly = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True)
        if len(poly) >= 3:
            polygons.append(poly)
    return polygons


def extract_objects(result):
    """รวมข้อมูลของแต่ละวัตถุ: class, ความมั่นใจ, กรอบ และ polygon"""
    objects = []
    if result.boxes is None:
        return objects
    masks = result.masks.data.cpu().numpy() if result.masks is not None else [None] * len(result.boxes)
    for box, mask in zip(result.boxes, masks):
        cls_id = int(box.cls)
        objects.append({
            "cls_id": cls_id,
            "label": model.names[cls_id],
            "confidence": float(box.conf),
            "box": box.xyxy[0].tolist(),
            "polygons": [] if mask is None else mask_to_polygons(mask),
        })
    return objects


def draw_polygons(bgr_img, objects):
    """วาดเส้น Polygon ของแต่ละวัตถุ (แทนกรอบสี่เหลี่ยมและ mask ทึบ)"""
    canvas = bgr_img.copy()
    overlay = canvas.copy()
    thickness = max(2, round(min(canvas.shape[:2]) / 250))
    font_scale = max(0.5, min(canvas.shape[:2]) / 900)

    # วาดวัตถุที่มั่นใจน้อยก่อน ให้ตัวที่มั่นใจมากอยู่ด้านบน
    objects = sorted((o for o in objects if o["polygons"]), key=lambda o: o["confidence"])
    for obj in objects:
        cv2.fillPoly(overlay, obj["polygons"], colors(obj["cls_id"]))

    # ระบายสีโปร่งใสเล็กน้อยภายใน polygon แล้วค่อยวาดเส้นขอบ + จุดยอด
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
    for obj in objects:
        color = colors(obj["cls_id"])
        cv2.polylines(canvas, obj["polygons"], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        for poly in obj["polygons"]:
            for x, y in poly[:, 0]:
                cv2.circle(canvas, (int(x), int(y)), thickness, color, -1, cv2.LINE_AA)

    # ป้ายชื่อวางไว้ที่จุดบนสุดของส่วนที่ใหญ่ที่สุด ถ้าทับกับป้ายอื่นให้เลื่อนลง
    placed = []
    for obj in sorted(objects, key=lambda o: -o["confidence"]):
        color = colors(obj["cls_id"])
        label = f"{obj['label']} {obj['confidence']:.2f}"
        pts = obj["polygons"][0]
        x, y = pts[pts[:, 0, 1].argmin(), 0]
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        w, h = tw + 6, th + base + 6
        x = int(min(max(x - w // 2, 0), canvas.shape[1] - w))
        y = int(max(y - 4, h))
        while any(x < px + pw and px < x + w and y - h < py and py - ph < y for px, py, pw, ph in placed):
            y += h + 2
        y = min(y, canvas.shape[0])
        placed.append((x, y, w, h))
        cv2.rectangle(canvas, (x, y - h), (x + w, y), color, -1)
        cv2.putText(canvas, label, (x + 3, y - base - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
    return canvas


def to_data_url(bgr_img):
    ok, buf = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@app.route("/")
def index():
    refresh_model()
    return render_template("index.html", names=list(model.names.values()))


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="กรุณาเลือกรูปภาพ"), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify(error="ไฟล์นี้ไม่ใช่รูปภาพที่รองรับ"), 400

    refresh_model()
    conf = float(request.form.get("conf", 0.25))
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # retina_masks=True ให้ mask มีขนาดเท่ารูปจริง ขอบ polygon จะละเอียดกว่า
    result = model.predict(bgr, conf=conf, retina_masks=True, verbose=False)[0]
    objects = extract_objects(result)

    detections = [{
        "label": o["label"],
        "confidence": round(o["confidence"], 4),
        "box": [round(v, 1) for v in o["box"]],
        "polygon_points": sum(len(p) for p in o["polygons"]),
    } for o in objects]
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return jsonify(
        image=to_data_url(draw_polygons(bgr, objects)),
        detections=detections,
        size=[img.width, img.height],
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
