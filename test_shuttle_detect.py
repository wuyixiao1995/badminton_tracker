"""
test_shuttle_detect.py — 测试 YOLOv8 是否能检测到羽毛球

用法: python test_shuttle_detect.py

用 YOLOv8 的对象检测模型 (yolov8n.pt / yolov8s.pt) 检测 "sports ball" 类别
同时也尝试用帧间差分法找快速移动的小物体
"""

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "test_match.mp4"
FRAMES_TO_TEST = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540]


def method1_yolo_detection(frame, model, conf_threshold=0.1):
    """方法1: YOLOv8 对象检测 - 找 sports ball (class 32)"""
    results = model(frame, verbose=False)
    detections = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                name = model.names.get(cls_id, str(cls_id))
                if conf > conf_threshold:
                    detections.append({
                        "class_id": cls_id,
                        "class_name": name,
                        "confidence": conf,
                        "bbox": [int(x) for x in xyxy],
                        "center": (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)),
                    })
    return detections


def method2_motion_detection(frame, prev_frame, min_area=5, max_area=200, threshold=25):
    """方法2: 帧间差分法 - 找快速移动的小物体"""
    if prev_frame is None:
        return []

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # 膨胀连接碎片
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                candidates.append({"center": (cx, cy), "area": area})

    return candidates


def method3_white_blob_detection(frame, min_area=3, max_area=100):
    """方法3: 白色小物体检测 (羽毛球通常是白色的)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 白色: 低饱和度 + 高亮度
    lower = np.array([0, 0, 200])
    upper = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                candidates.append({"center": (cx, cy), "area": area})

    return candidates


def main():
    print("=" * 70)
    print("羽毛球检测测试 — 对比 3 种方法")
    print("=" * 70)

    # 加载 YOLO 检测模型 (不是pose, 是detect)
    print("\n[INFO] Loading YOLOv8n (detect)...")
    model_detect = YOLO("yolov8n.pt")
    print(f"  COCO classes: {len(model_detect.names)}")
    sports_ball_id = None
    for k, v in model_detect.names.items():
        if "ball" in v.lower() or "sport" in v.lower():
            print(f"  Class {k}: {v}")
            if v == "sports ball":
                sports_ball_id = k

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nVideo: {width}x{height} @ {fps}fps, {total_frames} frames\n")

    os.makedirs("debug_shuttle", exist_ok=True)

    prev_frame = None
    hit_count_yolo = 0
    hit_count_motion = 0
    hit_count_white = 0
    tested = 0

    for target_frame in FRAMES_TO_TEST:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"  Frame {target_frame}: READ FAILED")
            continue

        tested += 1
        timestamp = target_frame / fps

        # 下采样加快检测
        scale = 1.0
        if max(width, height) > 1280:
            scale = 1280 / max(width, height)
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small_frame = frame

        # ---- 方法1: YOLO detect ----
        yolo_dets = method1_yolo_detection(small_frame, model_detect, conf_threshold=0.05)
        # 筛选出 ball 相关
        ball_dets = [d for d in yolo_dets if d["class_name"] == "sports ball"]

        # ---- 方法2: 帧间差分 ----
        motion_cands = method2_motion_detection(small_frame, prev_frame)

        # ---- 方法3: 白色小物体 ----
        white_cands = method3_white_blob_detection(small_frame)

        # ---- 绘制 ----
        vis = frame.copy()
        y_offset = 30

        # 方法1: YOLO sports ball — 绿色
        for d in ball_dets:
            x1, y1, x2, y2 = [int(v / scale) for v in d["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = int(d["center"][0] / scale), int(d["center"][1] / scale)
            cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(vis, f"BALL {d['confidence']:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 方法2: 运动检测 — 红色
        for c in motion_cands[:10]:
            cx = int(c["center"][0] / scale)
            cy = int(c["center"][1] / scale)
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), 1)

        # 方法3: 白色小物体 — 蓝色
        for c in white_cands[:10]:
            cx = int(c["center"][0] / scale)
            cy = int(c["center"][1] / scale)
            cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

        # Legend
        cv2.putText(vis, "GREEN=YOLO ball  RED=motion  BLUE=white blob",
                   (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(vis, f"Frame {target_frame} | {timestamp:.1f}s",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 所有 YOLO 检测到的物体
        other_texts = [f"{d['class_name']}:{d['confidence']:.2f}" for d in yolo_dets if d["class_name"] != "sports ball"]
        if other_texts:
            cv2.putText(vis, "Other: " + ", ".join(other_texts[:5]),
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        out_path = f"debug_shuttle/frame_{target_frame:04d}.png"
        cv2.imwrite(out_path, vis)

        if ball_dets:
            hit_count_yolo += 1
            print(f"  Frame {target_frame} ({timestamp:.1f}s): YOLO sports ball={len(ball_dets)}, "
                  f"motion={len(motion_cands)}, white={len(white_cands)} ★ BALL!")
        else:
            print(f"  Frame {target_frame} ({timestamp:.1f}s): YOLO sports ball=0, "
                  f"motion={len(motion_cands)}, white={len(white_cands)}")

        prev_frame = small_frame

    cap.release()

    print(f"\n{'='*70}")
    print(f"Summary ({tested} frames tested):")
    print(f"  YOLO sports ball detected: {hit_count_yolo}/{tested} frames")
    print(f"  Motion candidates per frame: avg ~{hit_count_motion/max(tested,1):.1f}")
    print(f"  White blob candidates per frame: avg ~{hit_count_white/max(tested,1):.1f}")
    print(f"\nOutput images saved to debug_shuttle/")
    print(f"{'='*70}")

    # 建议
    print("\n[建议]")
    if hit_count_yolo > 0:
        print(f"  ✅ YOLOv8 能检测到 'sports ball' ({hit_count_yolo}/{tested})")
        print(f"  可以考虑使用 YOLO detect + sports ball 类别来跟踪羽毛球")
    else:
        print(f"  ❌ YOLOv8 COCO 模型检测不到羽毛球 ({hit_count_yolo}/{tested})")
        print(f"  方案A: 用 TrackNet 专门做球类追踪")
        print(f"  方案B: 用帧间差分 + 运动预测 (卡尔曼滤波)")
        print(f"  方案C: 微调 YOLOv8 在羽毛球数据集上 (如 Shuttlecock Dataset)")
        print(f"  方案D: 训练一个轻量球检测器 (几 MB 的小模型)")


if __name__ == "__main__":
    import os
    main()
