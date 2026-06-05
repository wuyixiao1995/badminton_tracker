"""
check_tracknet.py — 可视化 TrackNet 检测结果

从 TrackNet 输出 JSON 中提取轨迹，在视频帧上叠加：
  - 红色圆圈: TrackNet 检测到的"球"位置
  - 红色轨迹线: 最近 N 帧的轨迹
  - 绿色虚线: 插值补齐的帧

用法: python check_tracknet.py
"""

import json, cv2, os, argparse
import numpy as np

JSON_PATH = "tracknet_output_track.json"
VIDEO_PATH = "test_match.mp4"
OUTPUT_DIR = "debug_tracknet"
SAMPLE_EVERY = 60  # 每60帧取一帧（每秒1帧 @ 60fps）


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=JSON_PATH)
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--every", type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load track data
    with open(args.json) as f:
        data = json.load(f)

    tracks = data["tracks"]
    print(f"Loaded {len(tracks)} frames, {data['detected_frames']} detected ({data['detection_rate']})")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_frames = list(range(0, len(tracks), args.every))

    # Stats
    total_checked = 0
    plausible = 0

    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        # postprocess uses scale=2, so output coords are in 1280x720 space
        scale_x = w / 1280.0
        scale_y = h / 720.0
        timestamp = frame_idx / fps

        vis = frame.copy()

        # Draw ball trajectory (last 30 frames)
        trail_len = 30
        points = []
        for i in range(max(0, frame_idx - trail_len), frame_idx + 1):
            t = tracks[i] if i < len(tracks) else None
            if t and t["detected"]:
                bx = int(t["x"] * scale_x)
                by = int(t["y"] * scale_y)
                points.append((bx, by, i == frame_idx))

        # Draw trail line
        for i in range(1, len(points)):
            cv2.line(vis, points[i-1][:2], points[i][:2], (0, 0, 255), 1)

        # Draw points along trail
        for bx, by, is_current in points:
            if is_current:
                # Current: big red circle + crosshair
                cv2.circle(vis, (bx, by), 10, (0, 0, 255), 2)
                cv2.circle(vis, (bx, by), 10, (0, 0, 255), -1)
                cv2.line(vis, (bx - 12, by), (bx + 12, by), (255, 255, 255), 1)
                cv2.line(vis, (bx, by - 12), (bx, by + 12), (255, 255, 255), 1)
            else:
                cv2.circle(vis, (bx, by), 3, (0, 0, 200), -1)

        # Current detection info
        t = tracks[frame_idx]
        if t["detected"]:
            bx, by = int(t["x"] * scale_x), int(t["y"] * scale_y)
            cv2.putText(vis, f"DETECTED ({bx},{by})", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Check if position is within court area (rough check)
            in_court = (w * 0.1 < bx < w * 0.9) and (h * 0.1 < by < h * 0.9)
            if in_court:
                plausible += 1
            total_checked += 1

        cv2.putText(vis, f"Frame: {frame_idx}  Time: {timestamp:.1f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Detection rate in this region
        nearby = tracks[max(0,frame_idx-30):min(len(tracks),frame_idx+30)]
        local_rate = sum(1 for t in nearby if t["detected"]) / len(nearby)
        cv2.putText(vis, f"Local detect rate: {100*local_rate:.0f}%",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, vis)

    cap.release()

    # ---- Summary ----
    print(f"\n{'='*50}")
    print(f"Generated {len(sample_frames)} debug frames in {args.output_dir}/")
    print(f"Plausible positions (in court area): {plausible}/{total_checked}")

    # Analyze trajectory continuity
    all_x = [t["x"] for t in tracks if t["detected"]]
    all_y = [t["y"] for t in tracks if t["detected"]]

    if all_x:
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        avg_x = np.mean(all_x)
        avg_y = np.mean(all_y)
        print(f"\nDetection position stats:")
        print(f"  X: mean={avg_x:.0f}, range=[{min(all_x):.0f}, {max(all_x):.0f}], span={x_range:.0f}")
        print(f"  Y: mean={avg_y:.0f}, range=[{min(all_y):.0f}, {max(all_y):.0f}], span={y_range:.0f}")
        print(f"  (TrackNet works in 640x360 space)")

        if x_range < 20 and y_range < 20:
            print(f"\n  [WARNING] Detections clustered in tiny area ({x_range:.0f}x{y_range:.0f})")
            print(f"  Likely fixating on static feature - NOT the shuttlecock.")
        elif x_range > 100 or y_range > 100:
            print(f"\n  [OK] Detections cover wide area ({x_range:.0f}x{y_range:.0f})")
            print(f"  Might be tracking actual movement.")
        else:
            print(f"\n  [?] Moderate spread ({x_range:.0f}x{y_range:.0f}). Need visual check.")

    # Segments analysis
    segments = []
    in_seg = False
    seg_start = 0
    for i, t in enumerate(tracks):
        if t["detected"] and not in_seg:
            seg_start = i
            in_seg = True
        elif not t["detected"] and in_seg:
            segments.append((seg_start, i - 1, i - seg_start))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(tracks) - 1, len(tracks) - seg_start))

    print(f"\nDetection segments: {len(segments)}")
    if segments:
        avg_len = np.mean([s[2] for s in segments])
        max_len = max(s[2] for s in segments)
        print(f"  Avg segment length: {avg_len:.0f} frames ({avg_len/fps:.1f}s)")
        print(f"  Max segment length: {max_len} frames ({max_len/fps:.1f}s)")
        # A good shuttlecock rally should have segments of 30-200 frames (0.5-3s)
        good_segs = [s for s in segments if 10 < s[2] < 300]
        print(f"  Segments 10-300 frames: {len(good_segs)}/{len(segments)}")

        if len(good_segs) > 3 and avg_len > 20:
            print(f"  [OK] Reasonable rally-like patterns detected")
        else:
            print(f"  [WARN]  Too many short segments — likely false positives")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()
