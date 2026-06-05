"""
visualize_tracknetv3.py — TrackNetV3 羽毛球检测结果全标注

在视频每一帧上标记：
  - 黄色圆圈虚线: 所有检测到的"球"位置（含置信度）
  - 圆越大 = 置信度越高
  - 虚线连接连续检测帧

用法: python visualize_tracknetv3.py
"""

import cv2, os, argparse
import numpy as np
import pandas as pd
from collections import deque

CSV_PATH = "tracknetv3_output_ball.csv"
VIDEO_PATH = "test_match.mp4"
OUTPUT_VIDEO = "tracknetv3_annotated.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--output", default=OUTPUT_VIDEO)
    parser.add_argument("--trail-len", type=int, default=30, help="Trajectory trail length in frames")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    n_detected = int(df['Visibility'].sum())
    print(f"CSV: {len(df)} frames, {n_detected} detected ({100*n_detected/len(df):.1f}%)")

    # Calculate per-frame "confidence" from detection continuity
    # A detection is more confident if it's part of a continuous segment
    confidences = [0.0] * len(df)
    segment_lengths = [0] * len(df)
    in_seg = False; seg_start = 0
    segments = []
    for i, v in enumerate(df['Visibility']):
        if v and not in_seg:
            seg_start = i; in_seg = True
        elif not v and in_seg:
            seg_len = i - seg_start
            for j in range(seg_start, i):
                segment_lengths[j] = seg_len
            segments.append((seg_start, i-1, seg_len))
            in_seg = False
    if in_seg:
        seg_len = len(df) - seg_start
        for j in range(seg_start, len(df)):
            segment_lengths[j] = seg_len
        segments.append((seg_start, len(df)-1, seg_len))

    # Confidence: longer continuous segment = more confident
    # Also check speed: too slow or too fast = less confident
    for i in range(len(df)):
        if df.iloc[i]['Visibility']:
            seg_len = segment_lengths[i]
            # Base confidence from segment length (max at 30+ frames)
            len_conf = min(1.0, seg_len / 30.0)

            # Speed check
            if i > 0 and df.iloc[i-1]['Visibility']:
                dx = df.iloc[i]['X'] - df.iloc[i-1]['X']
                dy = df.iloc[i]['Y'] - df.iloc[i-1]['Y']
                spd = np.sqrt(dx*dx + dy*dy)
                # Very slow (<2px) or very fast (>400px) = lower confidence
                if spd < 2:
                    speed_conf = 0.3  # probably static false positive
                elif spd > 400:
                    speed_conf = 0.5  # probably outlier
                else:
                    speed_conf = min(1.0, spd / 20.0)  # 20px/frame is good
            else:
                speed_conf = 0.5

            confidences[i] = 0.5 * len_conf + 0.5 * speed_conf

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    max_frames = min(args.max_frames if args.max_frames > 0 else total, len(df))
    print(f"Video: {w}x{h} @ {fps:.1f}fps")
    print(f"Rendering {max_frames} frames to {args.output}...")

    # Trail history as deque of (x, y, confidence)
    trail = deque(maxlen=args.trail_len)

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()
        r = df.iloc[frame_idx]
        is_detected = bool(r['Visibility'])
        conf = confidences[frame_idx]

        # Add current position to trail
        if is_detected:
            trail.appendleft((int(r['X']), int(r['Y']), conf))
        else:
            trail.appendleft(None)

        # ---- Draw trajectory (dashed lines + circles) ----
        # First collect valid trail points
        valid_trail = [(p[0], p[1], p[2]) for p in trail if p is not None]

        # Draw dashed lines between consecutive detections
        for i in range(1, len(valid_trail)):
            x1, y1, c1 = valid_trail[i-1]
            x2, y2, c2 = valid_trail[i]

            # Dashed line: draw small segments
            dx = x2 - x1; dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1:
                continue
            ux, uy = dx/length, dy/length
            dash_len = 4; gap_len = 6
            pos = 0
            while pos < length:
                seg_end = min(pos + dash_len, length)
                sx1 = int(x1 + ux*pos)
                sy1 = int(y1 + uy*pos)
                sx2 = int(x1 + ux*seg_end)
                sy2 = int(y1 + uy*seg_end)
                # Color fade from old to new
                alpha = 1.0 - i / args.trail_len
                color = (0, int(255*alpha), int(255*alpha))  # yellow fading
                cv2.line(vis, (sx1, sy1), (sx2, sy2), color, 1, cv2.LINE_AA)
                pos += dash_len + gap_len

        # Draw circles for each trail point
        for i, (px, py, c) in enumerate(valid_trail):
            is_current = (i == 0)
            alpha = 1.0 - i / args.trail_len

            if is_current:
                # Current detection: large circle, size proportional to confidence
                radius = int(4 + c * 8)  # 4-12 px
                # Outer ring
                cv2.circle(vis, (px, py), radius + 2, (0, 255, 255), 2)
                # Filled circle
                cv2.circle(vis, (px, py), radius, (0, 200, 255), -1)
                # Crosshair
                cv2.line(vis, (px - radius - 4, py), (px + radius + 4, py), (0, 255, 255), 1)
                cv2.line(vis, (px, py - radius - 4), (px, py + radius + 4), (0, 255, 255), 1)
            else:
                # Past detection: small circle, fading
                r_small = max(1, int(3 * alpha))
                color = (0, int(200 * alpha), int(255 * alpha))
                cv2.circle(vis, (px, py), r_small, color, -1)

        # ---- Info Panel ----
        # Background
        overlay = vis.copy()
        cv2.rectangle(overlay, (5, 5), (300, 120), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)

        cv2.putText(vis, f"Frame: {frame_idx}  {frame_idx/fps:.1f}s",
                   (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if is_detected:
            cv2.putText(vis, f"DETECT ({int(r['X'])},{int(r['Y'])}) conf={conf:.2f}",
                       (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(vis, "NO DETECTION",
                       (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Segment info
        seg_len = segment_lengths[frame_idx]
        cv2.putText(vis, f"Segment len: {seg_len:.0f}f",
                   (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        # Local detection rate
        start_i, end_i = max(0, frame_idx - 60), min(len(df), frame_idx + 60)
        local_df = df.iloc[start_i:end_i]
        local_rate = local_df['Visibility'].sum() / len(local_df)
        cv2.putText(vis, f"Detect rate (+-1s): {100*local_rate:.0f}%",
                   (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        # Global stats
        detected_so_far = df.iloc[:frame_idx+1]['Visibility'].sum()
        cv2.putText(vis, f"Overall: {detected_so_far}/{frame_idx+1} ({100*detected_so_far/(frame_idx+1):.0f}%)",
                   (12, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        # Legend
        cv2.putText(vis, "Circle=detection / Dashed line=trajectory / Size=confidence",
                   (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        out.write(vis)

        if frame_idx % 300 == 0 and frame_idx > 0:
            print(f"  {frame_idx}/{max_frames}")

    cap.release()
    out.release()
    print(f"\nDone! Output: {args.output}")

    # Quick stats
    print(f"\nDetection summary:")
    print(f"  Total segments: {len(segments)}")
    if segments:
        lens = [s[2] for s in segments]
        print(f"  Avg segment: {np.mean(lens):.0f}f ({np.mean(lens)/fps:.1f}s)")
        print(f"  Long segments (>30f): {sum(1 for l in lens if l > 30)}")
    print(f"  High-confidence detections (conf>0.6): {sum(1 for c in confidences if c > 0.6)}")


if __name__ == "__main__":
    main()
