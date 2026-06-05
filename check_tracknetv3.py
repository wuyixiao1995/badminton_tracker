"""
check_tracknetv3.py — 可视化 TrackNetV3 检测结果

在每个采样帧上叠加：
  - 黄色圆圈 + 十字: 当前帧检测到的球位置
  - 黄色轨迹线: 最近 N 帧的球轨迹
  - 红色段: 高速跳跃 (>50px/帧)
  - 绿色段: 中速移动 (10-50px/帧)
  - 灰色段: 慢速/静止 (<10px/帧, 可能是假阳性)
"""

import json, cv2, os, argparse
import numpy as np
import pandas as pd

VIDEO_PATH = "test_match.mp4"
CSV_PATH = "tracknetv3_output_ball.csv"
OUTPUT_DIR = "debug_tracknetv3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--every", type=int, default=30, help="Sample every N frames")
    parser.add_argument("--segments-only", action="store_true", help="Only sample from detected segments")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} frames, {df['Visibility'].sum()} detected ({100*df['Visibility'].sum()/len(df):.1f}%)")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame-to-frame speed for coloring
    speeds = [0.0]
    for i in range(1, len(df)):
        if df.iloc[i]['Visibility'] and df.iloc[i-1]['Visibility']:
            dx = df.iloc[i]['X'] - df.iloc[i-1]['X']
            dy = df.iloc[i]['Y'] - df.iloc[i-1]['Y']
            speeds.append(np.sqrt(dx*dx + dy*dy))
        else:
            speeds.append(0.0)

    # Find segments
    segments = []
    in_seg = False; seg_start = 0
    for i, v in enumerate(df['Visibility']):
        if v and not in_seg: seg_start = i; in_seg = True
        elif not v and in_seg: segments.append((seg_start, i-1)); in_seg = False
    if in_seg: segments.append((seg_start, len(df)-1))

    # Find interesting segments (with significant movement)
    interesting_segs = []
    for start, end in segments:
        seg_speeds = [s for s in speeds[start:end+1] if s > 0]
        if seg_speeds and max(seg_speeds) > 30:
            interesting_segs.append((start, end, max(seg_speeds)))
    interesting_segs.sort(key=lambda x: -x[2])

    print(f"\nSegments: {len(segments)} total, {len(interesting_segs)} with movement >30px/frame\n")

    # Sample frames to generate
    sample_frames = set()

    # Always sample from interesting segments (every 10 frames within them)
    for start, end, _ in interesting_segs[:3]:
        for f in range(start, min(end + 1, len(df)), 5):
            sample_frames.add(f)

    # Also sample every N frames globally
    if not args.segments_only:
        for f in range(0, len(df), args.every):
            sample_frames.add(f)

    sample_frames = sorted(sample_frames)
    print(f"Generating {len(sample_frames)} debug frames...")

    for frame_idx in sample_frames:
        if frame_idx >= total:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        timestamp = frame_idx / fps

        vis = frame.copy()

        # Draw trajectory (last 60 frames)
        trail_len = 60
        trail_points = []
        for i in range(max(0, frame_idx - trail_len), frame_idx + 1):
            if i < len(df) and df.iloc[i]['Visibility']:
                px, py = int(df.iloc[i]['X']), int(df.iloc[i]['Y'])
                spd = speeds[i] if i < len(speeds) else 0

                # Color by speed
                if spd > 50:
                    color = (0, 0, 255)       # Red: fast
                elif spd > 10:
                    color = (0, 255, 255)     # Yellow: medium
                else:
                    color = (128, 128, 128)   # Gray: slow/static (false positive)

                is_current = (i == frame_idx)
                trail_points.append((px, py, color, is_current, spd))

        # Draw trail lines
        for i in range(1, len(trail_points)):
            pt1 = trail_points[i-1]
            pt2 = trail_points[i]
            cv2.line(vis, (pt1[0], pt1[1]), (pt2[0], pt2[1]), pt2[2], 1, cv2.LINE_AA)

        # Draw trail points
        for px, py, color, is_current, spd in trail_points:
            if is_current:
                # Current position: big circle + crosshair
                cv2.circle(vis, (px, py), 10, color, 2)
                cv2.circle(vis, (px, py), 12, color, -1)
                cv2.line(vis, (px-15, py), (px+15, py), (255,255,255), 1)
                cv2.line(vis, (px, py-15), (px, py+15), (255,255,255), 1)
            else:
                radius = 3 if spd > 10 else 1
                cv2.circle(vis, (px, py), radius, color, -1)

        # Info overlay
        r = df.iloc[frame_idx]
        if r['Visibility']:
            cv2.putText(vis, f"BALL: ({int(r['X'])},{int(r['Y'])}) speed={speeds[frame_idx]:.0f}px/f",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Speed color code
        spd = speeds[frame_idx]
        if spd > 50:
            speed_label = "FAST (likely shuttlecock)"
            speed_color = (0, 0, 255)
        elif spd > 10:
            speed_label = "MEDIUM"
            speed_color = (0, 255, 255)
        else:
            speed_label = "SLOW (likely false positive)"
            speed_color = (128, 128, 128)

        cv2.putText(vis, f"Frame: {frame_idx}  {timestamp:.1f}s",
                   (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if r['Visibility']:
            cv2.putText(vis, speed_label, (10, 72),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, speed_color, 1)

        # Detection rate nearby
        start_i = max(0, frame_idx - 30)
        end_i = min(len(df), frame_idx + 30)
        local_df = df.iloc[start_i:end_i]
        local_rate = local_df['Visibility'].sum() / len(local_df)
        cv2.putText(vis, f"Local detect: {100*local_rate:.0f}%",
                   (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        out_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, vis)

    cap.release()

    # Summary
    print(f"\nInteresting segments:")
    for start, end, max_speed in interesting_segs[:5]:
        print(f"  Frame {start:3d}-{end:3d} ({(end-start+1)/60:.1f}s): "
              f"max_speed={max_speed:.0f}px/frame")

    print(f"\nDebug frames saved to: {args.output_dir}/")
    print(f"Open and check if yellow/red circles match actual shuttlecock position.")


if __name__ == "__main__":
    main()
