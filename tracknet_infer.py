"""
tracknet_infer.py — TrackNet 羽毛球轨迹推理

用法:
  python tracknet_infer.py --video test_match.mp4 --max-frames 1800 --output output.mp4

用 TrackNet (U-Net 热力图) 检测视频中的球类轨迹。
虽然是网球训练的权重，但可以测试对羽毛球小目标的泛化能力。
"""

import sys, os, argparse
import cv2
import torch
import numpy as np
from scipy.spatial import distance
from itertools import groupby

# 添加 tracknet_repo 到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tracknet_repo"))
from model import BallTrackerNet
from general import postprocess


def read_frames(path_video, max_frames=0):
    """读取视频帧"""
    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {path_video}")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames")

    frames = []
    max_frames = max_frames if max_frames > 0 else total
    for i in range(min(max_frames, total)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Read {len(frames)} frames")
    return frames, fps


def infer_model(frames, model, device):
    """TrackNet 推理: 3帧输入 → 热力图 → 球坐标"""
    height, width = 360, 640
    ball_track = [(None, None)] * 2
    dists = [-1.0] * 2

    for num in range(2, len(frames)):
        # 缩放3帧到模型输入尺寸
        f0 = cv2.resize(frames[num], (width, height))
        f1 = cv2.resize(frames[num - 1], (width, height))
        f2 = cv2.resize(frames[num - 2], (width, height))

        # 拼接 3×RGB = 9 通道
        imgs = np.concatenate((f0, f1, f2), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        with torch.no_grad():
            out = model(torch.from_numpy(inp).float().to(device), testing=True)
        output = out.argmax(dim=1).detach().cpu().numpy()

        x_pred, y_pred = postprocess(output)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)

        if num % 300 == 0:
            detected = sum(1 for b in ball_track[-300:] if b[0] is not None)
            print(f"  Frame {num}/{len(frames)}: "
                  f"recent detection rate = {detected}/300 ({100*detected/300:.0f}%)")

    return ball_track, dists


def remove_outliers(ball_track, dists, max_dist=100):
    """异常值移除"""
    for i in range(2, len(ball_track) - 1):
        if dists[i] > max_dist:
            ball_track[i] = (None, None)
    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """分割轨迹为子段"""
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if k == 1 and i > 0 and i < len(groups) - 1:
            dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + l])
            if l >= max_gap or dist / l > max_dist_gap:
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1
        cursor += l
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result


def interpolation(coords):
    """缺失帧插值"""
    x = np.array([c[0] if c[0] is not None else np.nan for c in coords])
    y = np.array([c[1] if c[1] is not None else np.nan for c in coords])

    nans_x = np.isnan(x)
    nans_y = np.isnan(y)
    if nans_x.any():
        ok = ~nans_x
        x[nans_x] = np.interp(np.flatnonzero(nans_x), np.flatnonzero(ok), x[ok])
    if nans_y.any():
        ok = ~nans_y
        y[nans_y] = np.interp(np.flatnonzero(nans_y), np.flatnonzero(ok), y[ok])

    return [*zip(x, y)]


def render_output(frames, ball_track, fps, path_output, trace=10):
    """渲染带球轨迹的输出视频 + 保存轨迹数据"""
    h, w = frames[0].shape[:2]
    scale_x = w / 640.0
    scale_y = h / 360.0

    # 视频输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_output, fourcc, fps, (w, h))

    # 轨迹数据输出
    track_data = []

    for num in range(len(frames)):
        frame = frames[num].copy()

        # 画球轨迹
        for i in range(trace):
            idx = num - i
            if idx >= 0 and ball_track[idx][0] is not None:
                bx = int(ball_track[idx][0] * scale_x)
                by = int(ball_track[idx][1] * scale_y)
                alpha = 1.0 - i / trace
                radius = max(1, int(5 * alpha))
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))  # 红→黄渐变
                cv2.circle(frame, (bx, by), radius, color, -1)

        # 当前帧球位置：大红圈 + 十字
        if ball_track[num][0] is not None:
            bx = int(ball_track[num][0] * scale_x)
            by = int(ball_track[num][1] * scale_y)
            cv2.circle(frame, (bx, by), 8, (0, 0, 255), 2)
            cv2.line(frame, (bx - 10, by), (bx + 10, by), (0, 0, 255), 1)
            cv2.line(frame, (bx, by - 10), (bx, by + 10), (0, 0, 255), 1)

        # 帧号 + 时间
        cv2.putText(frame, f"Frame: {num}  Time: {num/fps:.1f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if ball_track[num][0] is not None:
            cv2.putText(frame, f"Ball: ({ball_track[num][0]:.0f}, {ball_track[num][1]:.0f})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)

        # 保存轨迹数据
        track_data.append({
            "frame": num,
            "timestamp": num / fps,
            "x": ball_track[num][0],
            "y": ball_track[num][1],
            "detected": ball_track[num][0] is not None,
        })

    out.release()

    # 保存 JSON
    import json
    json_path = path_output.replace(".mp4", "_track.json")
    # 统计
    detected_frames = sum(1 for t in track_data if t["detected"])
    total_frames = len(track_data)
    stats = {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": f"{100*detected_frames/total_frames:.1f}%" if total_frames > 0 else "0%",
        "tracks": track_data,
    }
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Track data saved to {json_path}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="test_match.mp4", help="Input video path")
    parser.add_argument("--model", default="tracknet_weights.pt", help="Pretrained model weights")
    parser.add_argument("--output", default="tracknet_output.mp4", help="Output video path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    parser.add_argument("--no-extrapolation", action="store_true", help="Skip interpolation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model weights not found: {args.model}")
        print("Download from: https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view")
        return

    device = args.device
    print(f"Device: {device}")

    # ---- Load Model ----
    print("\n[1/4] Loading TrackNet model...")
    model = BallTrackerNet()
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("  Model loaded OK")

    # ---- Read Video ----
    print(f"\n[2/4] Reading video: {args.video}")
    frames, fps = read_frames(args.video, max_frames=args.max_frames)
    if len(frames) < 3:
        print("ERROR: Need at least 3 frames")
        return

    # ---- Inference ----
    print(f"\n[3/4] Running TrackNet inference on {len(frames)} frames...")
    ball_track, dists = infer_model(frames, model, device)

    # ---- Post-process ----
    print("\n[4/4] Post-processing...")
    ball_track = remove_outliers(ball_track, dists)

    if not args.no_extrapolation:
        subtracks = split_track(ball_track)
        print(f"  Found {len(subtracks)} subtracks for interpolation")
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack

    # ---- Stats ----
    total_detected = sum(1 for b in ball_track if b[0] is not None)
    print(f"\n{'='*50}")
    print(f"Results: {total_detected}/{len(ball_track)} frames with ball detected "
          f"({100*total_detected/len(ball_track):.1f}%)")
    print(f"{'='*50}")

    # ---- Render ----
    print(f"\nRendering output video: {args.output}")
    stats = render_output(frames, ball_track, fps, args.output)

    print(f"\nDone! Check: {args.output}")
    print(f"Detection rate: {stats['detection_rate']}")


if __name__ == "__main__":
    main()
