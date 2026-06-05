"""
tracknetv3_infer.py — TrackNetV3 羽毛球专用轨迹推理 (CPU 兼容)

TrackNetV3 = TrackNet (heatmap tracking) + InpaintNet (trajectory rectification)
论文: https://dl.acm.org/doi/10.1145/3595916.3626370
性能: 97.5% Accuracy, 98.6% F1 on Shuttlecock Trajectory Dataset

用法:
  python tracknetv3_infer.py --video test_match.mp4 --max-frames 600 --output tracknetv3_output.mp4
"""

import sys, os, argparse
import cv2
import numpy as np
from collections import deque
from PIL import Image
from tqdm import tqdm

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tracknetv3_repo"))
from model import TrackNet, InpaintNet
from utils.general import HEIGHT, WIDTH, get_model, write_pred_csv, draw_traj


def load_video(video_path, max_frames=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return [], 0, (0, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

    frames = []
    max_frames = max_frames if max_frames > 0 else total
    for _ in range(min(max_frames, total)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[:, :, ::-1])  # BGR -> RGB
    cap.release()
    print(f"Read {len(frames)} frames")
    return frames, fps, (w, h)


def get_median_image(frames, max_samples=1800):
    n = len(frames)
    if n > max_samples:
        step = n // max_samples
        sampled = frames[::step]
    else:
        sampled = frames
    return np.median(np.array(sampled), axis=0).astype(np.uint8)


def predict_location(heatmap):
    """Get ball bounding box from binary heatmap (uint8, 0/255).
    Same logic as test.py:predict_location."""
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    cnts, _ = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return 0, 0, 0, 0
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    max_area_idx = 0
    max_area = rects[0][2] * rects[0][3]
    for i in range(1, len(rects)):
        area = rects[i][2] * rects[i][3]
        if area > max_area:
            max_area_idx = i
            max_area = area
    return rects[max_area_idx]


def run_tracknet_nonoverlap(frames, model, median_img, device, seq_len=8):
    """Simple non-overlap sliding window. Fast and reliable. """
    h_scaler = frames[0].shape[0] / HEIGHT
    w_scaler = frames[0].shape[1] / WIDTH

    # Preprocess median
    median_pil = Image.fromarray(median_img.astype(np.uint8))
    median_resized = np.array(median_pil.resize(size=(WIDTH, HEIGHT)))
    median_chw = np.moveaxis(median_resized, -1, 0) / 255.0  # (3, H, W)

    frame_count = len(frames)
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

    # Preprocess all frames
    print(f"Preprocessing {frame_count} frames...")
    processed = []
    for frame in tqdm(frames):
        img = Image.fromarray(frame.astype(np.uint8))
        img = np.array(img.resize(size=(WIDTH, HEIGHT)))
        img = np.moveaxis(img, -1, 0) / 255.0  # (3, H, W)
        processed.append(img)

    n_sequences = (frame_count + seq_len - 1) // seq_len
    print(f"Running TrackNet on {n_sequences} sequences (non-overlap)...")

    for seq_idx in tqdm(range(0, frame_count, seq_len)):
        # Build input
        seq_frames = []
        for f in range(seq_len):
            idx = min(seq_idx + f, frame_count - 1)
            seq_frames.append(processed[idx])

        frames_chw = np.concatenate(seq_frames, axis=0)  # (L*3, H, W)
        input_chw = np.concatenate([median_chw, frames_chw], axis=0)  # ((L+1)*3, H, W)
        input_tensor = torch.from_numpy(input_chw).float().unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(input_tensor).cpu().numpy()[0]  # (L, H, W)

        for f_offset in range(min(seq_len, frame_count - seq_idx)):
            heatmap = y_pred[f_offset]
            heatmap_bin = ((heatmap > 0.5) * 255).astype(np.uint8)
            x, y, bw, bh = predict_location(heatmap_bin)

            frame_idx = seq_idx + f_offset
            if bw > 0 and bh > 0:
                cx = int((x + bw / 2) * w_scaler)
                cy = int((y + bh / 2) * h_scaler)
                pred_dict['Frame'].append(frame_idx)
                pred_dict['X'].append(cx)
                pred_dict['Y'].append(cy)
                pred_dict['Visibility'].append(1)
            else:
                pred_dict['Frame'].append(frame_idx)
                pred_dict['X'].append(0)
                pred_dict['Y'].append(0)
                pred_dict['Visibility'].append(0)

    return pred_dict


def run_tracknet_ensemble(frames, model, median_img, device, seq_len=8):
    """Ensemble mode: sliding step=1 with weighted average.
    Based on predict.py ensemble logic, fixed for CPU single-batch. """
    h_scaler = frames[0].shape[0] / HEIGHT
    w_scaler = frames[0].shape[1] / WIDTH

    median_pil = Image.fromarray(median_img.astype(np.uint8))
    median_resized = np.array(median_pil.resize(size=(WIDTH, HEIGHT)))
    median_chw = np.moveaxis(median_resized, -1, 0) / 255.0  # (3, H, W)

    frame_count = len(frames)

    print(f"Preprocessing {frame_count} frames...")
    processed = []
    for frame in tqdm(frames):
        img = Image.fromarray(frame.astype(np.uint8))
        img = np.array(img.resize(size=(WIDTH, HEIGHT)))
        img = np.moveaxis(img, -1, 0) / 255.0  # (3, H, W)
        processed.append(img)

    # Ensemble weight: higher weight for frames closer to center of sequence
    weight = np.linspace(1, 2, seq_len)
    weight = weight / weight.sum()

    # Initialize buffer
    buffer_size = seq_len - 1  # 7
    y_pred_buffer = np.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=np.float32)

    # For accumulating per-frame results
    accumulated = [np.zeros((HEIGHT, WIDTH), dtype=np.float32) for _ in range(frame_count)]
    counts = [0] * frame_count

    n_sequences = frame_count - seq_len + 1
    print(f"Running TrackNet ensemble on {n_sequences} sequences...")

    # Pre-built indexing arrays (fix: use simple batch_i, not batch_i + i - buffer_size)
    batch_i = np.arange(seq_len)     # [0,1,2,3,4,5,6,7]
    frame_i = np.arange(seq_len-1, -1, -1)  # [7,6,5,4,3,2,1,0]

    sample_count = 0
    for i in tqdm(range(n_sequences)):
        # Build input for sequence starting at frame i
        seq_chw = np.concatenate(
            [processed[j] for j in range(i, i + seq_len)], axis=0
        )
        input_chw = np.concatenate([median_chw, seq_chw], axis=0)
        input_tensor = torch.from_numpy(input_chw).float().unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(input_tensor).cpu().numpy()[0]  # (L, H, W)

        # Append to buffer: buffer grows by 1 row each iteration
        y_pred_buffer = np.concatenate([y_pred_buffer, y_pred[np.newaxis, ...]], axis=0)

        if sample_count < buffer_size:
            # Incomplete buffer: average available predictions
            valid = sample_count + 1
            bi = np.arange(valid)
            fi = np.arange(sample_count, -1, -1)
            ensemble = y_pred_buffer[bi, fi].sum(0) / valid
        else:
            # Full buffer: weighted ensemble over last 8 predictions
            # buffer always has exactly buffer_size + 1 = 8 rows at this point
            # (trimmed to 7 then had 1 new appended)
            ensemble = (y_pred_buffer[batch_i, frame_i] * weight[:, None, None]).sum(0)

        accumulated[i] += ensemble
        counts[i] += 1
        sample_count += 1

        # Last sequence: predict tail frames
        if sample_count == n_sequences:
            zero_pad = np.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=np.float32)
            y_pred_buffer = np.concatenate([y_pred_buffer, zero_pad], axis=0)

            for f in range(1, seq_len):
                frame_idx = i + f
                if frame_idx < frame_count:
                    bi = batch_i + f  # [f, f+1, ..., f+7]
                    preds = y_pred_buffer[bi, frame_i]
                    ensemble = preds.sum(0) / (seq_len - f)
                    accumulated[frame_idx] += ensemble
                    counts[frame_idx] += 1

        # Trim buffer: keep only last buffer_size rows for next iteration
        y_pred_buffer = y_pred_buffer[-buffer_size:]

    # Generate final predictions
    print("Generating final predictions...")
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

    for i in range(frame_count):
        if counts[i] > 0:
            avg_heatmap = accumulated[i] / counts[i]
        else:
            avg_heatmap = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

        heatmap_bin = ((avg_heatmap > 0.5) * 255).astype(np.uint8)
        x, y, bw, bh = predict_location(heatmap_bin)

        if bw > 0 and bh > 0:
            cx = int((x + bw / 2) * w_scaler)
            cy = int((y + bh / 2) * h_scaler)
            pred_dict['Frame'].append(i)
            pred_dict['X'].append(cx)
            pred_dict['Y'].append(cy)
            pred_dict['Visibility'].append(1)
        else:
            pred_dict['Frame'].append(i)
            pred_dict['X'].append(0)
            pred_dict['Y'].append(0)
            pred_dict['Visibility'].append(0)

    return pred_dict


def render_output_video(video_path, pred_dict, output_path, traj_len=8):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']
    pred_queue = deque()
    i = 0

    while True:
        success, frame = cap.read()
        if not success or i >= len(x_pred):
            break

        if len(pred_queue) >= traj_len:
            pred_queue.pop()

        if vis_pred[i]:
            pred_queue.appendleft([x_pred[i], y_pred[i]])
        else:
            pred_queue.appendleft(None)

        frame = draw_traj(frame, pred_queue, radius=4, color='yellow')

        cv2.putText(frame, f"Frame: {i}  Time: {i/fps:.1f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if vis_pred[i]:
            cv2.putText(frame, f"Ball: ({x_pred[i]}, {y_pred[i]})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.circle(frame, (x_pred[i], y_pred[i]), 6, (0, 255, 255), -1)
            cv2.circle(frame, (x_pred[i], y_pred[i]), 8, (0, 200, 200), 2)

        local_start = max(0, i - 30)
        local_end = min(len(vis_pred), i + 30)
        local_vis = vis_pred[local_start:local_end]
        if local_vis:
            rate = sum(local_vis) / len(local_vis)
            cv2.putText(frame, f"Detect rate: {100*rate:.0f}%",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        out.write(frame)
        i += 1

    out.release()
    cap.release()
    print(f"Output video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="test_match.mp4")
    parser.add_argument("--tracknet-weights", default="ckpts/TrackNet_best.pt")
    parser.add_argument("--output", default="tracknetv3_output.mp4")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ensemble", action="store_true", default=True)
    parser.add_argument("--no-ensemble", dest="ensemble", action="store_false")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device} | Ensemble: {args.ensemble}")

    # ---- Load Model ----
    if not os.path.exists(args.tracknet_weights):
        print(f"ERROR: Weights not found: {args.tracknet_weights}")
        return

    print("\n[1/5] Loading TrackNetV3...")
    ckpt = torch.load(args.tracknet_weights, map_location=device)
    seq_len = ckpt['param_dict']['seq_len']
    bg_mode = ckpt['param_dict']['bg_mode']
    print(f"  seq_len={seq_len}, bg_mode={bg_mode}")

    model = get_model('TrackNet', seq_len, bg_mode).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("  OK")

    # ---- Load Video ----
    print(f"\n[2/5] Loading video: {args.video}")
    frames, fps, (vw, vh) = load_video(args.video, args.max_frames)
    if len(frames) < seq_len:
        print(f"ERROR: Need at least {seq_len} frames")
        return

    # ---- Median ----
    print("\n[3/5] Generating median background...")
    median_img = get_median_image(frames)
    print(f"  shape={median_img.shape}")

    # ---- Inference ----
    print(f"\n[4/5] Running inference ({len(frames)} frames)...")
    if args.ensemble:
        pred_dict = run_tracknet_ensemble(frames, model, median_img, device, seq_len)
    else:
        pred_dict = run_tracknet_nonoverlap(frames, model, median_img, device, seq_len)

    # ---- Stats ----
    n_detected = sum(pred_dict['Visibility'])
    n_total = len(pred_dict['Visibility'])
    print(f"\n{'='*60}")
    print(f"Results: {n_detected}/{n_total} frames detected ({100*n_detected/n_total:.1f}%)")

    if n_detected > 0:
        xs = [x for x, v in zip(pred_dict['X'], pred_dict['Visibility']) if v]
        ys = [y for y, v in zip(pred_dict['Y'], pred_dict['Visibility']) if v]
        if xs:
            print(f"  X: min={min(xs)} max={max(xs)} span={max(xs)-min(xs)}")
            print(f"  Y: min={min(ys)} max={max(ys)} span={max(ys)-min(ys)}")
            pts = [(x,y) for x,y,v in zip(pred_dict['X'],pred_dict['Y'],pred_dict['Visibility']) if v]
            if len(pts) > 1:
                dists = [np.sqrt((pts[i][0]-pts[i-1][0])**2+(pts[i][1]-pts[i-1][1])**2)
                         for i in range(1,len(pts))]
                print(f"  Frame-to-frame: mean={np.mean(dists):.1f}px median={np.median(dists):.1f}px")

    # Segments analysis
    segments = []
    in_seg = False; seg_start = 0
    for i, v in enumerate(pred_dict['Visibility']):
        if v and not in_seg: seg_start = i; in_seg = True
        elif not v and in_seg: segments.append((seg_start, i-1, i-seg_start)); in_seg = False
    if in_seg: segments.append((seg_start, n_total-1, n_total-seg_start))
    if segments:
        avg_len = np.mean([s[2] for s in segments])
        print(f"  Segments: {len(segments)} avg_len={avg_len:.0f} frames ({avg_len/fps:.1f}s)")

    print(f"{'='*60}")

    # ---- Save CSV ----
    csv_path = args.output.replace('.mp4', '_ball.csv')
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    write_pred_csv(pred_dict, save_file=csv_path)
    print(f"CSV saved: {csv_path}")

    # ---- Render ----
    print(f"\n[5/5] Rendering video: {args.output}")
    render_output_video(args.video, pred_dict, args.output)
    print(f"Done!")


if __name__ == "__main__":
    main()
