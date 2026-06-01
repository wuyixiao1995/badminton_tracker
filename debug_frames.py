"""
debug_frames.py — 生成调试帧图片

在原始视频帧上叠加:
  - 绿色四边形: 检测到的球场边界
  - 红色大圆点: 球员A脚部位置 (画面下方/近侧)
  - 蓝色大圆点: 球员B脚部位置 (画面上方/远侧)
  - 黄色文字: 坐标信息

用法: python debug_frames.py demo.MP4 --frames 0,300,600,900

输出: debug_frames/ 目录下的 PNG 图片
"""

import os, sys, argparse
import cv2
import numpy as np

from calibrator import Calibrator
from tracker import PlayerTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--frames", default="0,300,600,900",
                        help="Comma-separated frame numbers to capture")
    parser.add_argument("--max-dim", type=int, default=960)
    parser.add_argument("--output-dir", default="debug_frames")
    args = parser.parse_args()

    frame_indices = [int(x.strip()) for x in args.frames.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {width}x{height} @ {fps:.1f}fps")

    # ---- Calibration on first frame ----
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        return

    calibrator = Calibrator()
    success = calibrator.calibrate(first_frame)
    if not success:
        print("Auto calibration FAILED. Trying manual?")
        cap.release()
        return

    print(f"Calibration OK: {calibrator.court_type}, corners detected:")
    for name, pt in [("tl", calibrator.corners.tl), ("tr", calibrator.corners.tr),
                      ("br", calibrator.corners.br), ("bl", calibrator.corners.bl)]:
        print(f"  {name}=({pt.x},{pt.y})")

    # ---- Tracking ----
    tracker = PlayerTracker()
    scale_factor = args.max_dim / max(width, height) if max(width, height) > args.max_dim else 1.0
    if scale_factor < 1.0:
        pw, ph = int(width * scale_factor), int(height * scale_factor)
        print(f"Downsampling: {width}x{height} -> {pw}x{ph} (scale={scale_factor:.2f})")
    else:
        pw, ph = width, height
        scale_factor = 1.0

    # Process requested frames
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {fi}: READ FAILED")
            continue

        timestamp = fi / fps
        orig = frame.copy()

        # Downsample for YOLO
        if scale_factor < 1.0:
            pf = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            pf = frame

        # Track
        player_frames = tracker.process_frame(pf, calibrator.M, timestamp, pixel_scale=scale_factor)

        # ---- Draw court boundary (green) ----
        if calibrator.corners:
            pts = np.array([
                [calibrator.corners.tl.x, calibrator.corners.tl.y],
                [calibrator.corners.tr.x, calibrator.corners.tr.y],
                [calibrator.corners.br.x, calibrator.corners.br.y],
                [calibrator.corners.bl.x, calibrator.corners.bl.y],
            ], dtype=np.int32)
            cv2.polylines(orig, [pts], True, (0, 255, 0), 3)  # Green border

            # Corner labels
            for label, px, py in [("TL", calibrator.corners.tl.x, calibrator.corners.tl.y),
                                   ("TR", calibrator.corners.tr.x, calibrator.corners.tr.y),
                                   ("BR", calibrator.corners.br.x, calibrator.corners.br.y),
                                   ("BL", calibrator.corners.bl.x, calibrator.corners.bl.y)]:
                cv2.circle(orig, (px, py), 8, (0, 255, 0), -1)
                cv2.putText(orig, label, (px + 12, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---- Draw player foot positions ----
        # Get raw pixel positions from tracker
        raw_a = tracker.prev_positions.get("A")
        raw_b = tracker.prev_positions.get("B")

        inv_scale = 1.0 / scale_factor if scale_factor > 0 else 1.0

        if player_frames and len(player_frames) >= 2:
            pa, pb = player_frames[0], player_frames[1]

            if raw_a is not None:
                # Scale back to original resolution
                pxa = int(raw_a[0] * inv_scale)
                pya = int(raw_a[1] * inv_scale)
            else:
                pxa, pya = -1, -1

            if raw_b is not None:
                pxb = int(raw_b[0] * inv_scale)
                pyb = int(raw_b[1] * inv_scale)
            else:
                pxb, pyb = -1, -1

            # Player A = near side = RED circle
            if pxa > 0:
                cv2.circle(orig, (pxa, pya), 20, (0, 0, 255), 3)     # Red outer ring
                cv2.circle(orig, (pxa, pya), 6, (0, 0, 255), -1)     # Red fill
                cv2.putText(orig, f"A conf={pa.confidence:.2f}", (pxa + 25, pya - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(orig, f"court=({pa.position.x:.1f},{pa.position.y:.1f})m",
                            (pxa + 25, pya + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)

            # Player B = far side = BLUE circle
            if pxb > 0:
                cv2.circle(orig, (pxb, pyb), 20, (255, 0, 0), 3)     # Blue outer ring
                cv2.circle(orig, (pxb, pyb), 6, (255, 0, 0), -1)     # Blue fill
                cv2.putText(orig, f"B conf={pb.confidence:.2f}", (pxb + 25, pyb - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(orig, f"court=({pb.position.x:.1f},{pb.position.y:.1f})m",
                            (pxb + 25, pyb + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 0), 1)

        # ---- Info overlay ----
        cv2.putText(orig, f"Frame: {fi}  Time: {timestamp:.1f}s  Court: {calibrator.court_type}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save
        out_path = os.path.join(args.output_dir, f"frame_{fi:04d}.png")
        cv2.imwrite(out_path, orig)
        print(f"Frame {fi}: saved -> {out_path}")

    cap.release()
    print(f"\nDone! Check {args.output_dir}/ for debug images.")


if __name__ == "__main__":
    main()
