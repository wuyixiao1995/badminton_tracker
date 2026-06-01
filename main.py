"""
main.py - Badminton Match Analysis Pipeline

Two-phase architecture:
  Phase 1 (track):  YOLO pose detection + calibration -> tracks.jsonl
  Phase 2 (replay): Load tracks.jsonl -> detector + scorer + overlay

This separation means you can iterate on scoring logic without re-running YOLO.
"""

import argparse
import sys
import json
import logging
import time

import numpy as np
import cv2

from calibrator import Calibrator
from tracker import PlayerTracker
from detector import ReadyDetector
from scorer import Scorer
from overlay import OverlayRenderer
from models import Point, PlayerFrame, ReadyEvent, ScoreUpdate, MatchState
from config import LOG_LEVEL, OUTPUT_FPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Badminton Match Analysis Pipeline")
    parser.add_argument("--video", default=None,
                        help="Input video path")
    parser.add_argument("--first-server", choices=["A", "B"], default="A",
                        help="First server (default: A)")
    parser.add_argument("--output-video", default="annotated.mp4",
                        help="Output annotated video path")
    parser.add_argument("--output-scores", default="scores.json",
                        help="Output scores JSON path")
    parser.add_argument("--manual-calibration", action="store_true",
                        help="Use interactive manual calibration")
    parser.add_argument("--max-dim", type=int, default=960,
                        help="Max dimension for YOLO processing. Default: 960")
    parser.add_argument("--no-downsample", action="store_true",
                        help="Disable downsampling")
    parser.add_argument("--frame-skip", type=int, default=3,
                        help="YOLO every N frames. Default: 3")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process (0=all)")
    parser.add_argument("--output-resolution", type=str, default=None,
                        help="Output video WxH (e.g. 1280x720)")
    # Track save/load (two-phase architecture)
    parser.add_argument("--save-tracks", type=str, default=None,
                        help="Save tracking data to JSONL file")
    parser.add_argument("--load-tracks", type=str, default=None,
                        help="Load tracking data from JSONL (skip YOLO)")
    return parser.parse_args()


# ======================================================================
#  PHASE 1: Tracking (YOLO)
# ======================================================================

def run_tracking(args) -> int:
    """Phase 1: Run YOLO tracking, optionally save to JSONL."""
    logger = logging.getLogger(__name__)

    if not args.video:
        logger.error("--video is required for tracking mode")
        return 1

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %dx%d @ %.2f fps, %d frames", width, height, fps, total_frames)

    # Calibration
    ret, first_frame = cap.read()
    if not ret:
        logger.error("Cannot read first frame")
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    calibrator = Calibrator()
    if args.manual_calibration:
        success = calibrator.manual_calibrate(first_frame)
    else:
        success = calibrator.calibrate(first_frame)

    if not success:
        logger.error("Calibration failed. Use --manual-calibration.")
        return 1
    logger.info("Calibration OK: type=%s, width=%.2fm", calibrator.court_type, calibrator.court_width)

    # Resolution adaptation
    if args.no_downsample:
        scale_factor = 1.0
        process_width, process_height = width, height
    else:
        largest = max(width, height)
        max_dim = args.max_dim
        if largest > max_dim:
            scale_factor = max_dim / largest
            process_width = int(width * scale_factor)
            process_height = int(height * scale_factor)
            logger.info("Downsampling: %dx%d -> %dx%d (%.2f)", width, height, process_width, process_height, scale_factor)
        else:
            scale_factor = 1.0
            process_width, process_height = width, height

    # Init tracker
    tracker = PlayerTracker()
    max_process_frames = args.max_frames if args.max_frames > 0 else total_frames

    # Open track file if saving
    track_file = None
    if args.save_tracks:
        track_file = open(args.save_tracks, "w", encoding="utf-8")
        # Write header
        json.dump({
            "video": args.video,
            "fps": fps,
            "width": width,
            "height": height,
            "court_type": calibrator.court_type,
            "court_width": calibrator.court_width,
            "scale_factor": scale_factor,
            "corners": {
                "tl": [calibrator.corners.tl.x, calibrator.corners.tl.y],
                "tr": [calibrator.corners.tr.x, calibrator.corners.tr.y],
                "br": [calibrator.corners.br.x, calibrator.corners.br.y],
                "bl": [calibrator.corners.bl.x, calibrator.corners.bl.y],
            },
        }, track_file)
        track_file.write("\n")
        logger.info("Saving tracks to: %s", args.save_tracks)

    # Frame loop
    frame_number = 0
    last_player_frames = None
    last_pixel_a = last_pixel_b = None
    t0 = time.time()

    logger.info("Tracking %d frames (frame_skip=%d)...", max_process_frames, args.frame_skip)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps

            # YOLO on keyframes only
            if frame_number % args.frame_skip == 0 or last_player_frames is None:
                if scale_factor < 1.0:
                    pf = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
                else:
                    pf = frame

                player_frames = tracker.process_frame(pf, calibrator.M, timestamp, pixel_scale=scale_factor)

                raw_a = tracker.prev_positions.get("A")
                raw_b = tracker.prev_positions.get("B")
                inv_scale = 1.0 / scale_factor if scale_factor > 0 else 1.0

                if raw_a:
                    pixel_a = (raw_a[0] * inv_scale, raw_a[1] * inv_scale)
                else:
                    pixel_a = None
                if raw_b:
                    pixel_b = (raw_b[0] * inv_scale, raw_b[1] * inv_scale)
                else:
                    pixel_b = None

                if (len(player_frames) >= 2 and player_frames[0].confidence > 0 and player_frames[1].confidence > 0):
                    last_player_frames = player_frames
                    last_pixel_a = pixel_a
                    last_pixel_b = pixel_b
                else:
                    player_frames = last_player_frames
                    pixel_a = last_pixel_a
                    pixel_b = last_pixel_b
            else:
                player_frames = last_player_frames
                pixel_a = last_pixel_a
                pixel_b = last_pixel_b

            # Save track data
            if track_file and player_frames and len(player_frames) >= 2:
                pa, pb = player_frames[0], player_frames[1]
                record = {
                    "frame": frame_number,
                    "timestamp": round(timestamp, 3),
                    "a_x": round(pa.position.x, 3),
                    "a_y": round(pa.position.y, 3),
                    "a_conf": round(pa.confidence, 3),
                    "a_vel": round(pa.velocity, 3),
                    "b_x": round(pb.position.x, 3),
                    "b_y": round(pb.position.y, 3),
                    "b_conf": round(pb.confidence, 3),
                    "b_vel": round(pb.velocity, 3),
                    "a_px": round(pixel_a[0], 1) if pixel_a else None,
                    "a_py": round(pixel_a[1], 1) if pixel_a else None,
                    "b_px": round(pixel_b[0], 1) if pixel_b else None,
                    "b_py": round(pixel_b[1], 1) if pixel_b else None,
                }
                track_file.write(json.dumps(record) + "\n")

            frame_number += 1
            if max_process_frames > 0 and frame_number >= max_process_frames:
                break
            if frame_number % 500 == 0:
                elapsed = time.time() - t0
                fps_eff = frame_number / elapsed if elapsed > 0 else 0
                print(f"Tracked {frame_number} frames ({fps_eff:.1f} fps)...", flush=True)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Saving progress...")

    cap.release()
    if track_file:
        track_file.close()
        logger.info("Tracks saved: %d frames to %s", frame_number, args.save_tracks)

    elapsed = time.time() - t0
    logger.info("Tracking complete: %d frames in %.1fs (%.1f fps)", frame_number, elapsed, frame_number / elapsed if elapsed > 0 else 0)
    return 0


# ======================================================================
#  PHASE 2: Scoring (replay from tracks)
# ======================================================================

def run_scoring(args) -> int:
    """Phase 2: Load tracks, run detector + scorer + overlay."""
    logger = logging.getLogger(__name__)

    if not args.load_tracks:
        logger.error("--load-tracks is required for scoring mode")
        return 1

    # Load tracks
    logger.info("Loading tracks from: %s", args.load_tracks)
    with open(args.load_tracks, "r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        records = [json.loads(line) for line in f]

    logger.info("Loaded %d frames", len(records))
    fps = header["fps"]
    court_type = header.get("court_type", "singles")
    court_width = header.get("court_width", 5.18)

    # Reconstruct calibrator
    from models import PixelPoint, CourtCorners, CourtZones
    corners = header["corners"]
    cal = Calibrator()
    cal.court_type = court_type
    cal.court_width = court_width
    court_corners = CourtCorners(
        tl=PixelPoint(corners["tl"][0], corners["tl"][1]),
        tr=PixelPoint(corners["tr"][0], corners["tr"][1]),
        br=PixelPoint(corners["br"][0], corners["br"][1]),
        bl=PixelPoint(corners["bl"][0], corners["bl"][1]),
    )
    cal.compute_transform(court_corners)
    cal.define_zones()
    logger.info("Calibrator reconstructed: type=%s", court_type)

    # Init scoring components
    detector = ReadyDetector(cal.zones)
    scorer = Scorer(args.first_server)

    # Run detection + scoring over all records
    total_ready = 0
    total_scores = 0
    t0 = time.time()

    for i, rec in enumerate(records):
        ts = rec["timestamp"]
        pa = PlayerFrame(
            player_id="A",
            position=Point(rec["a_x"], rec["a_y"]),
            confidence=rec["a_conf"],
            timestamp=ts,
            velocity=rec["a_vel"],
            zone=None,
        )
        pb = PlayerFrame(
            player_id="B",
            position=Point(rec["b_x"], rec["b_y"]),
            confidence=rec["b_conf"],
            timestamp=ts,
            velocity=rec["b_vel"],
            zone=None,
        )

        ready_event = detector.update(pa, pb, scorer.state, ts)
        if ready_event:
            total_ready += 1
            score_update = scorer.process_event(ready_event)
            if score_update:
                total_scores += 1
                print(f"[{ts:.1f}s] {score_update.description}", flush=True)

        if i % 1000 == 0:
            elapsed = time.time() - t0
            print(f"Scored {i}/{len(records)} frames, {total_ready} ready events, {total_scores} scores", flush=True)

    elapsed = time.time() - t0
    logger.info("Scoring complete in %.1fs", elapsed)
    logger.info("Ready events: %d, Score updates: %d", total_ready, total_scores)
    logger.info("Final score: %s", scorer.get_final_score_string())

    # Export scores
    scorer.export_score_json(args.output_scores)
    print(f"\nFinal: {scorer.get_final_score_string()}, Server: {scorer.state.server}")
    print(f"Ready events: {total_ready}, Scores: {total_scores}")

    # Generate score curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        from overlay import OverlayRenderer
        ov = OverlayRenderer("dummy.mp4", args.output_scores)
        for su in scorer.score_history:
            ov.add_score_update(su)
        ov.export_score_curve("score_curve.png")
        print("Score curve: score_curve.png")
    except Exception as e:
        logger.warning("Score curve generation failed: %s", e)

    return 0


# ======================================================================
#  COMBINED MODE: Track + Score + Render (original full pipeline)
# ======================================================================

def run_combined(args) -> int:
    """Full pipeline: track + score + render in one pass."""
    logger = logging.getLogger(__name__)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %dx%d @ %.2f fps, %d frames", width, height, fps, total_frames)

    # Calibration
    ret, first_frame = cap.read()
    if not ret:
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    calibrator = Calibrator()
    if args.manual_calibration:
        calibrator.manual_calibrate(first_frame)
    else:
        if not calibrator.calibrate(first_frame):
            logger.error("Auto calibration failed. Use --manual-calibration.")
            return 1
    logger.info("Calibration: %s (%.2fm)", calibrator.court_type, calibrator.court_width)

    # Resolution
    if args.no_downsample:
        scale_factor = 1.0
        pw, ph = width, height
    else:
        largest = max(width, height)
        if largest > args.max_dim:
            scale_factor = args.max_dim / largest
            pw, ph = int(width * scale_factor), int(height * scale_factor)
            logger.info("Downsampling: %dx%d -> %dx%d", width, height, pw, ph)
        else:
            scale_factor = 1.0
            pw, ph = width, height

    # Output resolution
    if args.output_resolution:
        out_w, out_h = map(int, args.output_resolution.split('x'))
        do_resize = True
    else:
        out_w, out_h = width, height
        do_resize = False

    # Init components
    tracker = PlayerTracker()
    detector = ReadyDetector(calibrator.zones)
    scorer = Scorer(args.first_server)
    overlay = OverlayRenderer(args.output_video, args.output_scores)
    output_fps = OUTPUT_FPS if OUTPUT_FPS else fps
    overlay.init_video_writer(out_w, out_h, output_fps)

    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    logger.info("Processing %d frames (frame_skip=%d)...", max_frames, args.frame_skip)

    frame_number = 0
    total_ready = 0
    total_scores = 0
    last_pfs = None
    last_pxa = last_pxb = None
    t0 = time.time()

    # Open track file if saving
    track_file = None
    if args.save_tracks:
        track_file = open(args.save_tracks, "w", encoding="utf-8")
        json.dump({
            "video": args.video, "fps": fps, "width": width, "height": height,
            "court_type": calibrator.court_type, "court_width": calibrator.court_width,
            "scale_factor": scale_factor,
            "corners": {
                "tl": [calibrator.corners.tl.x, calibrator.corners.tl.y],
                "tr": [calibrator.corners.tr.x, calibrator.corners.tr.y],
                "br": [calibrator.corners.br.x, calibrator.corners.br.y],
                "bl": [calibrator.corners.bl.x, calibrator.corners.bl.y],
            },
        }, track_file)
        track_file.write("\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps

            # Track (YOLO on keyframes)
            if frame_number % args.frame_skip == 0 or last_pfs is None:
                if scale_factor < 1.0:
                    pf = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
                else:
                    pf = frame

                player_frames = tracker.process_frame(pf, calibrator.M, timestamp, pixel_scale=scale_factor)

                raw_a = tracker.prev_positions.get("A")
                raw_b = tracker.prev_positions.get("B")
                inv_scale = 1.0 / scale_factor if scale_factor > 0 else 1.0
                pxa = (raw_a[0] * inv_scale, raw_a[1] * inv_scale) if raw_a else None
                pxb = (raw_b[0] * inv_scale, raw_b[1] * inv_scale) if raw_b else None

                if (len(player_frames) >= 2 and player_frames[0].confidence > 0 and player_frames[1].confidence > 0):
                    last_pfs = player_frames
                    last_pxa = pxa
                    last_pxb = pxb
                else:
                    player_frames = last_pfs
                    pxa = last_pxa
                    pxb = last_pxb
            else:
                player_frames = last_pfs
                pxa = last_pxa
                pxb = last_pxb

            # Save track
            if track_file and player_frames and len(player_frames) >= 2:
                pa, pb = player_frames[0], player_frames[1]
                track_file.write(json.dumps({
                    "frame": int(frame_number), "timestamp": float(round(timestamp, 3)),
                    "a_x": float(round(pa.position.x, 3)), "a_y": float(round(pa.position.y, 3)),
                    "a_conf": float(round(pa.confidence, 3)), "a_vel": float(round(pa.velocity, 3)),
                    "b_x": float(round(pb.position.x, 3)), "b_y": float(round(pb.position.y, 3)),
                    "b_conf": float(round(pb.confidence, 3)), "b_vel": float(round(pb.velocity, 3)),
                    "a_px": float(round(pxa[0], 1)) if pxa else None,
                    "a_py": float(round(pxa[1], 1)) if pxa else None,
                    "b_px": float(round(pxb[0], 1)) if pxb else None,
                    "b_py": float(round(pxb[1], 1)) if pxb else None,
                }) + "\n")

            # Detect + Score
            if player_frames and len(player_frames) >= 2 and player_frames[0].confidence > 0:
                pa, pb = player_frames[0], player_frames[1]
                ready_event = detector.update(pa, pb, scorer.state, timestamp)
                if ready_event:
                    total_ready += 1
                    score_update = scorer.process_event(ready_event)
                    if score_update:
                        total_scores += 1
                        print(f"[{timestamp:.1f}s] {score_update.description}", flush=True)
                        overlay.add_score_update(score_update)

                # Render
                rendered = overlay.render_frame(frame, pa, pb, scorer.state, pxa, pxb, timestamp)
            else:
                rendered = frame

            if do_resize:
                rendered = cv2.resize(rendered, (out_w, out_h))
            overlay.write_frame(rendered)

            frame_number += 1
            if frame_number >= max_frames:
                break
            if frame_number % 500 == 0:
                elapsed = time.time() - t0
                print(f"Processed {frame_number} frames ({frame_number/elapsed:.1f} fps)...", flush=True)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Saving progress...")

    cap.release()
    overlay.release()
    if track_file:
        track_file.close()

    scorer.export_score_json(args.output_scores)

    elapsed = time.time() - t0
    logger.info("=" * 50)
    logger.info("  COMPLETE: %d frames in %.1fs", frame_number, elapsed)
    logger.info("  Final: %s, Server: %s", scorer.get_final_score_string(), scorer.state.server)
    logger.info("  Ready: %d, Scores: %d", total_ready, total_scores)
    logger.info("  Output: %s, %s", args.output_video, args.output_scores)
    logger.info("=" * 50)
    return 0


# ======================================================================
#  Main
# ======================================================================

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    if args.load_tracks:
        # Phase 2 only: scoring from saved tracks
        sys.exit(run_scoring(args))
    elif args.video:
        # Phase 1 + 2 combined (track + score + render)
        sys.exit(run_combined(args))
    else:
        print("Error: --video or --load-tracks required")
        sys.exit(1)


if __name__ == "__main__":
    main()
