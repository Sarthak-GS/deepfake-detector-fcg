"""
trim_videos.py
--------------
Trims all videos in data_subset/ to a max of 10 seconds.

Input:  data_subset/fake/  data_subset/real/
Output: data_subset_trimmed/fake/  data_subset_trimmed/real/

Requirements: ffmpeg must be installed (conda install -c conda-forge ffmpeg OR apt install ffmpeg)

Usage:
  python trim_videos.py --in-dir data_subset --out-dir data_subset_trimmed --max-seconds 10
"""

import os
import sys
import glob
import argparse
import subprocess


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir',      default='../data_subset')
    p.add_argument('--out-dir',     default='../data_subset_trimmed')
    p.add_argument('--max-seconds', type=float, default=10.0)
    return p.parse_args()


def get_video_duration(path: str) -> float:
    """Use ffprobe to get duration in seconds. Returns -1 on error."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', path],
            capture_output=True, text=True, timeout=30
        )
        import json
        info = json.loads(result.stdout)
        for stream in info.get('streams', []):
            dur = stream.get('duration')
            if dur:
                return float(dur)
    except Exception as e:
        print(f"  [WARN] ffprobe failed for {path}: {e}")
    return -1.0


def trim_video(src: str, dst: str, max_sec: float) -> bool:
    """
    Trim video to max_sec seconds using ffmpeg.
    If video is already ≤ max_sec, just copies it (fast stream copy).
    Returns True on success.
    """
    duration = get_video_duration(src)

    if duration <= 0:
        print(f"  [WARN] Could not determine duration for {os.path.basename(src)} — copying as-is.")
        # Still try to copy with ffmpeg (it validates the file too)
        cmd = ['ffmpeg', '-y', '-i', src, '-c', 'copy', dst]
    elif duration <= max_sec:
        # Already short enough — stream copy (instant)
        cmd = ['ffmpeg', '-y', '-i', src, '-c', 'copy', dst]
        # print(f"  [OK ] {os.path.basename(src)} ({duration:.1f}s ≤ {max_sec}s) → copy")
    else:
        # Trim with re-encode to ensure clean output
        cmd = [
            'ffmpeg', '-y',
            '-i', src,
            '-t', str(max_sec),
            '-c:v', 'libx264',   # re-encode for clean cut
            '-crf', '23',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            dst
        ]
        # print(f"  [TRIM] {os.path.basename(src)} ({duration:.1f}s → {max_sec}s)")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  [ERROR] ffmpeg failed for {os.path.basename(src)}:")
            print(f"    {result.stderr[-300:] if result.stderr else 'no stderr'}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] ffmpeg timed out for {os.path.basename(src)}")
        return False
    except FileNotFoundError:
        print("[FATAL] ffmpeg not found. Install with: conda install -c conda-forge ffmpeg")
        sys.exit(1)


def process_split(in_dir: str, out_dir: str, max_sec: float, split: str):
    """Process one split folder (fake or real)."""
    src_dir = os.path.join(in_dir, split)
    dst_dir = os.path.join(out_dir, split)

    if not os.path.isdir(src_dir):
        print(f"[WARN] Source folder not found: {src_dir} — skipping.")
        return 0, 0

    os.makedirs(dst_dir, exist_ok=True)

    videos = sorted(
        glob.glob(os.path.join(src_dir, '*.mp4')) +
        glob.glob(os.path.join(src_dir, '*.avi')) +
        glob.glob(os.path.join(src_dir, '*.mov'))
    )

    if not videos:
        print(f"[WARN] No videos found in {src_dir}")
        return 0, 0

    print(f"\n[INFO] Processing {len(videos)} {split.upper()} videos...")
    ok, fail = 0, 0
    for src in videos:
        # Keep same extension as source (mp4 preferred for trimmed)
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(dst_dir, base + '.mp4')

        success = trim_video(src, dst, max_sec)
        if success:
            ok += 1
        else:
            fail += 1

    return ok, fail


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_dir  = os.path.normpath(os.path.join(script_dir, args.in_dir))
    out_dir = os.path.normpath(os.path.join(script_dir, args.out_dir))

    print("=" * 60)
    # print("  VIDEO TRIMMING (max {} seconds)".format(args.max_seconds))
    print("=" * 60)
    print(f"  Input  : {in_dir}")
    print(f"  Output : {out_dir}")

    total_ok = total_fail = 0
    for split in ['fake', 'real']:
        ok, fail = process_split(in_dir, out_dir, args.max_seconds, split)
        total_ok += ok
        total_fail += fail

    print("\n" + "=" * 60)
    print(f"  ✓ Trimming complete!")
    print(f"    Success : {total_ok}")
    print(f"    Failed  : {total_fail}")
    print(f"    Output  : {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
