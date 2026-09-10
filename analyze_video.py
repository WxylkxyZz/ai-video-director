#!/usr/bin/env python3
"""
analyze_video.py — 固化 TikTok 服装视频分析流程。

一条命令跑完：ffprobe 元数据 → ffmpeg 抽帧 → vision_reader 视觉读图 → 拼接 analysis.txt。
stdout 只打印精简摘要，不刷屏；导演报告（report.md）由 Claude 读 analysis.txt 后合成。

用法：
  python analyze_video.py <video.mp4>
  python analyze_video.py <video.mp4> --fps 2 --workers 6 --no-fallback

产物统一写入 tmp/<视频文件名不含扩展名>/（frames/、prompt.txt、analysis.txt）。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import vision_reader  # 复用 get_env / run_batch

DEFAULT_PROMPT = (
    "客观描述这一帧画面：人物（数量、性别、外貌、穿着、姿态）、"
    "正在进行的动作（手部/身体/脚步的具体状态）、场景（室内/室外、背景物品、光线方向）。"
    "这是视频帧序列中的一帧，人物可能处于运动中，请描述此刻的瞬间状态。"
    "不要给穿搭推荐、购物链接、品牌建议或风格评价，只描述看到的内容。"
)


def probe(video: Path) -> dict:
    """用文本行解析元数据，避开 JSON 对中文/反斜杠路径的转义崩溃。"""
    cmd = [
        "ffprobe", "-hide_banner",
        "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate",
        "-of", "default=noprint_wrappers=1",
        "-v", "quiet", str(video),
    ]
    out = subprocess.check_output(cmd, encoding="utf-8", timeout=60).splitlines()
    meta = {"duration": "", "codec": "", "width": "", "height": "", "fps": ""}
    for line in out:
        key, _, val = line.partition("=")
        key = key.strip()
        if key == "duration" and not meta["duration"]:
            meta["duration"] = val.strip()
        elif key == "codec_name" and not meta["codec"]:
            meta["codec"] = val.strip()
        elif key == "width":
            meta["width"] = val.strip()
        elif key == "height":
            meta["height"] = val.strip()
        elif key == "avg_frame_rate" and val.strip() not in ("0/0", ""):
            meta["fps"] = val.strip()
    return meta


def extract_frames(video: Path, workdir: Path, fps: int, scale: int, q: int) -> int:
    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # 清空旧帧，防止上次同目录残留
    for old in frames_dir.glob("*"):
        if old.is_file():
            old.unlink()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", f"fps={fps},scale={scale}:-2",
        "-q:v", str(q),
        str(frames_dir / "frame_%04d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(frames_dir.glob("frame_*.jpg")))


def read_frames(workdir: Path, workers: int, timeout: int, no_fallback: bool,
                shard: bool):
    frames = sorted((workdir / "frames").glob("frame_*.jpg"))
    prompt = workdir / "prompt.txt"
    prompt.write_text(DEFAULT_PROMPT, encoding="utf-8")

    if shard:
        channels = vision_reader.load_channels()
        if not channels:
            print("错误：--shard 但未找到任何渠道（检查 .env）", file=sys.stderr)
            sys.exit(2)
        vision_reader.run_batch_sharded(frames, prompt.read_text(encoding="utf-8"),
                                        channels, workers, timeout)
        return channels[0][2], channels  # model_name, channels

    primary = (
        vision_reader.get_env("OPENAI_BASE_URL"),
        vision_reader.get_env("OPENAI_API_KEY"),
        vision_reader.get_env("OPENAI_VISION_MODEL") or "gemini-2.5-flash",
    )
    fallback = None
    if not no_fallback:
        fb_url = vision_reader.get_env("FALLBACK_BASE_URL")
        fb_key = vision_reader.get_env("FALLBACK_API_KEY")
        fb_model = vision_reader.get_env("FALLBACK_VISION_MODEL")
        if fb_url and fb_key and fb_model:
            fallback = (fb_url, fb_key, fb_model)

    if not primary[0] or not primary[1]:
        print("错误：需要设置 OPENAI_BASE_URL 和 OPENAI_API_KEY（.env）", file=sys.stderr)
        sys.exit(2)

    vision_reader.run_batch(frames, prompt.read_text(encoding="utf-8"),
                            primary, fallback, workers, timeout)
    return primary[2], fallback


def concat_analysis(workdir: Path):
    frames = sorted((workdir / "frames").glob("frame_*.jpg"))
    analysis = workdir / "analysis.txt"
    lines = []
    for f in frames:
        txt = f.with_suffix(".txt")
        lines.append(f"===== {f.name} =====")
        lines.append(txt.read_text(encoding="utf-8").strip() if txt.exists() else "[无描述]")
        lines.append("")
    analysis.write_text("\n".join(lines), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="固化 TikTok 服装视频分析流程")
    ap.add_argument("video", help="参考视频路径")
    ap.add_argument("--fps", type=int, default=1, help="抽帧频率（默认 1，快动作可 2）")
    ap.add_argument("--scale", type=int, default=640, help="帧图宽度（默认 640）")
    ap.add_argument("--q", type=int, default=6, help="JPEG 质量 2-31，越小质量越高（默认 6）")
    ap.add_argument("--workers", type=int, default=6, help="全局视觉 API 并发数（默认 6，越多越快但易撞限流）")
    ap.add_argument("--timeout", type=int, default=180, help="单帧 API 超时（秒）")
    ap.add_argument("--no-fallback", action="store_true", help="禁用备用视觉模型")
    ap.add_argument("--shard", action="store_true",
                    help="多渠道分片负载：每帧轮流发不同渠道，避免单渠道限流")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        print(f"错误：未找到视频 {video}", file=sys.stderr)
        sys.exit(2)

    workdir = Path("tmp") / video.stem
    workdir.mkdir(parents=True, exist_ok=True)

    # ---- 分阶段计时 ----
    t_all = time.perf_counter()
    timings = {}

    t0 = time.perf_counter()
    meta = probe(video)
    (workdir / "duration.txt").write_text(meta["duration"], encoding="utf-8")
    timings["probe"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n = extract_frames(video, workdir, args.fps, args.scale, args.q)
    timings["extract"] = time.perf_counter() - t0
    if n == 0:
        print("错误：抽帧结果为空", file=sys.stderr)
        sys.exit(2)

    t0 = time.perf_counter()
    primary_model, channels = read_frames(
        workdir, args.workers, args.timeout, args.no_fallback, args.shard)
    timings["vision_read"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    concat_analysis(workdir)
    timings["concat"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_all

    # read_frames 返回 (primary_model, channels)：
    #   shard 模式 -> channels 为渠道列表;  非 shard 模式 -> channels 为 fallback 元组或 None
    if args.shard and isinstance(channels, list) and channels:
        model_mode = "shard(" + ",".join(c[3] for c in channels) + ")"
    else:
        model_mode = primary_model
        if isinstance(channels, tuple) and channels:
            model_mode += f" + fallback({channels[2]})"
        elif args.no_fallback:
            model_mode += "（no-fallback）"

    dur = meta["duration"] or "未知"
    res = f"{meta['width']}x{meta['height']}" if meta["width"] else "未知"
    print(f"视频: {workdir.name}")
    print(f"分辨率: {res} | 时长: {dur}s | codec: {meta['codec'] or '未知'}")
    print(f"帧数: {n}")
    print(f"模型: {model_mode}")
    print(f"产物: {workdir / 'frames'} / {workdir / 'analysis.txt'}")
    print("耗时(墙钟, 即真实经过时间):")
    print(f"  probe(元数据探测):      {timings['probe']:.2f}s")
    print(f"  extract(抽帧):          {timings['extract']:.2f}s")
    print(f"  vision_read(视觉读图):  {timings['vision_read']:.2f}s  ({n}帧 @ {args.workers}路并发)")
    print(f"  concat(拼 analysis):    {timings['concat']:.2f}s")
    print(f"  total(总耗时):          {timings['total']:.2f}s")

    (workdir / "timing.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()