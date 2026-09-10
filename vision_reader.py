#!/usr/bin/env python3
"""
vision_reader.py — 读取视频帧图片，通过 OpenAI 兼容的视觉接口返回文字描述。

用于解决主模型不支持读图的问题：
抽帧由 ffmpeg 完成，本脚本只负责把每一帧图发给外部视觉模型，
拿回结构化文字描述，再交给主模型合成时间线与动作流程。

支持自动 fallback：如果主模型返回穿搭/购物建议而非画面描述，
自动用备选模型重试。

环境变量（.env 文件）：
  OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_VISION_MODEL      主配置
  FALLBACK_BASE_URL / FALLBACK_API_KEY / FALLBACK_VISION_MODEL  备选配置

用法：
  python vision_reader.py frame_0001.jpg                       # 单图 + 默认 prompt
  python vision_reader.py *.jpg --workers 4                     # 批量并行
  python vision_reader.py frames/frame_%04d.jpg --start 1 --end 7  # printf 格式
  python vision_reader.py frame.jpg --prompt-file prompt.txt    # 从文件读指令
  python vision_reader.py frame.jpg --no-fallback               # 禁用 fallback
  python vision_reader.py frame.jpg --force-fallback            # 强制用 fallback 模型
"""

import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT = 180

# ---------- 环境变量加载 ----------

def load_env_file(path: Path) -> dict:
    """读取 KEY=VALUE 形式的 .env 文件，忽略注释与空行。"""
    env = {}
    if not path.exists():
        return env
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip("'").strip('"')
    except Exception as e:
        print(f"[warn] 读取 {path} 失败: {e}", file=sys.stderr)
    return env


def get_env(name: str) -> str:
    """优先取系统环境变量，其次取 .env 文件。"""
    val = os.environ.get(name)
    if val:
        return val
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        env = load_env_file(base / ".env")
        if name in env:
            return env[name]
    return ""


# ---------- 图片处理 ----------

def image_to_data_url(path: str) -> str:
    """读取图片为 base64 data URL。不做缩放——抽帧时已压到合适尺寸。"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")
    return f"data:{mime};base64,{b64}"


# ---------- 共享 HTTP Session ----------

_shared_session: requests.Session | None = None

def _get_session() -> requests.Session:
    """获取全局共享 Session（复用连接池）。"""
    global _shared_session
    if _shared_session is None:
        _shared_session = requests.Session()
        retry = Retry(
            total=2, read=2, connect=2, backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        _shared_session.mount("http://", adapter)
        _shared_session.mount("https://", adapter)
    return _shared_session


# ---------- 调用视觉接口 ----------

SYSTEM_PROMPT = (
    "你是一个专业的视频帧序列分析工具。"
    "你收到的图片是一个**视频片段中按时间顺序抽出的帧序列中的一张**。"
    "你的任务：客观描述这一帧画面的视觉内容。"
    "\n\n"
    "描述重点："
    "1. 人物：数量、性别、外貌、穿着、表情、姿态"
    "2. 动作：人物正在做什么动作？手的位置、身体朝向、脚步状态"
    "3. 与前一帧对比：有什么变化？(人物移动了位置？姿势变了？场景变了？)"
    "4. 场景：室内/室外、背景物品、建筑、光线方向、天气"
    "\n\n"
    "注意：既然这是视频中的一帧，画面中的人物可能在运动中。"
    "请描述你看到的瞬间状态，不要只说'这是一张静态图片'。"
    "绝对禁止：不要给出穿搭推荐、购物链接、品牌建议、风格评价、"
    "时尚趋势分析或任何购买建议。只描述你看到的画面。"
    "如果无法确定，就说'无法确定'。"
)


def _chat_payload(model: str, prompt: str, image_path: str,
                  frame_index: int = 0, total_frames: int = 0):
    frame_context = ""
    if total_frames > 0:
        frame_context = (
            f"\n[帧位置] 这是本视频帧序列的第 {frame_index}/{total_frames} 帧，"
            f"时间戳约 {frame_index - 1} 秒。\n"
        )
    user_text = frame_context + prompt
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url",
                     "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1,
    }


def call_one(image_path: str, prompt: str,
             base_url: str, api_key: str, model: str,
             timeout: int = DEFAULT_TIMEOUT,
             label: str = "",
             frame_index: int = 0, total_frames: int = 0) -> str:
    """单张图读一次，返回文字描述。label 用于日志标识主/备。"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = _chat_payload(model, prompt, image_path,
                            frame_index=frame_index,
                            total_frames=total_frames)

    tag = label or model
    sess = _get_session()
    try:
        resp = sess.post(url, headers=headers, json=payload,
                         timeout=timeout, stream=True)
        resp.raise_for_status()
        raw = resp.content  # bytes, avoids charset issues
        data = json.loads(raw.decode("utf-8"))
        try:
            text = data["choices"][0]["message"]["content"].strip()
            # safe print: strip non-ASCII for stderr on GBK terminals
            safe_name = Path(image_path).name
            print(f"  [{tag}] {safe_name} -> {len(text)} chars",
                  file=sys.stderr)
            return text, tag
        except (KeyError, IndexError, TypeError):
            return json.dumps(data, ensure_ascii=False)[:2000], tag
    except Exception as e:
        raise RuntimeError(f"[{tag}] 读取 {Path(image_path).name} 失败: {e}")


# ---------- 购物/穿搭建议检测 ----------

_SHOP_KEYWORDS = [
    "search for", "buy", "shop", "purchase", "where to find", "look for",
    "available at", "carry similar", "相似单品", "购买", "同款",
    "搜索关键词", "品牌", "Reformation", "Zara", "ASOS", "H&M",
    "price", "affordable", "购物", "预算", "穿搭推荐",
    "搭配建议", "search terms", "关键词搜索", "在哪里买",
]


def _is_shopping_advice(text: str) -> bool:
    """检测响应是否是穿搭/购物建议而非画面描述。"""
    count = sum(1 for kw in _SHOP_KEYWORDS if kw.lower() in text.lower())
    return count >= 3


# ---------- 带 fallback 的读图 ----------

def read_with_fallback(image_path: str, prompt: str,
                       primary: tuple, fallback: tuple | None,
                       timeout: int = DEFAULT_TIMEOUT,
                       frame_index: int = 0, total_frames: int = 0) -> str:
    """
    primary = (base_url, api_key, model)
    fallback = (base_url, api_key, model) 或 None
    返回描述文本 + 使用的模型标签。
    """
    # 尝试主模型
    try:
        text, tag = call_one(image_path, prompt, *primary,
                             timeout=timeout, label="primary",
                             frame_index=frame_index, total_frames=total_frames)
        if fallback and _is_shopping_advice(text):
            print(f"  [*] 主模型返回疑似购物建议，切换到 fallback",
                  file=sys.stderr)
            text2, tag2 = call_one(image_path, prompt, *fallback,
                                   timeout=timeout, label="fallback",
                                   frame_index=frame_index, total_frames=total_frames)
            # 如果 fallback 也返回购物建议，就返回第一个（可能是模型倾向问题）
            if _is_shopping_advice(text2):
                print(f"  [*] Fallback 同样返回购物建议，使用主模型结果",
                      file=sys.stderr)
                return text, tag
            return text2, tag2
        return text, tag
    except Exception as e:
        if fallback:
            print(f"  [*] 主模型失败，切换到 fallback: {e}", file=sys.stderr)
            try:
                text2, tag2 = call_one(image_path, prompt, *fallback,
                                       timeout=timeout, label="fallback",
                                       frame_index=frame_index, total_frames=total_frames)
                return text2, tag2
            except Exception as e2:
                raise RuntimeError(f"主/备模型均失败. 主: {e} | 备: {e2}")
        raise


# ---------- 批量 ----------

def run_batch(image_paths, prompt,
              primary, fallback, workers, timeout=DEFAULT_TIMEOUT):
    results = {}
    total = len(image_paths)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for i, p in enumerate(image_paths, start=1):
            fut = ex.submit(read_with_fallback, p, prompt, primary, fallback,
                            timeout, frame_index=i, total_frames=total)
            futs[fut] = p
        for fut in cf.as_completed(futs):
            p = futs[fut]
            try:
                text, tag = fut.result()
                results[p] = text
                out = Path(p).with_suffix(".txt")
                out.write_text(text, encoding="utf-8")
            except Exception as e:
                results[p] = f"[ERROR] {e}"
    return results


# ---------- 展开文件列表 ----------

def expand_paths(spec: str, start: int | None, end: int | None,
                 step: int | None) -> list[str]:
    p = Path(spec)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(str(x) for x in p.glob("*.jpg")) + \
               sorted(str(x) for x in p.glob("*.png"))
    import glob as g
    files = sorted(g.glob(spec))
    if not files and ("%" in spec):
        m = re.search(r"%0?(\d+)d", spec)
        if m:
            pad = int(m.group(1))
            head, _, tail = spec.partition("%0" + str(pad) + "d") or \
                            spec.partition("%" + str(pad) + "d")
            lo = start if start is not None else 1
            hi = end if end is not None else lo + 499
            files = [f"{head}{i:0{pad}d}{tail}" for i in range(lo, hi + 1, step or 1)]
    return [f for f in files if Path(f).exists()]


# ---------- main ----------

def main():
    # Windows GBK 终端安全：stdout/stderr 用 utf-8 写，非 ASCII 替换而非崩溃
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="OpenAI 兼容视觉接口读图（带自动 fallback）")
    ap.add_argument("input", help="图片路径 / 目录 / glob / printf格式")
    ap.add_argument("prompt", nargs="?", default=None,
                    help="分析指令。留空则从 --prompt-file 或 stdin 读取")
    ap.add_argument("--prompt-file", default=None,
                    help="从文件读取分析指令（推荐，避免中文编码问题）")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--no-fallback", action="store_true",
                    help="禁用自动 fallback")
    ap.add_argument("--force-fallback", action="store_true",
                    help="强制使用 fallback 模型")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"请求超时（秒，默认 {DEFAULT_TIMEOUT}）")
    args = ap.parse_args()

    # 读取 prompt
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        prompt = "请描述这个画面：人物、动作、场景、关键细节。"

    # 展开文件
    files = expand_paths(args.input, args.start, args.end, args.step)
    if not files:
        print(f"错误：未找到图片（输入: {args.input}）", file=sys.stderr)
        sys.exit(2)

    # 构建配置
    primary = (
        get_env("OPENAI_BASE_URL"),
        get_env("OPENAI_API_KEY"),
        get_env("OPENAI_VISION_MODEL") or DEFAULT_MODEL,
    )
    fallback = None
    if not args.no_fallback and not args.force_fallback:
        fb_url = get_env("FALLBACK_BASE_URL")
        fb_key = get_env("FALLBACK_API_KEY")
        fb_model = get_env("FALLBACK_VISION_MODEL")
        if fb_url and fb_key and fb_model:
            fallback = (fb_url, fb_key, fb_model)
    if args.force_fallback:
        fb_url = get_env("FALLBACK_BASE_URL")
        fb_key = get_env("FALLBACK_API_KEY")
        fb_model = get_env("FALLBACK_VISION_MODEL")
        if not fb_url or not fb_key or not fb_model:
            print("错误：--force-fallback 但未配置 FALLBACK_* 环境变量", file=sys.stderr)
            sys.exit(2)
        primary = (fb_url, fb_key, fb_model)
        fallback = None  # 本身就是 fallback,不需要再 fallback

    if not primary[0] or not primary[1]:
        print("错误：需要设置 OPENAI_BASE_URL 和 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    mode = "force-fallback" if args.force_fallback else \
           ("no-fallback" if args.no_fallback else
            f"primary+fallback({fallback[2]})" if fallback else "primary-only")
    print(f"[info] 模式={mode} 模型={primary[2]} 图片={len(files)} workers={args.workers}",
          file=sys.stderr)
    if fallback:
        print(f"[info] fallback={fallback[2]} {fallback[0]}",
              file=sys.stderr)

    results = run_batch(files, prompt, primary, fallback, args.workers,
                        timeout=args.timeout)

    if args.json:
        # json 输出带元数据
        out = {"_meta": {"model": primary[2], "mode": mode, "files": len(files)}}
        out.update(results)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for p in files:
            print(f"===== {Path(p).name} =====")
            print(results[p])
            print()


if __name__ == "__main__":
    main()