# TikTok AI 服装视频导演 Agent

> TikTok AI Video Director
> 专注于服装类 TikTok UGC 视频的参考视频拆解、镜头重构、产品展示设计，以及 Wan 2.2 图生视频 Prompt 编写。

> **一句话**: 把 TikTok 服装 UGC 视频**重新导演**成 Wan 2.2 图生视频可执行的分段方案（不是"描述视频"，而是"理解它为什么有效，再让 AI 拍出来"）。

## 快速开始

```bash
# 1. 依赖
#    - ffmpeg / ffprobe（抽帧、元数据）
#    - python 3 + requests（读图脚本）
#    - 一个 OpenAI 兼容的视觉 API（读图，因主模型不支持读图）

# 2. 配置视觉 API（.env，不入 git）
cp .env.example .env   # 或手动创建，见下方「视觉模型配置」

# 3. 把参考视频放到仓库根目录，然后让 Claude 分析：
#    分析 <视频文件名>.mp4
# 产物会写到 tmp/<视频文件名>/ 下（帧图、逐帧描述、report.md 导演报告）
```

详细流程见下文「工程说明」。

---

# 工程说明

## 概述

将 TikTok 服装类 UGC 视频**逐帧拆解**，获取人物动作流程、时间线与产品展示信息，输出可供 Wan 2.2 图生视频使用的结构化分析报告。

## 工作流

### 快速模式（推荐）
```text
python analyze_video.py <参考视频.mp4>
    │
    ├─ ffprobe → 元数据
    ├─ ffmpeg  → 抽帧 → tmp/<视频名>/frames/
    ├─ vision_reader → 视觉读帧 → frames/*.txt
    └─ 拼接 → tmp/<视频名>/analysis.txt （精简 stdout）
            │
            ▼
        Claude 读 analysis.txt → 合成 report.md
```

### 手动完整流程
```text
参考视频.mp4
    │
    ├─ ffprobe → 元数据（时长、分辨率、帧率）
    ├─ ffmpeg  → 抽帧（640px, 1fps, q:v 6）→ tmp/<视频名>/frames/
    └─ vision_reader.py → 视觉模型读帧 → tmp/<视频名>/frames/*.txt
            │
            ▼
        主模型合成时间线 + 动作流程报告 → tmp/<视频名>/report.md
```

## 文件说明

| 文件 | 用途 |
|---|---|
| `vision_reader.py` | 抽帧结果发视觉 API，支持主/备双模型自动 fallback |
| `analyze_video.py` | 固化流程脚本：一条命令跑完probe→抽帧→读图→拼analysis.txt，输出精简 |
| `.env` | API 配置（key、base_url、模型名）**不入 git** |
| `.env` | API 配置（key、base_url、模型名）**不入 git** |
| `.env.example` | 配置模板（无真实密钥，可入库） |
| `.gitignore` | 忽略 `.env` 与 `tmp/` |
| `README.md` | 项目介绍 + 工程说明 |
| `角色.md` | Agent 角色规范（30 条导演规则），供主模型遵循 |
| `tmp/` | 临时目录：每视频独立子目录存放全部分析产物 |

## 视觉模型配置（.env）

主模型 `gemini-2.5-flash`（ai.hybgzs.com/v1），备选 `vision`（octopus.ollia.top/v1）。
自动检测穿搭/购物建议并 fallback。

修改 `.env` 即可切换模型。

## 使用

```bash
# === 快速模式（推荐）===
python analyze_video.py <video.mp4> [--fps 1] [--scale 640] [--q 6]

# === 手动单步 ===
# 1. 放视频到当前目录
cp /path/to/video.mp4 .

# 2. 让 Claude 分析（一次性全流程）
# 直接告诉 Claude：分析 video.mp4

# 3. 或手动跑单帧
python vision_reader.py tmp/<视频名>/frames/frame_0003.jpg --prompt-file tmp/<视频名>/prompt.txt
```

## tmp/ 目录约定（重要）

每视频分析产物必须放在**各自独立子目录**下，禁止共用同一份 report/frames/prompt：

```text
tmp/<视频文件名(不含扩展名)>/
    frames/           # 抽帧 jpg + 逐帧 txt
    analysis.txt      # 逐帧描述的拼接
    report.md         # 导演报告
    prompt.txt        # 视觉 prompt
```

> **Why:** 曾因所有视频共用 tmp/report.md、frames/，导致第二个视频覆盖了第一个视频的帧和报告。
>
> **How:** 分析每个视频前先 `mkdir -p tmp/<视频名>`，所有产物（含首帧图）只写进该目录。

## 注意

- **主模型（deepseek-v4-flash）不支持读图**，所以读图步骤外包给云端视觉 API
- 所有中间产物放在 `tmp/` 下，分析完可删除（.gitignore 已排除）

