"""测试远程教师模型（Qwen3.8-Omni API）在 CH-SIMS 数据集上的情感分类准确率。

背景:
  CH-SIMS 是中文多模态情感分析数据集，2281 个短视频片段，每个片段含文本(transcript)、
  音频(语音)和人脸视频。情感三分类: Negative / Neutral / Positive，另有连续回归值 label。
  教师模型复用 gen_soft_labels.py 里配置的 Qwen Omni 多模态 API。

输入:
  --csv        CH-SIMS 的 label.csv（列: video_id,clip_id,text,label,annotation,mode）
  --media-root 原始视频根目录，结构: <media-root>/Raw/<video_id>/<clip_id>.mp4
               （与 MMSA 官方结构一致）

视频下载（本项目 CH-SIMS 下载只含 label.csv，原始 mp4 需另行下载）:
  Google Drive: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk
  BaiduYun:     https://pan.baidu.com/s/1a1bDX5htPsZjsRyHcvCKHw  提取码 qq0b
  下载后把解压出的 Raw 目录放到与 label.csv 同级（即 --media-root 下）。
  --split      测哪个划分: train / valid / test / all（默认 test，457 条）
  --limit      最多测多少条（默认 0 = 全部）

流程:
  1. 读 label.csv 并按 split 过滤、可选采样
  2. ffmpeg 从 mp4 抽 16kHz 单声道 wav + 1 帧 jpg（缓存到 --cache-dir，支持断点续跑）
  3. 把 文本+音频+图像 发给多模态教师，让其输出 Negative/Neutral/Positive
  4. 与 annotation 对比，输出 accuracy / F1 / 混淆矩阵；再用连续 label 算回归 MAE 与相关系数

依赖: openai、ffmpeg；API 密钥复用 gen_soft_labels.py（也可用 --api-key / --base-url 覆盖）。

用法:
  python test_teacher_sims.py --csv CH-SIMS/raw/MSA-Datasets/SIMS/label.csv ^
      --media-root CH-SIMS/raw/MSA-Datasets/SIMS --split test --limit 50

断点续跑: 结果写进 --out 指定的 jsonl，已处理的 id 下次自动跳过；若想重跑请删除该文件。
"""
import os, csv, json, argparse, base64, time, random, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# ---- 复用 gen_soft_labels 的默认教师配置，避免在新文件里复制密钥 ----
try:
    from gen_soft_labels import client as _gs_client, MODEL as _gs_model
    _DEFAULT_KEY = getattr(_gs_client, "api_key", None)
    _DEFAULT_BASE = getattr(_gs_client, "base_url", None)
    DEFAULT_MODEL = _gs_model
except Exception:  # 独立运行时的兜底
    _DEFAULT_KEY = os.environ.get("DASHSCOPE_API_KEY")
    _DEFAULT_BASE = os.environ.get("DASHSCOPE_BASE_URL")
    DEFAULT_MODEL = "qwen3.8-omni-flash"

SENTIMENTS = ["Negative", "Neutral", "Positive"]
SENT2VAL = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}

PROMPT = (
    "You are an expert in multimodal sentiment analysis. Based on the speaker's voice, "
    "facial expression and the transcript below, judge the overall sentiment polarity "
    "of this video clip.\n"
    'Transcript: "{text}"\n'
    "Output ONLY one word among: Negative, Neutral, Positive."
)


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def parse_pred(txt):
    up = (txt or "").strip().upper()
    for s in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        if s in up:
            return s.capitalize()
    return None


def extract_audio(video, wav):
    subprocess.run(["ffmpeg", "-y", "-i", video, "-ac", "1", "-ar", "16000",
                    "-loglevel", "quiet", wav], capture_output=True)
    return os.path.exists(wav) and os.path.getsize(wav) > 0


def extract_frame(video, jpg, t):
    for tt in (t, 0.1, 0.0):  # 帧时间太靠后取不到就逐步回退
        subprocess.run(["ffmpeg", "-y", "-ss", str(tt), "-i", video, "-frames:v", "1",
                        "-vf", "scale=-2:360", "-loglevel", "quiet", jpg], capture_output=True)
        if os.path.exists(jpg) and os.path.getsize(jpg) > 0:
            return True
    return False


def ask_once(client, model, text, wav, jpg, temperature):
    content = []
    if jpg and os.path.exists(jpg):
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64(jpg)}"}})
    if wav and os.path.exists(wav):
        content.append({"type": "input_audio",
                        "input_audio": {"data": f"data:;base64,{b64(wav)}", "format": "wav"}})
    content.append({"type": "text", "text": PROMPT.format(text=text)})
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        modalities=["text"], stream=True, temperature=temperature,
    )
    txt = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    return parse_pred(txt)


def process_one(row, args, client):
    vid, cid = row["video_id"], row["clip_id"]
    uid = f"{vid}_{cid}"
    video = os.path.join(args.media_root, "Raw", vid, cid + ".mp4")
    base = {"id": uid, "text": row["text"], "true": row["annotation"],
            "label": float(row["label"])}
    if not os.path.exists(video):
        base["pred"], base["error"] = None, "no_video"
        return base

    d = os.path.join(args.cache_dir, vid)
    os.makedirs(d, exist_ok=True)
    wav = os.path.join(d, cid + ".wav")
    jpg = os.path.join(d, cid + ".jpg")
    if not os.path.exists(wav) and not extract_audio(video, wav):
        base["pred"], base["error"] = None, "audio_fail"
        return base
    if not os.path.exists(jpg):
        extract_frame(video, jpg, args.frame_time)

    for k in range(args.retries):
        try:
            base["pred"] = ask_once(client, args.model, row["text"], wav, jpg, args.temperature)
            base.pop("error", None)
            return base
        except Exception as e:
            if k == args.retries - 1:
                base["pred"], base["error"] = None, str(e)[:200]
            else:
                time.sleep(2 * (k + 1))
    return base


def pearson(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def report(results):
    from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                                 classification_report)
    ok = [r for r in results if r.get("pred") is not None]
    err = [r for r in results if r.get("pred") is None]
    if err:
        from collections import Counter
        print(f"[统计] 成功 {len(ok)} / 共 {len(results)}，失败原因: {dict(Counter(r.get('error') for r in err))}")
    if not ok:
        print("[结果] 没有成功预测的样本")
        return

    true = [r["true"] for r in ok]
    pred = [r["pred"] for r in ok]
    print(f"[结果] 样本 {len(ok)}  accuracy {accuracy_score(true, pred):.4f}  "
          f"macro-F1 {f1_score(true, pred, average='macro'):.4f}  "
          f"weighted-F1 {f1_score(true, pred, average='weighted'):.4f}")
    print(classification_report(true, pred, labels=SENTIMENTS, zero_division=0))
    print("混淆矩阵 (行=真实, 列=预测) 顺序:", SENTIMENTS)
    print(confusion_matrix(true, pred, labels=SENTIMENTS))

    tv = [float(r["label"]) for r in ok]
    pv = [SENT2VAL[r["pred"]] for r in ok]
    mae = sum(abs(a - b) for a, b in zip(pv, tv)) / len(tv)
    print(f"[回归] 预测极性映射到 {{-1,0,1}} 后  MAE {mae:.4f}  Pearson {pearson(pv, tv):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--media-root", required=True)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    ap.add_argument("--limit", type=int, default=0, help="最多测多少条，0=全部")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--frame-time", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=_DEFAULT_KEY)
    ap.add_argument("--base-url", default=_DEFAULT_BASE)
    ap.add_argument("--dry-run", action="store_true", help="只检查数据与媒体文件，不调用 API")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("[错误] 未找到 API 密钥：请在 gen_soft_labels.py 配置，或用 --api-key 传入。")
    if not shutil.which("ffmpeg"):
        print("[警告] 未找到 ffmpeg，抽音频/取帧会失败（纯文本仍可测）。")

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    if args.split != "all":
        rows = [r for r in rows if r.get("mode") == args.split]
    if not rows:
        raise SystemExit(f"[错误] split={args.split} 下没有样本，请检查 label.csv 的 mode 列。")
    if args.limit and args.limit < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.limit)

    cache_dir = args.cache_dir or os.path.join(args.media_root, "_test_cache")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                   f"teacher_preds_{args.split}.jsonl")

    done = {}
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r
    todo = [r for r in rows if f"{r['video_id']}_{r['clip_id']}" not in done]

    def video_path(r):
        return os.path.join(args.media_root, "Raw", r["video_id"], r["clip_id"] + ".mp4")

    if args.dry_run:
        missing = sum(1 for r in rows if not os.path.exists(video_path(r)))
        print(f"[dry-run] split={args.split} 共 {len(rows)} 条，缺视频 {missing} 条")
        return

    args.cache_dir = cache_dir
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    results = list(done.values())
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(lambda r: process_one(r, args, client), todo):
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()

    report(results)
    print(f"[完成] 结果已写入 {out}")


if __name__ == "__main__":
    main()
