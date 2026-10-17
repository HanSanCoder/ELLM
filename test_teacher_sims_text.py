"""测试远程教师模型（Qwen Omni API）仅凭文本在 CH-SIMS 上的情感三分类准确率。

背景:
  CH-SIMS 的 label.csv 自带 text 列（转录文本），因此无需下载原始视频/音频，
  即可用纯文本模态评测教师模型。情感三分类: Negative / Neutral / Positive，
  另有连续回归值 label。

与 test_teacher_sims.py（多模态）的区别:
  本脚本只把 transcript 发给教师，不抽音频、不取帧，可在无媒体文件时直接运行。

用法:
  python test_teacher_sims_text.py --csv CH-SIMS/raw/MSA-Datasets/SIMS/label.csv --split test --limit 50

断点续跑: 结果写进 --out 指定的 jsonl，已处理的 id 下次自动跳过；重跑请删除该文件。
"""
import os, csv, json, argparse, time, random
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# ---- 复用 gen_soft_labels 的默认教师配置，避免复制密钥 ----
try:
    from gen_soft_labels import client as _gs_client, MODEL as _gs_model
    _DEFAULT_KEY = getattr(_gs_client, "api_key", None)
    _DEFAULT_BASE = getattr(_gs_client, "base_url", None)
    DEFAULT_MODEL = _gs_model
except Exception:
    _DEFAULT_KEY = os.environ.get("DASHSCOPE_API_KEY")
    _DEFAULT_BASE = os.environ.get("DASHSCOPE_BASE_URL")
    DEFAULT_MODEL = "qwen3.8-omni-flash"

SENTIMENTS = ["Negative", "Neutral", "Positive"]
SENT2VAL = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}

PROMPT = (
    "You are an expert in sentiment analysis. Based on the transcript below, judge the "
    "overall sentiment polarity.\n"
    'Transcript: "{text}"\n'
    "Output ONLY one word among: Negative, Neutral, Positive."
)


def parse_pred(txt):
    up = (txt or "").strip().upper()
    for s in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        if s in up:
            return s.capitalize()
    return None


def ask_once(client, model, text, temperature):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": [{"type": "text", "text": PROMPT.format(text=text)}]}],
        modalities=["text"], stream=True, temperature=temperature,
    )
    txt = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    return parse_pred(txt)


def process_one(row, args, client):
    uid = f"{row['video_id']}_{row['clip_id']}"
    base = {"id": uid, "text": row["text"], "true": row["annotation"],
            "label": float(row["label"])}
    for k in range(args.retries):
        try:
            base["pred"] = ask_once(client, args.model, row["text"], args.temperature)
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
    ap.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    ap.add_argument("--limit", type=int, default=0, help="最多测多少条，0=全部")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=_DEFAULT_KEY)
    ap.add_argument("--base-url", default=_DEFAULT_BASE)
    ap.add_argument("--dry-run", action="store_true", help="只检查数据，不调用 API")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("[错误] 未找到 API 密钥：请在 gen_soft_labels.py 配置，或用 --api-key 传入。")

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    if args.split != "all":
        rows = [r for r in rows if r.get("mode") == args.split]
    if not rows:
        raise SystemExit(f"[错误] split={args.split} 下没有样本，请检查 label.csv 的 mode 列。")
    if args.limit and args.limit < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.limit)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                   f"teacher_text_preds_{args.split}.jsonl")

    done = {}
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r
    todo = [r for r in rows if f"{r['video_id']}_{r['clip_id']}" not in done]

    if args.dry_run:
        print(f"[dry-run] split={args.split} 共 {len(rows)} 条，待测 {len(todo)} 条（纯文本，无需媒体文件）")
        return

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
