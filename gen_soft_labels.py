"""用 Qwen2.5-Omni (API) 为每个样本生成情感软标签（概率分布）。
输入 jsonl 每行: {"id","text","audio":"x.wav","image":"x.jpg","label":"happy"(可选)}
输出 jsonl 每行: {"id","probs":[...]}  顺序与 LABELS 一致
用法: python gen_soft_labels.py --data data/train.jsonl --out data/train_soft.jsonl --n 3
"""
import os, re, json, base64, argparse, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

LABELS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
MODEL = "qwen3.8-omni-flash"
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url='https://llm-gpsckmaeplaws7x6.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
)
PROMPT = (
    "You are an expert in multimodal emotion recognition. Based on the speaker's voice, "
    "facial expression and the transcript below, estimate the probability of each emotion.\n"
    'Transcript: "{text}"\nEmotions: {labels}\n'
    "Output ONLY a JSON object mapping every emotion to a probability, summing to 1."
)


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def ask_once(s, temperature):
    content = []
    if s.get("image"):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(s['image'])}"}})
    if s.get("audio"):
        content.append({"type": "input_audio", "input_audio": {"data": f"data:;base64,{b64(s['audio'])}", "format": "wav"}})
    content.append({"type": "text", "text": PROMPT.format(text=s.get("text", ""), labels=", ".join(LABELS))})
    # Omni 模型要求 stream=True；只要文本输出
    stream = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": content}],
        modalities=["text"], stream=True, temperature=temperature,
    )
    txt = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    d = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    p = [max(float(d.get(l, 0.0)), 0.0) for l in LABELS]
    return [x / sum(p) for x in p]


def soft_label(s, n, eps=0.01):
    outs = []
    for _ in range(n):
        for k in range(3):  # 重试
            try:
                outs.append(ask_once(s, 0.2 if n == 1 else 0.7))
                break
            except Exception as e:
                time.sleep(2 * (k + 1))
    if not outs:
        return None
    p = [sum(o[i] for o in outs) / len(outs) for i in range(len(LABELS))]
    p = [(x + eps) for x in p]  # 平滑，避免 KL 中出现 0
    z = sum(p)
    return {"id": s["id"], "probs": [x / z for x in p]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="每个样本采样次数，取平均让软标签更平滑")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    data = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    done = set()
    if os.path.exists(a.out):  # 断点续跑
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")}
    todo = [s for s in data if s["id"] not in done]
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda s: soft_label(s, a.n), todo):
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
    print(f"done: {len(todo)} samples")
