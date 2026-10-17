"""Qwen3-0.6B 多模态学生模型 + 软标签蒸馏。
学生结构: Whisper-tiny 编码器(冻结) + CLIP-ViT-B/32(冻结) -> 线性投影 -> 作为前缀 token
          与文本 embedding 拼接 -> Qwen3-0.6B (LoRA) -> 均值池化 -> 分类头
损失: alpha * CE(hard) + (1-alpha) * T^2 * KL(teacher || student)
用法: python train.py --train data/train.jsonl --train_soft data/train_soft.jsonl --val data/val.jsonl
"""
import json, argparse, torch, torch.nn as nn, torch.nn.functional as F, librosa
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModel, AutoTokenizer, WhisperModel, WhisperFeatureExtractor,
                          CLIPVisionModel, CLIPImageProcessor)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
L2I = {l: i for i, l in enumerate(LABELS)}
LLM = "d:/项目/ELLM/models/Qwen3-0.6B"
WHISPER = "d:/项目/ELLM/models/whisper-tiny"
CLIP = "d:/项目/ELLM/models/clip-vit-base-patch3"  # 本地目录名少了末尾的"2"，按实际填写


# ---------------- 数据 ----------------
class DS(Dataset):
    def __init__(self, path, soft_path=None):
        self.d = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.soft = {}
        if soft_path:
            for l in open(soft_path, encoding="utf-8"):
                r = json.loads(l); self.soft[r["id"]] = r["probs"]
            self.d = [s for s in self.d if s["id"] in self.soft]

    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]


class Collate:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(LLM)
        self.fe = WhisperFeatureExtractor.from_pretrained(WHISPER)
        self.ip = CLIPImageProcessor.from_pretrained(CLIP)

    def __call__(self, batch):
        wavs = [librosa.load(s["audio"], sr=16000)[0][:16000 * 30] for s in batch]
        a = self.fe(wavs, sampling_rate=16000, return_tensors="pt").input_features  # B,80,3000
        a_len = torch.tensor([min(1500, max(1, int(len(w) / 16000 * 50))) for w in wavs])
        pix = self.ip([Image.open(s["image"]).convert("RGB") for s in batch], return_tensors="pt").pixel_values
        t = self.tok([s.get("text", "") for s in batch], padding=True, truncation=True,
                     max_length=128, return_tensors="pt")
        hard = torch.tensor([L2I.get(s.get("label"), -100) for s in batch])
        soft = torch.tensor([s.get("probs", [1 / len(LABELS)] * len(LABELS)) for s in batch])
        return dict(a=a, a_len=a_len, pix=pix, ids=t.input_ids, mask=t.attention_mask, hard=hard, soft=soft)


def with_soft(ds):
    for s in ds.d:
        if s["id"] in ds.soft: s["probs"] = ds.soft[s["id"]]
    return ds


# ---------------- 学生模型 ----------------
class Student(nn.Module):
    def __init__(self, n_cls=len(LABELS), na=16, nv=8, lora=True):
        super().__init__()
        self.na, self.nv = na, nv
        lm = AutoModel.from_pretrained(LLM, torch_dtype=torch.float32)
        self.emb = lm.get_input_embeddings()
        H = lm.config.hidden_size
        if lora:
            lm = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        self.lm = lm
        self.aenc = WhisperModel.from_pretrained(WHISPER).encoder
        self.venc = CLIPVisionModel.from_pretrained(CLIP)
        for m in (self.aenc, self.venc):
            for p in m.parameters(): p.requires_grad = False
        self.ap = nn.Linear(self.aenc.config.d_model, H)
        self.vp = nn.Linear(self.venc.config.hidden_size, H)
        self.head = nn.Linear(H, n_cls)

    def train(self, mode=True):
        super().train(mode)
        self.aenc.eval(); self.venc.eval()  # 编码器始终 eval
        return self

    def forward(self, a, a_len, pix, ids, mask):
        with torch.no_grad():
            ah = self.aenc(a).last_hidden_state                 # B,1500,384
            vh = self.venc(pixel_values=pix).last_hidden_state  # B,50,768
        at = torch.stack([F.adaptive_avg_pool1d(ah[i, :a_len[i]].T[None], self.na)[0].T
                          for i in range(len(ah))])             # 只池化有效音频帧
        vt = F.adaptive_avg_pool1d(vh.transpose(1, 2), self.nv).transpose(1, 2)
        x = torch.cat([self.ap(at), self.vp(vt), self.emb(ids)], 1)
        m = torch.cat([torch.ones(len(ids), self.na + self.nv, device=mask.device, dtype=mask.dtype), mask], 1)
        h = self.lm(inputs_embeds=x, attention_mask=m).last_hidden_state
        m = m.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1))


# ---------------- 蒸馏损失 ----------------
def distill_loss(logits, soft, hard, T, alpha):
    logits = logits.float()
    t = soft.clamp_min(1e-8) ** (1 / T)
    t = t / t.sum(-1, keepdim=True)                      # 教师概率按温度重整
    kd = F.kl_div(F.log_softmax(logits / T, -1), t, reduction="batchmean") * T * T
    if (hard != -100).any():
        ce = F.cross_entropy(logits, hard, ignore_index=-100)
        return alpha * ce + (1 - alpha) * kd
    return kd


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); P, Y = [], []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            lg = model(b["a"], b["a_len"], b["pix"], b["ids"], b["mask"])
        P += lg.argmax(-1).tolist(); Y += b["hard"].tolist()
    keep = [i for i, y in enumerate(Y) if y != -100]
    P, Y = [P[i] for i in keep], [Y[i] for i in keep]
    return accuracy_score(Y, P), f1_score(Y, P, average="weighted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--train_soft", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=float, default=2.0); ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--out", default="student.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    print(f"[运行] device={dev}")

    col = Collate()
    tr = DataLoader(with_soft(DS(a.train, a.train_soft)), a.bs, shuffle=True, collate_fn=col, num_workers=0)
    va = DataLoader(DS(a.val), a.bs, collate_fn=col, num_workers=0)
    model = Student().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    total = a.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.1)

    best = 0
    for ep in range(a.epochs):
        model.train()
        for i, b in enumerate(tr):
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lg = model(b["a"], b["a_len"], b["pix"], b["ids"], b["mask"])
            loss = distill_loss(lg, b["soft"], b["hard"], a.T, a.alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if i % 20 == 0: print(f"ep{ep} step{i} loss {loss.item():.4f}")
        acc, f1 = evaluate(model, va, dev)
        print(f"== ep{ep} val acc {acc:.4f}  w-F1 {f1:.4f}")
        if f1 > best:
            best = f1
            train_keys = {n for n, p in model.named_parameters() if p.requires_grad}
            torch.save({k: v for k, v in model.state_dict().items() if k in train_keys}, a.out)  # 仅存可训练部分(LoRA/投影/分类头)


if __name__ == "__main__":
    main()
