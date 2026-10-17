"""把 RAVDESS 转成本项目的 jsonl 格式，支持音频/视频两种输入，可混放，自动识别。

文件名: 模态-声道-情感-强度-语句-重复-演员.扩展名
- 视频 (Video_Speech/Song_Actor_XX): 01-01-*.mp4(语音) / 01-02-*.mp4(歌唱) / 02-02-*.mp4(仅视频)
  提取 16kHz 单声道 wav，并截取 1.2s 处的一帧 jpg（需要系统装有 ffmpeg）
- 音频 (Audio_Speech_Actors_01-24): 03-01-*.wav
  本身就是 16kHz 单声道，直接引用原文件，不需要 ffmpeg

每条记录含 id/text/audio/label/modality；只有视频输入才有 image 字段。
按演员划分（说话人独立）：--train_actors 1-6 --val_actors 7-8
calm 并入 neutral，保持 7 类
用法: python prepare_ravdess.py --root ravdess_raw --train_actors 1-6 --val_actors 7-8 [--only_video]
"""
import os, glob, json, argparse, subprocess, shutil

EMO = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
       "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"}
SENT = {"01": "Kids are talking by the door.", "02": "Dogs are sitting by the door."}


def rng(s):
    a, _, b = s.partition("-")
    return set(range(int(a), int(b or a) + 1))


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="解压后的目录，递归查找 mp4/wav")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train_actors", default="1-6")
    ap.add_argument("--val_actors", default="7-8")
    ap.add_argument("--only_video", action="store_true", help="只处理视频(mp4)，跳过纯音频(wav)")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        raise SystemExit(f"[错误] 数据目录不存在: {a.root}\n"
                         "       请先把 RAVDESS 的 Audio_Speech_Actors_01-24 或\n"
                         "       Video_Speech_Actor_XX 解压到该目录。")

    files = sorted(glob.glob(os.path.join(a.root, "**", "*.mp4"), recursive=True)
                   + glob.glob(os.path.join(a.root, "**", "*.wav"), recursive=True))
    if a.only_video:
        files = [f for f in files if f.lower().endswith(".mp4")]
    if not files:
        raise SystemExit(f"[错误] 在 {a.root} 下没找到任何 .mp4 / .wav 文件。")
    if any(f.lower().endswith(".mp4") for f in files) and not shutil.which("ffmpeg"):
        raise SystemExit("[错误] 输入含 mp4，需要 ffmpeg 抽音频和取帧，但 PATH 里找不到 ffmpeg。")

    splits = {"train": rng(a.train_actors), "val": rng(a.val_actors)}
    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    rows = {"train": [], "val": []}
    skip = {"bad_name": 0, "modality": 0, "ext": 0, "actor": 0, "emotion": 0}

    for src in files:
        ext = os.path.splitext(src)[1].lower()
        name = os.path.splitext(os.path.basename(src))[0]
        p = name.split("-")
        if len(p) != 7 or p[4] not in SENT:
            skip["bad_name"] += 1; continue
        modality, channel, emo = p[0], p[1], p[2]
        if modality not in ("01", "02", "03"):  # 01=全AV, 02=仅视频, 03=纯音频
            skip["modality"] += 1; continue
        # 声道: 01=语音 / 02=歌唱，两种都接受
        if ext != (".mp4" if modality in ("01", "02") else ".wav"):
            skip["ext"] += 1; continue
        try:
            actor = int(p[6])
        except ValueError:
            skip["bad_name"] += 1; continue
        split = next((k for k, v in splits.items() if actor in v), None)
        if split is None:
            skip["actor"] += 1; continue
        if emo not in EMO:
            skip["emotion"] += 1; continue

        row = {"id": name, "text": SENT[p[4]], "label": EMO[emo],
               "modality": "av" if modality in ("01", "02") else "audio",
               "channel": "speech" if channel == "01" else "song"}
        if modality in ("01", "02"):
            wav = os.path.join(a.out, "audio", name + ".wav")
            jpg = os.path.join(a.out, "img", name + ".jpg")
            if not os.path.exists(wav):
                run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav])
            if not os.path.exists(jpg):
                run(["ffmpeg", "-y", "-ss", "1.2", "-i", src, "-frames:v", "1", "-vf", "scale=-2:360", jpg])
            row["audio"], row["image"] = wav, jpg
        else:
            row["audio"] = src          # RAVDESS 音频已是 16kHz 单声道，直接用
        rows[split].append(row)

    for k, v in rows.items():
        with open(os.path.join(a.out, f"{k}.jsonl"), "w", encoding="utf-8") as f:
            for r in v: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_av = sum(1 for r in v if r["modality"] == "av")
        print(f"{k}: {len(v)} 条 (音频 {len(v) - n_av}, 音视频 {n_av})")

    print(f"[统计] 扫描 {len(files)} 个文件 | 跳过: 文件名不合规 {skip['bad_name']}, "
          f"模态非 01/02/03 {skip['modality']}, "
          f"扩展名与模态不符 {skip['ext']}, 演员不在划分内 {skip['actor']}, "
          f"未知情感 {skip['emotion']}")
    if not rows["train"] and not rows["val"]:
        print("[提示] 没有匹配到任何样本，请检查 --root 路径、RAVDESS 原始文件名，"
              "以及 --train_actors/--val_actors 是否覆盖了数据里的演员编号。")


if __name__ == "__main__":
    main()
