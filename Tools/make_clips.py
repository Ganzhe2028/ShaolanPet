"""薄肌桌宠台词总表 —— 两期素材合并重切。

选材换了标准：上一版挑的是「好听的金句」，这一版挑的是「能接话的句子」。
answer=True 的进捧哏池，其余只在自己被点时用。

素材：
  A = 《为什么我劝所有男人练薄肌？》 BV1xa4k6vEpp  17:20
  B = 《如何练出薄肌？训练、饮食、减脂一次讲透》 BV1httT6LEfy  18:22
"""
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

# 复现这 37 句需要的输入，放在仓库的 Work/ 下（不进版本库，见 .gitignore）：
#   Work/v1/full44k.wav  Work/v1/transcript.json   《为什么我劝所有男人练薄肌？》
#   Work/v2/full44k.wav  Work/v2/transcript.json   《如何练出薄肌？训练、饮食、减脂一次讲透》
# wav 由 ffmpeg 从原视频音频解码得到，transcript.json 由 Tools/transcribe.py 生成。
ROOT = Path(__file__).resolve().parent.parent
P1 = ROOT / "Work" / "v1"
P2 = ROOT / "Work" / "v2"
OUT = ROOT / "Assets"
OUT.mkdir(exist_ok=True)
SR = 44100
TARGET_RMS = 0.10
PEAK_CEIL = 0.95
TRIM = 0.012

SOURCES = {
    "A": {"wav": P1 / "full44k.wav", "json": P1 / "transcript.json"},
    "B": {"wav": P2 / "full44k.wav", "json": P2 / "xunlian.json"},
}

# (源, 时间码, 字幕, 能接话)
CANDIDATES = [
    # ---------- 接话句：对别人的话做反应 ----------
    ("A", 100.10, "是我我也崩", True),
    ("A", 859.90, "怎么可能", True),
    ("A", 822.70, "没有意义", True),
    ("A", 900.80, "天赋没什么好聊的", True),
    ("A", 755.30, "不要找理由", True),
    ("A", 852.50, "不要去骗自己", True),
    ("A", 570.70, "AI 救不了你", True),
    ("A", 1037.16, "Let's do it", True),
    ("A", 845.70, "bro", True),
    ("B", 43.20, "何必呢 对吧", True),
    ("B", 132.00, "比你想象的还要难", True),
    ("B", 171.50, "总感觉不得劲", True),
    ("B", 179.70, "下一个", True),
    ("B", 257.20, "恰恰相反", True),
    ("B", 387.00, "doesn't matter", True),
    ("B", 392.50, "数字不会骗人", True),
    ("B", 450.20, "我不相信你没有空健身", True),
    ("B", 719.00, "不要怕", True),
    ("B", 732.50, "蛋白粉不是科技", True),
    ("B", 863.30, "当然可以吃", True),
    ("B", 888.60, "不用问我", True),
    ("B", 896.50, "没有欺骗餐", True),
    ("B", 916.80, "不开倒车", True),
    ("B", 957.90, "I don't do it", True),
    ("B", 1100.40, "我们顶峰相见", True),
    # ---------- 自说：梗和金句，接在别人后面会别扭 ----------
    ("B", 39.90, "过度健身吸引到的是同性", False),
    ("B", 178.20, "你 420 匹谁有耐心看", False),
    ("B", 226.80, "你就是版本气质了", False),
    ("B", 266.30, "把简单的事情复杂化", False),
    ("B", 561.60, "薄肌的核心逻辑就是要帅", False),
    ("B", 676.20, "我已经是版本之子了", False),
    ("B", 870.10, "中国人必须得吃饭", False),
    ("A", 290.82, "主权感就是一个超能力", False),
    ("A", 24.02, "刷牙 is everything", False),
    ("A", 423.88, "丝滑连招", False),
    ("A", 1034.30, "给我好好健身", False),
    ("A", 1035.18, "好好上分", False),
]

cache = {}
for key, info in SOURCES.items():
    with wave.open(str(info["wav"])) as w:
        cache[key] = (
            np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768,
            json.load(open(info["json"], encoding="utf-8")),
        )


def save(path: Path, data: np.ndarray):
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


clips, report = [], []
for idx, (src, t, label, answer) in enumerate(CANDIDATES, 1):
    audio, rows = cache[src]
    row = min(rows, key=lambda r: abs(r["start"] - t))
    if abs(row["start"] - t) > 0.6:
        report.append(f"!! {src}{t} 找不到，最近 {row['start']}「{row['text']}」")
        continue
    words = row["words"]
    first = words[0]["start"] if words else row["start"]
    last = words[-1]["end"] if words else row["end"]
    a, b = max(0.0, first - 0.05), min(len(audio) / SR, last + 0.18)

    chunk = audio[int(a * SR):int(b * SR)].copy()
    if len(chunk) < SR * 0.15:
        report.append(f"!! {label} 太短")
        continue
    loud = np.where(np.abs(chunk) > TRIM)[0]
    if len(loud) == 0:
        report.append(f"!! {label} 静音")
        continue
    chunk = chunk[max(0, loud[0] - int(SR * 0.02)): min(len(chunk), loud[-1] + int(SR * 0.06))]

    rms = float(np.sqrt(np.mean(chunk ** 2)))
    gain = TARGET_RMS / rms if rms > 1e-6 else 1.0
    if float(np.abs(chunk).max()) * gain > PEAK_CEIL:
        gain = PEAK_CEIL / float(np.abs(chunk).max())
    chunk = chunk * gain

    save(OUT / f"{idx:02d}.wav", chunk)
    dur = len(chunk) / SR
    clips.append({"id": idx, "label": label, "file": f"{idx:02d}.wav",
                  "duration": round(dur, 3), "answer": answer, "source": row["text"][:24]})
    report.append(f'{idx:02d} {dur:4.2f}s {"接话" if answer else "自说"}  {label}')

print("\n".join(report))
n_ans = sum(1 for c in clips if c["answer"])
print(f"\n共 {len(clips)} 句，能接话 {n_ans} 句，自说 {len(clips)-n_ans} 句")
(OUT / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
