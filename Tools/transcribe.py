"""转写一个音频文件。用法: transcribe_one.py <input> <output.json> <label>

跟主项目同样的坑：40 行那句注释不是废话 —— faster-whisper 收到 numpy 数组
会当成 16k，所以必须先用 ffmpeg 转 16k 单声道再喂，绕开 PyAV 也绕开慢放。
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

src = Path(sys.argv[1])
out = Path(sys.argv[2])
base = src.parent

wav = base / (src.stem + "_16k.wav")
if not wav.exists():
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-ar", "16000", "-ac", "1", str(wav)], check=True)
    print(f"wav ready {wav.stat().st_size/1e6:.1f}MB", flush=True)

with wave.open(str(wav)) as w:
    rate = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
print(f"{len(a)/rate:.1f}s", flush=True)

model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8,
                     download_root=str(src.parent / "models"))
segments, info = model.transcribe(a, language="zh", word_timestamps=True, beam_size=5, vad_filter=True)

rows = []
for seg in segments:
    rows.append({
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": seg.text.strip(),
        "words": [{"start": round(x.start, 3), "end": round(x.end, 3), "word": x.word} for x in (seg.words or [])],
    })
    print(f"[{seg.start:7.1f}] {seg.text.strip()}", flush=True)

out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n{len(rows)} segments -> {out}", flush=True)
