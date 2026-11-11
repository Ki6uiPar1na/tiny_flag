# Offline ASR with Vosk (Kaldi). Transcribes audio and reports word error rate against the reference.
import wave, json, sys
from vosk import Model, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)
MODEL = sys.argv[1]
reference = "natural language processing allows computers to understand human speech"
wf = wave.open("../audio/tts_output.wav", "rb")
rec = KaldiRecognizer(Model(MODEL), wf.getframerate()); rec.SetWords(True)
while True:
    data = wf.readframes(4000)
    if not data: break
    rec.AcceptWaveform(data)
res = json.loads(rec.FinalResult())
hyp = res["text"]
print("Reference :", reference)
print("Recognised:", hyp)
for w in res.get("result", []):
    print(f"  {w['word']:12s} {w['start']:.2f}-{w['end']:.2f}s  conf={w['conf']:.2f}")

def wer(r, h):
    r, h = r.split(), h.split()
    d = [[i+j if i*j == 0 else 0 for j in range(len(h)+1)] for i in range(len(r)+1)]
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(r[i-1] != h[j-1]))
    return d[-1][-1] / len(r)
print(f"Word Error Rate: {wer(reference, hyp):.2%}")
