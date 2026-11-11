# Text-to-Speech using gTTS; output saved as MP3 and converted to 16 kHz WAV for experiment 10
import subprocess, os
from gtts import gTTS
text = "Natural language processing allows computers to understand human speech."
os.makedirs("../audio", exist_ok=True)
gTTS(text=text, lang="en").save("../audio/tts_output.mp3")
subprocess.run(["ffmpeg","-y","-loglevel","error","-i","../audio/tts_output.mp3","-ar","16000","-ac","1","../audio/tts_output.wav"], check=True)
print("Input text :", text)
for f in ["tts_output.mp3","tts_output.wav"]:
    print(f"Created ../audio/{f}: {os.path.getsize('../audio/'+f)} bytes")
print(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1","../audio/tts_output.wav"],capture_output=True,text=True).stdout.strip())
