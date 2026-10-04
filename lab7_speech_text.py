# Lab 7: Audio -> Text and Text -> Audio
# NLTK itself has no audio support; it is used here for text processing,
# while SpeechRecognition (speech->text) and gTTS (text->speech) do the audio part.
#
# Install:  pip install nltk SpeechRecognition gTTS pydub
# Note: SpeechRecognition reads WAV/AIFF/FLAC. Convert mp3 -> wav with pydub (needs ffmpeg).

import nltk
import speech_recognition as sr
from gtts import gTTS
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

TEXT = ("Jatiya Kabi Kazi Nazrul Islam University (JKKNIU) is a premier public "
        "university in Trishal, Mymensingh, established in 2006.")

# ---------- b) Text -> Audio ----------
def text_to_audio(text, out_file="jkkniu.mp3"):
    gTTS(text=text, lang="en").save(out_file)      # needs internet
    print("Audio saved as:", out_file)
    return out_file

# ---------- a) Audio -> Text ----------
def audio_to_text(wav_file):
    r = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)           # needs internet
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"API error: {e}"

if __name__ == "__main__":
    mp3 = text_to_audio(TEXT)

    # convert mp3 -> wav so SpeechRecognition can read it
    from pydub import AudioSegment
    AudioSegment.from_mp3(mp3).export("jkkniu.wav", format="wav")

    recognized = audio_to_text("jkkniu.wav")
    print("\nRecognized text:", recognized)
    print("NLTK tokens    :", word_tokenize(recognized))
