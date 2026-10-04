# Lab 3: Word Sense Disambiguation (Lesk algorithm)
import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# Sample list: ambiguous word -> sentences using different senses
samples = [
    ("bank",   "I deposited my salary at the bank this morning."),
    ("bank",   "The fisherman sat on the bank of the river."),
    ("bat",    "He hit the ball with a wooden bat."),
    ("bat",    "A bat flew out of the cave at night."),
    ("bark",   "The dog began to bark loudly at strangers."),
    ("bark",   "The bark of the old tree was rough."),
    ("spring", "Flowers bloom in spring after the winter."),
    ("spring", "The mattress has a metal spring inside."),
    ("light",  "Please switch on the light, it is dark."),
    ("light",  "This bag is very light and easy to carry."),
]

print("Ambiguous words:", sorted({w for w, _ in samples}), "\n")

for word, sentence in samples:
    sense = lesk(word_tokenize(sentence), word)
    print("Sentence :", sentence)
    print("Word     :", word)
    if sense:
        print("Sense    :", sense.name())
        print("Meaning  :", sense.definition())
    else:
        print("Sense    : not found")
    print("-" * 60)

# Interactive
w = input("Enter an ambiguous word (or press Enter to skip): ").strip()
if w:
    s = input("Enter a sentence containing it: ")
    sense = lesk(word_tokenize(s), w)
    print("Predicted sense:", sense.definition() if sense else "None")
