# Lab 5: POS tagging of 10+ words and finding POS of a given word
import nltk
from nltk import pos_tag, word_tokenize

for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "universal_tagset"]:
    nltk.download(pkg, quiet=True)

# Sample list (12 words)
words = ["university", "student", "runs", "beautiful", "quickly", "and",
         "in", "he", "teach", "poet", "dedicated", "honoring"]

tags = pos_tag(words)
tags_univ = dict(pos_tag(words, tagset="universal"))

print(f"{'Word':<14}{'Penn Tag':<10}{'Universal'}")
for w, t in tags:
    print(f"{w:<14}{t:<10}{tags_univ[w]}")

# Find POS for any given word / sentence
lookup = dict(tags)
while True:
    user = input("\nEnter a word or sentence (q to quit): ").strip()
    if user.lower() == "q":
        break
    if user in lookup:
        print("From sample list:", user, "->", lookup[user])
    print("Tagger output   :", pos_tag(word_tokenize(user)))
