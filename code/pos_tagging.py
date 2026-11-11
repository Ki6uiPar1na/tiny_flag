import nltk
from nltk.tokenize import word_tokenize
text = "The quick brown fox jumps over the lazy dog while she was reading a book."
tags = nltk.pos_tag(word_tokenize(text))
print("Tagged:", tags)
print("\nUniversal tagset:", nltk.pos_tag(word_tokenize(text), tagset="universal"))
print("\nWord -> POS lookup")
for w in ["fox", "jumps", "quick", "over", "she"]:
    print(f"  {w:6s}", dict(tags)[w])
from collections import Counter
print("\nTag counts:", Counter(t for _, t in tags))
print("\nTag meaning: NN noun, VBZ verb 3rd sg, JJ adjective, DT determiner, IN preposition, PRP pronoun")
nltk.help.upenn_tagset("NN")
