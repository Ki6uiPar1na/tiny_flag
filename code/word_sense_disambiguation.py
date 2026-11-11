from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

tests = [("I went to the bank to deposit my money", "bank"),
         ("The boat was tied up on the bank of the river", "bank"),
         ("He used a bat to hit the ball", "bat"),
         ("A bat flew out of the cave at night", "bat"),
         ("Python is a popular programming language", "python")]
for sent, w in tests:
    s = lesk(word_tokenize(sent), w)
    print(f"Sentence : {sent}\nWord     : {w}\nSense    : {s.name() if s else None}\nGloss    : {s.definition() if s else None}\n")
