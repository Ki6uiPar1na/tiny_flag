import math
from collections import Counter
from nltk import ngrams, FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.lm import Laplace, MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

# Morphological analysis
lem, st = WordNetLemmatizer(), PorterStemmer()
print("Morphological analysis:")
for w in ["unhappiness", "replayed", "cars", "mice", "walking"]:
    print(f"  {w:12s} stem={st.stem(w):9s} lemma(n)={lem.lemmatize(w):9s} lemma(v)={lem.lemmatize(w,'v')}")

corpus = ["i love natural language processing", "i love machine learning",
          "natural language processing is fun", "machine learning is fun"]
tok = [word_tokenize(s) for s in corpus]
text = " ".join(corpus).split()
print("\nBigrams :", list(ngrams(text[:7], 2)))
print("Trigrams:", list(ngrams(text[:7], 3)))

uni = Counter(text); bi = Counter(ngrams(text, 2)); V = len(uni)
def p_mle(w1, w2): return bi[(w1, w2)] / uni[w1] if uni[w1] else 0
def p_add1(w1, w2): return (bi[(w1, w2)] + 1) / (uni[w1] + V)
print(f"\nVocabulary size V = {V}")
print(f"{'bigram':28s}{'count':>6s}{'MLE':>9s}{'Laplace':>10s}")
for pair in [("i","love"), ("love","natural"), ("machine","learning"), ("learning","processing"), ("fun","i")]:
    print(f"{str(pair):28s}{bi[pair]:6d}{p_mle(*pair):9.3f}{p_add1(*pair):10.3f}")

# NLTK language model with Laplace smoothing; perplexity on an unseen sentence
train, vocab = padded_everygram_pipeline(2, tok)
lm = Laplace(2); lm.fit(train, vocab)
test = ["i", "love", "machine", "learning"]
print("\nNLTK Laplace P(love|i)       =", round(lm.score("love", ["i"]), 4))
print("NLTK Laplace P(processing|i) =", round(lm.score("processing", ["i"]), 4), "(unseen bigram, non-zero)")
from nltk.util import bigrams
print("Perplexity of 'i love machine learning' =", round(lm.perplexity(list(bigrams(test))), 3))
