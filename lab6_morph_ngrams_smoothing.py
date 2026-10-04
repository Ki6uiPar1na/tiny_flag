# Lab 6: a) Morphological analysis  b) N-grams  c) N-gram smoothing
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.util import ngrams
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

text = ("Jatiya Kabi Kazi Nazrul Islam University is a premier public university "
        "in Trishal Mymensingh established in 2006 and the university is growing")
tokens = [t.lower() for t in word_tokenize(text) if t.isalnum()]

# ---------- a) Morphological Analysis ----------
print("=== a) MORPHOLOGICAL ANALYSIS ===")
stemmer, lem = PorterStemmer(), WordNetLemmatizer()
prefixes = ("un", "re", "pre", "dis", "in")
suffixes = ("ing", "ed", "ly", "s", "er", "ment", "ness", "ion")

for w in ["established", "growing", "universities", "happily", "unhappiness", "teachers"]:
    stem = stemmer.stem(w)
    lemma = lem.lemmatize(w, pos="v") if w.endswith(("ed", "ing")) else lem.lemmatize(w)
    pre = next((p for p in prefixes if w.startswith(p) and len(w) > len(p) + 2), "-")
    suf = next((s for s in suffixes if w.endswith(s)), "-")
    print(f"{w:<14} stem={stem:<10} lemma={lemma:<12} prefix={pre:<4} suffix={suf:<5} "
          f"morphy={wn.morphy(w)}")

# ---------- b) N-grams ----------
print("\n=== b) N-GRAMS ===")
for n, name in [(1, "Unigrams"), (2, "Bigrams"), (3, "Trigrams")]:
    grams = list(ngrams(tokens, n))
    print(f"{name} ({len(grams)}): {grams[:6]} ...")

# ---------- c) N-gram Smoothing ----------
print("\n=== c) SMOOTHING ===")
# Manual Add-One (Laplace) smoothing for bigrams
unigram_counts = Counter(tokens)
bigram_counts = Counter(ngrams(tokens, 2))
V = len(unigram_counts)

def laplace_bigram(w1, w2):
    return (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)

def mle_bigram(w1, w2):
    return bigram_counts[(w1, w2)] / unigram_counts[w1] if unigram_counts[w1] else 0

for pair in [("is", "a"), ("university", "is"), ("is", "premier")]:
    print(f"P{pair}: MLE={mle_bigram(*pair):.4f}  Laplace={laplace_bigram(*pair):.4f}")

# NLTK built-in Laplace model (bigram)
train, vocab = padded_everygram_pipeline(2, [tokens])
model = Laplace(2)
model.fit(train, vocab)
print("\nNLTK Laplace P(a | is)        =", round(model.score("a", ["is"]), 4))
print("NLTK Laplace P(poet | is) [unseen word] =", round(model.score("poet", ["is"]), 4))
