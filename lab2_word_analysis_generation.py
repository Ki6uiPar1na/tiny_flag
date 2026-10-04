# Lab 2: a) Word Analysis  b) Word Generation
import nltk, itertools, random
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

text = ("Jatiya Kabi Kazi Nazrul Islam University is a premier public university "
        "in Trishal, Mymensingh, established in 2006.")

# ---------- a) Word Analysis ----------
print("=== a) WORD ANALYSIS ===")
words = [w for w in word_tokenize(text) if w.isalpha()]
stemmer, lemmatizer = PorterStemmer(), WordNetLemmatizer()
vowels = set("aeiouAEIOU")

print(f"{'Word':<14}{'Len':<5}{'Vowels':<8}{'Cons.':<7}{'Stem':<12}{'Lemma':<12}")
for w in words:
    v = sum(c in vowels for c in w)
    print(f"{w:<14}{len(w):<5}{v:<8}{len(w)-v:<7}{stemmer.stem(w):<12}{lemmatizer.lemmatize(w.lower()):<12}")

freq = Counter(w.lower() for w in words)
print("\nWord frequency:", dict(freq))
print("Longest word  :", max(words, key=len))
print("Average length: %.2f" % (sum(map(len, words)) / len(words)))

# ---------- b) Word Generation ----------
print("\n=== b) WORD GENERATION ===")
roots = ["play", "work", "teach", "walk"]
suffixes = ["s", "ed", "ing", "er"]
prefixes = ["re", "un", "pre"]

print("Suffix-based generation:")
for r in roots:
    print(" ", r, "->", [r + s for s in suffixes])

print("Prefix + root generation:")
for p, r in itertools.product(prefixes, roots[:2]):
    print(f"  {p} + {r} = {p + r}")

print("Permutation-based generation from 'cat':",
      sorted({"".join(p) for p in itertools.permutations("cat")}))

random.seed(1)
letters = "abcdefghijklmnopqrstuvwxyz"
print("Random 5-letter strings:", ["".join(random.choices(letters, k=5)) for _ in range(3)])
