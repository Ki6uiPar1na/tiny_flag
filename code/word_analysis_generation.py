# Word analysis (frequency, length, vocabulary) and word generation (affixes + WordNet derivations)
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

text = ("The quick brown fox jumps over the lazy dog. The dog barks and the fox runs away. "
        "Running foxes are quick and clever.")
words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
print("Total words      :", len(words))
print("Unique words     :", len(set(words)))
print("Lexical diversity:", round(len(set(words))/len(words), 3))
print("Avg word length  :", round(sum(map(len, words))/len(words), 2))
print("Top 5 frequent   :", Counter(words).most_common(5))
print("Longest word     :", max(words, key=len))

# Word generation: rule based prefix/suffix attachment
prefixes = ["un", "re", "pre", "dis"]
suffixes = ["ing", "ed", "er", "ly", "ness"]
def generate(root):
    out = [root]
    out += [p + root for p in prefixes]
    out += [root + s for s in suffixes]
    return out
for root in ["play", "happy", "kind"]:
    print(f"\nRoot '{root}' ->", generate(root))

# Generation via WordNet lemmas (derivationally related forms)
print("\nDerivationally related forms (WordNet):")
for w in ["act", "create", "decide"]:
    forms = {l.name() for s in wn.synsets(w) for lem in s.lemmas()
             for l in lem.derivationally_related_forms()}
    print(f"  {w:7s}->", sorted(forms)[:10])
