# Grapheme-to-Phoneme conversion: dictionary lookup (CMUdict) + rule-based fallback for OOV words
import re
from nltk.corpus import cmudict
cmu = cmudict.dict()

# Simplified letter-to-sound rules (ordered; longest grapheme first)
RULES = [("tch","CH"),("sh","SH"),("ch","CH"),("th","TH"),("ph","F"),("ng","NG"),("ck","K"),("qu","K W"),
         ("ee","IY"),("oo","UW"),("ai","EY"),("ay","EY"),("oa","OW"),("ou","AW"),
         ("a","AE"),("e","EH"),("i","IH"),("o","AA"),("u","AH"),("b","B"),("c","K"),("d","D"),("f","F"),
         ("g","G"),("h","HH"),("j","JH"),("k","K"),("l","L"),("m","M"),("n","N"),("p","P"),("r","R"),
         ("s","S"),("t","T"),("v","V"),("w","W"),("x","K S"),("y","Y"),("z","Z")]

def rule_g2p(word):
    out, i = [], 0
    while i < len(word):
        for g, p in RULES:
            if word.startswith(g, i):
                out.append(p); i += len(g); break
        else:
            i += 1
    return " ".join(out).split()

def g2p(word):
    w = word.lower()
    if w in cmu: return cmu[w][0], "dictionary"
    return rule_g2p(w), "rules"

for sent in ["speech technology changes language", "the quick brown fox", "blorpting chatbots tokenizer"]:
    print("\nSentence:", sent)
    for w in re.findall(r"[a-z']+", sent):
        ph, src = g2p(w)
        print(f"  {w:12s} /{' '.join(ph):32s}/ [{src}]")
