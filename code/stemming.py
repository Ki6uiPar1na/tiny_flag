from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
words = ["running", "runs", "ran", "easily", "fairly", "studies", "studying", "happiness",
         "generously", "connection", "connected", "connecting", "better", "caring"]
p, l, s, lem = PorterStemmer(), LancasterStemmer(), SnowballStemmer("english"), WordNetLemmatizer()
print(f"{'Word':14s}{'Porter':12s}{'Lancaster':12s}{'Snowball':12s}{'Lemma(v)':12s}")
for w in words:
    print(f"{w:14s}{p.stem(w):12s}{l.stem(w):12s}{s.stem(w):12s}{lem.lemmatize(w,'v'):12s}")
