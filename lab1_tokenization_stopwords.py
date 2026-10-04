# Lab 1: a) Tokenization  b) Stop word removal
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

for pkg in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)

text = ("JKKNIU stands as a dedicated center for higher education, honoring the "
        "profound cultural and literary legacy of Bangladesh's national poet.")

# a) Tokenization
print("Sentences:", sent_tokenize(text))
tokens = word_tokenize(text)
print("\nWord tokens:", tokens)
print("Total tokens:", len(tokens))

# b) Stop word removal
stop_words = set(stopwords.words("english"))
filtered = [w for w in tokens if w.lower() not in stop_words and w.isalnum()]
removed = [w for w in tokens if w.lower() in stop_words]

print("\nStop words removed:", removed)
print("After stop word removal:", filtered)
