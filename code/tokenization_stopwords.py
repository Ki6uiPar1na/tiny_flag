import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

text = ("Natural Language Processing enables computers to understand human language. "
        "It is widely used in chatbots, search engines, and translation systems!")
print("Original:", text)
print("\nSentence tokens:")
for s in sent_tokenize(text): print("  ", s)
words = word_tokenize(text)
print("\nWord tokens:", words)
stop = set(stopwords.words("english"))
filtered = [w for w in words if w.lower() not in stop and w.isalnum()]
print("\nAfter stop-word removal:", filtered)
print(f"\nTokens before: {len(words)}  after: {len(filtered)}")
