# Lab 4: Install NLTK and perform stemming
# Install first:  pip install nltk
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
print("NLTK version:", nltk.__version__)

porter, lancaster, snowball = PorterStemmer(), LancasterStemmer(), SnowballStemmer("english")

words = ["running", "runs", "easily", "studies", "studying", "generous",
         "generation", "happily", "connected", "connection", "universities"]

print(f"\n{'Word':<15}{'Porter':<14}{'Lancaster':<14}{'Snowball':<14}")
for w in words:
    print(f"{w:<15}{porter.stem(w):<14}{lancaster.stem(w):<14}{snowball.stem(w):<14}")

sentence = "Students are studying natural language processing at the university"
print("\nSentence stemming:")
print([porter.stem(t) for t in word_tokenize(sentence)])
