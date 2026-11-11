# Viterbi decoding for POS tag ambiguity using transition and emission probabilities
states = ["N", "V", "D"]
start = {"N": 0.3, "V": 0.1, "D": 0.6}
trans = {"D": {"N": 0.8, "V": 0.1, "D": 0.1},
         "N": {"N": 0.2, "V": 0.6, "D": 0.2},
         "V": {"N": 0.4, "V": 0.1, "D": 0.5}}
emit = {"D": {"the": 0.7, "a": 0.3},
        "N": {"fish": 0.3, "can": 0.1, "saw": 0.1, "book": 0.2, "dog": 0.3},
        "V": {"fish": 0.2, "can": 0.3, "saw": 0.3, "book": 0.2}}

def viterbi(obs):
    V = [{s: start[s] * emit[s].get(obs[0], 1e-6) for s in states}]
    back = [{}]
    for t in range(1, len(obs)):
        V.append({}); back.append({})
        for s in states:
            p, prev = max((V[t-1][r] * trans[r][s] * emit[s].get(obs[t], 1e-6), r) for r in states)
            V[t][s], back[t][s] = p, prev
    last = max(V[-1], key=V[-1].get)
    path = [last]
    for t in range(len(obs)-1, 0, -1): path.insert(0, back[t][path[0]])
    return path, V

for sent in ["the dog saw a fish", "the fish can", "the book fish"]:
    obs = sent.split()
    path, V = viterbi(obs)
    print(f"\nSentence: {sent}")
    for t, w in enumerate(obs):
        print(f"  {w:5s} " + "  ".join(f"{s}:{V[t][s]:.6f}" for s in states))
    print("  Best tags:", list(zip(obs, path)))
