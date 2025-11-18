import json
import sys
from pathlib import Path

def load_preds(paths):
    preds = []
    for p in paths:
        with Path(p).open("r", encoding="utf-8") as f:
            preds.append(json.load(f))
    return preds

def weighted_average(preds, weights):
    out = []
    for i in range(len(preds[0])):
        scores = {}
        for j, pr in enumerate(preds):
            hyp = pr[i]["hypothesis"]
            scores[hyp] = scores.get(hyp, 0.0) + weights[j]
        best = max(scores.items(), key=lambda x: x[1])[0]
        out.append({"hypothesis": best})
    return out

def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print("usage: blend.py out.json w1,p1 w2,p2 ...")
        sys.exit(1)
    out = args[0]
    pairs = args[1:]
    paths = []
    weights = []
    for pair in pairs:
        w,p = pair.split(",")
        weights.append(float(w))
        paths.append(p)
    preds = load_preds(paths)
    blended = weighted_average(preds, weights)
    with Path(out).open("w", encoding="utf-8") as f:
        json.dump(blended, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
