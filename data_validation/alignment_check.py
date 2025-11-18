import sys
from pathlib import Path
import json

def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/processed_large_simple/train_data.json"
    p = Path(inp)
    if not p.exists():
        print("missing")
        sys.exit(1)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    bad = 0
    for item in data:
        k = item.get("korean", "").strip()
        e = item.get("english", "").strip()
        if not k or not e:
            bad += 1
    print(f"empty_or_misaligned,{bad},total,{len(data)}")

if __name__ == "__main__":
    main()
