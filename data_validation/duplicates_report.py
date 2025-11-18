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
    seen = set()
    dup = 0
    for item in data:
        key = (item.get("korean", "").strip(), item.get("english", "").strip())
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    print(f"duplicates,{dup},total,{len(data)}")

if __name__ == "__main__":
    main()
