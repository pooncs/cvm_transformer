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
    kr_count = 0
    en_count = 0
    for item in data:
        k = item.get("korean", "")
        e = item.get("english", "")
        if any("\uac00" <= ch <= "\ud7a3" for ch in k):
            kr_count += 1
        if any(ch.isascii() for ch in e):
            en_count += 1
    print(f"kr_ok,{kr_count},en_ok,{en_count},total,{len(data)}")

if __name__ == "__main__":
    main()
