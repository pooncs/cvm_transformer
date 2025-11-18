import json
import sys
from pathlib import Path

def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/processed_large_simple/train_data.json"
    p = Path(inp)
    if not p.exists():
        print("missing")
        sys.exit(1)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ok = 0
    for item in data:
        if isinstance(item, dict) and "korean" in item and "english" in item:
            ok += 1
    print(f"valid_items,{ok},total,{len(data)}")

if __name__ == "__main__":
    main()
