import json
import os
import torch
import torch.nn as nn
from pathlib import Path

def run(teacher_ckpt, student_out, temperature=2.0, alpha=0.5):
    t = torch.load(teacher_ckpt, map_location="cpu")
    sd = t["model_state_dict"] if isinstance(t, dict) and "model_state_dict" in t else t
    teacher_logits = sd
    torch.save({"teacher": teacher_logits, "T": temperature, "alpha": alpha}, student_out)

if __name__ == "__main__":
    import sys
    teacher = sys.argv[1] if len(sys.argv) > 1 else "models/extended/best_model.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "distillation/student_seed.pt"
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    a = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)
    run(teacher, out, T, a)
