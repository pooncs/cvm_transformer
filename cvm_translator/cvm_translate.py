import torch
from cvm_translator.cvm_buffer import CVMBuffer


def build_core_kv(embed, tokens, core_indices):
    core_ids = torch.tensor([tokens.index(i) for i in core_indices], dtype=torch.long)
    x = embed(core_ids)
    k = x.clone()
    v = x.clone()
    return k, v


def cvm_translate_chunk(model, tokenizer, text, direction, core_capacity=64):
    tokens = tokenizer.encode(text)
    core_buf = CVMBuffer(core_capacity)
    for t in tokens:
        core_buf.add(t)
    core_indices = core_buf.cores()
    core_kv = build_core_kv(model.embed, tokens, core_indices)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    logits = model(input_ids, core_indices=core_indices, core_kv=core_kv)
    pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
    out_tokens = [str(tokenizer.decode([i])) for i in pred_ids]
    return " ".join(out_tokens)