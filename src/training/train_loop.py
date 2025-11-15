import torch
from torch.utils.data import Dataset, DataLoader
from cvm_translator.cvm_transformer import CVMTransformer
from cvm_translator.kd_losses import compute_loss


class BiTextDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.tokenizer.encode(src)[:self.max_len]
        tgt_ids = self.tokenizer.encode(tgt)[:self.max_len]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }


def collate(batch):
    src_ids = torch.nn.utils.rnn.pad_sequence([b["src_ids"] for b in batch], batch_first=True, padding_value=0)
    tgt_ids = torch.nn.utils.rnn.pad_sequence([b["tgt_ids"] for b in batch], batch_first=True, padding_value=0)
    return {"src_ids": src_ids, "tgt_ids": tgt_ids}


def train_epoch(student, teacher, loader, optimizer, device, core_capacity=64):
    student.train()
    teacher.eval()
    total_loss = 0
    for batch in loader:
        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)
        optimizer.zero_grad()
        s_logits, s_h, s_attn = student(src, return_hidden=True, return_attn=True)
        with torch.no_grad():
            t_logits, t_h, t_attn = teacher(src, return_hidden=True, return_attn=True)
        # Align target length with student output length for CE stability in demo
        tgt_aligned = tgt[:, :s_logits.size(1)]
        loss = compute_loss((s_logits, s_h, s_attn), (t_logits, t_h, t_attn), tgt_aligned)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 32000
    student = CVMTransformer(vocab_size, d_model=768, n_layers=6, core_capacity=64).to(device)
    teacher = CVMTransformer(vocab_size, d_model=768, n_layers=12, core_capacity=256).to(device)
    pairs = [("안녕하세요", "Hello"), ("오늘 날씨 좋네요", "Today weather is nice")] * 100
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {"<pad>": 0, "<unk>": 1}
        def encode(self, text):
            return [self.vocab.get(ch, self.vocab["<unk>"]) for ch in text]
    tokenizer = SimpleTokenizer()
    dataset = BiTextDataset(pairs, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    for epoch in range(3):
        loss = train_epoch(student, teacher, loader, optimizer, device)
        print(f"Epoch {epoch+1}, loss={loss:.4f}")


if __name__ == "__main__":
    main()
