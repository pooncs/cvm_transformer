import sentencepiece as spm
import os


def train_spm(corpus_files, prefix="kr_en", vocab_size=200):
    spm.SentencePieceTrainer.train(
        input=corpus_files,
        model_prefix=prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


class SPTokenizer:
    def __init__(self, model_path="kr_en.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def vocab_size(self):
        return self.sp.vocab_size()


if __name__ == "__main__":
    if not os.path.exists("kr_en.model"):
        train_spm(["data/kr.txt", "data/en.txt"], vocab_size=200)
    tok = SPTokenizer()
    print(tok.encode("안녕하세요"))
    print(tok.decode(tok.encode("Hello")))