class SLMTranslator:
    def __init__(self):
        self.kr_en = {
            "안녕하세요": "Hello",
            "오늘": "Today",
            "날씨": "weather",
            "좋네요": "is nice",
            "입니다": "is",
            "한국": "Korea",
            "영어": "English",
        }
        self.en_kr = {
            "Hello": "안녕하세요",
            "Today": "오늘",
            "weather": "날씨",
            "is": "입니다",
            "nice": "좋네요",
            "Korea": "한국",
            "English": "영어",
        }

    def tokenize(self, text):
        return text.strip().split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def translate_tokens_kren(self, tokens):
        out = []
        for t in tokens:
            if t in self.kr_en:
                out.append(self.kr_en[t])
            else:
                out.append(t)
        return out

    def translate_tokens_enkr(self, tokens):
        out = []
        for t in tokens:
            if t in self.en_kr:
                out.append(self.en_kr[t])
            else:
                out.append(t)
        return out

    def translate(self, text, direction):
        toks = self.tokenize(text)
        if direction == "KR_EN":
            out = self.translate_tokens_kren(toks)
        else:
            out = self.translate_tokens_enkr(toks)
        return self.detokenize(out)

