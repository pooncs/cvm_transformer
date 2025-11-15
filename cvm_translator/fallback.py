import re


def repair_punctuation(text):
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?])(\w)", r"\1 \2", text)
    return text.strip()


def lm_denoise(text):
    return repair_punctuation(text)


def fallback_translate(text, direction):
    if direction == "KR_EN":
        return repair_punctuation(text.replace("안녕하세요", "Hello"))
    else:
        return repair_punctuation(text.replace("Hello", "안녕하세요"))


def process_asr_output(item, direction):
    if item["confidence"] < 0.5:
        item["text"] = lm_denoise(item["text"])
        item["fallback"] = True
        item["translation"] = fallback_translate(item["text"], direction)
    else:
        item["fallback"] = False
    return item