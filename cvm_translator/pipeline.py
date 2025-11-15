import time
from cvm_translator.cvm_buffer import CVMBuffer
from cvm_translator.slm_model import SLMTranslator
from cvm_translator.whisper_interface import transcribe_chunk


def select_core_tokens(tokens, capacity):
    b = CVMBuffer(capacity)
    for t in tokens:
        b.add(t)
    return b.cores()


def translate_stream(chunks, direction, capacity):
    model = SLMTranslator()
    outputs = []
    t0 = time.perf_counter()
    for ch in chunks:
        asr = transcribe_chunk(ch)
        toks = model.tokenize(asr["text"])
        cores = select_core_tokens(toks, capacity)
        core_text = model.detokenize(cores)
        out = model.translate(core_text, direction)
        outputs.append({"input": ch, "asr": asr, "cores": cores, "translation": out})
    t1 = time.perf_counter()
    return outputs, (t1 - t0) * 1000.0


def main():
    chunks = ["안녕하세요", "오늘 날씨 좋네요"]
    out, ms = translate_stream(chunks, "KR_EN", 4)
    print("latency_ms", ms)
    for o in out:
        print(o["translation"]) 


if __name__ == "__main__":
    main()

