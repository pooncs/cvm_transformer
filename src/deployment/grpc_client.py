import grpc
import sys
sys.path.insert(0, '.')
import cvm_translator.realtime_translation_pb2 as realtime_translation_pb2
import cvm_translator.realtime_translation_pb2_grpc as realtime_translation_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = realtime_translation_pb2_grpc.TranslatorStub(channel)
        def gen():
            texts = ["안녕하세요", "오늘 날씨 좋네요"]
            for t in texts:
                yield realtime_translation_pb2.TranslationRequest(text=t, direction="KR_EN")
        for resp in stub.StreamTranslate(gen()):
            print(f"KR: {resp.translation} latency={resp.latency_ms:.1f}ms cores={resp.core_count}")


if __name__ == "__main__":
    run()