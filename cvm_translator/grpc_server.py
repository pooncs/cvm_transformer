import grpc
from concurrent import futures
import realtime_translation_pb2
import realtime_translation_pb2_grpc
import queue
import threading
import time


class TranslatorServicer(realtime_translation_pb2_grpc.TranslatorServicer):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def worker(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            self.process(item)

    def process(self, item):
        # stub: call cvm_translate_chunk
        pass

    def StreamTranslate(self, request_iterator, context):
        for req in request_iterator:
            self.q.put(req)
            yield realtime_translation_pb2.TranslationResponse(
                translation="stub",
                latency_ms=0.0,
                core_count=0
            )


def serve(model, tokenizer, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    realtime_translation_pb2_grpc.add_TranslatorServicer_to_server(
        TranslatorServicer(model, tokenizer), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC server started on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve(None, None)