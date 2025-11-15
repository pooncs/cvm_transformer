import torch
import sys
sys.path.insert(0, '.')
from cvm_translator.cvm_transformer import CVMTransformer
from cvm_translator.quantization import QuantizedCVMTransformer, prepare_qat, convert_int8
from cvm_translator.sp_tokenizer import SPTokenizer
from cvm_translator.telemetry import TelemetryDashboard
import cvm_translator.realtime_translation_pb2 as realtime_translation_pb2
import cvm_translator.realtime_translation_pb2_grpc as realtime_translation_pb2_grpc
import grpc
from concurrent import futures
import time


class TranslatorServicer(realtime_translation_pb2_grpc.TranslatorServicer):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dash = TelemetryDashboard()

    def StreamTranslate(self, request_iterator, context):
        for req in request_iterator:
            t0 = time.perf_counter()
            ids = self.tokenizer.encode(req.text)
            input_ids = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(input_ids)
            pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
            translation = self.tokenizer.decode(pred_ids)
            latency = (time.perf_counter() - t0) * 1000
            self.dash.log("latency_ms", latency)
            yield realtime_translation_pb2.TranslationResponse(
                translation=translation,
                latency_ms=latency,
                core_count=len(ids)
            )


def serve(model_path="kr_en.model", port=50051):
    tokenizer = SPTokenizer(model_path)
    base = CVMTransformer(vocab_size=tokenizer.vocab_size(), d_model=256, n_layers=4)
    model = base  # skip quantization for now
    model.eval()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    realtime_translation_pb2_grpc.add_TranslatorServicer_to_server(
        TranslatorServicer(model, tokenizer), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"FP32 gRPC server ready on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()