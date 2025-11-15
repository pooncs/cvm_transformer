# Local edge deployment without Docker

## 1. Install deps
pip install -r requirements.txt

## 2. Compile proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cvm_translator/realtime_translation.proto

## 3. Train SentencePiece (optional, already done)
python -m cvm_translator.sp_tokenizer

## 4. Start INT8 gRPC server
python -m cvm_translator.grpc_int8_server

## 5. Run client
python -m cvm_translator.grpc_client

## 6. View telemetry
python -m cvm_translator.telemetry