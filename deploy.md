# Build
docker build -t cvm-translator .

# Run
docker run -p 50051:50051 cvm-translator

# Client example (Python)
python -m cvm_translator.grpc_client