# Build and package for edge deployment

FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cvm_translator/realtime_translation.proto

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build /app
WORKDIR /app

ENV PYTHONPATH=/app
EXPOSE 50051

CMD ["python", "-m", "cvm_translator.grpc_int8_server"]