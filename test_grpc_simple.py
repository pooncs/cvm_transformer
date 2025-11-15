#!/usr/bin/env python3
"""
Simple gRPC client test to verify the server is working
"""

import sys
sys.path.insert(0, '.')
import cvm_translator.realtime_translation_pb2 as realtime_translation_pb2
import cvm_translator.realtime_translation_pb2_grpc as realtime_translation_pb2_grpc
import grpc
import time

def test_grpc_connection():
    """Test gRPC connection with detailed output"""
    print("🌐 Testing gRPC Connection")
    print("=" * 40)
    
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = realtime_translation_pb2_grpc.TranslatorStub(channel)
        
        # Test sentences
        test_sentences = [
            "안녕하세요",
            "Hello world", 
            "실시간 번역 시스템",
            "CVM algorithm test"
        ]
        
        print("Connected to gRPC server")
        print(f"Testing {len(test_sentences)} sentences...")
        
        for i, text in enumerate(test_sentences, 1):
            print(f"\n{i}. Testing: '{text}'")
            
            def generate_requests():
                yield realtime_translation_pb2.TranslationRequest(text=text)
                time.sleep(0.1)  # Small delay
            
            try:
                responses = stub.StreamTranslate(generate_requests())
                for resp in responses:
                    print(f"   Translation: '{resp.translation}'")
                    print(f"   Latency: {resp.latency_ms:.2f}ms")
                    print(f"   Core count: {resp.core_count}")
                    
            except Exception as e:
                print(f"   Error: {e}")
                
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure the gRPC server is running on port 50051")

if __name__ == "__main__":
    test_grpc_connection()