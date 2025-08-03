#!/usr/bin/env python3
"""
Open-source LLM Testing with Hugging Face
Test and compare different open-source LLMs including Llama and Mistral models.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
import time
import warnings
import subprocess
import threading
import requests
import json
from openai import OpenAI
import os
import signal
warnings.filterwarnings("ignore")


class LLMTester:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_model(self, model_name, use_quantization=True):
        """Load a model with optional quantization for memory efficiency"""
        print(f"\nLoading {model_name}...")
        
        try:
            # Configure quantization for memory efficiency
            if use_quantization and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto" if self.device == "cuda" else None
            )
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.pipelines[model_name] = pipe
            
            print(f"✓ Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {str(e)}")
            return False
    
    def generate_text(self, model_name, prompt, max_length=200, temperature=0.7):
        """Generate text using a loaded model"""
        if model_name not in self.pipelines:
            print(f"Model {model_name} not loaded!")
            return None
        
        try:
            start_time = time.time()
            
            result = self.pipelines[model_name](
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizers[model_name].eos_token_id,
                num_return_sequences=1
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            generated_text = result[0]['generated_text']
            # Remove the original prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return {
                'text': generated_text,
                'time': generation_time,
                'tokens_per_second': len(generated_text.split()) / generation_time
            }
            
        except Exception as e:
            print(f"Error generating text with {model_name}: {str(e)}")
            return None
    
    def test_model(self, model_name, test_prompts):
        """Test a model with multiple prompts"""
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print(f"{'='*50}")
        
        if model_name not in self.pipelines:
            print(f"Model {model_name} not loaded!")
            return
        
        results = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt[:50]}...")
            result = self.generate_text(model_name, prompt)
            
            if result:
                print(f"Response: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
                print(f"Time: {result['time']:.2f}s | Speed: {result['tokens_per_second']:.1f} tokens/sec")
                results.append(result)
            else:
                print("Failed to generate response")
        
        return results
    
    def compare_models(self, model_names, prompt):
        """Compare multiple models on the same prompt"""
        print(f"\n{'='*60}")
        print(f"Model Comparison: '{prompt[:50]}...'")
        print(f"{'='*60}")
        
        results = {}
        
        for model_name in model_names:
            if model_name in self.pipelines:
                result = self.generate_text(model_name, prompt)
                if result:
                    results[model_name] = result
                    print(f"\n{model_name}:")
                    print(f"Response: {result['text'][:150]}...")
                    print(f"Time: {result['time']:.2f}s | Speed: {result['tokens_per_second']:.1f} tokens/sec")
        
        return results


class vLLMServer:
    def __init__(self, port=8000):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}/v1"
        
    def start_server(self, model_name, gpu_memory_utilization=0.8, max_model_len=2048):
        """Start vLLM server with OpenAI API interface"""
        print(f"\nStarting vLLM server for {model_name}...")
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--served-model-name", model_name.split("/")[-1]  # Use just the model name without org
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            print("Waiting for server to start...")
            for i in range(60):  # Wait up to 60 seconds
                try:
                    response = requests.get(f"{self.base_url}/models", timeout=2)
                    if response.status_code == 200:
                        print(f"✓ vLLM server started successfully on port {self.port}")
                        return True
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
                print(f"  Waiting... ({i+1}/60)")
            
            print("✗ Server failed to start within 60 seconds")
            self.stop_server()
            return False
            
        except Exception as e:
            print(f"✗ Failed to start vLLM server: {e}")
            return False
    
    def stop_server(self):
        """Stop the vLLM server"""
        if self.process:
            print("Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            print("✓ vLLM server stopped")
    
    def is_running(self):
        """Check if server is running"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=2)
            return response.status_code == 200
        except:
            return False


class OpenAITester:
    def __init__(self, base_url="http://localhost:8000/v1", api_key="dummy"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key  # vLLM doesn't require real API key
        )
        self.base_url = base_url
    
    def list_models(self):
        """List available models"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate_text(self, model_name, prompt, max_tokens=200, temperature=0.7):
        """Generate text using OpenAI API interface"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            generated_text = response.choices[0].message.content
            
            return {
                'text': generated_text,
                'time': generation_time,
                'tokens_per_second': len(generated_text.split()) / generation_time if generation_time > 0 else 0,
                'usage': response.usage.dict() if response.usage else None
            }
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return None
    
    def test_model_openai(self, model_name, test_prompts):
        """Test model using OpenAI API interface"""
        print(f"\n{'='*50}")
        print(f"Testing {model_name} via OpenAI API")
        print(f"{'='*50}")
        
        results = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt[:50]}...")
            result = self.generate_text(model_name, prompt)
            
            if result:
                print(f"Response: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
                print(f"Time: {result['time']:.2f}s | Speed: {result['tokens_per_second']:.1f} tokens/sec")
                if result['usage']:
                    print(f"Tokens used: {result['usage']['total_tokens']}")
                results.append(result)
            else:
                print("Failed to generate response")
        
        return results


def test_vllm_mode():
    """Test models using vLLM server with OpenAI API"""
    print("=== vLLM + OpenAI API Testing Mode ===")
    
    # Models that work well with vLLM
    vllm_models = [
        "microsoft/DialoGPT-medium",
        "EleutherAI/gpt-neo-1.3B",
        # "meta-llama/Llama-2-7b-chat-hf",  # Uncomment if you have access
        # "mistralai/Mistral-7B-Instruct-v0.2",  # Uncomment if you have access
    ]
    
    test_prompts = [
        "Explain quantum computing in simple terms",
        "Write a short story about a robot",
        "What are the benefits of renewable energy?"
    ]
    
    # Choose model
    model_name = vllm_models[0]  # Start with smallest model
    print(f"Using model: {model_name}")
    
    # Start vLLM server
    server = vLLMServer(port=8000)
    
    try:
        if server.start_server(model_name, gpu_memory_utilization=0.7, max_model_len=1024):
            # Test with OpenAI client
            client = OpenAITester()
            
            # List available models
            available_models = client.list_models()
            print(f"Available models: {available_models}")
            
            if available_models:
                model_id = available_models[0]
                results = client.test_model_openai(model_id, test_prompts)
                
                print(f"\n{'='*60}")
                print("vLLM Testing completed!")
                print(f"Model: {model_name}")
                print(f"Total tests: {len(results)}")
                print(f"{'='*60}")
        else:
            print("Failed to start vLLM server")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        server.stop_server()


def main():
    """Main function to demonstrate LLM testing"""
    print("Choose testing mode:")
    print("1. Traditional Transformers testing")
    print("2. vLLM + OpenAI API testing")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == "2":
        test_vllm_mode()
        return
    
    # Traditional mode
    tester = LLMTester()
    
    # Available models (choose based on your GPU memory)
    models_to_test = [
        "microsoft/DialoGPT-medium",  # Smaller model for testing
        "microsoft/DialoGPT-large",   # Medium-sized model
        # "meta-llama/Llama-2-7b-chat-hf",  # Requires approval
        # "mistralai/Mistral-7B-v0.1",      # Requires approval
    ]
    
    # Note: For Llama and Mistral models, you need to:
    # 1. Request access on Hugging Face
    # 2. Login with: huggingface-cli login
    
    # Alternative open models that don't require approval:
    open_models = [
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "bigscience/bloom-1b7",
    ]
    
    # Test prompts
    test_prompts = [
        "Explain quantum computing in simple terms:",
        "Write a short story about a robot:",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis:",
        "How does machine learning work?"
    ]
    
    print("Loading models...")
    loaded_models = []
    
    # Try to load open models first
    for model_name in open_models[:2]:  # Load first 2 to save memory
        if tester.load_model(model_name):
            loaded_models.append(model_name)
    
    if not loaded_models:
        print("No models loaded successfully!")
        return
    
    # Test each model
    for model_name in loaded_models:
        results = tester.test_model(model_name, test_prompts[:3])  # Test with first 3 prompts
    
    # Compare models on a single prompt
    if len(loaded_models) > 1:
        comparison_prompt = "Explain artificial intelligence:"
        tester.compare_models(loaded_models, comparison_prompt)
    
    print(f"\n{'='*60}")
    print("Testing completed!")
    print(f"Models tested: {', '.join(loaded_models)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Open-source LLM Tester with Hugging Face")
    print("=========================================")
    
    # Check requirements
    print("\nChecking requirements...")
    try:
        import torch
        import transformers
        print("✓ PyTorch and Transformers installed")
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Install with: pip install torch transformers accelerate bitsandbytes")
        exit(1)
    
    try:
        import vllm
        print("✓ vLLM installed")
    except ImportError:
        print("⚠ vLLM not installed (required for option 2)")
        print("Install with: pip install vllm")
    
    try:
        import openai
        print("✓ OpenAI client installed")
    except ImportError:
        print("⚠ OpenAI client not installed (required for option 2)")
        print("Install with: pip install openai")
    
    main()