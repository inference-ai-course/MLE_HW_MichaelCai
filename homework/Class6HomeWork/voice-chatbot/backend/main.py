from statistics import median

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers/PEFT not available. Install with: pip install transformers peft")
    exit(1)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisper
import requests
from gtts import gTTS
# Commenting out problematic Coqui TTS for now
# from TTS.api import TTS                                                                                                                         
import logging
import torch
from pathlib import Path
import json
from sympy import sympify
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from cosyvoice.cli.cosyvoice import CosyVoice
from langgraph_agent import create_langgraph_agent


app = FastAPI()

# TTS using gTTS - no initialization needed

#Load Whisper model
logging.info("Loading Whisper model...")
#I am using base instead of small
asr_model = whisper.load_model("base")

#Using Ollama API for LLM
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  # or whatever model name you have in Ollama
#instead of using OLLAMA Model, load model directly

def load_trained_model(model_dir="../../hybrid_trainer/hybrid_apple_model"):
    """load trained model and tokenizer"""
    print(f"load model from {model_dir}")

    try:
        #Load base model
        base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        print(f"Loading base model {base_model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

        #add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        #load peft adapter if exist
        adapter_path = Path(model_dir)
        if adapter_path.exists():
            print(f"loading PEFT adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            print(f"No adapter found at {adapter_path}, using base model")

        #set device
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else  "cpu"
        print(f"Using device {device}")

        return model, tokenizer, device
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav")
    return result['text']

def model_inference(user_text):
    global model, tokenizer, device
    
    print(f"Model Inference {user_text}")

    # Check if model is loaded
    if model is None:
        raise Exception("Model failed to load. Check if hybrid_apple_model exists and model files are valid.")

    model.eval()

    try:
        #format for inference
        message = [{"role": "user", "content": user_text}]

        inputs = tokenizer.apply_chat_template(
            message, 
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        attention_mask = torch.ones_like(inputs).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask= attention_mask,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

        #Extract just the generated part (after the prompt)
        input_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True))
        generated_response = response[input_length:].strip()

        print(f"generated response for {user_text}: {generated_response}")
        return generated_response
    
    except Exception as e:
        print(f"Error doing inference with {user_text}: {e}")
        return f"Error during model inference: {str(e)}"

def route_llm_output(llm_output: str) -> str:

    """
    Route LLM response to the correct tool if it's a function call, else return the text.
    Expects LLM output in JSON format like {'function': ..., 'arguments': {...}}.
    """
    print("*"*20)
    print(f"llm_output is {llm_output}")
    try:
        output = json.loads(llm_output)
        func_name = output.get("function")
        args = output.get("arguments", {})
    except (json.JSONDecodeError, TypeError):
        #Not a JSON functin call; return the text directly
        return llm_output
    
    if func_name == "search_arxiv":
        query = args.get("query", "")
        return search_arxiv(query)
    elif func_name == "calculate":
        query = args.get("expression", "")
        return calculate(query)
    else:
        return f"Error: Unknow function {func_name}"

def search_arxiv(query: str) -> str:

    return f"Simulation Search Arxiv Query {query}"

def calculate(expression: str) -> str:
    """
    evaluate a mathematical expression and return result
    """
    try:
        result = sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error with calcualtion: {e}"


def synthesize_speech(text, filename="response.wav"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

def synthesize_speech_coqui(text, filename="response.wav"):
    """Alternative TTS using Coqui TTS - fallback to gTTS due to compatibility issues"""
    try:
        # Fallback to gTTS since Coqui TTS has compatibility issues
        return synthesize_speech(text, filename)
    except Exception as e:
        raise Exception(f"TTS failed: {e}")


# tts_engine = CozyVoice()

# def synthesize_speech(text, filename="response.wav"):
#     tts_engine.generate(text, output_file=filename)
#     return filename
    
print("🔄 Loading trained model...")
model, tokenizer, device = load_trained_model()
print(f"✅ Model loading result: model={model is not None}, tokenizer={tokenizer is not None}, device={device}")

# Initialize LangGraph agent
print("🔄 Initializing LangGraph agent...")
langgraph_agent = create_langgraph_agent(model_inference)
print("✅ LangGraph agent initialized")


@app.get("/")
async def root():
    return {"message": "Voice Chatbot API is running", "docs": "/docs"}


# @app.post("/chat")
# async def chat_endpoint(file: UploadFile = File(...)):

#     audio_data = await  file.read()
#     user_text = transcribe_audio(audio_data)
#     print(f"user text: {user_text}")
#     # bot_text = generate_response(user_text)
#     inference_text = model_inference(user_text)
#     print(f"bot_text is {inference_text}")
#     route_llm_output_response = route_llm_output(inference_text);
#     print(f"After route llm response: {route_llm_output_response}")
#     python_audio_path = synthesize_speech_coqui(route_llm_output_response)

#     return FileResponse(python_audio_path, media_type="audio/wav")


@app.post("/chat_langgraph")
async def chat_langgraph_endpoint(file: UploadFile = File(...)):
    """Chat endpoint using LangGraph agent for tool calling"""
    audio_data = await file.read()
    user_text = transcribe_audio(audio_data)
    print(f"user text: {user_text}")
    
    # Use LangGraph agent instead of manual routing
    response = langgraph_agent.invoke(user_text)
    print(f"LangGraph response: {response}")
    
    python_audio_path = synthesize_speech_coqui(response)
    return FileResponse(python_audio_path, media_type="audio/wav")
 
