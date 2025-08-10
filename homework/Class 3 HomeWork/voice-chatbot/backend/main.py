from statistics import median
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisper
import requests
from gtts import gTTS
# Uncomment below for Coqui TTS
from TTS.api import TTS                                                                                                                         
import logging
# from cosyvoice.cli.cosyvoice import CosyVoice


app = FastAPI()

# TTS using gTTS - no initialization needed

#Load Whisper model
logging.info("Loading Whisper model...")
#I am using base instead of small
asr_model = whisper.load_model("base")

# Using Ollama API for LLM
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  # or whatever model name you have in Ollama


conversation_history = []

@app.get("/")
async def root():
    return {"message": "Voice Chatbot API is running", "docs": "/docs"}

def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav")
    return result['text']

def generate_response(user_text):
    conversation_history.append({"role": "user", "text": user_text})
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"
    
    # Call Ollama API
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0
    })
    
    if response.status_code == 200:
        bot_response = response.json()["response"]
    else:
        bot_response = f"Error: Could not generate response. Status: {response.status_code}"
    
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response

def synthesize_speech(text, filename="response.wav"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

def synthesize_speech_coqui(text, filename="response.wav"):
    """Alternative TTS using Coqui TTS (requires: pip install coqui-tts)"""
    try:

        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        tts.tts_to_file(text=text, file_path=filename)
        return filename
    except ImportError:
        raise ImportError("Coqui TTS not installed. Run: pip install coqui-tts")
    except Exception as e:
        raise Exception(f"Coqui TTS failed: {e}")

# tts_engine = CozyVoice()

# def synthesize_speech(text, filename="response.wav"):
#     tts_engine.generate(text, output_file=filename)
#     return filename

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_data = await  file.read()
    user_text = transcribe_audio(audio_data)
    bot_text = generate_response(user_text)
    python_audio_path = synthesize_speech_coqui(bot_text)

    return FileResponse(python_audio_path, media_type="audio/wav")

