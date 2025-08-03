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
import logging


warnings.filterwarnings("ignore")

class llmTester:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models={}
        self.tokenizers = {}
        self.pipelines = {}

        logging.info(f"Using device: {self.device}")
        if(self.device=="cuda"):
            logging.info(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        
    def load_model(self, model_name, use_quantization=True):

        logging.info(f"\nLoading {model_name}")

        try:
            #configure quantization for memory efficiency
            if use_quantization and self.device=="cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bits=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
            
            #load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            #Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config = quantization_config,
                device_map="auto" if self.device=="cuda" else None
                torch_dtype=torch.float16 if self.device=="cuda" or torch.float32
                trust_remote_code=True
            )

            #create a pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer = tokenizer,
                device_map="auto" if self.device=="cuda" else None
            )

            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.pipelines[model_name] = pipe

            logging.info(f"successfully load model {model_name}")

            return True
        
        except Exception as e:
            logging.error(f"Failed to load {model_name}: {str(e)}")
            return False
    
    def generate_text(self, model_name:str, prompt:str, max_length: int=200, temperature: float=0.7):

        if model_name not in self.pipelines:
            logging.info(f"Model {model_name} not loaded")
            return None
        
        try:
            start_time = time.time()

            result = self.pipelines[model_name](
                prompt,

            )



