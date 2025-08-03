import requests

def generate_with_ollama(model_name, prompt, max_token=100):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_token,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json().get('response', 'No response')
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    prompt = "Explain machine learning in simple terms"
    
    print("Testing llama3.1:8b:")
    result = generate_with_ollama("llama3.1:8b", prompt)
    print(result)
    
    print("\n" + "="*50 + "\n")
    
    print("Testing llama2:latest:")
    result = generate_with_ollama("llama2:latest", prompt)
    print(result)


        