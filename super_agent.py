import requests
import json
import random
import time
import uuid

# --- CONFIGURATION ---
COMFY_SERVER = "127.0.0.1:8188"
OLLAMA_SERVER = "http://localhost:11434/api/generate"
CLIENT_ID = str(uuid.uuid4())

# --- MODULE A: THE BRAIN ---
def call_ollama_brain(user_request):
    print(f"   (Consulting Ollama to refine: '{user_request}')...")
    # Ask the LLM to refine the prompt
    consultant_prompt = f"Write a highly detailed text-to-image prompt for Stable Diffusion based on this idea: '{user_request}'. Mention lighting, style, and resolution. Output ONLY the prompt text, no intro."
    
    payload = {
        "model": "llama3", 
        "prompt": consultant_prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_SERVER, json=payload)
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Brain Error: {e}"

# --- MODULE B: THE ARTIST (ComfyUI) ---
def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow, "client_id": CLIENT_ID}
    data = json.dumps(p).encode('utf-8')
    req = requests.post(f"http://{COMFY_SERVER}/prompt", data=data)
    return req.json()

def get_default_workflow(positive_prompt_text):
    # The standard workflow, customized for your specific Model
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 8,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": random.randint(1, 1000000000),
                "steps": 20
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            # THIS MATCHES YOUR SCREENSHOT EXACTLY
            "inputs": {"ckpt_name": "Realistic_Vision_V6.0_NV_B1.safetensors"} 
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": 512, "width": 512}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": positive_prompt_text} 
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": "text, watermark, blurry, low quality, deformed"} 
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "DGX_Agent_Gen", "images": ["8", 0]}
        }
    }
    return workflow

# --- MAIN CONTROLLER ---
if __name__ == "__main__":
    print("\n--- DGX MULTI-MODAL AGENT ---")
    
    # 1. Get User Idea
    user_idea = input(">> Describe the image you want: ")
    
    # 2. Refine with Brain
    print(f"\n[Brain] Refining prompt with Llama 3...")
    rich_prompt = call_ollama_brain(user_idea)
    print(f"--> Generated Prompt: {rich_prompt}")
    
    # 3. Send to Artist
    print(f"\n[Artist] Sending to ComfyUI...")
    workflow = get_default_workflow(rich_prompt)
    
    try:
        response = queue_prompt(workflow)
        print(f"--> Success! Image queued. Prompt ID: {response['prompt_id']}")
        print("--> Check your ComfyUI Output folder in a moment!")
    except Exception as e:
        print(f"--> Error connecting to ComfyUI: {e}")
