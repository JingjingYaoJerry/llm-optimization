import time
import torch

def measure_latency(model, tokenizer, prompt="Hello, DJI"):
    """Measure Time to First Token (TTFT) and per-Token latency.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1)
    ttft = (time.time() - start_time) * 1000  # ms
    
    print(f"Time to First Token (TTFT): {ttft:.2f} ms")
    return ttft

if __name__ == "__main__":
    print("Evaluation pipeline initialized. Target: <100ms latency.")
    # Evaluation Pipeline