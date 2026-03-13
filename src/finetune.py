import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def setup_model(model_name):
    # Configure LoRA
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=.05,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    # Load 8-bit quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=True, 
        device_map="auto"
    )
    
    model = get_peft_model(model, lora_config)
    return model

if __name__ == "__main__":
    print("Initializing QLoRA fine-tuning pipeline...")
    # Training Loop