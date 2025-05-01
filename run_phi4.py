from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import torch
from threading import Thread
import sys
import argparse

def create_quantization_config(quantization_type):
    """
    Creates quantization configuration based on the specified type.
    """
    if quantization_type == "4":
        # 4-bit quantization configuration
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization_type == "8":
        # 8-bit quantization configuration
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        # No quantization
        return None

def load_model(quantization_type):
    """
    Loads the model with the specified quantization configuration.
    """
    print(f"Loading Phi-4 model with {quantization_type}-bit quantization...")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-reasoning-plus")
    
    quantization_config = create_quantization_config(quantization_type)
    
    if quantization_config:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-reasoning-plus", 
            device_map="auto",
            quantization_config=quantization_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-reasoning-plus", 
            device_map="auto",
            torch_dtype="auto"
        )
    
    print("Model loaded successfully.\n")
    return tokenizer, model

# Initial system message
system_message = {
    "role": "system", 
    "content": "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion."
}

def generate_response(messages, tokenizer, model):
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    
    # Configure streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Set generation parameters
    generation_kwargs = {
        "inputs": inputs.to(model.device),
        "max_new_tokens": 4096,
        "temperature": 0.8,
        "top_p": 0.95,
        "do_sample": True,
        "streamer": streamer
    }
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Print tokens as they are generated
    response = ""
    for token in streamer:
        print(token, end="", flush=True)
        sys.stdout.flush()
        response += token
    
    thread.join()
    return response

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Chat with Phi-4 using different quantization levels')
    parser.add_argument('-q', '--quantization', choices=['4', '8', 'none'], default='4',
                        help='Quantization type: 4 for 4-bits, 8 for 8-bits, none for no quantization')
    args = parser.parse_args()
    
    # Load model and tokenizer
    tokenizer, model = load_model(args.quantization)
    
    messages = [system_message]
    
    print("\n" + "="*50)
    print(f"Chat with Phi-4 (Quantized {args.quantization}-bit)")
    print("="*50)
    print("Type 'exit' or 'quit' to end the conversation.")
    print("="*50 + "\n")
    
    while True:
        # User input
        print("\n" + "-"*50)
        user_input = input("User: ")
        print("-"*50 + "\n")
        
        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        print("\n" + "-"*50)
        print("Phi-4: ", end="")
        response = generate_response(messages, tokenizer, model)
        print("\n" + "-"*50)
        
        # Add assistant response
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 