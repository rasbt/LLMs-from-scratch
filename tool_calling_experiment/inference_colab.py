import torch
from architecture import GPTModel
from tokenizer_utils import TokenizerWrapper
from config import GPT_CONFIG_124M, SPECIAL_TOKENS
from execution_sandbox import execute_code_safe

def load_trained_model(model_path="tool_llm.pth", cfg=GPT_CONFIG_124M):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Tokenizer to get vocab size
    tokenizer = TokenizerWrapper()
    new_vocab_size = tokenizer.base_tokenizer.n_vocab + len(tokenizer.special_tokens)
    
    # Init Model
    # Note: We need to manually resize the config or model layers to match the saved state
    # because the saved state has 50259 vocab size, but config has 50257.
    
    # Option 1: Update config
    cfg_copy = cfg.copy()
    cfg_copy["vocab_size"] = new_vocab_size
    
    model = GPTModel(cfg_copy)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate_tool_call(model, tokenizer, device, instruction, max_new_tokens=200):
    start_token = SPECIAL_TOKENS["<CODE_START>"]
    end_token = SPECIAL_TOKENS["<CODE_END>"]
    
    # Prompt format
    prompt = f"Instruction: {instruction}\n<CODE_START>\n"
    
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    # Generate
    # We implement a custom loop to stop at <CODE_END>
    
    generated_ids = []
    
    print("Generating code...", end="", flush=True)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_tensor)
        
        last_logits = logits[:, -1, :]
        next_id = torch.argmax(last_logits, dim=-1).item()
        
        # Stop if end token
        if next_id == end_token:
            break
            
        generated_ids.append(next_id)
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_id]]).to(device)], dim=1)
        print(".", end="", flush=True)
    
    print(" Done.")
    
    # Decode
    code = tokenizer.decode(generated_ids)
    return code

def main():
    print("Loading model...")
    # Ensure model exists or warn
    import os
    if not os.path.exists("tool_llm.pth"):
        print("Warning: 'tool_llm.pth' not found. Please train the model first using train_colab.py")
        return

    model, tokenizer, device = load_trained_model()
    
    print("\nModel loaded. Enter an instruction (or 'q' to quit).")
    
    while True:
        instruction = input("\nUser: ")
        if instruction.lower() in ('q', 'quit', 'exit'):
            break
            
        code = generate_tool_call(model, tokenizer, device, instruction)
        
        print(f"\n[Generated Code]\n{code}\n")
        
        # Execute?
        confirm = input("Execute this code? (y/n): ")
        if confirm.lower() == 'y':
            result = execute_code_safe(code)
            print(f"\n[Execution Output]\n{result}")

if __name__ == "__main__":
    main()
