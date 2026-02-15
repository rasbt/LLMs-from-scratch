import torch
from architecture import GPTModel
from tokenizer_utils import TokenizerWrapper
from config import GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M, SPECIAL_TOKENS
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

def main(args):
    print(f"Loading model from {args.model_path} with size {args.model_size}...")
    # Ensure model exists or warn
    import os
    if not os.path.exists(args.model_path):
        print(f"Warning: '{args.model_path}' not found. Please train the model first using train_colab.py")
        return

    # Select config
    if args.model_size == "124M":
        cfg = GPT_CONFIG_124M
    elif args.model_size == "355M":
        cfg = GPT_CONFIG_355M
    elif args.model_size == "774M":
        cfg = GPT_CONFIG_774M
    elif args.model_size == "1558M":
        cfg = GPT_CONFIG_1558M
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
        
    cfg["context_length"] = args.max_length

    model, tokenizer, device = load_trained_model(model_path=args.model_path, cfg=cfg)
    
    print("\nModel loaded. Enter an instruction (or 'q' to quit).")
    
    while True:
        instruction = input("\nUser: ")
        if instruction.lower() in ('q', 'quit', 'exit'):
            break
            
        code = generate_tool_call(model, tokenizer, device, instruction)
        
        print(f"\n[Generated Code]\n{code}\n")
        
        # Execute?
        
        # Detect function name
        import re
        func_matches = re.findall(r"def\s+(\w+)\(", code)
        last_func = func_matches[-1] if func_matches else None
        
        if last_func:
            print(f"Detected function: {last_func}")
            
        confirm = input(f"Execute? (y/n, 'e' to edit/append, 'c' to call print({last_func}())): ")
        
        if confirm.lower() in ('y', 'e', 'c'):
            appended_code = ""
            if confirm.lower() == 'e':
                appended_code = input("Enter code to append: ")
            elif confirm.lower() == 'c' and last_func:
                appended_code = f"print({last_func}())"
            
            full_code = code + "\n" + appended_code
            print(f"Executing:\n{full_code}")
            result = execute_code_safe(full_code)
            print(f"\n[Execution Output]\n{result}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="tool_llm.pth", help="Path to trained model checkpoint")
    parser.add_argument("--model_size", type=str, default="124M", help="GPT-2 model size (124M, 355M, 774M, 1558M)")
    parser.add_argument("--max_length", type=int, default=1024, help="Context length used during training")
    
    args = parser.parse_args()
    main(args)
