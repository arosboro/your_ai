
import os
import json
from transformers import PreTrainedTokenizerFast

def fix_tokenizer(model_path):
    print(f"Checking tokenizer at: {model_path}")
    tokenizer_path = os.path.join(model_path, "tokenizer.json")

    if not os.path.exists(tokenizer_path):
        print("tokenizer.json not found")
        return

    # Load with transformers (which handles the discrepancy logic)
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        print("Tokenizer loaded successfully.")

        # Re-save it. This forces consistency between vocab and added_tokens
        tokenizer.save_pretrained(model_path)
        print("Tokenizer re-saved. Inconsistencies should be resolved.")

    except Exception as e:
        print(f"Error processing tokenizer: {e}")

# Path to the specific snapshot causing issues
target_dir = "/Users/arosboro/.cache/huggingface/hub/models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated/snapshots/368c8ed94ce4c986e7b9ca5c159651ef753908ce"

fix_tokenizer(target_dir)
