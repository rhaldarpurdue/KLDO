import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

def main(args):
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path)

    # Load the LoRA adapter
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)

    # Merge the LoRA adapter with the base model
    merged_model = peft_model.merge_and_unload()

    # Save the state dict of the merged model 
    torch.save(merged_model.state_dict(), args.merged_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine LoRA adapter with base model")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--merged_model_path", type=str, required=True, help="Path to save the merged model")

    args = parser.parse_args()
    main(args)
