import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
import torch
from evaluate_llm import ask_judge_harmbench
from jailbreak import check_jailbreak_success
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from eval_utils import load_dataset, load_model_and_tokenizer



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the local model")
    parser.add_argument("--model_name", type=str, help="Model Name. Used for load tokenizer.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Name of the column in the CSV dataset containing the text. If not specified, the first column will be used.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        nargs="+",
        default=None,
        help="Index of the prompt to jailbreak. If not specified, all prompts will be jailbroken.",
    )
    parser.add_argument(
        "--model_judge",
        type=str,
        default="cais/HarmBench-Llama-2-13b-cls",
        help="Model to judge the jailbreak success. Please use cais/HarmBench-Llama-2-13b-cls only.",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Argument validation
    if not os.path.exists(args.dataset):
        raise ValueError(f"Dataset not found: {args.dataset}")
    if args.idx is not None and len(args.idx) != 2:
        raise ValueError("Invalid index format. Please provide two integers.")
    # assert output file is a csv
    assert os.path.splitext(args.output)[1] in [".csv", ".txt"], "Output file must be either .csv or .txt"
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    if args.column is not None:
        args.column = None if args.column == "None" else args.column

    return args


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {repr(value)}")

    # Load the model and tokenizer
    print(f"Loading model and tokenizer from {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    if args.model_path != 'base':
        weights_path = os.path.join(os.getcwd(),args.model_path, 'merged.pt')
        print(f"Loading the {weights_path}")
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    else:
        print('Just loaded the base model')
    model.eval()
    model.requires_grad_(False)
    model_judge, tokenizer_judge = load_model_and_tokenizer(args.model_judge)
    model_judge.eval()
    model_judge.requires_grad_(False)

    dataset = load_dataset(args.dataset, args.column).to_numpy()
    if args.idx is not None:
        lower_bound = max(0, args.idx[0])
        lower_bound = min(len(dataset), lower_bound)
        upper_bound = min(len(dataset), args.idx[1])
        upper_bound = max(0, upper_bound)
        dataset = dataset[lower_bound:upper_bound]

    success_count = 0
    results = []
    total_length = len(dataset)

    for idx, prompt in tqdm(enumerate(dataset), total=total_length):
        is_jailbroken, response = check_jailbreak_success(
            model,
            tokenizer,
            prompt,
            generate_length=32,
            return_response=True,
        )
        # print(f"[{idx}, {jailbroken}]: {repr(prompt)} -> {repr(response)}")

        if is_jailbroken:
            # Double-check
            _, response = check_jailbreak_success(
                model=model,
                tokenizer=tokenizer,
                prompt=f"{prompt}",
                generate_length=512,
                return_response=True,
            )
            rating, _ = ask_judge_harmbench(
                model_judge,
                tokenizer_judge,
                prompt,
                response,
            )
            is_jailbroken = rating == 1
            if is_jailbroken:
                success_count += 1

        results.append(
            {
                "prompt": prompt,
                "adv_string": "",
                "response": response,
                "jailbroken": is_jailbroken,
                "epoch": 0,
                "loss": 0,
                "attack_time": 0,
            }
        )

        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False, escapechar="\\", quoting=1)

    asr = success_count / total_length
    print(f"Clean ASR: {asr} ({success_count}/{total_length})")


if __name__ == "__main__":
    main()
