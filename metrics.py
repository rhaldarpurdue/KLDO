import os
import sys
#change current directory and hf cache folder accordingly
#os.chdir('/depot/qfsong/LLM/scratch/rhaldar/representation-space-jailbreak/')
os.environ['HF_HOME'] = '/depot/qfsong/LLM/scratch/rhaldar/hf_cache/'

# Get the current working directory 
current_directory = os.getcwd() 
print("Current Directory:", current_directory)
sys.path.append(os.getcwd())

import argparse
import os
import random
from itertools import cycle
from matplotlib.patches import Ellipse
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import load_dataset, load_model_and_tokenizer
from bhatta_dist import *
import gc

parser = argparse.ArgumentParser(
        description="Visualize the last hidden states of the last token of a Transformer model with PCA"
    )
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--viz", action="store_true", help="To plot visualization")
parser.add_argument("--base_model",type=str,default='ContextualAI/archangel_sft_llama7b')
parser.add_argument(
        "--anchors",
        type=str,
        nargs="+",
        default=[".dataset_generation/data/prompt-driven_benign.txt", ".dataset_generation/data/prompt-driven_harmful.txt"],
        help="Paths to the anchor datasets. Should exactly be two datasets, first "
        "one being harmless, second one being harmful, differs only in harmfulness.",
    )
parser.add_argument(
        "--system_prompt",
        type=str,
        help="(Optional) System prompt to use for the chat template",
    )
parser.add_argument(
        "--alignments",
        type=str,
        nargs="+",
        required=True,
        help="names of the aligned models",
    )
parser.add_argument(
        "--base_path",
        type=str,
        default="/depot/qfsong/LLM/scratch/rhaldar/HALOs/data/models/",
        help="base path where the aligned models are stored",
    )
args = parser.parse_args()
for key, value in vars(args).items():
    print(f"{key}: {repr(value)}")

def apply_chat_template(tokenizer, texts, system_prompt=None):
    full_prompt_list = []
    for idx, text in enumerate(texts):
        if system_prompt is not None:
            messages = [
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
        else:
            messages = [{"role": "user", "content": text}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        full_prompt_list.append(full_prompt)
        # print(f"==>> [{idx}]: {full_prompt}")
    return full_prompt_list

def get_hidden_states(model, tokenizer, full_prompt_list):
    model.eval()
    hidden_state_list = []

    with torch.no_grad():
        progress_bar = tqdm(full_prompt_list, desc="Calculating hidden states")
        for full_prompt in progress_bar:
            inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(
                model.device
            )
            outputs = model(**inputs, output_hidden_states=True)
            # Get the last hidden state of the last token for each sequence
            # We use -1 to index the last layer, and -1 again to index the hidden state of the last token
            hidden_state_list.append(outputs.hidden_states[-1][:, -1, :])
    hidden_state_list = torch.stack(hidden_state_list)
    return hidden_state_list


def pca_reduce_dimensions(hidden_states, num_components):
    pca = PCA(n_components=num_components)
    # reduced = pca.fit_transform(hidden_states)    # .fit_transform() will return the reduced data
    pca.fit(hidden_states)  # .fit() will not return the reduced data
    return pca


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        raise ValueError(f"Invalid covariance shape {covariance.shape}")

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs)
        )


def plot_pca(
    embedding_list,
    pca_object,
    num_components_to_draw,
    color_list_string,
    embedding_labels,
    filename,
    scores=None,plt_name=None
):
    assert num_components_to_draw in [2, 3], "Only 2D and 3D plots are supported."
    assert len(embedding_list) == len(color_list_string), "Color must match embedding."
    assert len(embedding_list) == len(embedding_labels), "Labels must match embedding."
    plt.figure(figsize=(10, 10))

    # Define a color cycle, e.g., "rgbcmyk"
    colors = cycle(color_list_string)

    if num_components_to_draw == 2:
        # 2D plot
        for embedding, color, label in zip(embedding_list, colors, embedding_labels):
            reduced = pca_object.transform(embedding.cpu().numpy())
            n = np.minimum(reduced.shape[0],100)
            reduced = reduced [:n,]
            if scores:
                sizes=torch.exp(torch.abs(scores[label])).detach().cpu().numpy()
                scaling_factor=1/10
                sizes=scaling_factor*sizes
                sizes=np.clip(sizes,a_min=0,a_max=500)
                print("USING ABS SCORE SCALING")
                if label=='success':
                    plt.scatter(reduced[:, 0], reduced[:, 1], color=color, label=label, s=sizes, edgecolor='red')
                elif label=='fail':
                    plt.scatter(reduced[:, 0], reduced[:, 1], color=color, label=label, s=sizes, edgecolor='lime')
                else:
                    plt.scatter(reduced[:, 0], reduced[:, 1], color=color, label=label, s=sizes)
            else:
                plt.scatter(reduced[:, 0], reduced[:, 1], color=color, label=label)
                print("NOT USING SCORE SCALING")
            center = reduced.mean(axis=0)
            # Draw center point
            plt.plot(
                center[0],
                center[1],
                "o",
                markerfacecolor="w",
                markeredgecolor=color,
                markersize=10,
            )
            # Draw ellipse
            reduced_2d = reduced[:, :2]
            if len(reduced_2d) > 1: # avoid error when there is only one point
                draw_ellipse(
                    center, np.cov(reduced_2d, rowvar=False), alpha=0.2, color=color
                )
        # plt.xlabel("Principal Component 1")
        # plt.ylabel("Principal Component 2")
        # make axis tick font bigger
        plt.tick_params(axis="both", which="major", labelsize="x-large")
    else:
        # 3D plot
        ax = plt.axes(projection="3d")
        for embedding, color, label in zip(embedding_list, colors, embedding_labels):
            reduced = pca_object.transform(embedding.cpu().numpy())
            ax.scatter3D(
                reduced[:, 0], reduced[:, 1], reduced[:, 2], color=color, label=label
            )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")

    # plt.title("PCA of Last Hidden States")
    plt.legend(fontsize="x-large")
    plt.title(plt_name)
    
    # Check if the directory of the output file exists, create if not
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.savefig(filename)

def get_metrics(embedding_list, pca_object):
    x0 = (pca_object.transform(embedding_list[0].cpu().numpy()))[:,:2]
    x1 = (pca_object.transform(embedding_list[1].cpu().numpy()))[:,:2]
    X = np.concatenate((x0,x1),axis=0)
    Y = np.concatenate((np.zeros(x0.shape[0]),np.ones(x1.shape[0])))
    bd = bhattacharyya_distance_multivariate(x0, x1)
    bc = bhattacharyya_coeff(x0, x1)
    sc = silhouette_score(X, Y)
    return sc, bd, bc

# Load the anchors
df_anchors_list = []
for idx, anchor_path in enumerate(args.anchors):
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(f"File not found: {anchor_path}")
    df_anchors = load_dataset(anchor_path, column_name=None)
    df_anchors = df_anchors.sample(
        min(args.num_samples, len(df_anchors)), random_state=args.seed
    )
    df_anchors_list.append(df_anchors)

#model alignment paths
base_path = args.base_path
model_list=['base']+ [f'{base_path}/{a}/FINAL/merged.pt' for a in args.alignments]
model_labels=[f'base_{args.base_model.split('/')[-1]}'] + [f'{a}' for a in args.alignments]
metric=dict()

for model_p, model_label in zip(model_list,model_labels):
    model, tokenizer = load_model_and_tokenizer(args.base_model)
    #load model 
    if 'base'!= model_p:
        print(f"Loading the {model_label}")
        state_dict = torch.load(model_p)
        model.load_state_dict(state_dict)
    else:
        print('Just loaded the base model')

    hidden_states_anchors_list = []
    for df_anchors in df_anchors_list:
        full_prompts = apply_chat_template(tokenizer, df_anchors, args.system_prompt)
        hidden_states_dataset = get_hidden_states(model, tokenizer, full_prompts)
        # Reshape hidden states to (num_texts, -1) to flatten the token dimension
        hidden_states_dataset = hidden_states_dataset.view(
            hidden_states_dataset.size(0), -1
        )
        hidden_states_anchors_list.append(hidden_states_dataset)
        # Concatenate the hidden states of all anchor datasets and reduce the dimensions with PCA
    hidden_states_anchor_cat = torch.cat(hidden_states_anchors_list, dim=0)
    pca = pca_reduce_dimensions(
        hidden_states_anchor_cat.cpu().numpy(), 2
    )

    # Update labels to remove those corresponding to skipped datasets
    #embedding_labels = ['safe','harm'] + valid_labels
    embedding_labels = ['safe','harm']

    # Update colors to remove those corresponding to skipped datasets
    color_list = ['r','b']

    if args.viz:
        # Plot the PCA visualization
        plot_pca(
            hidden_states_anchors_list,
            pca,
            num_components_to_draw=2,
            color_list_string=color_list,
            embedding_labels=embedding_labels,
            filename=f'../experiments/anchor_train/{model_label}.png',plt_name=model_label
        )
    sc, bd ,bc = get_metrics(hidden_states_anchors_list,
                             pca)
    metric[model_label]={'silhouette_score': sc, 'bhatta_dist': bd, 'bhatta_coeff': bc}
        
    #delete model after use
    del model
    gc.collect()
    torch.cuda.empty_cache()

import json 

file_path = f"../experiments/{args.alignments[0][5:]}.json" 
# Write the dictionary to a JSON file 
with open(file_path, "w") as json_file:
    json.dump(metric, json_file, indent=4)