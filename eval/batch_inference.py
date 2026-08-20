import torch
import sys
import os
import json
import time
from tqdm import tqdm
from scipy.stats import pearsonr, kendalltau

sys.path.append(os.getcwd())
from eval.reward_adaptor_loader import load_reward_adaptor, inference_process_phi3v, preference_compute

class Args:
    pass

args = Args()
# Path to the trained reward model checkpoint
args.pm_path = "./checkpoints/phi35_cultural_ultra" 
args.pretrain = "microsoft/Phi-3.5-vision-instruct"
args.cache_dir = None
args.ft_projector = True
args.seed = 1234
args.disable_fast_tokenizer = False

# 1. Initialize and load the model
args, model, processor, tokenizer = load_reward_adaptor(
    args, 
    model_type='phi3v', 
    reward_config_path=os.path.join(args.pm_path, "reward_config.yaml"), 
    load_tokenizer=True
)
model.to('cuda')
model.eval()

# 2. Load the evaluation dataset
json_path = "./data/test_dataset.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Successfully loaded {len(data)} test pairs. Starting evaluation...")

# 3. Initialize metrics tracking
correct_pairwise = 0
total_pairwise = 0
total_time = 0.0

all_human_scores = []
all_pred_scores = []

# 4. Evaluation loop
for entry in tqdm(data):
    prompt = entry['prompt']
    img_dir_list = [entry['chosen_path'], entry['reject_path']]
    
    # Retrieve human cultural alignment scores
    human_c_rate = float(entry.get('c_rate', 0))
    human_r_rate = float(entry.get('r_rate', 0))
    
    try:
        # Vision-language preprocessing
        img_inputs = inference_process_phi3v(args, processor, tokenizer, img_dir_list, prompt, device='cuda')
        img_inputs_c = img_inputs[0] # Chosen
        img_inputs_r = img_inputs[1] # Rejected

        start_time = time.time()
        
        with torch.no_grad():
            chosen_rewards, _ = model.custom_forward(**img_inputs_c)
            reject_rewards, _ = model.custom_forward(**img_inputs_r)
        
        # Calculate preference probability (prob > 0.5 indicates 'Chosen' is preferred)
        prob = preference_compute(args, chosen_rewards, reject_rewards)
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        # Update pairwise accuracy
        total_pairwise += 1
        if prob > 0.5:
            correct_pairwise += 1
        
        # Extract absolute scalar rewards for correlation analysis
        try:
            score_c_pred = chosen_rewards.float().mean().item()
            score_r_pred = reject_rewards.float().mean().item()
        except:
            score_c_pred = float(prob)
            score_r_pred = float(1 - prob)
        
        all_human_scores.extend([human_c_rate, human_r_rate])
        all_pred_scores.extend([score_c_pred, score_r_pred])
        
    except Exception as e:
        print(f"\nError processing sample: {prompt[:30]}... Error: {e}")

# 5. Compute and report final metrics
accuracy = correct_pairwise / total_pairwise if total_pairwise > 0 else 0
avg_inference_time = total_time / (total_pairwise * 2) if total_pairwise > 0 else 0

pearson_corr, _ = pearsonr(all_human_scores, all_pred_scores)
kendall_corr, _ = kendalltau(all_human_scores, all_pred_scores)

print("\n" + "="*50)
print("Implicit Cultural Alignment Reward Model - Results")
print("="*50)
print(f"Pairwise Accuracy:       {accuracy * 100:.2f}%")
print(f"Avg Inference Time:      {avg_inference_time:.4f} sec / image")
print(f"Pearson Correlation:     {pearson_corr:.4f}")
print(f"Kendall Correlation:     {kendall_corr:.4f}")
print("="*50)