# %%
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from transformer_lens.utils import tokenize_and_concatenate
from tqdm import tqdm
import os

# %%

torch.set_grad_enabled(False)

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model and SAE
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
sae, cfg_dict, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device,
)

# Load dataset
dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,
    tokenizer=model.tokenizer,
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)

# %%

def accumulate_feature_acts(sae, model, tokens, batch_size=32, num_batches=None):
    sae.eval()
    
    # Calculate total number of samples
    total_samples = tokens.shape[0] if num_batches is None else min(tokens.shape[0], num_batches * batch_size)
    
    n_feats = sae.W_enc.shape[-1]
    context_size = sae.cfg.context_size
    # Pre-allocate tensor for all activations, now including context_size
    all_feature_acts = torch.empty((total_samples * context_size, n_feats), dtype=torch.float32, device='cpu')
    
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size)):
            batch = tokens[i:i+batch_size]
            _, cache = model.run_with_cache(batch, prepend_bos=False, stop_at_layer=9)
            
            feature_acts = sae.encode(cache[sae.cfg.hook_name])
            # Reshape feature_acts to (batch_size * context_size, n_feats)
            feature_acts_flat = feature_acts.reshape(-1, n_feats)
            all_feature_acts[i*context_size:(i+feature_acts.shape[0])*context_size] = feature_acts_flat.cpu()
            
            del cache
    
    return all_feature_acts

# Generate and save activations
def save_sparse_activations(output_dir, num_samples=10000, batch_size=32):
    os.makedirs(output_dir, exist_ok=True)
    
    tokens = token_dataset[:num_samples]["tokens"]
    activations = accumulate_feature_acts(sae, model, tokens, batch_size)
    
    # Convert to sparse tensor
    sparse_activations = activations.to_sparse()
    
    # Save sparse tensor
    torch.save(sparse_activations, os.path.join(output_dir, f"sparse_activations_{num_samples * sae.cfg.context_size}.pt"))
    print(f"Saved sparse activations for {num_samples * sae.cfg.context_size} token positions")

# %%

tokens = token_dataset[:320]["tokens"]
acts = accumulate_feature_acts(sae, model, tokens, batch_size=32)


# %%
# Usage
# output_directory = "sae_activations"
# save_sparse_activations(output_directory, num_samples=10000)

# %%

total_samples = 320
batch_size=32

n_feats = sae.W_enc.shape[-1]
context_size = sae.cfg.context_size
# Pre-allocate tensor for all activations, now including context_size
all_feature_acts = torch.empty((total_samples * context_size, n_feats), dtype=torch.float32, device='cpu')

with torch.no_grad():
    for i in tqdm(range(0, total_samples, batch_size)):
        batch = tokens[i:i+batch_size]
        _, cache = model.run_with_cache(batch, prepend_bos=False, stop_at_layer=9)
        
        feature_acts = sae.encode(cache[sae.cfg.hook_name])
        # Reshape feature_acts to (batch_size * context_size, n_feats)
        feature_acts_flat = feature_acts.reshape(-1, n_feats)
        all_feature_acts[i*context_size:(i+feature_acts.shape[0])*context_size] = feature_acts_flat.cpu()
        
        del cache

all_feature_acts
