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
    """
    Accumulate feature activations for a given SAE and model.

    Args:
        sae (SAE): The Sparse Autoencoder object.
        model (HookedTransformer): The transformer model.
        tokens (torch.Tensor): Input tokens to process.
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        num_batches (int, optional): Number of batches to process. If None, process all. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (total_samples * context_size, n_feats) containing all feature activations.
    """
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

# %%

def accumulate_sparse_feature_acts(sae, model, tokens, batch_size=32, num_batches=None):
    """
    Accumulate sparse feature activations for a given SAE and model.

    Args:
        sae (SAE): The Sparse Autoencoder object.
        model (HookedTransformer): The transformer model.
        tokens (torch.Tensor): Input tokens to process.
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        num_batches (int, optional): Number of batches to process. If None, process all. Defaults to None.

    Returns:
        torch.Tensor: A sparse tensor of shape (total_samples * context_size, n_feats) containing non-zero feature activations.
    """
    sae.eval()
    
    # Calculate total number of samples
    total_samples = tokens.shape[0] if num_batches is None else min(tokens.shape[0], num_batches * batch_size)
    
    n_feats = sae.W_enc.shape[-1]
    context_size = sae.cfg.context_size
    total_positions = total_samples * context_size

    # Initialize lists to store indices and values for sparse tensor
    indices = []
    values = []
    
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size)):
            batch = tokens[i:i+batch_size]
            _, cache = model.run_with_cache(batch, prepend_bos=False, stop_at_layer=9)
            
            feature_acts = sae.encode(cache[sae.cfg.hook_name])
            # Reshape feature_acts to (batch_size * context_size, n_feats)
            feature_acts_flat = feature_acts.reshape(-1, n_feats)
            
            # Find non-zero elements
            non_zero = feature_acts_flat.nonzero()
            batch_indices = non_zero[:, 0] + i * context_size
            feat_indices = non_zero[:, 1]
            
            # Append to lists
            indices.append(torch.stack([batch_indices, feat_indices]))
            values.append(feature_acts_flat[non_zero[:, 0], non_zero[:, 1]])
            
            del cache
    
    # Concatenate all indices and values
    indices = torch.cat(indices, dim=1)
    values = torch.cat(values)
    
    # Create sparse tensor
    sparse_activations = torch.sparse_coo_tensor(indices, values, (total_positions, n_feats))
    
    return sparse_activations.coalesce() 

# %%

def save_sparse_activations(output_dir, sparse_activations, num_samples):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sparse tensor
    torch.save(sparse_activations, os.path.join(output_dir, f"sparse_activations_{num_samples * sae.cfg.context_size}.pt"))
    print(f"Saved sparse activations for {num_samples * sae.cfg.context_size} token positions")


# %%

OUTPUT_DIR = "sae_activations"
tokens = token_dataset[:640]["tokens"]
sparse_activations = accumulate_sparse_feature_acts(sae, model, tokens, batch_size=32)
save_sparse_activations(OUTPUT_DIR, sparse_activations, num_samples=640)

# %%
