#%%
## Imports & Installs
import einops
import numpy as np
import torch
from tqdm import tqdm
import plotly.express as px
import matplotlib.pyplot as plt

# Imports for displaying vis in Colab / notebook
import webbrowser
import http.server
import socketserver
import threading

PORT = 8000

torch.set_grad_enabled(False);

#%%

## Set Up

if torch.backends.mps.is_available():
    import os
    device = "mps"
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

#%%

# Loading a pretrained Sparse Autoencoder
# Below we load a Transformerlens model, a pretrained SAE and a dataset from huggingface.

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
    device=device,
)

#%%

from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)
#%%

# ## Basic Analysis
#
# Let's check some basic stats on this SAE in order to see how some basic functionality in the codebase works.
#
# We'll calculate:
# - L0 (the number of features that fire per activation)
# - The cross entropy loss when the output of the SAE is used in place of the activations

# + [markdown] id="xOcubgsRv611"
# ### L0 Test and Reconstruction Test

# + id="gAUR5CRBv611"
sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with torch.no_grad():
    # activation store can give us tokens.
    batch_tokens = token_dataset[:32]["tokens"]
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    # Use the SAE
    feature_acts = sae.encode(cache[sae.cfg.hook_name])
    sae_out = sae.decode(feature_acts)

    # save some room
    del cache

    # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
    l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
    print("average l0", l0.mean().item())
    px.histogram(l0.flatten().cpu().numpy()).show()

# + [markdown] id="ijoelLtdv611"
# Note that while the mean L0 is 64, it varies with the specific activation.
#
# To estimate reconstruction performance, we calculate the CE loss of the model with and without the SAE being used in place of the activations. This will vary depending on the tokens.
# -

#%%

# ## Calculate PCA of Activations
# from torch.utils.data import DataLoader, TensorDataset
from PCA import PCA

#%%
sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with torch.no_grad():
    # activation store can give us tokens.
    batch_tokens = token_dataset[:32]["tokens"]
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    # Use the SAE
    model_acts = cache[sae.cfg.hook_name]
    # feature_acts = sae.encode(model_acts)
    # sae_out = sae.decode(feature_acts)

    # save some room
    del cache

# +

flattened_acts = einops.rearrange(model_acts, 'b s ... -> (b s) ...')
print(f"Original shape: {model_acts.shape}")
print(f"Flattened shape: {flattened_acts.shape}")

#%%

def get_model_acts(tokens):
    _, cache = model.run_with_cache(tokens, prepend_bos=True)
    model_acts = cache[sae.cfg.hook_name]
    del cache
    return model_acts

def fit_pca(model_acts, n_components=10) -> PCA:
    pca = PCA(n_components=n_components)
    pca.fit(model_acts.cpu())
    return pca

#%%

batch_tokens = token_dataset[:32]["tokens"]
model_acts = get_model_acts(batch_tokens)
flattened_acts = einops.rearrange(model_acts, 'b s ... -> (b s) ...')
pca = fit_pca(flattened_acts)
print(pca.explained_variance_ratio)

#%%

def get_model_acts_dataset(tokens, batch_size=32):
    all_acts = []
    
    for i in tqdm(range(0, len(tokens), batch_size)):
        batch_tokens = tokens[i:i+batch_size]
        batch_acts = get_model_acts(batch_tokens)
        all_acts.append(batch_acts.cpu())  # Move to CPU to save GPU memory
    
    return torch.cat(all_acts, dim=0)

#%%

# Use the function to get all model activations
all_model_acts = get_model_acts_dataset(token_dataset[:20000]["tokens"])
print(f"Shape of all model activations: {all_model_acts.shape}")

#%%

# all_model_acts = get_model_acts_dataset(token_dataset["tokens"])
# print(f"Shape of all model activations: {all_model_acts.shape}")

#%% 
# Now you can use all_model_acts for your PCA
flattened_acts = einops.rearrange(all_model_acts, 'b s ... -> (b s) ...')
pca = fit_pca(flattened_acts)
print(pca.explained_variance_ratio)

plt.plot(pca.explained_variance_ratio);
plt.yscale('log');

#%%

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a,axis=1) * np.linalg.norm(b))

def calc_pc_weight_cosine_sim(pca, weights, component=0):
    first_pc = pca.components_[:, component]
    cosine_similarities = cosine_sim(weights, first_pc)
    return cosine_similarities

def plot_cosine_sim(sims, sorted=True, ax=None, component=0):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    sims = np.abs(sims)
    if sorted:
        sims = np.sort(sims)[::-1]
    ax.plot(sims)
    ax.set_title(f'Sorted Absolute Cosine Similarities between Decoder Weights Principal Component {component}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Cosine Similarity')
    ax.grid(True)

#%% 

fig, ax = plt.subplots(figsize=(8, 12), nrows=3)

decoder_weights = sae.W_dec.cpu().detach().numpy()
pc1_cosine_sim = calc_pc_weight_cosine_sim(pca, decoder_weights, component=0)
plot_cosine_sim(pc1_cosine_sim, ax=ax[0], component=0)

pc2_cosine_sim = calc_pc_weight_cosine_sim(pca, decoder_weights, component=1)
plot_cosine_sim(pc2_cosine_sim, ax=ax[1], component=1)

pc3_cosine_sim = calc_pc_weight_cosine_sim(pca, decoder_weights, component=2)
plot_cosine_sim(pc3_cosine_sim, ax=ax[2], component=2)

plt.tight_layout()

# %%
# Zoom in on top decoder weights.
pc1_cosine_sim[np.argsort(pc1_cosine_sim)[::-1][:10]]
decoder_weights = sae.W_dec.cpu().detach().numpy()
pc1_cosine_sim = calc_pc_weight_cosine_sim(pca, decoder_weights, component=0)
abs_sim = np.abs(pc1_cosine_sim)
print(np.sort(abs_sim)[::-1])
plt.plot(np.sort(abs_sim)[::-1][:100]);
plt.xlabel('Decoder Weights')
plt.ylabel('Abs Cosine Similarity with PC1');
# %% 

WWT = decoder_weights @ decoder_weights.T
norms = np.linalg.norm(decoder_weights, axis=1)
C = WWT / (norms[:, None] * norms[None, :])

# %%
plt.hist(C[np.triu_indices(len(C), k=1)], bins=100);
# %%

# There are a moderate number of weights that are pretty similar to one another.
# - [ ] Check if they are both actually active.
plt.hist(C[np.triu_indices(len(C), k=1)], bins=100, log=True);
# %%

sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with torch.no_grad():
    # activation store can give us tokens.
    batch_tokens = token_dataset[:32]["tokens"]
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    # Use the SAE
    feature_acts = sae.encode(cache[sae.cfg.hook_name])
    sae_out = sae.decode(feature_acts)

    # save some room
    del cache

    # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
    l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
    print("average l0", l0.mean().item())
    px.histogram(l0.flatten().cpu().numpy()).show()
# %%

# TODO: I am wasting cycles here by calculating the activations twice - once for saes and then again for PCA.

def accumulate_feature_acts(sae, model, tokens, batch_size=32, num_batches=None):
    sae.eval()
    all_feature_acts = []
    
    with torch.no_grad():
        for i in tqdm(range(0, tokens.shape[0], batch_size)):
            if num_batches is not None and i // batch_size >= num_batches:
                break
            
            batch = tokens[i:i+batch_size]
            _, cache = model.run_with_cache(batch, prepend_bos=True)
            
            feature_acts = sae.encode(cache[sae.cfg.hook_name])
            all_feature_acts.append(feature_acts.cpu())  # Move to CPU to save GPU memory
            
            del cache
    
    return torch.cat(all_feature_acts, dim=0)

# Usage example:
# Assuming token_dataset[:100_000]["tokens"] returns a tensor
tokens = token_dataset[:1000]["tokens"]
accumulated_feature_acts = accumulate_feature_acts(sae, model, tokens, batch_size=32, num_batches=None)

# Calculate and display L0 stats
l0 = (accumulated_feature_acts[:, 1:] > 0).float().sum(-1).detach()
print("average l0", l0.mean().item())
px.histogram(l0.flatten().cpu().numpy()).show()


# def accumulate_feature_acts(sae, model, tokens, batch_size=32, num_batches=None):
#     sae.eval()
#     all_feature_acts = []
    
#     with torch.no_grad():
#         for i, batch in enumerate(tqdm(tokens.iter(batch_size=batch_size))):
#             if num_batches is not None and i >= num_batches:
#                 break
            
#             _, cache = model.run_with_cache(batch, prepend_bos=True)
            
#             feature_acts = sae.encode(cache[sae.cfg.hook_name])
#             all_feature_acts.append(feature_acts.cpu())  # Move to CPU to save GPU memory
            
#             del cache
    
#     return torch.cat(all_feature_acts, dim=0)

# # Usage example:
# accumulated_feature_acts = accumulate_feature_acts(sae, model, token_dataset, batch_size=32, num_batches=20)

# # Calculate and display L0 stats
# l0 = (accumulated_feature_acts[:, 1:] > 0).float().sum(-1).detach()
# print("average l0", l0.mean().item())
# px.histogram(l0.flatten().cpu().numpy()).show()
# %%

flat_sae_acts = einops.rearrange(accumulated_feature_acts, 'b s ... -> (b s) ...')
sae_latent_freqs = (flat_sae_acts > 0).numpy().mean(0)

# %%

# %%
from PCA import IncrementalPCA

def fast_incremental_pca_on_dataset(get_model_acts, tokens, n_components=10, batch_size=32):
    pca = IncrementalPCA(n_components=n_components)
    
    # Calculate total number of batches
    total_samples = tokens.shape[0]
    total_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
    
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for i in range(0, total_samples, batch_size):
            batch_tokens = tokens[i:i+batch_size]
            batch_acts = get_model_acts(batch_tokens)
            flattened_acts = einops.rearrange(batch_acts, 'b s ... -> (b s) ...')
            pca.partial_fit(flattened_acts.cpu())  # Move to CPU if necessary
            pbar.update(1)
    
    pca.finalize()
    return pca

# Usage
pca = fast_incremental_pca_on_dataset(get_model_acts, token_dataset[:100_000]['tokens'])
print(pca.explained_variance_ratio)

# %%

# TODO: Add marginal histograms

pc1_cosine_sim = calc_pc_weight_cosine_sim(pca, decoder_weights, component=0)
plt.plot(sae_latent_freqs, np.abs(pc1_cosine_sim), '.', alpha=0.2);
plt.xlabel('Frequency of SAE Latent');
plt.ylabel('Abs Cosine Similarity with PC1');

#%%

pc2_cosine_sim = calc_pc_weight_cosine_sim(pca, decoder_weights, component=1)
plt.plot(sae_latent_freqs, np.abs(pc2_cosine_sim), '.', alpha=0.2);
plt.xlabel('Frequency of SAE Latent');
plt.ylabel('Abs Cosine Similarity with PC2');

#%%

pc2_cosine_sim = calc_pc_weight_cosine_sim(pca, decoder_weights, component=2)
plt.plot(sae_latent_freqs, np.abs(pc2_cosine_sim), '.', alpha=0.2);
plt.xlabel('Frequency of SAE Latent');
plt.ylabel('Abs Cosine Similarity with PC3');

# %%

cosine_sims = []
for comp in range(10):
    pc_cosine_sim = calc_pc_weight_cosine_sim(pca, decoder_weights, component=comp)
    cosine_sims.append(pc_cosine_sim)

stacked_cosine_sims = np.stack(cosine_sims)
most_sim_pc = np.argmax(np.abs(stacked_cosine_sims), axis=0)
# max_cosine_sims = np.max(np.abs(stacked_cosine_sims), axis=0)


# %%
stacked_cosine_sims[most_sim_pc].shape

# %%

plt.plot(sae_latent_freqs, stacked_cosine_sims[most_sim_pc], '.', alpha=0.2);
plt.xlabel('PC');
plt.ylabel('Max Abs Cosine Similarity');

# %%
def project_onto_pca_subspace(decoder_weights, pca, n_components=10):
    """
    Project each row of decoder_weights onto the subspace defined by the top n_components PCA components.
    
    Args:
    decoder_weights (np.ndarray): The decoder weights matrix, shape (n_features, n_dims)
    pca (PCA): The fitted PCA object
    n_components (int): Number of top PCA components to use for the subspace
    
    Returns:
    np.ndarray: Projected decoder weights, shape (n_features, n_dims)
    np.ndarray: Magnitude of the projection for each feature, shape (n_features,)
    """
    # Ensure we're not using more components than available
    n_components = min(n_components, pca.n_components)
    
    # Get the top n_components
    components = pca.components_.T
    top_components = components[:n_components]
    
    # Project each row of decoder_weights onto the PCA subspace
    projections = np.dot(decoder_weights, top_components.T)
    
    # Reconstruct the projected vectors in the original space
    projected_weights = np.dot(projections, top_components)
    
    # Calculate the magnitude of the projection for each feature
    projection_magnitudes = np.linalg.norm(projected_weights, axis=1)
    
    return projected_weights, projection_magnitudes

# Usage example:
projected_weights, projection_magnitudes = project_onto_pca_subspace(decoder_weights, pca)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(sae_latent_freqs, projection_magnitudes, alpha=0.5)
plt.xlabel('SAE Latent Frequency')
plt.ylabel('Magnitude of Projection onto PCA Subspace')
plt.title('SAE Latent Frequency vs Projection Magnitude')
plt.colorbar(label='Feature Index')
plt.show()

# Print some statistics
print(f"Mean projection magnitude: {projection_magnitudes.mean():.4f}")
print(f"Median projection magnitude: {np.median(projection_magnitudes):.4f}")
print(f"Max projection magnitude: {projection_magnitudes.max():.4f}")
print(f"Min projection magnitude: {projection_magnitudes.min():.4f}")
# %%

pca.components_[0]


