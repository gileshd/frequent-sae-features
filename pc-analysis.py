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

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components_ = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # Center the data
        self.mean = torch.mean(X, dim=0)
        X = X - self.mean

        # Compute the covariance matrix
        cov = torch.mm(X.t(), X) / (X.size(0) - 1)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort eigenvectors by decreasing eigenvalues
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store the first n_components
        self.components_ = eigenvectors[:, :self.n_components]

        # Calculate and store the proportion of variance explained
        total_variance = torch.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] 
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance

    def transform(self, X):
        if self.components_ is None:
            raise ValueError("PCA model has not been fitted yet. Call fit() first.")
        X = X - self.mean
        return torch.mm(X, self.components_)



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

def fit_pca(model_acts, n_components=10):
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
all_model_acts = get_model_acts_dataset(token_dataset[:1000]["tokens"])
print(f"Shape of all model activations: {all_model_acts.shape}")

# %%

all_model_acts = get_model_acts_dataset(token_dataset["tokens"])
print(f"Shape of all model activations: {all_model_acts.shape}")

#%% 
# Now you can use all_model_acts for your PCA
flattened_acts = einops.rearrange(all_model_acts, 'b s ... -> (b s) ...')
pca = fit_pca(flattened_acts)
print(pca.explained_variance_ratio)

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

def accumulate_feature_acts(sae, model, token_dataset, batch_size=32, num_batches=None):
    sae.eval()
    all_feature_acts = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(token_dataset.iter(batch_size=batch_size))):
            if num_batches is not None and i >= num_batches:
                break
            
            batch_tokens = batch["tokens"]
            _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
            
            feature_acts = sae.encode(cache[sae.cfg.hook_name])
            all_feature_acts.append(feature_acts.cpu())  # Move to CPU to save GPU memory
            
            del cache
    
    return torch.cat(all_feature_acts, dim=0)

# Usage example:
accumulated_feature_acts = accumulate_feature_acts(sae, model, token_dataset, batch_size=32, num_batches=20)

# Calculate and display L0 stats
l0 = (accumulated_feature_acts[:, 1:] > 0).float().sum(-1).detach()
print("average l0", l0.mean().item())
px.histogram(l0.flatten().cpu().numpy()).show()
# %%

flat_sae_acts = einops.rearrange(accumulated_feature_acts, 'b s ... -> (b s) ...')

# %%

import torch
from tqdm import tqdm

class FastIncrementalPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.n_samples_seen = 0
        self.sum = None
        self.sum_squares = None

    def partial_fit(self, X):
        if self.mean is None:
            self.mean = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            self.sum = torch.zeros_like(self.mean)
            self.sum_squares = torch.zeros_like(self.mean)

        batch_size = X.shape[0]
        self.n_samples_seen += batch_size

        # Update sum and sum of squares
        self.sum += torch.sum(X, dim=0)
        self.sum_squares += torch.sum(X ** 2, dim=0)

    def finalize(self):
        # Compute final mean and covariance
        self.mean = self.sum / self.n_samples_seen
        total_variance = (self.sum_squares / self.n_samples_seen) - (self.mean ** 2)
        
        # Perform SVD on the covariance matrix
        cov_matrix = torch.diag(total_variance)
        U, S, V = torch.svd(cov_matrix)

        # Store results
        self.components = V[:, :self.n_components]
        self.explained_variance = S[:self.n_components]
        total_variance = torch.sum(S)
        self.explained_variance_ratio = self.explained_variance / total_variance

    def transform(self, X):
        if self.components is None:
            raise ValueError("PCA model has not been finalized. Call finalize() first.")
        X_centered = X - self.mean
        return torch.mm(X_centered, self.components)

def fast_incremental_pca_on_dataset(get_model_acts, token_dataset, n_components=10, batch_size=32):
    pca = FastIncrementalPCA(n_components=n_components)
    
    # Calculate total number of batches
    total_samples = len(token_dataset)
    total_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
    
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for batch in token_dataset.iter(batch_size=batch_size):
            batch_tokens = batch["tokens"]
            batch_acts = get_model_acts(batch_tokens)
            flattened_acts = einops.rearrange(batch_acts, 'b s ... -> (b s) ...')
            pca.partial_fit(flattened_acts.cpu())  # Move to CPU if necessary
            pbar.update(1)
    
    pca.finalize()
    return pca

# Usage
pca = fast_incremental_pca_on_dataset(get_model_acts, token_dataset[:100_000])
print(pca.explained_variance_ratio)

# %%
token_dataset
# %%

token_dataset.iter(batch_size=10)
token_dataset[:10]
# %%
