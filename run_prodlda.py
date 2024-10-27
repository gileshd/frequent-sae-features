# %%
import math
import torch
import pyro
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange

from prodlda import ProdLDA

# setting global variables
smoke_test = False
seed = 0
torch.manual_seed(seed)
pyro.set_rng_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_topics = 20 if not smoke_test else 3

# %%

# LOAD DATA
USE_NEWGROUPS = False
if USE_NEWGROUPS:
    from load_newsgroups_docs import load_newsgroups_docs
    docs, vocab = load_newsgroups_docs()
    docs = docs.float()
else:
    # Load sparse tensor and convert to dense array
    sparse_docs = torch.load('sae_activations/sparse_activations_122880.pt', weights_only=True)
    docs = sparse_docs.to_dense()
    docs = (docs > 0).float().to(device)

# %%

batch_size = 64
learning_rate = 1e-3
num_epochs = 100 if not smoke_test else 1

pyro.clear_param_store()

prodLDA = ProdLDA(
    vocab_size=docs.shape[1],
    num_topics=num_topics,
    hidden=100 if not smoke_test else 10,
    dropout=0.2
)
prodLDA.to(device)

optimizer = pyro.optim.Adam({"lr": learning_rate})
svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
num_batches = int(math.ceil(docs.shape[0] / batch_size)) if not smoke_test else 1

# %%

bar = trange(num_epochs)
loss_history = []
for epoch in bar:
    running_loss = 0.0
    for i in range(num_batches):
        batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
        loss = svi.step(batch_docs)
        running_loss += loss / batch_docs.size(0)
    loss_history.append(running_loss)

    bar.set_postfix(epoch_loss='{:.5e}'.format(running_loss))

# %%
beta = prodLDA.beta()
# %%

# %%



# %%

def save_pyro_model(model, optimizer, filename):
    """
    Save a trained Pyro model along with its parameters and optimizer state.
    
    Args:
        model: The Pyro model module/class instance
        optimizer: The optimizer used for training
        filename: Path where the model should be saved
    """
    # Get model state dict
    model_state_dict = model.state_dict()
    
    # Get optimizer state dict
    if optimizer is not None:
        optimizer_state_dict = optimizer.state_dict()
    else:
        optimizer_state_dict = None
    
    # Get all Pyro params
    pyro_params = {}
    for name, param in pyro.get_param_store().items():
        pyro_params[name] = param.detach().cpu()
    
    # Save everything in a single checkpoint file
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'pyro_params': pyro_params
    }
    
    torch.save(checkpoint, filename)

def load_pyro_model(model, filename, optimizer=None):
    """
    Load a saved Pyro model along with its parameters and optimizer state.
    
    Args:
        model: The Pyro model module/class instance to load into
        optimizer: The optimizer instance to load state into  
        filename: Path to the saved model checkpoint
    
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
    """
    # Load checkpoint
    checkpoint = torch.load(filename)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load Pyro params
    pyro.clear_param_store()
    for name, param in checkpoint['pyro_params'].items():
        pyro.get_param_store()[name] = param
    
    return model, optimizer

# %%

MODEL_CHECKPOINT_DIR = "prodlda_checkpoints"
# Saving a model
save_pyro_model(
    model=prodLDA,
    optimizer=None,
    filename=f'{MODEL_CHECKPOINT_DIR}/prodlda_checkpoint_big.pt'
)

# # Loading a model
# model, optimizer = load_pyro_model(
#     model=prodLDA,
#     optimizer=optimizer,
#     filename=f'{MODEL_CHECKPOINT_DIR}/prodlda_checkpoint.pt'
# )
# %%
loss_history
