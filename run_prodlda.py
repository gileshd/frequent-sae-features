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
smoke_test = True
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
    sparse_docs = torch.load('sae_activations/sparse_activations_40960.pt', weights_only=True)
    docs = sparse_docs.to_dense()
    docs = (docs > 0).float()  

# %%

batch_size = 32
learning_rate = 1e-3
num_epochs = 50 if not smoke_test else 1

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
for epoch in bar:
    running_loss = 0.0
    for i in range(num_batches):
        batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
        loss = svi.step(batch_docs)
        running_loss += loss / batch_docs.size(0)

    bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))

# %%
beta = prodLDA.beta()
# %%
