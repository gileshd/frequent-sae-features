# %%
import math
import torch
import pyro
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
import argparse

from prodlda import ProdLDA
from save_load_pyro_model import save_pyro_model

# setting global variables
# smoke_test = False
seed = 0
torch.manual_seed(seed)
pyro.set_rng_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# LOAD DATA
# Load sparse tensor and convert to dense array
sparse_docs = torch.load('sae_activations/sparse_activations_122880.pt', weights_only=True)
docs = sparse_docs.to_dense()
docs = (docs > 0).float().to(device)


batch_size = 128
learning_rate = 1e-3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ProdLDA model')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save the model checkpoint (e.g., prodlda_checkpoints/model.pt)')
    parser.add_argument('--smoke-test', action='store_true',
                       help='Run in smoke test mode (smaller model, fewer epochs)')
    args = parser.parse_args()

    # Initialize model and training components
    pyro.clear_param_store()

    # Use args.smoke_test instead of the global variable
    num_topics = 3 if args.smoke_test else 300
    num_epochs = 1 if args.smoke_test else 500
    
    prodLDA = ProdLDA(
        vocab_size=docs.shape[1],
        num_topics=num_topics,
        hidden=10 if args.smoke_test else 300,
        dropout=0.2
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = 1 if args.smoke_test else int(math.ceil(docs.shape[0] / batch_size))

    print("Model info:")
    print(f"num_topics: {num_topics}")

    print("Training info:")
    print(f"num_batches: {num_batches}")
    print(f"num_epochs: {num_epochs}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")

    print("Starting training loop...")
    # Training loop
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

    # Get final beta
    beta = prodLDA.beta()

    # Save the model locally first
    save_pyro_model(
        model=prodLDA,
        optimizer=None,
        filename=args.output
    )
