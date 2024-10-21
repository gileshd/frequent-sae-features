import torch
from typing import Optional

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


class IncrementalPCA:
    def __init__(self, n_components: int):
        self.n_components: int = n_components
        self.mean: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None
        self.explained_variance: Optional[torch.Tensor] = None
        self.explained_variance_ratio: Optional[torch.Tensor] = None
        self.n_samples_seen: int = 0
        self.sum: Optional[torch.Tensor] = None
        self.sum_squares: Optional[torch.Tensor] = None

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
        assert self.sum is not None
        assert self.sum_squares is not None
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
