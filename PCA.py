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
        # Instead of sum_squares, we'll maintain the covariance sum
        self.cov_sum: Optional[torch.Tensor] = None

    def partial_fit(self, X: torch.Tensor):
        batch_size = X.shape[0]
        n_features = X.shape[1]
        
        # Initialize parameters on first call
        if self.mean is None:
            self.mean = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            self.sum = torch.zeros_like(self.mean)
            self.cov_sum = torch.zeros((n_features, n_features), 
                                     dtype=X.dtype, 
                                     device=X.device)

        # Update mean using the Welford's online algorithm
        old_mean = self.mean.clone()
        old_sample_count = self.n_samples_seen
        new_sample_count = old_sample_count + batch_size
        
        # Update sum and mean
        batch_sum = torch.sum(X, dim=0)
        self.sum += batch_sum
        self.mean = self.sum / new_sample_count
        
        # Center the batch data
        X_centered = X - self.mean.unsqueeze(0)
        
        # Update covariance sum
        # Using the formula: cov = (X - mean)^T @ (X - mean)
        batch_cov = torch.mm(X_centered.t(), X_centered)
        self.cov_sum += batch_cov
        
        # If this isn't the first batch, apply correction for the mean shift
        if old_sample_count > 0:
            # Correction term for mean shift
            mean_diff = self.mean - old_mean
            correction = (old_sample_count * batch_size) / new_sample_count * \
                        torch.outer(mean_diff, mean_diff)
            self.cov_sum += correction * new_sample_count
        
        self.n_samples_seen = new_sample_count

    def finalize(self):
        if self.n_samples_seen == 0:
            raise ValueError("Cannot finalize PCA without any data")
            
        # Compute final covariance matrix
        cov_matrix = self.cov_sum / (self.n_samples_seen - 1)
        
        # Perform eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by decreasing eigenvalues
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store results
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        total_variance = torch.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance

    def transform(self, X):
        if self.components_ is None:
            raise ValueError("PCA model has not been finalized. Call finalize() first.")
        X_centered = X - self.mean
        return torch.mm(X_centered, self.components_)