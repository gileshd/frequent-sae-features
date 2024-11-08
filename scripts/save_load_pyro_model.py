import pyro
import torch

def save_pyro_model(model, filename, optimizer=None):
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

def load_pyro_model(model, filename, optimizer=None, device=torch.device('cpu')):
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
    checkpoint = torch.load(filename, map_location=device)
    
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

