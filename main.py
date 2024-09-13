# main.py
import torch
from omegaconf import OmegaConf
from src.data_generation import generate_data

def main():
    # Load configuration
    cfg = OmegaConf.load('config/config.yaml')
    
    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility
    # set_seed(cfg.seed)
    
    # Generate data
    X, y, graph_filters_flat, weight_matrix, filter_coefficients = generate_data(cfg)

    print(X.shape, y.shape, graph_filters_flat.shape, weight_matrix.shape, filter_coefficients.shape)

if __name__ == "__main__":
    main()