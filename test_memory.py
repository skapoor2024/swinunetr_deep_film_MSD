import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Tuple, Dict
import gc

def calculate_model_memory(model: torch.nn.Module, 
                         input_size: Tuple,
                         batch_size: int = 1,
                         device: str = 'cuda') -> Dict:
    """
    Calculate memory consumption of a PyTorch model during forward and backward passes.
    
    Args:
        model: The PyTorch model to analyze
        input_size: Input size excluding batch dimension (e.g., (1, 96, 96, 96) for 3D volume)
        batch_size: Batch size to use for calculation
        device: Device to run the calculation on ('cuda' or 'cpu')
        
    Returns:
        Dict containing memory statistics in MB
    """
    # Clear cache and garbage collect
    torch.cuda.empty_cache()
    gc.collect()
    
    # Move model to specified device
    model = model.to(device)
    model.train()  # Set to training mode
    
    # Create dummy input
    x = torch.randn(batch_size, *input_size).to(device)
    
    # Get initial memory allocation
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated()
    
    def get_memory_stats():
        """Get current memory statistics"""
        return {
            'current_mb': torch.cuda.memory_allocated() / 1024**2,
            'peak_mb': torch.cuda.max_memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2
        }
    
    memory_stats = {}
    
    # Profile forward pass
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True, record_shapes=True) as prof:
        with record_function("forward_pass"):
            output = model(x)
            memory_stats['forward'] = get_memory_stats()
    
    # Calculate loss for backward pass
    criterion = torch.nn.MSELoss()
    target = torch.randn_like(output)
    
    # Profile backward pass
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True, record_shapes=True) as prof:
        with record_function("backward_pass"):
            loss = criterion(output, target)
            loss.backward()
            memory_stats['backward'] = get_memory_stats()
    
    # Calculate model parameters memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    # Calculate buffer memory (for batch norm, etc.)
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
    
    # Get gradient memory
    grad_memory = sum(p.grad.numel() * p.grad.element_size() 
                     for p in model.parameters() if p.grad is not None) / 1024**2
    
    # Compile final statistics
    final_stats = {
        'model_parameters_mb': param_memory,
        'model_buffers_mb': buffer_memory,
        'gradients_mb': grad_memory,
        'forward_pass_peak_mb': memory_stats['forward']['peak_mb'],
        'backward_pass_peak_mb': memory_stats['backward']['peak_mb'],
        'total_peak_mb': memory_stats['backward']['peak_mb'],
        'reserved_memory_mb': memory_stats['backward']['reserved_mb']
    }
    
    # Clean up
    del x, output, target
    torch.cuda.empty_cache()
    gc.collect()
    
    return final_stats

# Usage example
def print_memory_stats(stats: Dict):
    """Print memory statistics in a readable format"""
    print("\nModel Memory Consumption Analysis:")
    print("-" * 50)
    print(f"Model Parameters Memory: {stats['model_parameters_mb']:.2f} MB")
    print(f"Model Buffers Memory: {stats['model_buffers_mb']:.2f} MB")
    print(f"Gradients Memory: {stats['gradients_mb']:.2f} MB")
    print(f"Forward Pass Peak Memory: {stats['forward_pass_peak_mb']:.2f} MB")
    print(f"Backward Pass Peak Memory: {stats['backward_pass_peak_mb']:.2f} MB")
    print(f"Total Peak Memory: {stats['total_peak_mb']:.2f} MB")
    print(f"Reserved CUDA Memory: {stats['reserved_memory_mb']:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    from model.SwinUNETR_DEEP_FILM_2 import SwinUNETR_DEEP_FILM
    
    # Initialize your model
    model = SwinUNETR_DEEP_FILM(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=32,
        precomputed_prompt_path='./pretrained_weights/embeddings_template_flare.pkl'
    )
    
    # Calculate memory consumption
    stats = calculate_model_memory(
        model=model,
        input_size=(1, 96, 96, 96),  # Adjust based on your input size
        batch_size=1,
        device='cuda'
    )
    
    # Print results
    print_memory_stats(stats)
