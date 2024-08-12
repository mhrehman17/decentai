import torch
import numpy as np

# Define a function to add Gaussian noise to per-sample stochastic gradients.
def GaussianMechanism(grad_tensor, sigma_g, max_norm, batch_size, use_cuda):
    """
    Adds Gaussian noise to per-ample stochastic gradients.

    Parameters:
        grad_tensor (torch tensor): A stochastic gradient.
        sigma_g (float): The variance of the Gaussian noise (defined by DP theory).
        max_norm (float): Clipping value.
        batch_size (int): Number of data points in the considered batch.

    Returns:
        torch tensor: The stochastic gradient with added Gaussian noise.
    """

    # Calculate the standard deviation for the Gaussian noise
    std_gaussian = (2 * sigma_g * max_norm) / batch_size

    # Generate Gaussian noise with the calculated standard deviation and the same shape as the input gradient tensor
    gaussian_noise = torch.normal(0, std_gaussian, grad_tensor.shape)

    # Move the generated Gaussian noise to the GPU if necessary
    if use_cuda:
        gaussian_noise = gaussian_noise.cuda()

    # Add the Gaussian noise to the original stochastic gradient
    return grad_tensor + gaussian_noise