import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(size, concentration):
    """ Initialize the grid with a concentration in the center. """
    grid = np.zeros((size, size))
    center = size // 2
    grid[center, center] = concentration
    return grid

def update_grid(grid, diffusion_rate):
    """ Update the grid based on diffusion rate. """
    new_grid = grid.copy()
    for i in range(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            # Apply a simple diffusion rule
            new_grid[i, j] = grid[i, j] + diffusion_rate * (
                grid[i+1, j] + grid[i-1, j] + grid[i, j+1] + grid[i, j-1] - 4 * grid[i, j]
            )
    return new_grid

def diffusion_simulation(size=100, concentration=100, diffusion_rate=0.1, steps=50):
    """ Run the diffusion simulation. """
    grid = initialize_grid(size, concentration)
    grids = [grid]
    
    for _ in range(steps):
        grid = update_grid(grid, diffusion_rate)
        grids.append(grid)

    return grids

# Run the simulation
grids = diffusion_simulation()

# Visualize the result
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
times = [0, 10, 20, 50]  # Time steps to visualize

for ax, t in zip(axes, times):
    ax.imshow(grids[t], cmap='hot', interpolation='nearest')
    ax.set_title(f"Step {t}")
    ax.axis('off')

plt.show()
