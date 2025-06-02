import numpy as np
import os
import matplotlib.pyplot as plt
from botorch.utils.transforms import unnormalize, normalize
import torch

# Path to the .npz file
npz_path = "0.npz"

lb = np.array([
                0.01,    # Torso radius
                0.01,     # Leg segment 1 x
                0.01,     # Leg segment 1 y
                0.01,     # Leg segment 2 x
                0.01,     # Leg segment 2 y
                0.01,     # Foot x
            ])
            
ub = np.array([
                0.5,     # Torso radius
                0.5,     # Leg segment 1 x
                0.5,     # Leg segment 1 y
                0.5,     # Leg segment 2 x
                0.5,     # Leg segment 2 y
                0.5,     # Foot x
            ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

bounds = torch.cat((torch.from_numpy(lb).reshape(1, -1), 
                        torch.from_numpy(ub).reshape(1, -1)), 0).to(dtype=dtype, device=device) 



# Check if the file exists
if not os.path.exists(npz_path):
    print(f"File {npz_path} does not exist.")
    exit(1)

# Load the .npz file
data = np.load(npz_path, allow_pickle=True)

# Print available keys in the file
print("Keys in the .npz file:")
print(data.files)

print(data['x'], data['x'].shape)

# Extract y data
y = data['y']

# Add diagnostic information
print(f"Shape of y: {y.shape}")
print(f"Type of y: {type(y)}")
print(f"First few y values: {y[:5]}")

# Fix: Flatten y if it's a 2D array
if len(y.shape) > 1:
    y_flat = y.flatten() if hasattr(y, 'flatten') else np.array([item[0] for item in y])
    print(f"Flattened y shape: {y_flat.shape}")
else:
    y_flat = y

iterations = np.arange(1, len(y_flat) + 1)  # Create iteration numbers

# Plot y vs iterations
plt.figure(figsize=(10, 6))
plt.plot(iterations, y_flat, linestyle='-', linewidth=2)  # Added linewidth for better visibility
plt.title('Y vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Y Value')
plt.grid(True)

# Save the figure
plt.savefig('y_vs_iterations.png', dpi=300)
# plt.show()

# Find the best x (corresponding to the largest y)
best_idx = np.argmax(y_flat)
best_y = y_flat[best_idx]
best_x = data['x'][best_idx]

best_x = unnormalize(torch.from_numpy(best_x).to(dtype=dtype, device=device), bounds)

print(f"Best y value: {best_y} at iteration {best_idx + 1}")
print(f"Corresponding x value: {best_x}")

# Save the best x to a txt file with each value on a separate row
with open('best_x.txt', 'w') as f:
    for value in best_x:
        f.write(f"{value}\n")

print(f"Best x values saved to best_x.txt")

# Find the best 20 y values and their corresponding x values using the flattened array
top_indices = np.argsort(y_flat)[-20:][::-1]  # Sort indices by y value in descending order
print(f"Top 20 indices: {top_indices}")
top_y = y_flat[top_indices]
top_x = data['x'][top_indices]
top_iterations = iterations[top_indices]

# Print the top 20 y values and their corresponding x values
print("\nTop 20 y values and their corresponding x values:")
for i in range(20):
    print(f"Rank {i+1}: y = {top_y[i]} at iteration {top_iterations[i]}")
    
    # Convert to tensor, unnormalize, and move back to CPU before saving
    unnormalized_x = unnormalize(torch.from_numpy(top_x[i]).to(dtype=dtype, device=device), bounds).cpu()
    
    # Save each x to a separate txt file
    with open(f'best_x_rank_{i+1}.txt', 'w') as f:
        for value in unnormalized_x:
            f.write(f"{value}\n")
    
    print(f"Saved x values to best_x_rank_{i+1}.txt")

# Also save all top 20 x values to a single file with rankings
with open('top_20_x_values.txt', 'w') as f:
    for i in range(20):
        f.write(f"Rank {i+1}: y = {top_y[i]} at iteration {top_iterations[i]}\n")
        f.write("x values:\n")
        
        # Convert to tensor, unnormalize, and move back to CPU before saving
        unnormalized_x = unnormalize(torch.from_numpy(top_x[i]).to(dtype=dtype, device=device), bounds).cpu()
        
        for value in unnormalized_x:
            f.write(f"{value}\n")
        f.write("\n")

print("\nSaved all top 20 x values to top_20_x_values.txt")

