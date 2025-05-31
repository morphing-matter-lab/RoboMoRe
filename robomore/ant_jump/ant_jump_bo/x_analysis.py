import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize array to store all data (20 files × 12 components)
data = np.zeros((20, 12))

# Read data from each file
for i in range(1, 21):
    filename = f'best_x_rank_{i}.txt'
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for j, line in enumerate(lines[:12]):  # Only first 12 lines
                data[i-1, j] = float(line.strip())
    except FileNotFoundError:
        print(f"Warning: {filename} not found")

# Calculate mean and std for each component
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)

# Create visualization
plt.figure(figsize=(12, 7))

# Bar plot with error bars
x = np.arange(1, 13)
plt.bar(x, means, yerr=stds, alpha=0.7, capsize=5)

# Add grid and labels
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Component', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Mean and Standard Deviation for Each x Component', fontsize=14)
plt.xticks(x, [f'x{i}' for i in range(1, 13)])

# Show data values
for i, (mean, std) in enumerate(zip(means, stds)):
    plt.text(i+1, mean + std + 5, f'{mean:.2f}±{std:.2f}', ha='center')

plt.tight_layout()
plt.savefig('x_components_stats.png', dpi=300)
# plt.show()

# Also create a boxplot for more distribution details
plt.figure(figsize=(12, 7))
sns.boxplot(data=data)
plt.xlabel('Component')
plt.ylabel('Value')
plt.title('Distribution of x Components Across Top 20 Ranks')
plt.xticks(range(12), [f'x{i+1}' for i in range(12)])
plt.savefig('x_components_boxplot.png', dpi=300)
# plt.show()