import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


# Load the data
df = pd.read_csv('homework_1.2.csv')

# Split into treatment (X=1) and control (X=0)
treated = df[df['X'] == 1].copy()
control = df[df['X'] == 0].copy()

# Use 'Z' for matching
Z_cols = [col for col in df.columns if col.startswith('Z')]
Z_treated = treated[Z_cols].values
Z_control = control[Z_cols].values

# Fit Nearest Neighbors model on control group
nn = NearestNeighbors(n_neighbors=1)
nn.fit(Z_control)

# Find nearest control neighbors for each treated sample
distances, indices = nn.kneighbors(Z_treated)

# Get matched control samples
matched_control = control.iloc[indices.flatten()].copy()
matched_control.index = treated.index  # align with treated for merging

# Combine treated with their matched control
matched_pairs = pd.concat([treated, matched_control.add_suffix('_matched')], axis=1)

# Compute distances
matched_pairs['Z_distance'] = abs(matched_pairs['Z'] - matched_pairs['Z_matched'])

# Show first few matched pairs with distances
print(matched_pairs[['Unnamed: 0', 'X', 'Z', 'Unnamed: 0_matched', 'X_matched', 'Z_matched', 'Z_distance']].head())

# Find the farthest match
max_distance = matched_pairs['Z_distance'].max()
print(f"\nFarthest match distance: {max_distance:.5f}")

# Calculate average Y values
avg_Y_treated = matched_pairs['Y'].mean()
avg_Y_matched_control = matched_pairs['Y_matched'].mean()

# Compute the effect
effect = avg_Y_treated - avg_Y_matched_control
print(f"Effect: {effect:.4f}")

Z_cols = [col for col in df.columns if col.startswith('Z')]
Z_treated = treated[Z_cols].values
Z_control = control[Z_cols].values
Y_control = control['Y'].values
Y_treated = treated['Y'].values

# Fit NearestNeighbors with radius = 0.2
nn = NearestNeighbors(radius=0.2)
nn.fit(Z_control)

# Find neighbors within radius for each treated row
distances, indices = nn.radius_neighbors(Z_treated)

# Count total matches and duplicates
all_matched_indices = [idx for group in indices for idx in group]
duplicates_count = len(all_matched_indices) - len(set(all_matched_indices))
print(f"Duplicate matches (excluding first): {duplicates_count}")

# Compute effect
group_means = []
for group in indices:
    if len(group) > 0:
        group_mean = Y_control[group].mean()
        group_means.append(group_mean)

effect = Y_treated.mean() - np.mean(group_means)
print(f"Effect: {effect:.4f}")