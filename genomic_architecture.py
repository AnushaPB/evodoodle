import numpy as np
import pandas as pd

# Parameters
num_loci = 10000
proportion_no_mutation = 0.98  # Adjust this value based on expected proportion of loci with no variants
alpha = 0.5  # Change this value to suit your specific needs
beta = 0.5   # Change this value to suit your specific needs

# Number of loci with no mutation
num_no_mutation = int(proportion_no_mutation * num_loci)

# Generate allele frequencies for loci with mutations
num_with_mutation = num_loci - num_no_mutation
allele_freqs_with_mutation = np.random.beta(alpha, beta, num_with_mutation)

# Create an array of zeros for loci with no mutations
allele_freqs_no_mutation = np.zeros(num_no_mutation)

# Combine the two arrays
allele_freqs = np.concatenate((allele_freqs_no_mutation, allele_freqs_with_mutation))

# Round the values to 2 decimal places
allele_freqs = np.round(allele_freqs, 2)

# Shuffle the array to mix the zero frequencies with the random frequencies
np.random.shuffle(allele_freqs)

num_adaptive = 4  # Number of adaptive loci
# Create the table for the first 8 rows
first_rows = {
    'locus': np.arange(num_adaptive),
    'p': np.repeat(0.5, num_adaptive),  # Fixed value of 0.5
    'dom': np.zeros(num_adaptive),     # Fixed value of 0
    'r': np.array([0, 0.5, 0.5, 0.5]),  # Specified values
    'trait': np.array(['trait_0', 'trait_0', 'trait_0', 'trait_0']),  # Specified traits
    'alpha': np.array([0.25, -0.25, 0.25, -0.25])  # Specified alpha values
}

# Create the table for the remaining rows
remaining_rows = {
    'locus': np.arange(num_adaptive, num_loci),  # Remaining loci indices
    'p': allele_freqs[num_adaptive:],  # Generated allele frequencies
    'dom': np.zeros(num_loci - num_adaptive),  # Default value of 0
    'r': np.repeat(0.5, num_loci - num_adaptive),  # Default value of 0.5
    'trait': np.repeat('NA', num_loci - num_adaptive),  # Default value of 'NA'
    'alpha': np.zeros(num_loci - num_adaptive)  # Default value of 0
}

# Combine first rows with the remaining rows
data = {key: np.concatenate((first_rows[key], remaining_rows[key])) for key in first_rows}

# Create the DataFrame
df = pd.DataFrame(data)

# Print the first few rows to check
print(df.head(20))

# Save to CSV if needed
df.to_csv('genomic_architecture.csv', index=False)
