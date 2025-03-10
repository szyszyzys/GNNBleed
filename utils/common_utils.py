import numpy as np

# Sample distance data
data = np.array([3, 6, 9, 12, 15])


# Min-Max Normalization
def min_max_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Z-Score Normalization
def z_score_normalize(data):
    return (data - np.mean(data)) / np.std(data)


# Decimal Scaling
def decimal_scaling_normalize(data):
    max_abs_value = np.max(np.abs(data))
    num_decimal_places = np.ceil(np.log10(max_abs_value))
    return data / (10 ** num_decimal_places)


# Clipping (set a threshold for clipping)
def clip_and_normalize(data, threshold=10):
    clipped_data = np.clip(data, None, threshold)
    return min_max_normalize(clipped_data)


# Normalization examples
min_max_normalized = min_max_normalize(data)
z_score_normalized = z_score_normalize(data)
decimal_scaling_normalized = decimal_scaling_normalize(data)
clipped_normalized = clip_and_normalize(data, threshold=10)

min_max_normalized, z_score_normalized, decimal_scaling_normalized, clipped_normalized
