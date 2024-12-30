import matplotlib.pyplot as plt
import numpy as np

# Number of random points to generate
num_points = 100

# Mean and standard deviation for the normal distribution
mean = 0.5
std_dev = 0.1

# Generate random points with a normal distribution for both sets
x1 = np.random.normal(mean, std_dev, num_points)
y1 = np.random.normal(mean, std_dev, num_points)
x2 = np.random.normal(1.0 - mean, std_dev, num_points)
y2 = np.random.normal(1.0 - mean, std_dev, num_points)

# Create a scatter plot with two colors
plt.scatter(x1, y1, c='red', label='Red Points')
plt.scatter(x2, y2, c='blue', label='Blue Points')

# Add labels and a legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Save the plot as an image file (e.g., PNG)
plt.savefig('random_points_normal_distribution.png')

# Display the plot (optional)
plt.show()
