import numpy as np
import matplotlib.pyplot as plt

def generate_color_palette(data):
    # Ensure the data is a numpy array for consistency
    data = np.array(data)
    
    # Normalize the data to the range [0, 1]
    norm = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Generate the colors using a blue to red gradient
    colors = plt.cm.coolwarm(norm)  # coolwarm colormap goes from blue to red

    # Convert colors to RGBA format
    rgba_colors = [list(color) for color in colors]

    return rgba_colors

# Example dataset ranging from 0.14 to 0.36
data = np.linspace(0.037, 0.097, 1000) 

#data = [0.149688, 0.149688, 0.139147, 0.221147, 0.196078, 0.360551, 0.360551, 0.156444, 0.155087, 0.191896, 0.151407, 0.212357, 0.202458, 0.295107, 0.295107, 0.167582, 0.153218, 0.206617, 0.149239, 0.141862, 0.141552]
# Generate the color palette
color_palette = generate_color_palette(data)

# Visualize the color palette
plt.figure(figsize=(10, 2))
plt.scatter(data, np.ones(len(color_palette)), c=color_palette, s=200)
plt.legend()
plt.title('Color Palette for Visualizing Continuous Data')
plt.show()

# Display the generated colors
color_palette
