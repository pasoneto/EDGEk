import json
import os
import re
import matplotlib.pyplot as plt

# Directory containing your files
directory = "/Users/pdealcan/Downloads/loss/"

# Dictionaries to store data, with file numbers as keys
train_losses = {}
v_losses = {}
fk_losses = {}
#foot_losses = {}

# Read each file in the directory and store data
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # Check for .txt files
        # Extract the number after "redo-" in the filename
        match = re.search(r'redo-(\d+)', filename)
        if match:
            file_number = int(match.group(1))  # Get the number after "redo-"
            filepath = os.path.join(directory, filename)
            
            with open(filepath, "r") as file:
                data = json.load(file)
                train_losses[file_number] = data["Train Loss"]
                v_losses[file_number] = data["V Loss"]
                fk_losses[file_number] = data["FK Loss"]
#                foot_losses[file_number] = data["Foot Loss"]

# Sort the losses by file numbers
sorted_numbers = sorted(train_losses.keys())
train_losses_sorted = [train_losses[num] for num in sorted_numbers]
v_losses_sorted = [v_losses[num] for num in sorted_numbers]
fk_losses_sorted = [fk_losses[num] for num in sorted_numbers]
#foot_losses_sorted = [foot_losses[num] for num in sorted_numbers]

# Plot each loss over time in separate subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Train Loss
axs[0].plot(sorted_numbers, train_losses_sorted, label="Train Loss", color="b")
axs[0].set_ylabel("Train Loss")
axs[0].legend()

# V Loss
axs[1].plot(sorted_numbers, v_losses_sorted, label="V Loss", color="g")
axs[1].set_ylabel("V Loss")
axs[1].legend()

# FK Loss
axs[2].plot(sorted_numbers, fk_losses_sorted, label="FK Loss", color="r")
axs[2].set_ylabel("FK Loss")
axs[2].legend()

for ax in axs:
    ax.set_ylim(0, 0.02)
    ax.legend()

plt.suptitle("Losses over File Number")
plt.show()
