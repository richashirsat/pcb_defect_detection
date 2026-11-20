import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = "data/patches_split"
splits = ["train", "val", "test"]
classes = sorted(os.listdir(os.path.join(data_dir, "train")))

counts = {}
for split in splits:
    counts[split] = [len(os.listdir(os.path.join(data_dir, split, c))) for c in classes]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, counts["train"], width, label="Train")
ax.bar(x, counts["val"], width, label="Val")
ax.bar(x + width, counts["test"], width, label="Test")

ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_ylabel("Number of Patches")
ax.set_title("Class Distribution Across Splits")
ax.legend()
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=150)
print("Saved figures/class_distribution.png")
