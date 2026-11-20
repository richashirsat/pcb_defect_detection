import os, cv2, random
import matplotlib.pyplot as plt

data_dir = "data/patches_split/test"
classes = sorted(os.listdir(data_dir))

fig, axes = plt.subplots(2, 7, figsize=(16, 5))
for col, cls in enumerate(classes):
    cls_dir = os.path.join(data_dir, cls)
    imgs = [f for f in os.listdir(cls_dir) if f.endswith(".jpg")]
    for row in range(2):
        img = cv2.imread(os.path.join(cls_dir, random.choice(imgs)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[row][col].imshow(img)
        axes[row][col].axis("off")
        if row == 0:
            axes[row][col].set_title(cls.replace("_", "\n"), fontsize=8)

plt.suptitle("Sample Patches per Class", fontsize=14)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/sample_patches.png", dpi=150)
print("Saved figures/sample_patches.png")
