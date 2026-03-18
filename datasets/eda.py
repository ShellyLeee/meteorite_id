import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_ROOT = "/mnt/sharedata/ssd_large/users/liyx/kaggle/meteorite-identification"

train_csv = os.path.join(DATA_ROOT, "train_labels.csv")
train_img_dir = os.path.join(DATA_ROOT, "train_images")
test_img_dir = os.path.join(DATA_ROOT, "test_images")

df = pd.read_csv(train_csv)

print("==== Basic Info ====")
print(df.head())
print(df.info())

# -----------------------
# 1. Label distribution
# -----------------------
print("\n==== Label Distribution ====")
print(df["label"].value_counts())

df["label"].value_counts().plot(kind="bar")
plt.title("Label Distribution")
plt.show()

# -----------------------
# 2. Image size stats
# -----------------------
print("\n==== Image Size Stats ====")

sizes = []
for img_name in tqdm(df["id"].values[:500]):  # 先抽样500
    img_path = os.path.join(train_img_dir, img_name)
    try:
        img = Image.open(img_path)
        sizes.append(img.size)  # (width, height)
    except:
        continue

sizes = np.array(sizes)
print("Width mean:", sizes[:, 0].mean())
print("Height mean:", sizes[:, 1].mean())

plt.scatter(sizes[:, 0], sizes[:, 1], alpha=0.3)
plt.title("Image Size Distribution")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

# -----------------------
# 3. Channel / RGB check
# -----------------------
print("\n==== Channel Check ====")

rgb_count = 0
gray_count = 0

for img_name in tqdm(df["id"].values[:200]):
    img_path = os.path.join(train_img_dir, img_name)
    try:
        img = Image.open(img_path)
        if img.mode == "RGB":
            rgb_count += 1
        else:
            gray_count += 1
    except:
        continue

print(f"RGB: {rgb_count}, Non-RGB: {gray_count}")

# -----------------------
# 4. Broken image check
# -----------------------
print("\n==== Broken Image Check ====")

broken = []
for img_name in tqdm(df["id"].values):
    img_path = os.path.join(train_img_dir, img_name)
    try:
        Image.open(img_path).verify()
    except:
        broken.append(img_name)

print("Broken images:", len(broken))

# -----------------------
# 5. Test set size
# -----------------------
print("\n==== Test Set ====")
test_files = os.listdir(test_img_dir)
print("Test size:", len(test_files))