import numpy as np
import glob
import shutil
import os

if __name__ == "__main__":
    np.random.seed(3699)
    files = glob.glob("data/train_gt/*")
    files.sort()

    val_indices = np.arange(len(files))
    np.random.shuffle(val_indices)
    np.save("train_set.npy", val_indices[100:])
    np.save("val_set.npy", val_indices[:100])

    os.makedirs("data/val_gt", exist_ok=True)
    os.makedirs("data/val_input", exist_ok=True)
    os.makedirs("data/val_mask", exist_ok=True)
    os.makedirs("data/val_gray", exist_ok=True)
    for i in val_indices[:100]:
        shutil.move(files[i], "data/val_gt")
        shutil.move(files[i].replace("gt", "input"), "data/val_input")
        shutil.move(files[i].replace("gt", "mask"), "data/val_mask")
        shutil.move(files[i].replace("gt", "gray"), "data/val_gray")
    print("Validation set is created.")
