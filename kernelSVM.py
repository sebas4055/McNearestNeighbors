# The skeleton codes for the ELEC378 Final Project to load the dataset and help you get started
# Author: Adilkhan Kairzhan, Pedro Unikovski, Sebastian Molina, Brandon Sung
# Gooood luck!

# Before you start, you should do an Ananconda environment setup. Search it up online, it will make collaborations 
# so much easier. 
# The libraries in your conda environment can be transferred as a .yml file, which is great.
 
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# Modify these paths to match your dataset. 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB    = os.path.join(BASE_DIR, "sample_submission.csv")
OUTPUT_CSV    = os.path.join(BASE_DIR, "submission.csv")
 
RANDOM_STATE = 42

def rbf_kernel(x1, x2, sigma=1.0):
    distance = np.sum((x1 - x2) ** 2)
    return np.exp(-distance / (2 * (sigma ** 2)))
# Loads the metadata from the csv file
def load_metadata():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} entries, {df['TARGET'].nunique()} classes")
    return df

# Loads the images 
def load_image(filename, size, folder=TRAIN_IMG_DIR):
    path = os.path.join(folder, filename)
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img)

# This loads one image and its label of your choice. 
# Input: the dataframe, the index of the imgae, and the size of the image. 
def load_image_label_pair(df, index, size=224):
    row = df.iloc[index]
    img = load_image(row["file_name"], size)
    label = row["TARGET"]
    return img, label

def main():
    print("Loading the ELEC378 Final Project Dataset")

    df = load_metadata()
    print("Loading csv done")

    img, label = load_image_label_pair(df, 0)

    X_train_files, X_val_files, y_train, y_val = train_test_split(
        df["file_name"].values,
        df["TARGET"].values,
        test_size=0.2,
        stratify=df["TARGET"].values,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train_files)}, Validation: {len(X_val_files)}")

    IMG_SIZE = 128

    X_train = []
    for filename in X_train_files:
        img = load_image(filename, IMG_SIZE)
        X_train.append(img.flatten())
    X_train = np.array(X_train)

    print("Extracting validation features...")
    X_val_feats = []
    for filename in X_val_files:
        img_arr = load_image(filename, IMG_SIZE)
        X_val_feats.append(img_arr.flatten())
    X_val_feats = np.array(X_val_feats)

    print(f"X_train shape: {X_train.shape}")      # check the shapes
    print(f"X_val shape:   {X_val_feats.shape}")  

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    scaler = StandardScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val_feats)

    pca = PCA(n_components=250, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
    svm.fit(X_train_pca, y_train_enc)

    val_score = svm.score(X_val_pca, y_val_enc)
    print(f"Validation Accuracy: {val_score:.4f}")
    
    train_acc = svm.score(X_train_pca, y_train_enc)
    val_acc = svm.score(X_val_pca, y_val_enc)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val   accuracy: {val_acc:.4f}")

    print("Loading test images and predicting...")
    sub_df = pd.read_csv(SAMPLE_SUB)

    X_test_list = []
    for id_val in sub_df["ID"]:
        filename = f"{id_val}.jpg" 
        img_arr = load_image(filename, IMG_SIZE, folder=TEST_IMG_DIR)
        X_test_list.append(img_arr.flatten())

    X_test = np.array(X_test_list)

    X_test_scaled = scaler.transform(X_test)
    X_test_pca    = pca.transform(X_test_scaled)

    y_pred_enc    = svm.predict(X_test_pca)
    y_pred_labels = le.inverse_transform(y_pred_enc)

    submission = pd.DataFrame({
        "ID": sub_df["ID"],
        "TARGET": y_pred_labels
    })

    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! {OUTPUT_CSV} created.")

if __name__ == "__main__":
    main()
