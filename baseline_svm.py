import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


def load_hog_features(split_path):
    features = []
    labels = []

    for class_name in os.listdir(split_path):
        class_dir = os.path.join(split_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if not fname.endswith(".jpg"):
                continue

            img_path = os.path.join(class_dir, fname)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            feat = hog(gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)
            features.append(feat)
            labels.append(class_name)

    return np.array(features), np.array(labels)


if __name__ == "__main__":
    data_dir = "data/patches_split"

    print("Loading training features...")
    x_train, y_train = load_hog_features(os.path.join(data_dir, "train"))
    print(f"  Train: {x_train.shape}")

    print("Loading val features...")
    x_val, y_val = load_hog_features(os.path.join(data_dir, "val"))
    print(f"  Val: {x_val.shape}")

    print("Loading test features...")
    x_test, y_test = load_hog_features(os.path.join(data_dir, "test"))
    print(f"  Test: {x_test.shape}")

    # reduce feature dimensions with PCA
    print("Reducing dimensions with PCA...")
    pca = PCA(n_components=500, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_val = pca.transform(x_val)
    x_test = pca.transform(x_test)
    print(f"  Reduced to {x_train.shape[1]} features")

    # train SVM with a few C values, pick best on val set
    print("Training SVM...")
    best_acc = 0
    best_svm = None
    best_C = None

    for C in [1, 10, 100]:
        svm = SVC(kernel='rbf', C=C, gamma='scale')
        svm.fit(x_train, y_train)
        val_acc = accuracy_score(y_val, svm.predict(x_val))
        print(f"  C={C}: val accuracy = {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_C = C
            best_svm = svm

    print(f"Best C={best_C}, val accuracy={best_acc:.4f}")

    # evaluate on test set
    y_pred = best_svm.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=best_svm.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=best_svm.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(f"HOG+SVM Baseline (C={best_C}) — Test Accuracy: {acc:.4f}")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/baseline_confusion_matrix.png", dpi=150)
    print("Saved figures/baseline_confusion_matrix.png")
