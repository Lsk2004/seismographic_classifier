import numpy as np
import pandas as pd
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from data_loader import SeismoDataset
from model import build_classifier
import matplotlib.pyplot as plt

def augment_noise_sample(sample, num_augments=3, noise_std=0.05):
    augmented = []
    for _ in range(num_augments):
        noise = np.random.normal(0, noise_std, sample.shape)
        augmented.append(sample + noise)
    return augmented

def train(
    csv_path, hdf5_path,
    epochs=10,
    batch_size=32,
    threshold=0.5   
):
    
    df = pd.read_csv(csv_path)
    earthquake_idx = df[df['source_type'] == 'earthquake'].index.tolist()
    noise_idx = df[df['source_type'] == 'noise'].index.tolist()
    np.random.shuffle(earthquake_idx)
    np.random.shuffle(noise_idx)
    n_samples = min(len(noise_idx), len(earthquake_idx))
    selected_eq_idx = earthquake_idx[:n_samples]
    selected_noise_idx = noise_idx[:n_samples]
    final_indices = selected_noise_idx + selected_eq_idx
    np.random.shuffle(final_indices)

    ds = SeismoDataset(csv_path, hdf5_path)
    ds.fit_scaler()

    X = np.array([ds[i][0] for i in final_indices])
    y = np.array([ds[i][1] for i in final_indices])

    augmented_X = []
    augmented_y = []
    for xi, yi in zip(X, y):
        augmented_X.append(xi)
        augmented_y.append(yi)
        if yi == 0:  # 0 = noise
            augmented_X.extend(augment_noise_sample(xi, num_augments=3, noise_std=0.05))
            augmented_y.extend([0] * 3)
    X = np.array(augmented_X)
    y = np.array(augmented_y)

    original_shape = X.shape
    X_flat = X.reshape(X.shape[0], -1)

    smote = SMOTE(random_state=42)
    X_smote_flat, y_smote = smote.fit_resample(X_flat, y)
    X_smote = X_smote_flat.reshape(-1, original_shape[1], original_shape[2])

    train_idx, test_idx = train_test_split(
        np.arange(len(y_smote)), test_size=0.2, random_state=42, stratify=y_smote
    )
    X_train, X_test = X_smote[train_idx], X_smote[test_idx]
    y_train, y_test = y_smote[train_idx], y_smote[test_idx]

    class_weights_array = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
    class_weight_dict = {0: class_weights_array[0], 1: class_weights_array[1]}
    print(f"Using class weights: {class_weight_dict}")

    model = build_classifier(X_train[0].shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        verbose=2
    )

    pred_probs = model.predict(X_test).flatten()
    preds = (pred_probs > threshold).astype(int)
    print(f"Threshold for 'earthquake' classification: {threshold}")
    print(classification_report(y_test, preds, target_names=['noise', 'earthquake']))

    noise_probs = pred_probs[y_test == 0]
    eq_probs = pred_probs[y_test == 1]
    plt.figure(figsize=(10,6))
    plt.hist(noise_probs, bins=50, alpha=0.6, label='Noise')
    plt.hist(eq_probs, bins=50, alpha=0.6, label='Earthquake')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.title('Model Output Probabilities - Noise vs Earthquake')
    plt.xlabel('Predicted Probability of Earthquake')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    model.save('seismo_classifier_balanced_augmented.h5')

if __name__ == "__main__":
    train(
        r"C:\Users\laksh\Downloads\metadata\combined_shuffled_dataset.csv",
        r"C:\Users\laksh\Downloads\data\combined_data.hdf5",
        threshold=0.5  
    )
