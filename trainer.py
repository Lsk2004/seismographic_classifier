import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from data_loader import SeismoDataset
from model import build_classifier

import matplotlib.pyplot as plt

def train(
    csv_path, hdf5_path,
    epochs=10,
    batch_size=32,
    threshold=0.3
):
    # Load CSV and get indices for both classes
    df = pd.read_csv(csv_path)
    earthquake_idx = df[df['source_type'] == 'earthquake'].index.tolist()
    noise_idx = df[df['source_type'] == 'noise'].index.tolist()
    np.random.shuffle(earthquake_idx)
    np.random.shuffle(noise_idx)
    num_noise = len(noise_idx)
    # STRICT 1:1 ratio: use min(len(earthquake_idx), len(noise_idx))
    n_samples = min(len(noise_idx), len(earthquake_idx))
    selected_eq_idx = earthquake_idx[:n_samples]
    selected_noise_idx = noise_idx[:n_samples]
    final_indices = selected_noise_idx + selected_eq_idx
    np.random.shuffle(final_indices)

    # Dataset and scaler
    ds = SeismoDataset(csv_path, hdf5_path)
    ds.fit_scaler()

    # Stratified train/test split
    labels = df.loc[final_indices, 'source_type'].map({'noise': 0, 'earthquake': 1}).values
    train_idx, test_idx = train_test_split(
        final_indices, test_size=0.2, random_state=42, stratify=labels
    )

    X_train = np.array([ds[i][0] for i in train_idx])
    y_train = np.array([ds[i][1] for i in train_idx])
    X_test = np.array([ds[i][0] for i in test_idx])
    y_test = np.array([ds[i][1] for i in test_idx])

    # Compute class weights
    class_weights_array = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
    class_weight_dict = {0: class_weights_array[0], 1: class_weights_array[1]}
    print(f"Using class weights: {class_weight_dict}")

    # Build model
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

    # Custom threshold for classification
    pred_probs = model.predict(X_test).flatten()
    preds = (pred_probs > threshold).astype(int)

    print(f"Threshold for 'earthquake' classification: {threshold}")
    print(classification_report(y_test, preds, target_names=['noise', 'earthquake']))

    # Plot histogram of model output probabilities for error analysis
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

    model.save('seismo_classifier_balanced.h5')

if __name__ == "__main__":
    train(
        r"C:\Users\laksh\Downloads\metadata\combined_shuffled_dataset.csv",
        r"C:\Users\laksh\Downloads\data\combined_data.hdf5",
        threshold=0.3
    )
