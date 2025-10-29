import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler

class SeismoDataset:
    def __init__(self, csv_path, hdf5_path, scaler=None):
        self.df = pd.read_csv(csv_path)
        
        # Encode labels: earthquake = 1, noise = 0
        self.df['label'] = self.df['source_type'].map({'earthquake': 1, 'noise': 0})
        self.labels = self.df['label'].values

        self.hdf5_path = hdf5_path
        self.scaler = scaler
        
        # Open the HDF5 file once in __getitem__, not in __init__

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            waveform = f['data'][idx].astype(np.float32)  # shape: (3, 12000)
            # Normalize channel-wise if scaler is provided
            if self.scaler:
                for i in range(3):
                    waveform[i] = self.scaler.transform(waveform[i].reshape(1, -1)).flatten()
        label = self.labels[idx]
        return waveform, label

    def fit_scaler(self, sample_size=1000):
        # Fit scaler using a random sample to avoid loading all data
        idxs = np.random.choice(len(self.df), min(sample_size, len(self.df)), replace=False)
        channel_data = []
        with h5py.File(self.hdf5_path, 'r') as f:
            for idx in idxs:
                channel_data.extend(f['data'][idx])
        all_channels = np.vstack(channel_data)
        self.scaler = StandardScaler().fit(all_channels)
