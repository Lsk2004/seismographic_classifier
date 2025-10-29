# Place for any extra helpers, e.g., plotting, metrics
import matplotlib.pyplot as plt

def plot_waveform(waveform, label=None):
    import numpy as np
    for i, ch in enumerate(['E', 'N', 'Z']):
        plt.plot(waveform[i], label=ch)
    if label is not None:
        plt.title(f'Label: {label}')
    plt.legend()
    plt.show()
