import os
import sys

import itertools
import wave
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy import signal
from scipy.io import wavfile

import matplotlib.pyplot as plt

class SpectogramDataset(Dataset):

    def __init__(self, data):
        self.data = data[:, 0]
        self.labels = data[:, 1]

        # whole dataset
        self.dataset = data

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.dataset.shape[0]

def get_spectrogram_data(train_fraction):

    root_dir = os.path.dirname(os.path.realpath(__file__))+'/data'
    tmp_dataset = []
    label_list = []

    root_dir = os.walk(root_dir)
    _, label_list, _ = next(root_dir) # step into first layer, and extract subdir names

    for label_id, (dir, _, files) in enumerate(root_dir):
        for filenum, filename in enumerate(files):
            try:
                sample_rate, samples = wavfile.read(os.path.join(dir, filename))
                frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

                spectrogram[spectrogram == 0] = 1
                spectrogram = 10 * np.log10(spectrogram)
                spectrogram = np.expand_dims(spectrogram, 0)

                if len(times) == 71:
                    tmp_dataset.append([spectrogram, label_id])

            except:
                print("Having trouble reading the file")

    tmp_dataset = np.vstack(tmp_dataset)

    ### shuffle the data
    np.random.shuffle(tmp_dataset)

    print(tmp_dataset[:,0][0].shape)

    training_dataset = SpectogramDataset(tmp_dataset[0:int(train_fraction * len(tmp_dataset))])
    validation_dataset = SpectogramDataset(tmp_dataset[int(train_fraction * len(tmp_dataset)):])

    return training_dataset, validation_dataset, label_list

def get_spectrogram_data_loaders(train_fraction, batch_size):

    training_dataset, validation_dataset, label_list = get_spectrogram_data(train_fraction)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=64)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=64)

    return training_dataloader, validation_dataloader, label_list

def get_filename(label_id, filenum):

    root_dir = 'data/'
    root_dir = os.walk(root_dir)
    _, label_list, _ = next(root_dir) # step into first layer, and extract subdir names

    _, _, files = next(itertools.islice(root_dir, label_id, None))

    return os.path.join(label_list[label_id], files[filenum])

def plot_spectrogram(filepath):

    sample_rate, samples = wavfile.read(filepath)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    x_start, x_end = (times.min(), times.max())
    y_start, y_end = (frequencies.min(), frequencies.max())
    extent = [x_start, x_end, y_start, y_end]

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram, origin='lower', extent=extent, aspect='auto')
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()
