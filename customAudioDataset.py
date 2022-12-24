import os
import pandas as pd
import torch
import torchaudio
import random


class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, audio_dir, transform=None, tensor_cut=0, fixed_length=None):
        self.audio_labels = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.fixed_length = fixed_length
        self.tensor_cut = tensor_cut

    def __len__(self):
        if self.fixed_length:
            return self.fixed_length
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 10])
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform(waveform)

        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1]-self.tensor_cut-1)
                waveform = waveform[:, start:start+self.tensor_cut]
        return waveform, sample_rate

