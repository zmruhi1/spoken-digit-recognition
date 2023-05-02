import torchaudio 
from torch.utils.data import Dataset




class AudioDataset(Dataset):
  """
  Creates a Pytorch dataset for handling the data and feeding to the model
  """
  def __init__(self, files, labels, n_mfcc=80, n_fft=200, hop_length=80, sr=8000, transforms=[]):
    """
    param files: list of audio file locations
    param labels: corresponding labels of the audio files
    param n_mfcc: number of frequency bins
    param n_fft: length of the FFT window
    param hop_length: no. of samples between successive frames
    param sr: the sampling rate
    transforms: list of torchaudio's transformation function to apply to the raw waveform

    returns: tuple(mel spectrogram, length of mel spectrogram, label)
    """
    self.waveform_files = files
    self.labels = labels
    self.sr = sr
    self.mel_transform = torchaudio.transforms.MFCC(
        sample_rate=self.sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mfcc,
            "hop_length": hop_length,
            "mel_scale": "htk"
        })
    self.transforms = []

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    wv, sr = torchaudio.load(self.waveform_files[idx])

    # transform to mel spectrogram
    wv = self.mel_transform(wv)

    # apply any other transformations provided
    for t in self.transforms:
      wv = t(wv)
    wv = wv.permute(0, 2, 1)
    return (wv, wv.shape[1], self.labels[idx])