"""utils.py"""

import os
import numpy as np
import argparse


import numpy as np

import tqdm

import torch
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd

from sklearn  import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay

# from dataset import AudioDataset


def play_sample(args):

  # read tsv file into a dataframe 
  sdr_df = pd.read_csv(args.data_dir + 'SDR_metadata.tsv', sep='\t', header=0, index_col='Unnamed: 0')
  sdr_df.file = sdr_df.file.apply(lambda x: args.data_dir + x)
  sdr_df.loc[sdr_df['identifier'] == '7_theo_0']
  sample_wav_file = sdr_df.loc[sdr_df['identifier'] == '7_theo_0'].file[700]
  # play and listen to a sample 
  SAMPLING_RATE = 8000 # This value is determined by the wav file, DO NOT CHANGE

  x, sr = librosa.load(sample_wav_file, sr=SAMPLING_RATE) #, 
  ipd.Audio(x, rate=sr)

  fig, ax = plt.subplots(figsize=(10, 2), sharex=True)

  img = librosa.display.waveshow(y=x, sr=sr, alpha=0.75, x_axis='time', color='blue')

  ax.set(title='Amplitude waveform')
  ax.set_ylabel('Amplitude')
  ax.label_outer()

  melspectrogram = extract_melspectrogram(x, sr, num_mels=13)
  fig, ax = plt.subplots(figsize=(10, 2), sharex=True)

  img = librosa.display.specshow(
      melspectrogram, 
      sr=sr, 
      x_axis='time', 
      y_axis='mel', 
      cmap='viridis', 
      fmax=4000, 
      hop_length=80
  )

  ax.set(title='Log-frequency power spectrogram')

  ax.label_outer()






'   '   '   '   '   Spectrogram    '   '   '   '  '


def extract_melspectrogram(signal, sr, num_mels):
    """
    Given a time series speech signal (.wav), sampling rate (sr), 
    and the number of mel coefficients, return a mel-scaled 
    representation of the signal as numpy array.
    """
    
    mel_features = librosa.feature.melspectrogram(y=signal,
        sr=sr,
        n_fft=200, # with sampling rate = 8000, this corresponds to 25 ms
        hop_length=80, # with sampling rate = 8000, this corresponds to 10 ms
        n_mels=num_mels, # number of frequency bins, use either 13 or 39
        fmin=50, # min frequency threshold
        fmax=4000 # max frequency threshold, set to SAMPLING_RATE/2
    )
    
    # for numerical stability added this line
    mel_features = np.where(mel_features == 0, np.finfo(float).eps, mel_features)

    # 20 * log10 to convert to log scale
    log_mel_features = 20*np.log10(mel_features)

    # feature scaling
    scaled_log_mel_features = preprocessing.scale(log_mel_features, axis=1)
    
    return scaled_log_mel_features



def downsample_spectrogram(X, N):
    """
    Given a spectrogram of an arbitrary length/duration (X ∈ K x T), 
    return a downsampled version of the spectrogram v ∈ K * N
    """
    # Split the spectrogram into N equal parts
    v = np.split(X, indices_or_sections=N, axis=1)
    v = np.array(v)

    # Take the average
    v = np.mean(v, axis=2)

    # Flatten the matrix into an array
    v = v.flatten()
    return v


def preprocess_extract_and_downsample_mfcc(files, sr=8000, num_mels=13, N=25):
  """
  Extract the MFCC from the raw waveform, then downsample

  returns: downsampled spectrogram of shape (len(files), K, N)
  """
  output = np.zeros((len(files), num_mels*N))
  for idx, f in enumerate((files)):
    x, sr = librosa.load(f, sr=sr)
    x_mfcc = extract_melspectrogram(x, sr=sr, num_mels=num_mels)

    # pad with zeros at end to make equal division possible
    x_mfcc = np.pad(x_mfcc, pad_width=((0,0), (0, N - x_mfcc.shape[1] % N)))
    
    mfcc_downsampled = downsample_spectrogram(x_mfcc, N=N)
    output[idx, :] = mfcc_downsampled
  return output



def plot_scores(args, clf, dev_p, dev_r, dev_f1, test_p, test_r, test_f1, dev_labels, dev_preds, test_labels, test_preds):

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))

        dataset_labels = ["dev", "test"]
        evaluation_labels = ["precision", "recall", "f1"]
        stacked_precision = np.stack([dev_p, test_p])
        stacked_recall = np.stack([dev_r, test_r])
        stacked_f1 = np.stack([dev_f1, test_f1])
        stacked_evaluations = [stacked_precision, stacked_recall, stacked_f1]
        for idx, ax in enumerate(axes):
            im = ax.imshow(stacked_evaluations[idx])
            ax.set_title(evaluation_labels[idx])
            ax.set_yticks(np.arange(len(dataset_labels)), dataset_labels)
            ax.set_xticks(np.arange(10), range(10))

        for i in range(stacked_evaluations[idx].shape[0]):
            for j in range(stacked_evaluations[idx].shape[1]):
                value = np.round(stacked_evaluations[idx][i][j], 2)
                text = ax.text(j, i, value, ha="center", va="center", color="w" if value < 0.7 else "black")
        fig.suptitle("Precision/Recall/F1 for each label for dev/test splits of "+args.model+" model")
        plt.savefig(args.out_dir+'/{}_scores.png'.format(args.model))

        # analyze the confusion matrix of the baseline 
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

        ConfusionMatrixDisplay.from_predictions(dev_labels, dev_preds, labels=clf.classes_, ax=axes[0])
        axes[0].set_title("Dev")

        ConfusionMatrixDisplay.from_predictions(test_labels, test_preds, labels=clf.classes_, ax=axes[1])
        axes[1].set_title("Test")

        fig.suptitle("Confusion matrix for "+args.model+" model")
        plt.savefig(args.out_dir+'/{}_confusion_matrix.png'.format(args.model))
        print("Plots saved")



'   '   '   '   '   Misc    '   '   '   '  '


def save_log(args, log, name):
    file_name = name + '_log_' + args.sys_name + '.txt'
    save_dir = os.path.join(args.output_dir, file_name)

    epoch = log['epoch']
    loss_z = log['loss_z']
    # loss_deg_E = log['loss_deg_E']
    # loss_deg_S = log['loss_deg_S']

    f = open(save_dir, "w")
    f.write('epoch loss_z\n')
    for idx in range(len(epoch)):
        f.write(str(loss_z[idx]) + " \n")
        # f.write(str(loss_deg_E[idx]) + " " + str(loss_deg_S[idx]) + "\n") 
    f.close()

def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



class EarlyStopping:
    """
    This implementation was adapted from: https://github.com/Bjarten/early-stopping-pytorch
    The original implementation stored and compared validation loss and this implementation
    uses the validation accuracy for early stopping
    
    Early stops the training if validation accuracy doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="../output/checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation accuracy improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_acc_min:.6f} --> {val_acc:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_acc_min = val_acc



  
