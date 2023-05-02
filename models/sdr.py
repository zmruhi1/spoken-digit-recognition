import random
import pickle

from collections import defaultdict, Counter

# add this to ignore warnings from Librosa
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
import seaborn as sns
# for preprocessing
from sklearn  import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# dimensionality reduction
from sklearn.manifold import TSNE

# linear models
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

# evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# for NN models
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange

# for handling data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# for preprocessing audio
import torchaudio


from utilities.utils import preprocess_extract_and_downsample_mfcc, plot_scores
from utilities.dataset import AudioDataset 
from utilities.model_utils import train_model, evaluate
from models.rnn import RNNClassifier
from models.conformer import ConformerClassifier





def train(args):
     
    # prepare data and split

    # read tsv file into a dataframe 
    sdr_df = pd.read_csv(args.data_dir + 'SDR_metadata.tsv', sep='\t', header=0, index_col='Unnamed: 0')
    sdr_df.file = sdr_df.file.apply(lambda x: args.data_dir + x)

    # These only contain the file location string
    # the mel spectrograms will be extracted next
    train_files = sdr_df.file[sdr_df.split == "TRAIN"].tolist()
    dev_files = sdr_df.file[sdr_df.split == "DEV"].tolist()
    test_files = sdr_df.file[sdr_df.split == "TEST"].tolist()

    # These contain the labels for train/dev/test set
    train_labels = sdr_df.label[sdr_df.split == "TRAIN"].tolist()
    dev_labels = sdr_df.label[sdr_df.split == "DEV"].tolist()
    test_labels = sdr_df.label[sdr_df.split == "TEST"].tolist()

    # Extract the MFCC and downsample it using the previous function
    train_set = preprocess_extract_and_downsample_mfcc(train_files)
    dev_set = preprocess_extract_and_downsample_mfcc(dev_files)
    test_set = preprocess_extract_and_downsample_mfcc(test_files)

    # Create dataset instances for train/dev/test set
    train_ds = AudioDataset(train_files, train_labels)
    dev_ds = AudioDataset(dev_files, dev_labels)
    test_ds = AudioDataset(test_files, test_labels)

    # Create dataloader instances that can be directly fed into the model
    # We are using a batch_size of 1, which would mean we'll be using 
    # stochastic gradient descent instead of minibactch SGD

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=1, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)



    # model training

    if args.model=='linear':
        predictions = {}

        # train a linear model 
        # We are using SGDClassifier with hinge loss, which is a SVM model
        clf = make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001, learning_rate="optimal", random_state=42))
        clf.fit(train_set, train_labels)

        # evaluate the model using accuracy metric
        train_preds = clf.predict(train_set)
        dev_preds = clf.predict(dev_set)
        test_preds = clf.predict(test_set)

        # Store the predictions
        predictions["svm"] = {"train": train_preds, "dev": dev_preds, "test": test_preds}

        print("Train accuracy: {:.2f}".format(accuracy_score(train_labels, train_preds)))
        print("Dev accuracy: {:.2f}".format(accuracy_score(dev_labels, dev_preds)))
        print("Test acc: {:.2f}\n".format(accuracy_score(test_labels, test_preds)))

        # report precision, recall, F1-score for each label 
        dev_p, dev_r, dev_f1, dev_support = precision_recall_fscore_support(dev_labels, dev_preds, average=None)
        test_p, test_r, test_f1, test_support = precision_recall_fscore_support(test_labels, test_preds, average=None)

        plot_scores(args, clf, dev_p, dev_r, dev_f1, test_p, test_r, test_f1, dev_labels, dev_preds, test_labels, test_preds)

    
    if args.model=='rnn':


        # Initialize the RNN model
        # In this case, the embedding size is 80, the hidden size is 256
        # there are 10 classes to predict and the dropout probabilty is 0.1
        rnn_model = RNNClassifier(embeds_dim=80, hidden_dim=256, n_classes=10, dropout=0.1) 

        num_epochs = args.epoch
        lr = args.lr

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)

        # train the model
        losses, val_acc = train_model(
            model=rnn_model, 
            train_dl=train_dl, 
            val_dl=dev_dl, 
            criterion=criterion, 
            optimizer=optimizer, 
            epochs=num_epochs, 
            checkpoint_name="models/outputs/rnn_lstm_full.pt",
            rnn=True, # for rnn
            )
        # plot the training loss
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
        axes[0].plot(losses)
        axes[0].set_title("Training loss")
        axes[0].set_ylabel("Cross Entropy Loss")
        axes[0].set_xlabel("Epoch")

        # plot the validation accuracy
        axes[1].plot(val_acc)
        axes[1].set_title("Validation accuracy")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xlabel("Epoch")

        # Load and save the best model from the checkpoint saved by early stopping
        rnn_model.load_state_dict(torch.load('rnn_lstm_full.pt'))

        # Evaluate on Dev set
        dev_preds, dev_labs = evaluate(rnn_model, dev_dl, rnn=True)
        dev_preds = np.array(dev_preds)
        dev_labs = np.array(dev_labs)

        # Evaluate on test set
        test_preds, test_labs = evaluate(rnn_model, test_dl, rnn=True)
        test_preds = np.array(test_preds)
        test_labs = np.array(test_labs)

        accuracy = accuracy_score(test_labs, test_preds)
        print("Test Accuracy (RNN): ", accuracy)

        predictions["rnn"] = {"dev": dev_preds, "test": test_preds}

        dev_p, dev_r, dev_f1, dev_support = precision_recall_fscore_support(dev_labs, dev_preds, average=None)
        test_p, test_r, test_f1, test_support = precision_recall_fscore_support(test_labs, test_preds, average=None)
        
        plot_scores(args, clf, dev_p, dev_r, dev_f1, test_p, test_r, test_f1, dev_labels, dev_preds, test_labels, test_preds)

    if args.model=='conformer':
        transformer_model = ConformerClassifier(
        input_dim=80, 
        n_classes=10, 
        embeds_dim=256, 
        n_layers=2, 
        conv_expansion_factor=2, 
        linear_expansion_factor=2, 
        n_heads=4, 
        attn_head_dim=64, 
        dropout=0.2, 
        padding="same"
        )

        num_epochs = args.epoch
        lr = args.lr

        criterion = nn.NLLLoss()

        optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr)

        losses, val_acc = train_model(
            model=transformer_model, 
            train_dl=train_dl, 
            val_dl=dev_dl, 
            criterion=criterion, 
            optimizer=optimizer, 
            epochs=num_epochs,
            checkpoint_name="models/outputs/conformer_full.pt"
            )

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
        axes[0].plot(losses)
        axes[0].set_title("Training loss")
        axes[0].set_ylabel("Cross Entropy Loss")
        axes[0].set_xlabel("Steps")

        axes[1].plot(val_acc)
        axes[1].set_title("Validation accuracy")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xlabel("Epoch")

        # Load the best model from the checkpoint saved by early stopping
        transformer_model.load_state_dict(torch.load('conformer_full.pt'))

        dev_preds, dev_labs = evaluate(transformer_model, dev_dl)
        dev_preds = np.array(dev_preds)
        dev_labs = np.array(dev_labs)

        test_preds, test_labs = evaluate(transformer_model, test_dl)
        test_preds = np.array(test_preds)
        test_labs = np.array(test_labs)
        print("Test accuracy (Conformer): ", accuracy_score(test_labs, test_preds))
        predictions["conformer"] = {"dev": dev_preds, "test": test_preds}

        dev_p, dev_r, dev_f1, dev_support = precision_recall_fscore_support(dev_labs, dev_preds, average=None)
        test_p, test_r, test_f1, test_support = precision_recall_fscore_support(test_labs, test_preds, average=None)

        plot_scores(args, clf, dev_p, dev_r, dev_f1, test_p, test_r, test_f1, dev_labels, dev_preds, test_labels, test_preds)
