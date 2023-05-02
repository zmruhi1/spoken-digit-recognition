
import numpy as np
from sklearn.metrics import accuracy_score

from tqdm import tqdm
import pickle 

import torch.functional as F
import torch
from torch.utils.data import DataLoader

from utilities.utils import EarlyStopping
from utilities.dataset import AudioDataset


def train_model(model, train_dl, val_dl, criterion, optimizer, epochs=5, early_stop=True, patience=5, checkpoint_name="checkpoint.pt", rnn=False):
    """
    Train the model on the given data

    param model: the instance of Pytorch NN model
    param train_dl: the training dataloader to be trained on
    param val_dl: the validation dataloader for evaluating (in training and in early stopping)
    param criterion: the loss function
    param optimizer: the optimizer function
    param epochs: the number of epochs to train the model for
    param early_stop: If the early stopping should be used or not
    param patience: (for early stopping) the number of times to wait before stopping training
    param checkpoint_name: (for early stopping) the name of the model checkpoint
    """

    print("Training started...")

    losses_to_plot = []
    validation_accuracy = []

    # initialize the earlystopping module
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_name)


    for epoch in range(epochs):
        epoch_loss = []
        for i, (inp_wave, inp_len, label) in enumerate(pbar := tqdm(train_dl)):  
          
            inp_wave = inp_wave.squeeze(0) 
            inp_len = inp_len 
            label = label 
            
            # Forward pass
            if rnn:
              # if model is rnn then send empty hidden & context values
              outputs = model(inp_wave, None, None)
            else:
              # for conformer, send input wave and input length
              outputs = model(inp_wave, inp_len)

            loss = criterion(outputs, label)

            # Save the step loss for plotting
            losses_to_plot.append(loss.item())

            # Save the step loss for calculating mean error in epoch
            epoch_loss.append(loss.item())
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        epoch_loss_mean = sum(epoch_loss)/len(epoch_loss)
        # add loss value to the tqdm progress bar
        pbar.set_postfix_str(f"Loss: {epoch_loss_mean}")

        # Evaluate on validation set
        preds, labs = evaluate(model, val_dl, rnn=rnn)
        accuracy = accuracy_score(preds, labs)
        validation_accuracy.append(accuracy)

        if early_stop:
          # if early stopping is enabled, store the validation scores
          early_stopping(accuracy, model)

          if early_stopping.early_stop:
            # if the patience has been crossed, the training is stopped and
            # the checkpoint file contains the best model according to the
            # validation accuracy
            print("Stopping early at epoch [{}/{}]".format(epoch+1, epochs))
            return losses_to_plot, validation_accuracy

        print("Epoch [{}/{}] Loss: {:.4f} Val. Acc: {:.4f}".format(epoch+1, epochs, epoch_loss_mean, accuracy))
          
    return losses_to_plot, validation_accuracy


def evaluate(model, dataloader, disable_progress=False, rnn=False):
    """
    Evaluate the model on the given dataloader

    param model: the instance of Pytorch NN model
    param dataloader: the instance of Pytorch dataloader (usually test/val dataloader)
    disable_progress: disables the tqdm progress bar

    returns: tuple(list of predictions, list of corresponding true labels)
    """
    predictions = []
    labels = []
    model.eval()
    for (inp_wave, inp_len, label) in tqdm(dataloader, position=0, leave=True, disable=disable_progress):
          
        # inp_wave = inp_wave.squeeze(0) 
        # inp_len = inp_len 
        # label = label 
        
        # Forward pass
        if rnn:
          # if model is rnn then send empty hidden & context values
          out = model(inp_wave, None, None)
        else:
          # for conformer, send input wave and input length
          out = model(inp_wave, inp_len)

        # Get correct softmax probs.
        # from the log softmax output
        out = torch.exp(out)

        out = torch.argmax(out, dim=1)

        out = list(out.cpu().numpy())
        labels_batch = list(label.cpu().numpy())
        
        predictions.extend(out)
        labels.extend(labels_batch)

    return predictions, labels




'   '   '   '   '   Bootstrap Evaluation    '   '   '   '  '
  
  
def calculate_bootstrap_indices(n):
    """
    Generates an array of indices randomly to be used in sampling from the dataset

    param n: the length of the dataset
    """
    indices = np.random.randint(0, n-1, n)
    return indices
    
def run_bootstrap(model, data, labels, delta_x, baseline_acc, b_iters=1e4):
    """
    Run the bootstrap approach for measuring the statistical significance

    param model: the model to evaluate (Pytorch model instance)
    param data: the original test dataset
    param labels: the corresponding labels of the test set
    param delta_x: the improvement made by the new model over the baseline (new_acc - base_acc)
    param baseline_acc: the baseline accuracy for comparing with the bootstrap accuracy
    param b_iters: the number of bootstrap iterations to run
    """
    dataset_size = len(data)

    # standard error
    s = 0.0
    acc_sum = 0.0

    if not isinstance(data, np.ndarray):
        data = np.array(data)
        labels = np.array(labels)

    for i in tqdm(range(b_iters), position=0, leave=True):
        # calculate the bootstrap indices
        indices = calculate_bootstrap_indices(dataset_size)
        
        # sample based on the indices
        bootstrap_dataset = data[indices]
        bootstrap_labels = labels[indices]

        # create a bootstrap dataset
        bootstrap_dataset = AudioDataset(bootstrap_dataset, bootstrap_labels)
        bootstrap_dl = DataLoader(bootstrap_dataset, batch_size=1, shuffle=False)

        # evaluate the model on this bootstrap dataset
        bootstrap_preds, bootstrap_labels = evaluate(model, bootstrap_dl, disable_progress=True)
        bootstrap_preds = np.array(bootstrap_preds)
        bootstrap_labels = np.array(bootstrap_labels)

        # calculate the accuracy for the bootstrap predictions
        bootstrap_acc = accuracy_score(bootstrap_labels, bootstrap_preds)
        acc_sum += bootstrap_acc

        # calculate bootstrap delta value
        bootstrap_delta = bootstrap_acc - baseline_acc

        # if bootstrap_delta is larger than twice our model delta, we update s
        if bootstrap_delta > 2 * delta_x:
            s += 1.0

        if i % 500 == 0:
            pickle.dump({"s": s, "acc": acc_sum, "b_iters": i}, open("bootstrap_result.pkl", "wb"))

    # calculate p-value
    p_value = np.divide(s, b_iters)

    # calculate average test accuracy
    average_acc = np.divide(acc_sum, b_iters)

    return p_value, average_acc
  
  
  
'   '   '   '   '   Contrastive Learning in Supervised Learning Setting    '   '   '   '  '




# https://github.com/haideraltahan/CLAR/blob/main/utils.py 

def nt_xent_loss(out_1, out_2, temperature=0.5):
    """
    Loss used in SimCLR
    """
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()

    return loss



def shared_step(y_hat1, y_hat2, y):
  y_hat1 = F.normalize(y_hat1, dim=1)
  y_hat2 = F.normalize(y_hat2, dim=1)

  sm_y_hat1 = F.log_softmax(y_hat1)
  sm_y_hat2 = F.log_softmax(y_hat2)

  label_loss = F.nll_loss(sm_y_hat1, y) + F.nll_loss(sm_y_hat2, y)
  loss = nt_xent_loss(y_hat1, y_hat2) + label_loss

  if torch.isnan(loss):
    print("smyh1:", sm_y_hat1, "smh2:", sm_y_hat2, "yh1:", y_hat1, "yh2:", y_hat2)
  return loss