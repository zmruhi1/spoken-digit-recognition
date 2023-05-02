import torch 
import torch.nn.functional as F
from torch import nn


class RNNClassifier(nn.Module):
  """
  A simple RNN model that uses LSTM for audio classification
  """
  def __init__(self, embeds_dim, hidden_dim, n_classes=10, dropout=0.1):
    """
    param embeds_dim: the size of the embeddings
    param hidden_dim: the size of the internal hidden layers
    param n_classes: the number of classes
    param dropout: the dropout probability
    """
    super(RNNClassifier, self).__init__()

    self.hidden_dim = hidden_dim
    self.lstm = nn.LSTM(embeds_dim, hidden_dim)
    self.dropout = nn.Dropout(p=dropout)
    self.linear = nn.Linear(hidden_dim, n_classes)

  def forward(self, inp, hid_prev=None, ctx_prev=None):

    # If hidden representation (from previous output) is not given,
    # initialize a zero vector to feed to the LSTM model
    if not isinstance(hid_prev, torch.Tensor):
      hid_prev = torch.zeros(1, inp.shape[1], self.hidden_dim)

    # If context representation (from previous output) is not given,
    # initialize a zero vector to feed to the LSTM model
    if not isinstance(ctx_prev, torch.Tensor):
      ctx_prev = torch.zeros(1, inp.shape[1], self.hidden_dim)

    out, (hid, ctx) = self.lstm(inp, (hid_prev, ctx_prev))
    
    out = self.dropout(out)

    out = self.linear(out)
    out = torch.mean(out, dim=1)
    out = F.log_softmax(out, dim=-1)
    return out