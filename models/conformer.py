""" Conformer Model """

import torch 
import torch.nn.functional as F
from torch import nn
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange

class DepthWiseConv1d(nn.Module):
    # Taken from: https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
    # Depth wise convolution calculates convolution of each input channel with a different kernel
    # and concatenates it along the channel axis. In this convolution information isn't mixed
    # across different channels
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# simple functions used in the following classes
def exists(val):
    """Checks if value is None"""
    return val is not None

def default(val, d):
    """Returns a default value d if val doesn't exist"""
    return val if exists(val) else d



class RelativeSelfAttention(nn.Module):
    # Taken from: https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
    # This is same as the multi-headed self-attention but we use relative position encoding
    # to represent the positional information between the input tokens better. 
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb

        # this embeddings stores the relativ positional information
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

    def forward(self, x, context = None, mask = None, context_mask = None):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class MHAModule(nn.Module):
  """
  The multihead attention module of Conformer. It uses the relative self attention defined above.
  """
  def __init__(self, embeds_dim=512, n_heads=8, attn_head_dim=64, p=0.2):
    """
    param embeds_dim: size of the embeddings
    param n_heads: the number of attention heads
    param attn_head_dim: the embedding size of each attention head
    param p: the dropout probability
    """
    super(MHAModule, self).__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=embeds_dim)
    self.mha = RelativeSelfAttention(dim=embeds_dim, heads=n_heads, dim_head=attn_head_dim)
    self.dropout = nn.Dropout(p=p)

  def forward(self, x):
    out = self.layer_norm(x)
    out = self.mha(out)
    out = self.dropout(out)
    return out

class FeedForwardModule(nn.Module):
  """
  The feed forward module of Conformer. It uses two linear layers along with dropout, layer normalization and swish activation func.
  """
  def __init__(self, embeds_dim, expansion_factor=4, p=0.2):
    """
    param embeds_dim: the size of the embedding
    param expansion_factor: determines the size of the intermediate representation for the linear layer
    param p: the dropout probability
    """
    super(FeedForwardModule, self).__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=embeds_dim) # out: (batch, seq_len, embeds_dim)
    self.ff1 = nn.Linear(embeds_dim, embeds_dim*expansion_factor) # out: (batch, seq_len, embeds_dim * expansion_factor)
    self.swish = nn.SiLU() # out: (batch, seq_len, embeds_dim * expansion_factor)
    self.dropout1 = nn.Dropout(p=p) # out: (batch, seq_len, embeds_dim * expansion_factor)
    self.ff2 = nn.Linear(embeds_dim*expansion_factor, embeds_dim) # out: (batch, seq_len, embeds_dim)
    self.dropout2 = nn.Dropout(p=p) # out: (batch, seq_len, embeds_dim)

  def forward(self, x):
    out = self.layer_norm(x)
    out = self.ff1(out)
    out = self.swish(out)
    out = self.dropout1(out)
    out = self.ff2(out)
    out = self.dropout2(out)
    return out

class ConvModule(nn.Module):
  """
  The Convolution module from Conformer
  """
  def __init__(self, embeds_dim, expansion_factor=2, kernel=3, padding="same", dropout_p=0.2):
    """
    param embeds_dim: the size of the embeddings
    param expansion_factor: the multiplier for the intermediate dimension
    param kernel: the filter size
    param padding: the padding to use (only 'same' is used in this implementation)
    param dropout_p: the dropout probability
    """
    super(ConvModule, self).__init__()

    inner_dim = embeds_dim * expansion_factor

    # To maintain shape, use same padding
    if padding == "same":
      pad_size = kernel // 2
      padding = (pad_size, pad_size - (kernel+1)%2)

    self.layer_norm = nn.LayerNorm(normalized_shape=embeds_dim)
    self.rearrange1 = Rearrange('b n c -> b c n')
    self.pointwise1 = nn.Conv1d(in_channels=embeds_dim, out_channels=inner_dim*2, kernel_size=1)
    self.glu = nn.GLU(dim=1)
    self.depthwise_conv = DepthWiseConv1d(chan_in=inner_dim, chan_out=inner_dim, kernel_size=kernel, padding=padding)
    self.batch_norm = nn.BatchNorm1d(inner_dim)
    self.swish = nn.SiLU()
    self.pointwise2 = nn.Conv1d(in_channels=inner_dim, out_channels=embeds_dim, kernel_size=1)
    self.rearrange2 = Rearrange('b c n -> b n c')
    self.dropout = nn.Dropout(p=dropout_p)

  def forward(self, x):
    out = self.layer_norm(x)
    out = self.rearrange1(out)
    out = self.pointwise1(out)
    out = self.glu(out)
    out = self.depthwise_conv(out)
    out = self.batch_norm(out)
    out = self.swish(out)
    out = self.pointwise2(out)
    out = self.rearrange2(out)
    out = self.dropout(out)
    return out


class ConformerBlock(nn.Module):
  """
  A single Conformer block. It combines the FF, MHA and CONV modules defined previously.
  """
  def __init__(self, embeds_dim, conv_expansion_factor=2, linear_expansion_factor=4, n_heads=8, attn_head_dim=64, dropout=0.2, padding="same"):
    """
    param embeds_dim: the size of the embeddings
    param conv_expansion_factor: the multiplier for the intermediate dimension for Conv module
    param linear_expansion_factor: the multiplier for the intermediate dimension for FF module
    param n_heads: the number of attention heads
    param attn_head_dim: the embedding dimension of the attention head
    param padding: the padding to use (only 'same' is used in this implementation)
    param dropout: the dropout probability
    """
    super(ConformerBlock, self).__init__()

    self.ff_mod1 = FeedForwardModule(embeds_dim, expansion_factor=linear_expansion_factor, p=dropout)
    self.mha_mod = MHAModule(embeds_dim, n_heads, attn_head_dim, p=dropout)
    self.conv_mod = ConvModule(embeds_dim, expansion_factor=conv_expansion_factor, padding=padding)
    self.ff_mod2 = FeedForwardModule(embeds_dim, expansion_factor=linear_expansion_factor, p=dropout)
    self.layer_norm = nn.LayerNorm(normalized_shape=embeds_dim)

  def forward(self, inp):
    out = self.ff_mod1(inp)
    out_res = inp + 0.5 * out
    out = self.mha_mod(out_res)
    out_res = out_res + out
    out = self.conv_mod(out_res)
    out_res = out + out_res
    out = self.ff_mod2(out_res)
    out_res = out_res + 0.5 * out
    out = self.layer_norm(out)
    return out


class Conv2dSubampling(nn.Module):
    """
    Taken from: https://github.com/sooftware/conformer/blob/main/conformer/convolution.py
    Subsamples the input to 1/4th its original size by using Conv layer with stride 2 twice

    Convolutional 2D subsampling (to 1/4 length)
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs
    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(self, in_channels, out_channels):
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs, input_lengths):
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return outputs, output_lengths



class ConformerClassifier(nn.Module):
  def __init__(
      self, 
      input_dim=80, 
      n_classes=10, 
      embeds_dim=512, 
      n_layers=5, 
      conv_expansion_factor=2, 
      linear_expansion_factor=4, 
      n_heads=8, 
      attn_head_dim=64, 
      dropout=0.2, 
      padding=None):
    """
    param input_dim: the size of the input vector
    param n_classes: the number of output classes
    param embeds_dim: the size of the embeddings
    param n_layers: the number of conformer blocks to use
    param conv_expansion_factor: the multiplier for the intermediate dimension for Conv module
    param linear_expansion_factor: the multiplier for the intermediate dimension for FF module
    param n_heads: the number of attention heads
    param attn_head_dim: the embedding dimension of the attention head
    param padding: the padding to use (only 'same' is used in this implementation)
    param dropout: the dropout probability
    """
    super(ConformerClassifier, self).__init__()

    self.conv_subsampler = Conv2dSubampling(in_channels=1, out_channels=embeds_dim)
    self.linear = nn.Linear(embeds_dim * (((input_dim - 1) // 2 - 1) // 2), embeds_dim)
    self.dropout = nn.Dropout(p=dropout)
    self.conformer_blocks = nn.ModuleList([
        ConformerBlock(
            embeds_dim=embeds_dim,
            conv_expansion_factor=2,
            linear_expansion_factor=4, 
            n_heads=8, 
            attn_head_dim=64, 
            dropout=0.2, 
            padding="same"
        )

        for _ in range(n_layers)
    ])
    self.linear_out = nn.Linear(embeds_dim, n_classes)


  def forward(self, inp, inp_lengths, return_logits=False):
    out, out_lengths = self.conv_subsampler(inp, inp_lengths)
    out = self.linear(out)
    out = self.dropout(out)
    for l in self.conformer_blocks:
      out = l(out)
    out = self.linear_out(out)
    out = torch.mean(out, dim=1)

    if not return_logits:
      out = F.log_softmax(out, dim=-1)
      
    return out