import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Embedding(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, vocab_size, input_encoding_size, glove = None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size,input_encoding_size)
        self.glove = glove
        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        self._init_hidden()

    def _init_hidden(self):
        if self.glove is not None:
            self.embedding.load_state_dict({'weight': np.load(self.glove)})

    def forward(self, input):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim
        Returns:
            Variable -- context vector of size batch_size x dim
        """
        return self.embedding(input)
