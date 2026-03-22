import numpy as np
import sys

sys.path.append("mytorch")
from rnn_cell import *
from nn.linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO: Understand then uncomment this code :)
        self.rnn = [
            RNNCell(input_size, hidden_size) if i == 0 
                else RNNCell(hidden_size, hidden_size)
                    for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.
        -----
        Input
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]
        linear_weights:
                        [W, b]
        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.
        -----
        Input (see writeup for explanation)
        x: (batch_size, seq_len, input_size)

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states - defaults to zeros if not specified
        -------
        Returns
        logits: (batch_size, output_size) 

        Output (y): logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None

        ### Add your code here --->
        # (More specific pseudocode may exist in the write-up and/or lecture slides)
        
        # TODO
        for t in range(seq_len):
            x_t = x[:, t, :]
            for l in range(self.num_layers):
                x_t = self.rnn[l](x_t, hidden[l])
                hidden[l] = x_t
            self.hiddens.append(hidden.copy())
        # Get the outputs from the last time step using the linear layer and return it
        logits = self.output_layer(hidden[-1])
        return logits
        # raise NotImplementedError

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Input (see writeup for explanation)
        ------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                
        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)


        # Pseudocode may exist in the write-up and/or lecture slides
        # WATCH out for off by 1 errors due to implementation decisions.
        
        # TODO
        for t in range(seq_len-1, -1, -1):
            for l in range(self.num_layers-1, -1, -1):
                if l == 0:
                    h_prev_l = self.x[:, t, :]
                else:
                    h_prev_l = self.hiddens[t+1][l-1]
                h_t = self.hiddens[t+1][l]
                h_prev_t = self.hiddens[t][l]
                dx, dh_prev = self.rnn[l].backward(dh[l], h_t, h_prev_l, h_prev_t)

                dh[l] = dh_prev

                if l != 0:
                    dh[l-1] += dx
        return dh / batch_size
        # raise NotImplementedError
