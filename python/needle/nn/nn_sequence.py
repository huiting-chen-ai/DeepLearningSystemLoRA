"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = (1+ops.exp(-x))**(-1)
        return out
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = 1/np.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.nonlinearity = nonlinearity
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = Tensor(init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype), 
                       device=self.device, dtype=self.dtype, requires_grad=False)
        new_W = X@self.W_ih+h@self.W_hh
        if self.bias_hh:
            bias = self.bias_hh+self.bias_ih
            bias = bias.reshape((1, self.hidden_size))
            bias = bias.broadcast_to(new_W.shape)
            new_W = new_W+bias
        if self.nonlinearity == "tanh":
            return ops.tanh(new_W)
        else:
            return ops.relu(new_W)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        for i in range(1, num_layers):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h0 is None:
            h0 = Tensor(init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype), 
                       device=self.device, dtype=self.dtype, requires_grad=False)
        temp_X = list(ops.split(X, 0))
        temp_h = list(ops.split(h0, 0))
        for i in range(seq_len):
            for j in range(self.num_layers):
                temp_X[i] = self.rnn_cells[j](temp_X[i], temp_h[j])
                temp_h[j] = temp_X[i]
        temp_X = ops.stack(temp_X, 0)
        temp_h = ops.stack(temp_h, 0)
        return temp_X, temp_h
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = np.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.sig = Sigmoid()
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h0 = Tensor(init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype), 
                       device=self.device, dtype=self.dtype, requires_grad=False)
            c0 = Tensor(init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype), 
                       device=self.device, dtype=self.dtype, requires_grad=False)
        else:
            h0, c0 = h[0], h[1]
        ifgo = X@self.W_ih+h0@self.W_hh # bs, 4*hidden_size: ifgo
        if self.bias_hh is not None:
            bias = self.bias_hh+self.bias_ih
            bias = bias.reshape((1, 4*self.hidden_size))
            bias = bias.broadcast_to(ifgo.shape)
            ifgo = ifgo+bias
        ifgo = list(ops.split(ifgo, 1))
        i = self.sig(ops.stack(ifgo[0:self.hidden_size], 1))
        f = self.sig(ops.stack(ifgo[self.hidden_size:2*self.hidden_size], 1))
        g = ops.tanh(ops.stack(ifgo[2*self.hidden_size:3*self.hidden_size], 1))
        o = self.sig(ops.stack(ifgo[3*self.hidden_size:4*self.hidden_size], 1))
        c_ = f*c0+i*g
        h_ = o*ops.tanh(c_)
        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for i in range(1, num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h is None:
            h0 = Tensor(init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype), 
                       device=self.device, dtype=self.dtype, requires_grad=False)
            c0 = Tensor(init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype), 
                       device=self.device, dtype=self.dtype, requires_grad=False)
        else:
            h0 = h[0]
            c0 = h[1]
        temp_X = list(ops.split(X, 0))
        temp_h = list(ops.split(h0, 0))
        temp_c = list(ops.split(c0, 0))
        for i in range(seq_len):
            for j in range(self.num_layers):
                h_, c_ = self.lstm_cells[j](temp_X[i], (temp_h[j], temp_c[j]))
                temp_h[j] = h_
                temp_c[j] = c_
                temp_X[i] = h_
        temp_X = ops.stack(temp_X, 0)
        temp_h = ops.stack(temp_h, 0)
        temp_c = ops.stack(temp_c, 0)
        return temp_X, (temp_h, temp_c)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype)
        seq_len, bs = one_hot.shape[0], one_hot.shape[1]
        one_hot = one_hot.reshape((seq_len*bs, self.num_embeddings))
        proj = one_hot@self.weight
        return proj.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION