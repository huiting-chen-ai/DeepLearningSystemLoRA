import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ConvBD(ndl.nn.Module):
    def __init__(self, a, b, k, s, device=None, dtype="float32"):
        super().__init__()
        self.conv = ndl.nn.Conv(a, b, k, stride=s, device=device, dtype=dtype)
        self.batchnorm = ndl.nn.BatchNorm2d(b, device=device, dtype=dtype)
        self.relu = ndl.nn.ReLU()
    def forward(self, x):
        device0 = str(x.device)
        x = self.conv(x)
        device1 = str(x.device)
        x = self.batchnorm(x)
        device2 = str(x.device)
        x = self.relu(x)
        device3 = str(x.device)
        assert device0 == device1
        assert device0 == device2
        assert device0 == device3
        return x

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.convBD1 = ConvBD(3, 16, 7, 4, device=device, dtype=dtype)
        self.convBD2 = ConvBD(16, 32, 3, 2, device=device, dtype=dtype)
        self.res3 = ndl.nn.Residual(
            ndl.nn.Sequential(ConvBD(32, 32, 3, 1, device=device, dtype=dtype),
                              ConvBD(32, 32, 3, 1, device=device, dtype=dtype))
        )
        self.convBD4 = ConvBD(32, 64, 3, 2, device=device, dtype=dtype)
        self.convBD5 = ConvBD(64, 128, 3, 2, device=device, dtype=dtype)
        self.res6 = ndl.nn.Residual(
            ndl.nn.Sequential(ConvBD(128, 128, 3, 1, device=device, dtype=dtype),
                              ConvBD(128, 128, 3, 1, device=device, dtype=dtype))
        )
        self.flatten = ndl.nn.Flatten()
        self.linear7 = ndl.nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu8 = ndl.nn.ReLU()
        self.linear9 = ndl.nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.convBD1(x)
        x = self.convBD2(x)
        x = self.res3(x)
        x = self.convBD4(x)
        x = self.convBD5(x)
        x = self.res6(x)
        x = self.flatten(x)
        x = self.linear7(x)
        x = self.relu8(x)
        x = self.linear9(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_layer = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == "rnn":
            self.model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        else:
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embedding_layer(x)
        x, h = self.model(x, h)
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        x = self.linear(x)
        return x, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
