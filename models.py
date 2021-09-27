import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class MLP(nn.Module):
    def __init__(self, args, num_inputs):           #
        super(MLP, self).__init__()
        self.args = args
        self.affine1 = nn.Linear(num_inputs, args.hid_size)
        self.affine2 = nn.Linear(args.hid_size, args.hid_size)
        self.heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])
        self.value_head = nn.Linear(args.hid_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, info={}):                  #   batch * nagents * num_inputs
        x = from_numpy(x)
        x = self.tanh(self.affine1(x))              #   batch * nagents * hid_size
        h = self.tanh(sum([self.affine2(x), x]))    #   batch * nagents * hid_size
        v = self.value_head(h)                      #   batch * nagents * 1

        return [F.log_softmax(head(h), dim=-1) for head in self.heads], v   #   [batch * nagents * n_actions], batch * nagents * 1

    def save(self, save_dir):
        torch.save(self.state_dict(), save_dir)

    def load(self, weight_dir):
        self.load_state_dict(torch.load(weight_dir))

class RNN(MLP):
    def __init__(self, args, num_inputs):
        super(RNN, self).__init__(args, num_inputs)
        self.nagents = self.args.nagents
        self.hid_size = self.args.hid_size
        if self.args.rnn_type == 'LSTM':
            del self.affine2
            self.lstm_unit = nn.LSTMCell(self.hid_size, self.hid_size)

    def forward(self, x, info={}):
        x, prev_hid = x                            # x: batch * nagents * num_inputs
                                                   # prev_hid: (batch * nagents , hid_dim)(batch * nagents, hid_dim)
        x = from_numpy(x)
        encoded_x = self.affine1(x)                # encoded_x: batch * nagents * hid_size

        if self.args.rnn_type == 'LSTM':
            batch_size = encoded_x.size(0)
            encoded_x = encoded_x.view(batch_size * self.nagents, self.hid_size)
            output = self.lstm_unit(encoded_x, prev_hid)
            next_hid = output[0]
            cell_state = output[1]
            ret = (next_hid.clone(), cell_state.clone())
            next_hid = next_hid.view(batch_size, self.nagents, self.hid_size)
                                                    # next_hid: batch_size, nagents, hid_size
        else:
            next_hid = F.tanh(self.affine2(prev_hid) + encoded_x)
            ret = next_hid

        v = self.value_head(next_hid)               # batch * nagents * 1
        return [F.log_softmax(head(next_hid), dim=-1) for head in self.heads], v, ret

    def init_hidden(self, batch_size):
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

    def save(self, save_dir):
        torch.save(self.state_dict(), save_dir)

    def load(self, weight_dir):
        self.load_state_dict(torch.load(weight_dir))
