import argparse
import datetime
import time
from models import *
from utils import *
from trainer import Trainer
import torch
from RoadEnv import *
from torch.utils.tensorboard import SummaryWriter
import os
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Transport spectrum Sharing RL')
# training parameters
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')

# log and weight dir
time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir = './results/{time}'.format(time=time_now)
parser.add_argument('--log_dir', type=str, default=log_dir, help='dir for log this experiment')
#model parameters
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')
parser.add_argument('--nagents', type=int, default=10, help='Number of agents')
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much cooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str, help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10, type=int,
                    help='detach hidden state and cell state for rnns at this interval.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')

#optimization
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
parser.add_argument('--tau', type=float, default=1.0, help='gae')

parser.add_argument('--seed', type=int, default=0,
                    help='random seed. Pass -1 for random seed')
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')

#environment
parser.add_argument('--car_num_max', type=int, default=5, help='number of cars in one line')
parser.add_argument('--freq_num', type=int, default=4, help='number of freq band')
parser.add_argument('--naction_heads', type=list, default=[4], help='number of action head of policy network')
args = parser.parse_args()

env = Road(args.car_num_max, args.freq_num)

if args.seed == -1:
    args.seed = np.random.randint(0, 10000)
torch.manual_seed(args.seed)

print(args)

if args.recurrent:
    policy_net = RNN(args, env.observation_dim)
else:
    policy_net = MLP(args, env.observation_dim)

trainer = Trainer(args, policy_net, env)

weight_dir, log_dir = args.log_dir + '/weights', args.log_dir + '/logs'
os.makedirs(weight_dir)
os.makedirs(log_dir)
writer = SummaryWriter(log_dir)
log = dict()

#def run(num_epochs):
for ep in range(args.num_epochs):
    epoch_begin_time = time.time()
    stat = dict()
    print('epoch num: {}'.format(ep))
    for n in range(args.epoch_size):
            # trainer display here
        s = trainer.train_batch(ep)
        merge_stat(s, stat)
        for k, val in s.items():
            writer.add_scalar(k, val, ep * args.epoch_size + n)
    epoch_time = time.time() - epoch_begin_time
    trainer.save(os.path.join(weight_dir, str(ep) + '.pth'))









