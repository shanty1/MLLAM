import argparse
import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_input', type=int, default=11, help='dimension of input')
parser.add_argument('--n_output', type=int, default=1, help='dimension of output')
parser.add_argument('--n_hidden', type=int, default=50, help='number of hidden layer')
parser.add_argument('--n_hiddens', type=list, default=[100, 200, 50], help='number of hiddens layer')
parser.add_argument('--num_epochs', type=int, default=1000000, help='num_epochs')
parser.add_argument('--save_epoch', type=int, default=500, help='how many epoch times before save')
parser.add_argument('--batch_size_train', type=int, default=70, help='training data batch_size')
parser.add_argument('--batch_size_val', type=int, default=70, help='validation batch_size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--show_each_epoch', type=bool,default=True, help='batch_size')

args = parser.parse_args()
