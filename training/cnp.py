import torch
import argparse
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from training import CNPTrainer
from datasets import CELEBA, get_dataset
from models import CNP, CNPDecoder, CNPEncoder, MeanAgg, SumAgg, SetTransformerAgg, ConsistentAggregator

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='', type=str, help='dataset root')
parser.add_argument('--save_dir', default='cnp', type=str, help='directory to store checkpoints and results in')
parser.add_argument('--dataset', default='celeba', type=str, help='dataset to load')
parser.add_argument('--run', default=0, type=int, help='experiment run')
parser.add_argument('--x_dim', default=2, type=int, help='dimension of context x')
parser.add_argument('--y_dim', default=3, type=int, help='dimension of context y')
parser.add_argument('--size', default=[32,32], nargs='+', type=int, help='input image size')
parser.add_argument('--num_points', default=200, type=int, help='number of pixels to use as context')
parser.add_argument('--num_points_eval', default=200, type=int, help='number of pixels to use as context')
#Encoder/Decoder
parser.add_argument('--aggregator', default='mean', type=str, help='aggregation function in cnp encoder mean/sum/settransformer/consistent')
parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension')
parser.add_argument('--num_layers_encoder', default=4, type=int, help='layers in encoder and decoder')
parser.add_argument('--num_layers_decoder', default=4, type=int, help='layers in encoder and decoder')
#SlotSetEncoder
parser.add_argument('--K', default=[1], nargs='+', type=int, help='number of slots in each hierarchy')
parser.add_argument('--h', default=[128], nargs='+', type=int, help='dimension of each slot')
parser.add_argument('--d', default=[64], nargs='+', type=int, help='dimension of input')
parser.add_argument('--d_hat', default=[64], nargs='+', type=int, help='linear projection dimension')
parser.add_argument('--g', default='sum', type=str, help='aggregation function to use.')
parser.add_argument('--ln', default=True, type=str2bool, help='use layernorm or not.')
parser.add_argument('--_slots', default='Random', type=str, help='type of slots to use for mini batch consistent model(Random/Learned)')
#Optimizer/DataLoader
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--eval_mode', default='none', type=str, help='num of training epochs')
parser.add_argument('--epochs', default=100, type=int, help='num of training epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-3, type=float, help='weight decay')
parser.add_argument('--comment', default='', type=str, help='comment string which makes a new directory in the hierarchy')
parser.add_argument('--debug', default=False, type=str2bool, help='debug mode or not')
args = parser.parse_args()

if __name__ == '__main__':
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataloaders
    print('loading data: {}'.format(args.dataset))
    trainloader, validloader = get_dataset(args)

    # Create Model
    if args.aggregator == 'mean':
        aggregator = MeanAgg()
    elif args.aggregator == 'sum':
        aggregator = SumAgg()
    elif args.aggregator == 'settransformer':
        aggregator = SetTransformerAgg(hidden_dim=args.s_hidden_dim, num_seeds=args.num_seeds, num_heads=args.num_heads)
    elif args.aggregator == 'consistent':
        aggregator = ConsistentAggregator(K=args.K, h=args.h, d=args.d, d_hat=args.d_hat, g=args.g, ln=args.ln, _slots=args._slots)
    else:
        raise ValueError('{} not implemented'.format(args.aggregator))
    
    encoder = CNPEncoder(x_dim=args.x_dim, y_dim=args.y_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers_encoder, aggregator=aggregator)
    decoder = CNPDecoder(x_dim=args.x_dim, y_dim=args.y_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers_decoder)
    model   = CNP(encoder=encoder, decoder=decoder).to(args.device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, [int(r*args.epochs) for r in [0.2, 0.6, 0.8]], gamma=0.1)
    
    print('{} : {}'.format(args.aggregator, model.num_params))
    trainer = CNPTrainer(model, optimizer, scheduler, trainloader, validloader, args)
    if args.debug:
        with torch.autograd.detect_anomaly():
            trainer.fit()
    else:
        trainer.fit()
    print('{} :done'.format(args.run))
