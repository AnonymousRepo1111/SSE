import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from models import Linear, BMM, clip_grad

__all__ = ['CNPTrainer']

class Trainer(object):
    def __init__(self, model, optimizer, scheduler, trainloader, validloader, args, clip=True):
        self.args = args
        self.model = model
        self.run = args.run
        self.epochs = args.epochs
        self.device = args.device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.validloader = validloader
        self.clip = clip
        self.debug = args.debug
        
        self.dataset_name = trainloader.dataset.name
        self.num_points = str(self.trainloader.dataset.num_points)
        self.model_name = self.model.module.name if isinstance(self.model, nn.DataParallel) else self.model.name
        self.hidden_dim = str(self.model.module.hidden_dim) if isinstance(self.model, nn.DataParallel) else str(self.model.hidden_dim)
        
        self.result_path = os.path.join('results', self.dataset_name, self.model_name, self.num_points, self.hidden_dim, str(self.args.lr), args.comment)
        self.checkpoint_path = os.path.join('checkpoints', self.dataset_name, self.model_name, self.num_points, self.hidden_dim, str(args.lr), args.comment)

        self.paths = ['checkpoints', 'results', self.result_path, self.checkpoint_path]

        if not self.debug:
            self.initialize_paths()
        
    def initialize_paths(self):
        for path in self.paths:
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                except FileExistsError as e:
                    pass

    def fit(self):
        raise NotImplementedError

    def train(self, epoch):
        raise NotImplementedError

    def valid(self, epoch):
        raise NotImplementedError
    
    def flops(self):
        flops = 0.0
        for name, module in self.model.named_modules():
            if isinstance(module, Linear) or isinstance(module, BMM):
                flops += module.flops()
        return flops

    def save_checkpoint(self, epoch):
        checkpoint = {
                'state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
                'epoch': epoch,
                'result_path': self.result_path
                }
        torch.save(checkpoint, os.path.join(self.checkpoint_path, str(self.run)+'.pth'))

    def log(self, data_dict):
        for key, value in data_dict.items():
            with open(os.path.join(self.result_path, key), 'w+') as f:
                for v in value:
                    f.write('{}\n'.format(v))

class CNPTrainer(Trainer):
    def __init__(self, *args, clipper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_nll = []
        self.valid_nll = []
        self.best_nll = float('inf')

    def fit(self):
        for epoch in range(self.epochs):
            train_nll = self.train(epoch)
            valid_nll = self.valid(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            print('Epoch: {:<3}({}) Train NLL: {:.4f} Valid NLL: {:.4f}'.format(epoch, self.run, train_nll, valid_nll))

            self.train_nll.append(train_nll)
            self.valid_nll.append(valid_nll)
            if not self.debug:
                self.log(data_dict={'train_nll_{}.txt'.format(self.run): self.train_nll, 'valid_nll_{}.txt'.format(self.run): self.valid_nll})

            #Log FLOPS and Parameter Counts
            if epoch == 2:
                flops = self.flops()
                param = self.model.module.num_params if isinstance(self.model, nn.DataParallel) else self.model.num_params
                if not self.debug:
                    with open(os.path.join(self.result_path, 'data_{}.data'.format(self.run)), 'w+') as f:
                        f.write('FLOPS  : {}\n'.format(flops))
                        f.write('PARAM  : {}\n'.format(param))
                print('FLOPS and PARAMS logged')
            
            if epoch >= int(0.8 * self.epochs):
                if valid_nll < self.best_nll:
                    self.best_nll = valid_nll
                    if not self.debug:
                        self.save_checkpoint(epoch)
        
        if not self.debug:
            with open(os.path.join(self.result_path, 'data_{}.data'.format(self.run)), 'a') as f:
                        f.write('NLL: {}\n'.format(self.best_nll))

    def train(self, epoch):
        self.model.train()
        losses, count = 0.0, 0
        for context_x, context_y, target_x, target_y in tqdm(self.trainloader, total=len(self.trainloader), ncols=75, leave=False):
            self.optimizer.zero_grad()
            output = self.model(context_x.to(self.device), context_y.to(self.device), target_x.to(self.device), target_y.to(self.device))
            output['loss'].mean().backward()
            self.optimizer.step()
            
            count += target_y.size(0)
            losses += output['loss'].mean().item()
        return losses / count

    def valid(self, epoch):
        self.model.eval()
        with torch.no_grad():
            losses, count = 0.0, 0
            for context_x, context_y, target_x, target_y in tqdm(self.validloader, total=len(self.validloader), ncols=75, leave=False):
                output = self.model(context_x.to(self.device), context_y.to(self.device), target_x.to(self.device), target_y.to(self.device))
                
                count += target_y.size(0)
                losses += output['loss'].mean().item()
        return losses / count
