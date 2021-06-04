import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

__all__ = ['CELEBA']

class CELEBA(Dataset):
    def __init__(self, root, train=True, transform=None, size=[32,32], num_points=200, eval_mode='none', num_points_eval=200, download=False):
        self.name = 'CELEBA'
        split = 'train' if train else 'test'
        self.size = size        #Original size: (218, 178)
        self.num_points = num_points
        self.eval_mode = eval_mode
        self.num_points_eval = num_points_eval

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        self.dataset = datasets.CelebA(root=root, split=split, download=download, transform=transform)
        
        self.coordinates = torch.from_numpy(np.array([[int(i/size[1])/size[0],(i%size[1])/size[1]] for i in range(size[0] * size[1])])).float()

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        target_y = image.view(image.size(0), -1).transpose(0,1)
        target_x = self.coordinates
        
        random_idx = torch.from_numpy(np.random.choice(np.product(self.size), size=self.num_points_eval if self.eval_mode == 'all' else self.num_points, replace=False))
        context_x = torch.index_select(target_x, dim=0, index=random_idx)
        context_y = torch.index_select(target_y, dim=0, index=random_idx)
        
        return context_x, context_y, target_x, target_y

    def __len__(self):
        return len(self.dataset)
