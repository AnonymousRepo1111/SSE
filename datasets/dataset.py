from torch.utils.data import DataLoader

from datasets import CELEBA, ModelNet40, ModelNetFixed, ModelNetFixedDataset

__all__ = ["get_dataset"]

def get_dataset(args):
    if args.dataset == "celeba":
        celeba_train = CELEBA(root=args.root, train=True, num_points=args.num_points, size=args.size)
        celeba_valid = CELEBA(root=args.root, train=False, num_points=args.num_points, size=args.size, eval_mode=args.eval_mode, num_points_eval=args.num_points_eval)
        trainloader = DataLoader(celeba_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        validloader = DataLoader(celeba_valid, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        return trainloader, validloader
    else:
        raise ValueError(f"name: {args.name} is not an implemented dataset")
