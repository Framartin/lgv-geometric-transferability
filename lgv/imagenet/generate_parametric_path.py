import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torchvision import models as tmodels
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.data import ImageNet
from utils.helpers import guess_and_load_model, guess_model
from utils.pca_weights import model2vector, models2tensor, vector2model
from utils.subspace_inference.utils import save_checkpoint



def parse_args():
    parser = argparse.ArgumentParser(description="Export models along a path connecting two models")
    parser.add_argument("path_model_1", help="Path to first model")
    parser.add_argument("path_model_2", help="Path to 2nd model")
    parser.add_argument("--names-models", nargs=2, metavar=('name1', 'name2'), help="Names of the two models")
    parser.add_argument("--alphas", default=None, nargs='*', type=float, help="Overwrite default list of alphas. If specified, n-models is ignored.")
    parser.add_argument("--n-models", default=100, type=int, help="Number of models to export")
    parser.add_argument("--export-dir", default=None, help="Dir to export models and CSV")
    parser.add_argument("--update-bn", action='store_true', help="Update BN of produced models")
    parser.add_argument("--data-path", default=None, help="Path of data. Only supported for ImageNet.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Try a lower value if out of memory (especially for high values of --ensemble-inner).")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Random seed passed to PCA.")
    args = parser.parse_args()
    return args

def main(args):
    np.random.seed(args.seed)
    data = ImageNet(batch_size=args.batch_size, path=args.data_path)
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    loaders = {'train': train_loader, 'test': val_loader}

    model1 = guess_and_load_model(args.path_model_1, data=data, load_as_ghost=False, input_diversity=False, skip_gradient=False)
    model2 = guess_and_load_model(args.path_model_2, data=data, load_as_ghost=False, input_diversity=False, skip_gradient=False)

    model_cfg = getattr(tmodels, guess_model(args.path_model_1))

    theta1 = model2vector(model1)
    theta2 = model2vector(model2)

    # alpha_min, alpha_max = args.alpha_range
    alpha_min, alpha_max = -1, 2  # hardcoded to avoid corner cases

    if args.alphas:
        alpha_list = list(args.alphas)
    else:
        alpha_list = np.linspace(alpha_min, alpha_max, args.n_models, endpoint=True).tolist()
    if 0. not in alpha_list:
        alpha_list += [0.]
    if 1. not in alpha_list:
        alpha_list += [1.]
    print(f'Start exporting the following alpha values: {["{0:0.2f}".format(i) for i in alpha_list]}')
    for alpha in tqdm(alpha_list, desc="Export interpolated models"):
        theta_alpha = torch.lerp(theta1, theta2, alpha).detach().clone()
        model_alpha = vector2model(theta_alpha, model_cfg, update_bn=args.update_bn, train_loader=loaders['train'])
        export_dir = os.path.join(args.export_dir, f'alpha_{alpha}')
        save_checkpoint(export_dir, name='model_interpolated', sample=alpha, state_dict=model_alpha.state_dict())


if __name__ == '__main__':
    args = parse_args()
    main(args)