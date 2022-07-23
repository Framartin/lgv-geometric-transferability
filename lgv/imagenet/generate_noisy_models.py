import pandas as pd
import os
from pathlib import Path
import argparse
import random
from tqdm import tqdm
import numpy as np
import torch
from torchvision import models as tmodels
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.data import ImageNet
from utils.helpers import guess_and_load_model, guess_model, list_models
from utils.pca_weights import model2vector, models2tensor, vector2model
from utils.subspace_inference.utils import save_checkpoint



def parse_args():
    parser = argparse.ArgumentParser(description="Export models with added randomness")
    parser.add_argument("path_model", help="Path to model")
    parser.add_argument("--xp", default='gaussian_noise',
                        choices=['gaussian_noise', 'random_1D', 'random_ensemble_equivalent', 'gaussian_subspace'],
                        help="Name of the experiment to run. gaussian_noise to add Gaussian noise to weights (iid noise"
                             " per model). random_1D to export n-models along a single random direction in weights. "
                             "space. random_ensemble_equivalent to export for all models of path-ref-ensemble a model "
                             "at the same L2 distance from path_model but in a random direction")
    parser.add_argument("--std", default=1, type=float, help="Standard deviation of parameter noise. Used only for 'gaussian_noise' XP")
    parser.add_argument("--max-norm", default=1, type=float, help="Max 2-norm in weights space to sample equality spaced n-models along the single random direction. Used only for 'random_1D' XP")
    parser.add_argument("--path-ref-ensemble", default=None, help="Ensemble of reference from which we generate a similar one but with random directions. Used only for 'random_ensemble_equivalent' XP")
    parser.add_argument("--n-models", default=10, type=int, help="Number of models to export")
    parser.add_argument("--export-dir", default=None, help="Dir to export models and CSV")
    parser.add_argument("--update-bn", action='store_true', help="Update BN of produced models")
    parser.add_argument("--data-path", default=None, help="Path of data. Only supported for ImageNet.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Random seed passed to PCA.")
    args = parser.parse_args()
    return args

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
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

    model = guess_and_load_model(args.path_model, data=data, load_as_ghost=False, input_diversity=False, skip_gradient=False)
    if args.path_ref_ensemble:
        paths_ref_ensemble = list_models(args.path_ref_ensemble)
        models_ref_ensemble = []
        for i, filename in enumerate(paths_ref_ensemble):
            model_tmp = guess_and_load_model(filename, data=data, load_as_ghost=False, input_diversity=False,
                                         skip_gradient=False)
            models_ref_ensemble.append(model_tmp)
        w_ref_ensemble_list = [model2vector(x) for x in models_ref_ensemble]

    model_cfg = getattr(tmodels, guess_model(args.path_model))

    w = model2vector(model)

    if args.xp == 'gaussian_noise':
        for i in tqdm(range(args.n_models), desc="Export models"):
            w_random = w.detach().clone() + torch.randn(w.shape) * args.std
            model_noisy = vector2model(w_random, model_cfg, update_bn=args.update_bn, train_loader=loaders['train'])
            save_checkpoint(args.export_dir, name='model_noisy', sample=i, state_dict=model_noisy.state_dict())
    elif args.xp == 'random_1D':
        rand_vect = torch.randn(w.shape)
        rand_vect = rand_vect / torch.linalg.norm(rand_vect, ord=2)
        norm_list = np.linspace(0, args.max_norm, num=args.n_models).tolist()
        for i, norm in enumerate(tqdm(norm_list, desc="Export models")):
            w_random = w.detach().clone() + rand_vect * norm
            model_noisy = vector2model(w_random, model_cfg, update_bn=args.update_bn, train_loader=loaders['train'])
            export_dir = os.path.join(args.export_dir, f'norm_{norm}')
            save_checkpoint(export_dir, name='model_noisy_1D', sample=i, state_dict=model_noisy.state_dict())
    elif args.xp == 'random_ensemble_equivalent':
        # for each model, we compute the distance b/w w_ref and w, generate a random direction (uniform from unit sphere)
        # add it to w, update BN, and export
        for i, w_ref in enumerate(w_ref_ensemble_list):
            filename_ref = Path(paths_ref_ensemble[i]).stem
            ref_dist = torch.linalg.norm(w_ref - w, ord=2)
            rand_vect = torch.randn(w.shape)
            rand_vect = rand_vect / torch.linalg.norm(rand_vect, ord=2)
            w_random = w.detach().clone() + rand_vect * ref_dist
            print(f"Export model {filename_ref}; ||w_ref - w||: {ref_dist}; ||w_random - w||: {torch.linalg.norm(w_random - w, ord=2)}  ")
            model_noisy = vector2model(w_random, model_cfg, update_bn=args.update_bn, train_loader=loaders['train'])
            save_checkpoint(args.export_dir, name=filename_ref, state_dict=model_noisy.state_dict())
    elif args.xp == 'gaussian_subspace':
        # random gaussian in the LGV deviation subspace
        n_samples = len(models_ref_ensemble)
        n_weights = w.shape[0]
        W = models2tensor(models_ref_ensemble)
        D = (W-w).T  # deviation matrix
        for i in tqdm(range(args.n_models), desc="Export models"):
            w_random = w.detach().clone() + 1 / (n_samples-1)**0.5 * (D @ torch.randn(n_samples)) * args.std
            model_noisy = vector2model(w_random, model_cfg, update_bn=args.update_bn, train_loader=loaders['train'])
            save_checkpoint(args.export_dir, name='model_gaussian_subspace', sample=i, state_dict=model_noisy.state_dict())


if __name__ == '__main__':
    args = parse_args()
    main(args)