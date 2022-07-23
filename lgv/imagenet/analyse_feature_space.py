"""
Interpolate between adv ex from 2 surrogate in feature space
"""
import os
import sys
import torch
import math
import random
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from tqdm import tqdm
from utils.n_sphere import convert_spherical, convert_rectangular
from utils.data import CIFAR10, ImageNet
from utils.helpers import guess_and_load_model, load_classifier, list_models, project_on_sphere, compute_accuracy_from_nested_list_models, compute_norm, flatten
from utils.attacks import ExtendedProjectedGradientDescentPyTorch


# parse args
parser = argparse.ArgumentParser(description="Interpolation of adversarial examples from two surrogate")
parser.add_argument("path_model_1", help="Path to model 1. Could be either a directory of models or a path to a model.")
parser.add_argument("path_model_2", help="Path to model 2. Could be either a directory of models or a path to a model.")
parser.add_argument("--xp", choices=['interpolation', 'disk'], default='interpolation', help="Type of L-norm to use. Default: 2")
parser.add_argument("--path_target", nargs='+', help="Path to target directory")
parser.add_argument("--norm", choices=['1', '2', 'inf'], default='2', help="Type of L-norm to use. Default: 2")
parser.add_argument("--max-norm", type=float, required=True, help="Max L-norm of the perturbation")
parser.add_argument("--csv-export", default=None, help="Path to CSV where to export data about target.")

# all xp
parser.add_argument("--n-examples", type=int, default=2000, help="Craft adv ex on a subset of test examples. If None "
                                                                 "(default), perturbate all the test set. If "
                                                                 "model-target-path is set, extract the subset from "
                                                                 "the examples correctly predicted by it.")
# xp interpolation
parser.add_argument("--n-interpolation", type=int, default=100, help="Number of adv ex to compute along the generated path (include the two original ones).")
parser.add_argument('--interpolation-method', choices=['linear', 'proj_sphere', 'polar', 'linear_hyperspherical_coord'], default='linear_hyperspherical_coord', help="Interpolation method between the 2 adv examples")
parser.add_argument("--alpha-range", type=float, default=1, help="Max alpha interpolation coef. alpha in [-alphamax, 1+alphamax]")
# xp disk
parser.add_argument("--n-points", type=int, default=500, help="Number of points to evaluate loss in the disk.")
parser.add_argument("--grid", choices=['grid', 'sunflower'], default='grid', help="Type of grid to generate points in the disk.")

# others
parser.add_argument("--data-path", default=None, help="Path of data. Only supported for ImageNet.")
parser.add_argument("--seed", type=int, default=None, help="Set random seed")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Try a lower value if out of memory (especially for high values of --ensemble-inner).")

args = parser.parse_args()

if args.norm == 'inf':
    args.norm = np.inf
else:
    args.norm = int(args.norm)

if args.seed:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


if 'CIFAR10' in args.path_model_1:
    data = CIFAR10(batch_size=args.batch_size)
elif 'ImageNet' in args.path_model_1:
    data = ImageNet(batch_size=args.batch_size, path=args.data_path)
else:
    raise ValueError('dataset not supported')


path_model_list_1 = list_models(args.path_model_1)
path_model_list_2 = list_models(args.path_model_2)

list_model_1 = [guess_and_load_model(path, data=data) for path in path_model_list_1]
list_model_2 = [guess_and_load_model(path, data=data) for path in path_model_list_2]
list_classifier_1 = [load_classifier(model, data=data) for model in list_model_1]
list_classifier_2 = [load_classifier(model, data=data) for model in list_model_2]

models_target = [guess_and_load_model(path_model=x, data=data) for x in args.path_target]
X, y = data.correctly_predicted_to_numpy(models=models_target, train=False,
                                         N=args.n_examples, seed=args.seed)

attack1 = ExtendedProjectedGradientDescentPyTorch(estimators=list_classifier_1, targeted=False, norm=args.norm,
                                                  eps=args.max_norm, eps_step=args.max_norm / 10., max_iter=50,
                                                  num_random_init=0,
                                                  batch_size=args.batch_size)
attack2 = ExtendedProjectedGradientDescentPyTorch(estimators=list_classifier_2, targeted=False, norm=args.norm,
                                                  eps=args.max_norm, eps_step=args.max_norm / 10., max_iter=50,
                                                  num_random_init=0,
                                                  batch_size=args.batch_size)
X_adv1 = attack1.generate(x=X, y=y)
X_adv2 = attack2.generate(x=X, y=y)

delta1 = X_adv1 - X
delta2 = X_adv2 - X

print('Natural accuracy:')
for i, model_target in enumerate(models_target):
    acc_target_original, loss_target_original = compute_accuracy_from_nested_list_models([[model_target, ], ], X=X, y=y,
                                                                                       data=data)
    print(f'   * {args.path_target[i]}: {acc_target_original*100}% ; loss: {loss_target_original}')


if args.xp == "interpolation":
    alpha_list = np.linspace(-args.alpha_range, 1+args.alpha_range, args.n_interpolation, endpoint=True).tolist()

    metrics_list = []
    if args.interpolation_method == 'linear_hyperspherical_coord':
        # get nb digits precision for n-sphere rounding
        precision_dtype = np.finfo(delta1.dtype).precision
        # perturbations in spherical coordinates
        delta1_sc = convert_spherical(flatten(delta1), digits=precision_dtype)
        delta2_sc = convert_spherical(flatten(delta2), digits=precision_dtype)

    for alpha in tqdm(alpha_list, desc="Interpolate adv ex"):
        if args.interpolation_method == 'linear':
            X_adv_interp = X + (1-alpha) * delta1 + alpha * delta2
        elif args.interpolation_method == 'proj_sphere':
            X_adv_interp = X + (1-alpha) * delta1 + alpha * delta2
            X_adv_interp = project_on_sphere(X=X, X_adv=X_adv_interp, data=data, size=args.max_norm, norm=args.norm)
        elif args.interpolation_method == 'polar':
            # https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/
            # valid under some condition
            X_adv_interp = X + sqrt(1-alpha)*delta1 + sqrt(alpha)*delta2
        elif args.interpolation_method == 'linear_hyperspherical_coord':
            if args.norm != 2:
                raise ValueError('linear_hyperspherical_coord interpolation only guarantee for 2-norm')
            # the most appropriate interpolation
            # https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
            # interpolate in spherical coordinates
            delta_inter_sc = (1-alpha) * delta1_sc + alpha * delta2_sc
            # if we extrapolate, the perturbation may be outside the L2 ball. We project it into the L2 sphere
            # the first dim is the l2 norm (radius of the sphere)
            examples_to_proj = (delta_inter_sc[:, 0] > args.max_norm)
            delta_inter_sc[examples_to_proj, 0] = args.max_norm
            delta_inter = convert_rectangular(delta_inter_sc, digits=precision_dtype)
            X_adv_interp = X + delta_inter.reshape(X.shape)
        else:
            raise ValueError('Interpolation method not supported')
        # check norm perturbation
        lpnorm = compute_norm(X_adv=X_adv_interp, X=X, norm=args.norm)
        if (lpnorm > args.max_norm + 1e-6).any():
            print(f'For alpha={alpha}, nb examples outside the Lp ball: {(lpnorm > args.max_norm + 1e-6).sum()}')

        acc_surrogate1, loss_surrogate1 = compute_accuracy_from_nested_list_models([list_model_1, ], X=X_adv_interp, y=y, data=data)
        metrics_list.append({'model': args.path_model_1, 'type_model': 'surrogate', 'alpha': alpha, 'adv_accuracy': acc_surrogate1, 'adv_loss': loss_surrogate1})
        acc_surrogate2, loss_surrogate2 = compute_accuracy_from_nested_list_models([list_model_2, ], X=X_adv_interp, y=y, data=data)
        metrics_list.append({'model': args.path_model_2, 'type_model': 'surrogate', 'alpha': alpha, 'adv_accuracy': acc_surrogate2, 'adv_loss': loss_surrogate2})
        for i, model_target in enumerate(models_target):
            acc_target, loss_target = compute_accuracy_from_nested_list_models([[model_target, ], ], X=X_adv_interp, y=y, data=data)
            metrics_list.append({'model': args.path_target[i], 'type_model': 'target', 'alpha': alpha, 'adv_accuracy': acc_target, 'adv_loss': loss_target})

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics['norm'] = args.norm
    df_metrics['max_norm'] = args.max_norm
    df_metrics['n_examples'] = args.n_examples
    df_metrics['n_interpolation'] = args.n_interpolation
    df_metrics['interpolation_method'] = args.interpolation_method
    df_metrics['seed'] = args.seed

    os.makedirs(os.path.dirname(args.csv_export), exist_ok=True)
    df_metrics.to_csv(args.csv_export, header=True, index=False)

    # for i, path_target in enumerate(args.path_target):
    #     df_ = df_metrics.query(f'model == "{path_target}"')
    #     experiment.log_curve(f"target_{path_target}", x=df_['alpha'], y=df_['adv_loss'])
    #
    # df_ = df_metrics.query(f'model == "{args.path_model_1}"')
    # experiment.log_curve(f"target_{args.path_model_1}", x=df_['alpha'], y=df_['adv_loss'])
    #
    # df_ = df_metrics.query(f'model == "{args.path_model_2}"')
    # experiment.log_curve(f"target_{args.path_model_2}", x=df_['alpha'], y=df_['adv_loss'])

elif args.xp == 'disk':
    metrics_list = []
    # define plane with 3 points: X, X_adv1, X_adv2 using Gram-Schmidt
    # quote from SWA paper "Suppose we have three weight vectors w1, w2, w3.
    # We set u = (w2−w1), v = (w3−w1)− <w3−w1, w2− w1>/||w2− w1||2 · (w2 − w1).
    # Then the normalized vectors u_ = u/||u||, v_ = v/||v|| form an orthonormal basis in the plane containing w1, w2, w3."
    u = flatten(delta1)  # (X_adv1−X)
    u_ = u / np.linalg.norm(u, axis=1, keepdims=True)  # normalised vector
    u__ = u / np.linalg.norm(u, axis=1, keepdims=True)**2  # tmp
    v = flatten(delta2) - flatten(np.diagonal(np.dot(flatten(delta2), u__.T))) * u  # delta2 = Xadv2-X
    # check with np.isclose( flatten(X_adv2 - X)[0] - np.dot(flatten(X_adv2 - X)[0], u__[0]) * u[0] , v[0] )
    v_ = v / np.linalg.norm(v, axis=1, keepdims=True)  # normalised vector

    if args.grid == 'grid':
        nb_points_per_axis = math.ceil(math.sqrt(args.n_points))
        x_values = np.linspace(-args.max_norm, args.max_norm, nb_points_per_axis, endpoint=True)
        x1, x2 = np.meshgrid(x_values, x_values)
        x1, x2 = x1.flatten(), x2.flatten()
    elif args.grid == 'sunflower':
        # then we generate N points in the disk of radius epsilon
        # 2D sunflower spiral algorithm https://stackoverflow.com/a/44164075/6253027
        indices = np.arange(0, args.n_points, dtype=float) + 0.5
        r = np.sqrt(indices / args.n_points) * args.max_norm  # max_norm = epsilon
        theta = np.pi * (1 + 5 ** 0.5) * indices
        x1, x2 = r * np.cos(theta), r * np.sin(theta)
    else:
        raise ValueError("Wrong type of grid")

    # save position of u and v
    df_Xadv1 = pd.DataFrame({
        'model': args.path_model_1,
        'type_model': 'surrogate_1',
        'x1': np.diagonal(np.dot(flatten(delta1), u_.T)),
        'x2': np.diagonal(np.dot(flatten(delta1), v_.T)),
    })
    df_Xadv2 = pd.DataFrame({
        'model': args.path_model_2,
        'type_model': 'surrogate_2',
        'x1': np.diagonal(np.dot(flatten(delta2), u_.T)),
        'x2': np.diagonal(np.dot(flatten(delta2), v_.T)),
    })
    df_Xadv = df_Xadv1.append(df_Xadv2, ignore_index=True)
    print(f'Position of Xadv1: {df_Xadv1["x1"].mean()}, {df_Xadv1["x2"].mean()}')
    print(f'Position of Xadv2: {df_Xadv2["x1"].mean()}, {df_Xadv2["x2"].mean()}')
    os.makedirs(os.path.dirname(args.csv_export), exist_ok=True)
    path_csv_xadv = args.csv_export.replace('.csv', '__ref_xadv.csv')
    df_Xadv.to_csv(path_csv_xadv, header=True, index=False)

    # and generate the examples to evaluate
    # "A point P with coordinates(x, y) in the plane would then be given by P = w1+x·u_ + y·v_"
    # for each point in disk
    for i in tqdm(range(args.n_points), desc="Predicting points"):
        X_disk = X + (x1[i] * u_ + x2[i] * v_).reshape(X.shape)

        if args.grid != 'grid':
            lpnorm = compute_norm(X_adv=X_disk, X=X, norm=args.norm)
            if (lpnorm > args.max_norm + 1e-6).any():
                print(f'At point #{i}, nb examples outside the Lp ball: {(lpnorm > args.max_norm + 1e-6).sum()}')

        acc_surrogate1, loss_surrogate1 = compute_accuracy_from_nested_list_models([list_model_1, ], X=X_disk,
                                                                                   y=y, data=data)
        metrics_list.append(
            {'model': args.path_model_1, 'type_model': 'surrogate', 'x1': x1[i], 'x2': x2[i], 'adv_accuracy': acc_surrogate1,
             'adv_loss': loss_surrogate1})
        acc_surrogate2, loss_surrogate2 = compute_accuracy_from_nested_list_models([list_model_2, ], X=X_disk,
                                                                                   y=y, data=data)
        metrics_list.append(
            {'model': args.path_model_2, 'type_model': 'surrogate', 'x1': x1[i], 'x2': x2[i], 'adv_accuracy': acc_surrogate2,
             'adv_loss': loss_surrogate2})

        for j, model_target in enumerate(models_target):
            acc_target, loss_target = compute_accuracy_from_nested_list_models([[model_target, ], ], X=X_disk,
                                                                               y=y, data=data)
            metrics_list.append(
                {'model': args.path_target[j], 'type_model': 'target', 'x1': x1[i], 'x2': x2[i], 'adv_accuracy': acc_target,
                 'adv_loss': loss_target})

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics['norm'] = args.norm
    df_metrics['max_norm'] = args.max_norm
    df_metrics['n_examples'] = args.n_examples
    df_metrics['n_points'] = args.n_points
    df_metrics['seed'] = args.seed
    df_metrics['xp'] = args.xp

    df_metrics.to_csv(args.csv_export, header=True, index=False)


else:
    raise ValueError("Wrong XP id")
