"""
Implementation of the attacks used in the article
"""

import numpy as np
import pandas as pd
import torch
import argparse
import time
import os
import sys
import re
from tqdm import tqdm
import random
from random import shuffle
from utils.data import CIFAR10, CIFAR100, ImageNet, MNIST
from utils.helpers import keyvalue, guess_model, guess_and_load_model, load_classifier, load_classifier_ensemble, list_models, \
    compute_accuracy_from_nested_list_models, save_numpy, compute_norm, guess_method, USE_CUDA, DEVICE
from utils.attacks import ExtendedProjectedGradientDescentPyTorch
from art.attacks.evasion import CarliniLInfMethod
from utils.models import LightNestedEnsemble
from torchattacks import Square, CW, AutoAttack, OnePixel, PGD, PGDL2, APGD, FAB, DeepFool, MultiAttack
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn as cudnn


cudnn.benchmark = True
cudnn.deterministic = True

# parse args
parser = argparse.ArgumentParser(description="Craft PGD adv ex with each update computed on a different samples from an ensemble of models trained with cSGLD")
parser.add_argument("dirs_models", nargs='+', help="Path to directory containing all the models file of the ensemble model. Also support single path to a model file.")
parser.add_argument('--attack', choices=['PGD', 'PGD_ta', 'APGD', 'FAB', 'Square', 'AutoAttack', 'CW', 'OnePixel', 'DeepFool'], default='PGD', help="Attack to craft adversarial examples. Only PGD supports momentum, .")
parser.add_argument('--n-iter', type=int, default=None, help="Number of iterations to perform. If None (default), set to the number of samples.")
parser.add_argument("--norm", choices=['1', '2', 'inf'], default='2', help="Type of L-norm to use. Default: 2")
parser.add_argument("--max-norm", type=float, required=True, help="Max L-norm of the perturbation")
parser.add_argument("--norm-step", type=float, required=True, help="Max norm at each step.")
parser.add_argument('--n-ensemble', type=int, default=1, help="Number of samples to ensemble. Default: 1")
parser.add_argument('--shuffle', action='store_true', help="Random order of models vs sequential order of the MCMC (default)")
parser.add_argument('--n-random-init', type=int, default=0, help="Number of random restarts to perform. 0: no random init.")
parser.add_argument('--grad-noise-std', type=float, default=None, help="Add Gaussian noise to gradients with the specified standard deviation.")
parser.add_argument('--temperature', type=float, default=None, help="Temperature scaling the logits of the surrogate model. Deactivated if None (default).")

parser.add_argument('--skip-first-n-models', type=int, default=0, help="Number of models samples to discard")
parser.add_argument('--n-models-cycle', type=int, help="Number of models samples per cycle (only used for limit-n-samples-per-cycle or limit-n-cycles)")
parser.add_argument('--limit-n-samples-per-cycle', type=int, default=None, help="Takes into account only the first n samples inside a cycle, droping off the last ones. Default: None (desactivated)")
parser.add_argument('--method-samples-per-cycle', choices=['interval', 'true_interval', 'first', 'last'], default='interval', help="Method to select samples inside cycle. Use interval for cycle based surrogate, true_interval for non-cyclical surrogate.")
parser.add_argument('--limit-n-cycles', type=int, default=None, help="Takes into account only the first n cycles, droping off the last ones. Default: None (desactivated)")

# test time transferability improvements
parser.add_argument('--ghost-attack', action='store_true', help="Load each model as a Ghost network (default: no model alteration)")
parser.add_argument('--input-diversity', action='store_true', help="Add input diversity to each model (default: no model alteration)")
parser.add_argument('--skip-gradient-method', action='store_true', help="Add Skip Gradient Method (SGM) backward hook to each surrogate model (default: no model alteration)")
parser.add_argument('--translation-invariant', action='store_true', help="Apply translation invariance kernel to gradient (default: regular gradient)")
parser.add_argument('--target-defense-randomization', action='store_true', help="The target model is loaded with defense randomization (default: regular target). Set to True with --translation-invariant.")
parser.add_argument("--momentum", type=float, default=None, help="Apply momentum to gradients (default: regular gradient)")

# target model
parser.add_argument("--model-target-path", nargs='+', default=None, help="Path to the target models.")
parser.add_argument("--csv-export", default=None, help="Path to CSV where to export data about target.")
parser.add_argument("--csv-key-val", nargs='*', metavar="KEY=VALUE", action=keyvalue, help="Add the keys as columns with the corresponding values to the exported CSV.")
parser.add_argument("--export-target-per-iter", type=int, default=None, help="Export target acc each N iterations in csv-export file. Default (None) 1 line for final data.")

# others
parser.add_argument("--n-examples", type=int, default=None, help="Craft adv ex on a subset of test examples. If None "
                                                                 "(default), perturbate all the test set. If "
                                                                 "model-target-path is set, extract the subset from "
                                                                 "the examples correctly predicted by it.")
parser.add_argument("--data-path", default=None, help="Path of data. Only supported for ImageNet.")
parser.add_argument("--validation", action='store_true', help="Craft adversarial examples from a validation set built from train set (of size: 2 x n_examples). Default: no validation set, examples from test set.")
parser.add_argument("--seed", type=int, default=None, help="Set random seed")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Try a lower value if out of memory (especially for high values of --ensemble-inner).")
parser.add_argument("--force-add", type=float, default=None, help="Add this scalar to the example. Use for compatibility with model trained on other range of pixels")
parser.add_argument("--force-divide", type=float, default=None, help="Divide the example ex by this scalar. Use for compatibility with model trained on other range of pixels")
parser.add_argument("--skip-accuracy-computation", action='store_true', help="Do not compute accuracies. To be used for full test set.")

args = parser.parse_args()
if args.norm == 'inf':
    args.norm = np.inf
else:
    args.norm = int(args.norm)

# check args
if args.limit_n_samples_per_cycle or args.limit_n_cycles:
    if not args.n_models_cycle:
        raise ValueError("If a limit is set in the number of models to consider, you have to precise the number of samples per cycle.")
if args.validation and not args.n_examples:
    raise ValueError('For validation set, please provide its size with n-examples arg')

# set random seed
if args.seed:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# detect models
if re.match('.+\\.pth?(\\.tar)?$', args.dirs_models[0]) and len(args.dirs_models) == 1:
    # link to single model
    if not os.path.isfile(args.dirs_models[0]):
        raise ValueError('Non-existing path surrogate file passed')
    paths_ensembles = [[args.dirs_models[0], ], ]
else:
    paths_ensembles = [list_models(x) for x in args.dirs_models]
print(f'Ensembles of models detected: {[len(x) for x in paths_ensembles]}')
if args.skip_first_n_models:
    print(f'Discarding the first {args.skip_first_n_models} models')
    paths_ensembles = [x[args.skip_first_n_models:] for x in paths_ensembles]
if any([len(x) == 0 for x in paths_ensembles]):
    raise ValueError('Empty model ensemble')
if args.n_models_cycle:
    if any([len(x) % args.n_models_cycle != 0 for x in paths_ensembles]):
        print('Warning: Number of models is not a multiple of the number of models per cycle')
    if args.limit_n_cycles:
        if any([len(x) < args.limit_n_cycles * args.n_models_cycle for x in paths_ensembles]):
            raise ValueError(f'One of the ensemble is smaller than what expected ({ args.limit_n_cycles * args.n_models_cycle })')
    if args.limit_n_samples_per_cycle:
        if args.limit_n_samples_per_cycle > args.n_models_cycle:
            raise ValueError('Limit to nb samples > nb of samples per cycle.')

# load test/val data
validation_size = 2 * args.n_examples if args.validation else None  # if validation data, set the size of val dataset 2xn_examples (to have enough correctly predicted examples)
if 'CIFAR100' in args.dirs_models[0]:
    data = CIFAR100(batch_size=args.batch_size, validation=validation_size, seed=args.seed)
elif 'CIFAR10' in args.dirs_models[0]:
    data = CIFAR10(batch_size=args.batch_size, validation=validation_size, seed=args.seed)
elif 'ImageNet' in args.dirs_models[0]:
    data = ImageNet(batch_size=args.batch_size, path=args.data_path, validation=validation_size, seed=args.seed)
elif 'MNIST' in args.dirs_models[0]:
    data = MNIST(batch_size=args.batch_size, validation=validation_size, seed=args.seed)
else:
    raise NotImplementedError('Dataset not supported')

model_target = None
if args.model_target_path:
    # load target and select n_examples correctly predicted by it
    # target model is loaded with randomization defense for translation invariance
    models_target = [guess_and_load_model(path_model=x, data=data, defense_randomization=(args.translation_invariant or args.target_defense_randomization)) for x in args.model_target_path]
    X, y = data.correctly_predicted_to_numpy(models=models_target, train=False, validation=args.validation, N=args.n_examples, seed=args.seed)
else:
    X, y = data.to_numpy(train=False, validation=args.validation, N=args.n_examples, seed=args.seed)

if args.force_add:
    X += args.force_add
    data.min_pixel_value += args.force_add
    data.max_pixel_value += args.force_add
if args.force_divide:
    X /= args.force_divide
    data.min_pixel_value /= args.force_divide
    data.max_pixel_value /= args.force_divide

# limit cycles or samples per cycles
if args.limit_n_cycles or args.limit_n_samples_per_cycle:
    paths_ensembles_lim = []
    for i_ens, paths_models in enumerate(paths_ensembles):
        paths_ensembles_lim.append([])
        for i, path_model in enumerate(paths_models):
            # stop if limit is set on the number of cycles to consider
            if args.limit_n_cycles:
                if i >= args.limit_n_cycles * args.n_models_cycle:
                    break
            # only add current model for selected indexes
            if args.limit_n_samples_per_cycle:
                # select index (at regular interval, always including the last)
                max_index = args.n_models_cycle-1
                if args.method_samples_per_cycle == 'interval':
                    indexes_to_keep = [int(x.left) for x in pd.interval_range(start=0, end=max_index, periods=args.limit_n_samples_per_cycle-1)] + [max_index]
                elif args.method_samples_per_cycle == 'true_interval':
                    indexes_to_keep = [int(x.left) for x in pd.interval_range(start=0, end=max_index+1, periods=args.limit_n_samples_per_cycle)]
                elif args.method_samples_per_cycle == 'last':
                    indexes_to_keep = list(range(max_index - args.limit_n_samples_per_cycle+1, max_index+1))
                elif args.method_samples_per_cycle == 'first':
                    indexes_to_keep = list(range(0, args.limit_n_samples_per_cycle))
                else:
                    raise NotImplementedError('Method not supported.')
                if (i % args.n_models_cycle) not in indexes_to_keep:
                    continue
            paths_ensembles_lim[i_ens].append(path_model)
    paths_ensembles = paths_ensembles_lim

if any([len(x) != len(paths_ensembles[0]) for x in paths_ensembles]):
    raise NotImplementedError('All ensembles should have the same number of models.')
print(f'Ensembles of models used: {[len(x) for x in paths_ensembles]}')


# shuffle models
if args.shuffle:
    for paths_models in paths_ensembles:
        shuffle(paths_models)

# don't load unused models (if nb models > nb iters)
if args.n_iter:
    max_nb_models_used = args.n_iter * args.n_ensemble
    for i, paths_models in enumerate(paths_ensembles):
        if len(paths_models) > max_nb_models_used:
            paths_ensembles[i] = paths_models[:max_nb_models_used]

if len(args.dirs_models) > 1 and args.n_ensemble > 1:
    raise ValueError('Attacking multiple ensembles doesn\'t support n-ensemble arg.')

# create nested list of models (ensemble > model)
# [ens1: [m1, m2, m3, m4], ens2: [m5, m6, m7, m8]]
ensemble_list = []
for i, path_model in enumerate(paths_ensembles[0]):
    # if we have multiple MCMC chains, we ensemble
    if len(paths_ensembles) > 1:
        ensemble_list.append([x[i] for x in paths_ensembles])
    else:
        # if args.n_ensemble, we ensemble models from the same MCMC chain
        if len(ensemble_list) == 0:
            # avoid IndexError at first iteration
            ensemble_list.append([path_model, ])
        elif len(ensemble_list[-1]) >= args.n_ensemble:
            ensemble_list.append([path_model, ])
        else:
            ensemble_list[-1].append(path_model)

# load each models and create ART classifier
ensemble_classifiers = []  # list of ART classifiers. Each one has the logits fused
list_ensemble_models = []  # nested list of torch models
for i, ensemble_path in enumerate(ensemble_list):
    # only 1 model to attack
    if len(ensemble_path) == 1:
        model = guess_and_load_model(ensemble_path[0], data=data, load_as_ghost=args.ghost_attack, input_diversity=args.input_diversity, skip_gradient=args.skip_gradient_method, temperature=args.temperature)
        classifier = load_classifier(model, data=data)
        list_ensemble_models.append([model])
    # if ensembling, store path_model to a list and build the ensembling model
    else:
        models_to_ensemble = []
        for j, path_model in enumerate(ensemble_path):
            # load next model and continue only if ensemble is done
            models_to_ensemble.append(guess_and_load_model(path_model, data=data, load_as_ghost=args.ghost_attack, input_diversity=args.input_diversity, temperature=args.temperature, force_cpu=False))
        classifier = load_classifier_ensemble(models_to_ensemble, data=data)
        list_ensemble_models.append(models_to_ensemble)
    ensemble_classifiers.append(classifier)
    del classifier

# compute benign acc
if not args.skip_accuracy_computation:
    acc_ens_prob, loss_ens_prob, predict_correct_ens = compute_accuracy_from_nested_list_models(list_ensemble=list_ensemble_models, X=X, y=y, data=data, export_predict=True)
    print(f"Accuracy on ensemble benign test examples: {acc_ens_prob*100:.3f}%   (loss: {loss_ens_prob:.3f}).")

# time code
if USE_CUDA:
    torch.cuda.synchronize()
start_time = time.perf_counter()

if args.attack == 'PGD':
    attack = ExtendedProjectedGradientDescentPyTorch(
        estimators=ensemble_classifiers, targeted=False, norm=args.norm, eps=args.max_norm, eps_step=args.norm_step,
        max_iter=args.n_iter, num_random_init=args.n_random_init, batch_size=args.batch_size,
        translation_invariant=args.translation_invariant, momentum=args.momentum, grad_noise_std=args.grad_noise_std,
        models_target_dict={name: models_target[i] for i,name in enumerate(args.model_target_path)} if args.export_target_per_iter else None,
        freq_eval_target=args.export_target_per_iter,
        data=data
    )
    X_adv = attack.generate(x=X, y=y)
elif args.attack == 'CW' and args.norm == np.inf:
    ensemble_models = LightNestedEnsemble(list_models=list_ensemble_models, order=None)
    ensemble_classifier = load_classifier(ensemble_models, data=data)
    attack = CarliniLInfMethod(
        classifier=ensemble_classifier, targeted=False, eps=args.max_norm, max_iter=args.n_iter,
        batch_size=args.batch_size, learning_rate=0.01
    )
    X_adv = attack.generate(x=X, y=y)
else:
    # attacks from torchattacks
    ensemble_models = LightNestedEnsemble(list_models=list_ensemble_models, order=None)  # we take care of the order before
    norm_ta = f'L{args.norm}'
    if args.attack == 'PGD_ta' and norm_ta == 'Linf':
        attacks_list = [PGD(ensemble_models, eps=args.max_norm, alpha=args.norm_step, steps=args.n_iter, random_start=args.n_random_init > 0) for x in range(max(1, args.n_random_init))]
        attack = MultiAttack(attacks_list)
    elif args.attack == 'APGD':
        attack = APGD(ensemble_models, norm=norm_ta, eps=args.max_norm, steps=args.n_iter, n_restarts=args.n_random_init, loss='ce', seed=args.seed)
    elif args.attack == 'FAB':
        attack = FAB(ensemble_models, norm=norm_ta, eps=args.max_norm, steps=args.n_iter, n_restarts=args.n_random_init, seed=args.seed, n_classes=data.num_classes)
    elif args.attack == 'Square':
        attack = Square(ensemble_models, norm=norm_ta, eps=args.max_norm, n_queries=args.n_iter, n_restarts=1, loss='ce', seed=args.seed)
    elif args.attack == 'AutoAttack':
        attack = AutoAttack(ensemble_models, norm=norm_ta, eps=args.max_norm, n_classes=data.num_classes, seed=args.seed)
    elif args.attack == 'DeepFool':
        if args.norm != 2:
            raise ValueError('Only L2 norm supported for DeepFool attack')
        print('Warming: max-norm ignored for DeepFool attack!')
        attack = DeepFool(ensemble_models, steps=args.n_iter)
    elif args.attack == 'CW':
        if args.norm != 2:
            raise ValueError('Only L2 norm supported for CW attack')
        print('Warming: max-norm ignored for CW attack!')
        attacks_list = [CW(ensemble_models, c=c, steps=1000, lr=0.1, kappa=30) for c in [0.1, 1, 10, 100]]
        attack = MultiAttack(attacks_list)
    elif args.attack == 'OnePixel':
        print('Warming: norm ignored for OnePixel attack, max_norm used as nb pixels!')
        attack = OnePixel(ensemble_models, pixels=args.max_norm, steps=75, popsize=400, inf_batch=args.batch_size)
    else:
        raise NotImplementedError(f'Attack not implemented.')
    # implement batchsize
    X_dataset = TensorDataset(torch.tensor(X).to(DEVICE), torch.tensor(y).to(DEVICE))
    X_loader = DataLoader(X_dataset, batch_size=args.batch_size, shuffle=False)
    X_adv = np.zeros((0,)+data.get_input_shape()[1:])
    for X_batch, y_batch in tqdm(X_loader, desc='Batch'):
        X_adv_batch = attack(X_batch, y_batch).detach().cpu().numpy()
        X_adv = np.vstack((X_adv, X_adv_batch))
    if X.shape != X_adv.shape:
        raise RuntimeError(f'X and X_adv do not have the same shape: {X.shape} ; {X_adv.shape}')


if USE_CUDA:
    torch.cuda.synchronize()
end_time = time.perf_counter()

model_name_list = [guess_model(x) for x in args.dirs_models]

# print stats
if not args.skip_accuracy_computation:
    acc_ens_prob_adv, loss_ens_prob_adv = compute_accuracy_from_nested_list_models(list_ensemble=list_ensemble_models, X=X_adv, y=y, data=data)
    lpnorm = compute_norm(X_adv=X_adv, X=X, norm=args.norm)
    print(
        f"Surrogate stats after {args.n_iter} iters: Accuracy: {acc_ens_prob_adv * 100:.3f}%, Loss: {loss_ens_prob_adv:.3f} (from {loss_ens_prob:.3f}), "
        f"L{args.norm}-norm: mean {lpnorm.mean():.5f} (min {lpnorm.min():.5f} max {lpnorm.max():.5f}), Nb examples: {X_adv.shape[0]}, "
        f"Time: {(end_time - start_time) / 60:.3f} min")
    if args.csv_export:
        if not args.model_target_path:
            raise ValueError('Target model should be specified to export CSV.')
        for i, model_target in enumerate(models_target):
            acc_target_adv, loss_target_adv, acc_target_adv_ensok = compute_accuracy_from_nested_list_models([[model_target,],], X=X_adv, y=y, data=data, export_mask=predict_correct_ens)
            # transfer_rate_target = 1 - accuracy on adversarial examples predicted correctly both by the target and the surrogate
            transfer_rate_target = 1 - acc_target_adv_ensok
            nb_examples_transfer_rate = predict_correct_ens.sum().cpu().item()
            acc_target_original, loss_target_original = compute_accuracy_from_nested_list_models([[model_target,],], X=X, y=y, data=data)
            print(f'* On target: { args.model_target_path[i] }')
            print(f"   Attack success rate: {(1-acc_target_adv) * 100:.3f} %         (transfer rate: {transfer_rate_target * 100:.3f}% on {nb_examples_transfer_rate} examples)")
            print(f"   Loss on target: {loss_target_adv:.3f} (vs. original {loss_target_original:.3f})")
            dict_metrics = args.csv_key_val.copy() if args.csv_key_val else dict()
            dict_metrics.update({
                'model_target': f"{'defense_randomization/' if (args.translation_invariant or args.target_defense_randomization) else ''}{args.model_target_path[i]}",
                'arch_target': guess_model(args.model_target_path[i]),
                'model_surrogate': args.dirs_models[0],
                'surrogate_type': guess_method(args.dirs_models[0]),
                'surrogate_archs': '_'.join(model_name_list),
                # 'surrogate_size_ensembles': len(paths_ensembles[0]),  # nb models per arch
                'surrogate_size_ensembles': args.n_models_cycle * args.limit_n_cycles if args.limit_n_cycles else len(paths_ensembles[0]),  # nb models per arch
                'norm_type': args.norm,
                'norm_max': args.max_norm,
                'norm_step': args.norm_step,
                'n_iter': args.n_iter,
                'n_ensemble': args.n_ensemble,
                'n_random_init': args.n_random_init,
                'momentum': args.momentum,
                'shuffle': args.shuffle,
                'ghost': args.ghost_attack,
                'input_diversity': args.input_diversity,
                'translation_invariant': args.translation_invariant,
                #'adv_fail_rate': acc_target_adv,  # X contains only correctly predicted examples
                'adv_success_rate': 1-acc_target_adv,
                'transfer_rate': transfer_rate_target,
                'loss_adv': loss_target_adv,
                'loss_original': loss_target_original,
                'adv_norm_mean': lpnorm.mean(),
                'adv_norm_min': lpnorm.min(),
                'adv_norm_max': lpnorm.max(),
                'limit_samples_cycle': args.limit_n_samples_per_cycle,
                'limit_cycles': args.limit_n_cycles,
                'surrogate_acc_original_ex': acc_ens_prob,
                'surrogate_acc_adv_ex': acc_ens_prob_adv,
                'surrogate_loss_original_ex': loss_ens_prob,
                'surrogate_loss_adv_ex': loss_ens_prob_adv,
                'target_acc_original_ex': acc_target_original,
                'acc_original_ex': acc_ens_prob,
                'nb_adv': X_adv.shape[0],
                'nb_adv_transf_rate': nb_examples_transfer_rate,  # different nb of examples to compute the transfer rate
                'dataset': 'val' if args.validation else 'test',
                'time': end_time - start_time,
                'transferability_techniques': f"{'MI_' if args.momentum else ''}{'ghost_' if args.ghost_attack else ''}{'DI_' if args.input_diversity else ''}{'TI_' if args.translation_invariant else ''}{'SGM_' if args.skip_gradient_method else ''}",
                'grad_noise_std': args.grad_noise_std,
                'temperature': args.temperature,
                'attack': args.attack,
                'args': ' '.join(sys.argv[1:]),
            })
            df_metrics = pd.DataFrame([dict_metrics, ])
            if args.export_target_per_iter:
                stats_target_dict = attack.get_target_accuracy_per_iter(args.model_target_path[i])
                # duplicate the df line to the number of iterations
                df_metrics = pd.concat([df_metrics] * len(stats_target_dict['acc']), ignore_index=True)
                df_metrics['n_iter'] = stats_target_dict['iter']
                df_metrics['adv_fail_rate'] = stats_target_dict['acc']
                df_metrics['adv_success_rate'] = 1 - df_metrics['adv_fail_rate']
                df_metrics['loss_adv'] = stats_target_dict['loss']
            # create dir and append one line to csv
            os.makedirs(os.path.dirname(args.csv_export), exist_ok=True)
            df_metrics.to_csv(args.csv_export, mode='a', header=not os.path.exists(args.csv_export), index=False)

