import os
import re
import glob
import argparse
import torch
import numpy as np
from collections import OrderedDict
try:
    from art.classifiers import PyTorchClassifier
except ModuleNotFoundError:
    from art.estimators.classification import PyTorchClassifier
from .models import TorchEnsemble, CifarLeNet, MnistCnn, MnistFc, MnistSmallFc, ModelWithTemperature
from .layers import RandomResizePad
from .utils_sgm import register_hook_for_resnet, register_hook_for_preresnet, register_hook_for_densenet

from pytorch_ensembles import models as pemodels
from utils import modelsghost as ghostmodels
from utils import modelsghostpreresnet as ghostpreresnet
from torchvision import models as tvmodels
import timm
from robustbench.utils import load_model as rb_load_model
from torchvision import transforms
from torch import nn



MCMC_OPTIMIZERS = ['SGLD', 'pSGLD']
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
PEMODELS_NAMES = ['BayesPreResNet110', 'PreResNet110', 'PreResNet164', 'VGG16BN', 'VGG19BN', 'WideResNet28x10',
                          'WideResNet28x10do']
TVMODELS_NAMES = ['BayesResNet50', "resnet50", "resnet18", "resnet152", "resnext50_32x4d", "mnasnet1_0", "densenet121", "densenet201",
                  "mobilenet_v2", "wide_resnet50_2", "vgg19", "inception_v3", "inception_v2", "googlenet"]
TIMODELS_NAMES = ["efficientnet_b0", "adv_inception_v3", "inception_resnet_v2", "tf_inception_v3",
                  "timm_resnet152", "timm_resnext50_32x4d", "timm_wide_resnet50_2", "vit_base_patch16_224"]
# timm_vgg19 timm_densenet201 timm_inception_v3 are imported from torchvision. do not use
RBMODELS_NAMES = ["Salman2020Do_50_2", "Salman2020Do_R50", "Engstrom2019Robustness", "Wong2020Fast", "Salman2020Do_R18"]
ALL_MODELS_NAMES = ['MnistFc', 'MnistSmallFc', 'MnistCnn', 'LeNet'] + PEMODELS_NAMES + TVMODELS_NAMES + TIMODELS_NAMES + RBMODELS_NAMES


class keyvalue(argparse.Action):
    """
    Parse key-values command line argument
    Code from https://www.geeksforgeeks.org/python-key-value-pair-using-argparse/
    """
    # Constructor calling
    def __call__(self, parser, namespace,
                 values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            # split it into key and value
            key, value = value.split('=')
            # assign into dictionary
            getattr(namespace, self.dest)[key] = value


def args2paths(args, index_model=None):
    """
    Create a string corresponding to path to save current model
    :param args: train.py command line arguments namespace
    :param index_model: int for a specific element of ensemble models or for a model sample
    :return: set of pytorch model path and metrics CSV file path
    """
    if args.optimizer in MCMC_OPTIMIZERS:
        filename = f'{index_model:04}.pth'
        filename_metrics = 'metrics.csv'
        relative_path = f'{args.dataset}/{args.architecture}/mcmc_samples/{args.optimizer}_bs{args.batch_size}_lr{args.lr}_lrd{"plateau" if args.lr_decay_on_plateau else args.lr_decay}_psig{args.prior_sigma:.1f}_s{args.samples}_si{args.sampling_interval}_bi{args.burnin}_seed{args.seed}'
    else:  # DNN
        if index_model is None:  # single DNN
            filename = f'model_{args.optimizer}_bs{args.batch_size}_lr{args.lr}_lrd{"plateau" if args.lr_decay_on_plateau else args.lr_decay}_psig{args.prior_sigma:.1f}_ep{args.epochs}_seed{args.seed}.pth'
            filename_metrics = filename.replace('model', 'metrics').replace('.pth', '.csv')
            relative_path = f'{args.dataset}/{args.architecture}/single_model/'
        else:  # Ensemble of DNN
            filename = f'{index_model:04}.pth'
            filename_metrics = 'metrics.csv'
            relative_path = f'{args.dataset}/{args.architecture}/dnn_ensemble/{args.optimizer}_bs{args.batch_size}_lr{args.lr}_lrd{"plateau" if args.lr_decay_on_plateau else args.lr_decay}_psig{args.prior_sigma:.1f}_ep{args.epochs}_seed{args.seed}'
    os.makedirs(os.path.join(args.output, relative_path), exist_ok=True)
    return os.path.join(args.output, relative_path, filename), \
           os.path.join(args.output, relative_path, filename_metrics)


def list_models(models_dir):
    # pretrained model
    if re.match(r"^ImageNet/pretrained/(\w+)$", models_dir):
        return [models_dir, ]
    # path to single model
    if re.match('.+\\.pth?(\\.tar)?$', models_dir):
        if not os.path.isfile(models_dir):
            raise ValueError('Non-existing path surrogate file passed')
        return [models_dir, ]
    # directory of models
    path_models = glob.glob(f'{models_dir}/*.pt')
    path_models.extend(glob.glob(f'{models_dir}/*.pth'))
    path_models.extend(glob.glob(f'{models_dir}/*.pt.tar'))
    path_models.extend(glob.glob(f'{models_dir}/*.pth.tar'))
    path_models = sorted(path_models)
    return path_models


def load_model(path_model, class_model, *args, **kwargs):
    model = class_model(*args, **kwargs)
    model.load_state_dict(torch.load(path_model))
    model.eval()
    return model


def load_list_models(models_dir, class_model, device=None, *args, **kwargs):
    path_models = list_models(models_dir)
    models = []
    for path_model in path_models:
        model = load_model(path_model=path_model, class_model=class_model)
        if device:
            model.to(device)
        models.append(model)
    return models


def guess_model(path_model):
    """
    Return the name of the model
    """
    candidates = [x for x in ALL_MODELS_NAMES if x in path_model]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # pick the longest one
        return max(candidates, key=len)
    raise ValueError('Not able to guess model name')


def guess_and_load_model(path_model, data, load_as_ghost=False, input_diversity=False, skip_gradient=False, defense_randomization=False, temperature=None, force_cpu=False):
    """
    Load model from its path only (guessing the model class)
    :param path_model: str, path to the pt file to load
    :param data: data class
    :param load_as_ghost: load model as a Ghost Network. Only Skip Connection Erosion of resnet on ImageNet supported.
    :param input_diversity: add input diversity as first layer (p=0.5)
    :param skip_gradient: apply Skip Gradient Method with backward hook (gamma=0.5)
    :param defense_randomization: add input diversity as first layer (p=1) to be used as defense
    :param temperature: temperature scaling value. Deactivated if None (default).
    :param force_cpu: don't send model to GPU if True
    :return: pytorch instance of a model
    """
    if load_as_ghost and not (('resnet' in path_model and 'ImageNet' in path_model) or ('PreResNet' in path_model and 'CIFAR10' in path_model)):
        raise ValueError('Ghost Networks only supports resnet on ImageNet, or PreResNet on CIFAR10.')
    if input_diversity and defense_randomization:
        raise ValueError('input_diversity and defense_randomization should not be set at the same time')
    if 'MNIST' in path_model:
        if load_as_ghost:
            raise NotImplementedError('Ghost MNIST models not supported')
        # model from utils/models.py
        if 'CNN' in path_model or 'MnistCnn' in path_model:
            model_name_list = ['MnistCnn']
            model = MnistCnn(pretrained=False, num_classes=data.num_classes)
        elif 'FC' in path_model or 'MnistFc' in path_model:
            model_name_list = ['MnistFc']
            model = MnistFc(pretrained=False, num_classes=data.num_classes)
        elif 'MnistSmallFc' in path_model:
            model_name_list = ['MnistSmallFc']
            model = MnistSmallFc(pretrained=False, num_classes=data.num_classes)
        else:
            raise NotImplementedError('Model class unknown')
        model_loaded = torch.load(path_model, map_location=DEVICE)
        if 'model_state' in model_loaded:
            model_loaded = model_loaded['model_state']
        try:
            model.load_state_dict(model_loaded)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in model_loaded.items():
                name = k[2:]  # remove `1.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
        model = add_normalization_layer(model=model, mean=(0.1307,), std=(0.3081,))
        arch = model_name_list[0]
    elif 'CIFAR' in path_model:
        # ghost models
        if load_as_ghost:
            if 'PreResNet' not in path_model:
                raise ValueError('Only PreResNet ghost models supported on CIFAR10')
            model_name_list = [x for x in PEMODELS_NAMES if x in path_model]
            if len(model_name_list) != 1:
                raise ValueError(f'Failed to extract model name: {model_name_list}')
            arch = getattr(ghostpreresnet, model_name_list[0])
            model = arch.base(num_classes=data.num_classes, **arch.kwargs)
        # model from utils/models.py
        elif 'LeNet' in path_model:
            model_name_list = ['LeNet']
            model = CifarLeNet(pretrained=False, num_classes=data.num_classes)
        # model from pytorch-ensembles
        elif 'BayesPreResNet110' in path_model:
            model_name_list = ['BayesPreResNet110']
            arch = getattr(pemodels, 'BayesPreResNet110')
            model = arch.base(num_classes=data.num_classes, **arch.kwargs)
        elif len([x for x in PEMODELS_NAMES if x in path_model]) >= 1:
            # list model name in pemodels: [x for x in dir(pemodels) if x[0:2] != '__']
            model_name_list = [x for x in PEMODELS_NAMES if x in path_model]
            if len(model_name_list) != 1:
                raise ValueError(f'Failed to extract model name: {model_name_list}')
            arch = getattr(pemodels, model_name_list[0])
            model = arch.base(num_classes=data.num_classes, **arch.kwargs)
        else:
            raise NotImplementedError('Model class unknown')
        model_loaded = torch.load(path_model, map_location=DEVICE)
        if 'model_state' in model_loaded:
            model_loaded = model_loaded['model_state']
        try:
            model.load_state_dict(model_loaded)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in model_loaded.items():
                name = k[2:]  # remove `1.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
        # cSGLD trained on [0,1] data range: nothing to do
        # old dnn_ensemble trained on [-1, 1] data range
        if 'dnn_ensemble' in path_model or 'single_model' in path_model:
            model.to(DEVICE)
            model = add_normalization_layer(model=model, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        arch = model_name_list[0]
    elif 'ImageNet' in path_model:
        if 'ImageNet/pretrained' in path_model:
            if load_as_ghost:
                raise ValueError('Ghost pretrained models not supported')
            a = re.match(r"^ImageNet/pretrained/(\w+)$", path_model)
            if a:
                arch = a.groups()[0]
            else:
                raise ValueError('Failed extracting name of pretrained model')
            if arch in TVMODELS_NAMES:
                model = tvmodels.__dict__[arch](pretrained=True)
                model = add_normalization_layer(model=model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            elif arch in TIMODELS_NAMES:
                arch_ = arch.replace('timm_', '')
                model = timm.create_model(arch_, pretrained=True)
                config = timm.data.resolve_data_config({}, model=model)
                model = add_normalization_layer(model=model, mean=config['mean'], std=config['std'])
            elif arch in RBMODELS_NAMES:
                # https://github.com/RobustBench/robustbench#imagenet
                model = rb_load_model(model_name=arch, model_dir='~/.cache/robustbench/models', dataset='imagenet',
                                      threat_model="Linf")
            else:
                raise ValueError(f'Model {arch} not supported.')
            model.to(DEVICE)
        else:
            checkpoint_dict = torch.load(path_model, map_location=DEVICE)
            source_model = tvmodels
            if load_as_ghost:
                source_model = ghostmodels
            if 'cSGLD' in path_model or 'single_model' in path_model:
                arch = checkpoint_dict['arch']
                if arch in TIMODELS_NAMES:
                    if load_as_ghost:
                        raise ValueError('Ghost timm models not supported')
                    model = timm.create_model(arch, pretrained=False)
                else:
                    model = source_model.__dict__[arch]()
                # try to load state_dir
                # some models were trained with dataparallel, some not.
                try:
                    model.load_state_dict(checkpoint_dict['state_dict'])
                except RuntimeError:
                    #model = torch.nn.parallel.DataParallel(model)
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint_dict['state_dict'].items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model.load_state_dict(new_state_dict)
                if 'withdatanorm' in path_model:
                    # models trained with regular data norm
                    model = add_normalization_layer(model=model, mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
            elif 'subspace_inference' in path_model:
                # all SI models are trained on normalized data
                a = re.match(r".*models(_target)?/ImageNet/(\w+)/.*", path_model)
                if a:
                    arch = a.group(2)
                else:
                    raise ValueError('Failed extracting archiecture from filename model')
                if arch in TIMODELS_NAMES:
                    if load_as_ghost:
                        raise ValueError('Ghost timm models not supported')
                    model = timm.create_model(arch.replace('timm_', ''), pretrained=False)
                else:
                    model = source_model.__dict__[arch]()
                state_dict = checkpoint_dict
                if 'state_dict' in checkpoint_dict:
                    state_dict = checkpoint_dict['state_dict']
                try:
                    model.load_state_dict(state_dict)
                except RuntimeError:
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model.load_state_dict(new_state_dict)
                model.to(DEVICE)
                # add normalization layer at first
                model = add_normalization_layer(model=model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            elif ('deepens_imagenet' in path_model) or ('FGE' in path_model) or ('SSE' in path_model):
                # pretrained resnet50 from pytorch-ensemble
                arch = 'resnet50'
                model = source_model.__dict__[arch]()
                # models trained with dataparallel
                #model = torch.nn.parallel.DataParallel(model)
                new_state_dict = OrderedDict()
                for k, v in checkpoint_dict['state_dict'].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                model.to(DEVICE)
                # add normalization layer at first
                model = add_normalization_layer(model=model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            elif 'SWAG' in path_model:
                # train by us from 1 deepens_imagenet checkpoint resnet50
                arch = 'resnet50'
                model = source_model.__dict__[arch]()
                if 'state_dict' in checkpoint_dict:
                    checkpoint_dict = checkpoint_dict['state_dict']
                model.load_state_dict(checkpoint_dict)
                model.to(DEVICE)
                # add normalization layer at first
                model = add_normalization_layer(model=model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            elif 'VI' in path_model:
                arch = 'BayesResNet50'
                model = pemodels.__dict__[arch]()
                new_state_dict = OrderedDict()
                model.load_state_dict(checkpoint_dict)
                model.to(DEVICE)
                # add normalization layer at first
                model = add_normalization_layer(model=model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                raise ValueError('ImageNet model not recognized.')
        if 'inception' in path_model:
            # resize input to to 299
            model = add_resize_layer(model=model, size=(299, 299))
    else:
        raise ValueError('dataset not supported')
    model.eval()
    # to GPU
    if USE_CUDA and not force_cpu:
        model.to(DEVICE)
    if input_diversity:
        # add as first layer that randomly resize between [90%, 100%] and 0-pad to original size with probability 0.5
        model = add_random_resize_layer(model=model, p=0.5, min_resize=round(data.get_input_shape()[2]*0.9))
        model.to(DEVICE)
        model.eval()
    if skip_gradient:
        if arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            register_hook_for_resnet(model, arch=arch, gamma=0.2)
        elif arch in ['PreResNet110']:
            gamma = float(os.getenv('ADV_TRANSFER_SGM_GAMMA_PRERESNET', '0.7').replace(',', '.'))
            register_hook_for_preresnet(model, arch=arch, gamma=gamma)
        elif arch in ['densenet121', 'densenet169', 'densenet201']:
            register_hook_for_densenet(model, arch=arch, gamma=0.5)
        else:
            raise ValueError(f'Arch { arch } not supported by Skip Gradient Method')
    if defense_randomization:
        # add as first layer that randomly resize between [90%, 100%] and 0-pad to original size with probability 1.
        # defense by https://arxiv.org/pdf/1711.01991.pdf
        model = add_random_resize_layer(model=model, p=1, min_resize=round(data.get_input_shape()[2]*0.9))
        model.to(DEVICE)
        model.eval()
    if temperature:
        model = ModelWithTemperature(model, temperature=temperature)
        model.to(DEVICE)
        model.eval()
    return model


def load_classifier(model, data):
    """
    Load ART PyTorch classifier from pytorch model
    :param model: pytorch model instance
    :param data: data class
    :return: ART classifier
    """
    # not used but mandatory for ART
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(data.min_pixel_value, data.max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=data.get_input_shape(),
        nb_classes=data.num_classes,
        device_type="gpu" if USE_CUDA else "cpu"
    )
    #classifier.set_learning_phase(False) # automatically set by predict() calls
    return classifier


def load_classifier_ensemble(models, **kwargs):
    """
    Build an ART classifier of an ensemble of PyTorch models
    :param models: list of pytorch model instances
    :return:
    """
    model = TorchEnsemble(models=models, ensemble_logits=True)
    if USE_CUDA:
        model.to(DEVICE)
    model.eval()
    return load_classifier(model, **kwargs)


def predict_ensemble(models_dir, X, data):
    """
    Compute prediction for each model inside models_dir

    :param models_dir: str path to pytorch's models
    :param X: pytorch tensor or numpy array
    :param data: data class instance
    :return: tuple of 2 numpy arrays of predicted labels with probs and logits ensembling
    """
    torchdataset = torch.utils.data.TensorDataset(torch.Tensor(X))
    loader = torch.utils.data.DataLoader(
        torchdataset,
        batch_size=data.batch_size, shuffle=False,
        num_workers=data.num_workers, pin_memory=USE_CUDA)
    #if not torch.is_tensor(X):
    #    X = torch.Tensor(X)
    #if USE_CUDA:
    #    X = X.to(DEVICE)
    path_models = list_models(models_dir)
    y_pred_ens_logit = torch.zeros((X.shape[0], data.num_classes))
    y_pred_ens_prob = torch.zeros((X.shape[0], data.num_classes))
    for path_model in path_models:
        model = guess_and_load_model(path_model=path_model, data=data)
        for i, (inputs,) in enumerate(loader):
            inputs = inputs.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                output = model(inputs)
            y_pred_ens_logit[i*data.batch_size:(i+1)*data.batch_size, :] += output.cpu()
            y_pred_ens_prob[i*data.batch_size:(i+1)*data.batch_size, :] += torch.nn.functional.softmax(output, dim=1).cpu()
        # clean
        del model, output
        if USE_CUDA:
            torch.cuda.empty_cache()
    y_pred_ens_logit /= len(path_models)
    y_pred_ens_logit = torch.nn.functional.softmax(y_pred_ens_logit, dim=1)
    y_pred_ens_prob /= len(path_models)
    label_pred_logit = np.argmax(y_pred_ens_logit.numpy(), axis=1)
    label_pred_prob = np.argmax(y_pred_ens_prob.numpy(), axis=1)
    return label_pred_prob, label_pred_logit

def compute_accuracy_ensemble(models_dir, X, y, data):
    label_pred_prob, label_pred_logit = predict_ensemble(models_dir=models_dir, X=X, data=data)
    acc_prob = (label_pred_prob == y).mean()
    acc_logit = (label_pred_logit == y).mean()
    return acc_prob, acc_logit


def compute_accuracy_multiple_ensemble(models_dirs, X, y, data):
    """
    Compute the mean of the accuracies of several ensembles.
    """
    acc_prob, acc_logit = 0., 0.
    for models_dir in models_dirs:
        acc_prob_tmp, acc_logit_tmp = compute_accuracy_ensemble(models_dir=models_dir, X=X, y=y, data=data)
        acc_prob += acc_prob_tmp
        acc_logit += acc_logit_tmp
    return acc_prob / len(models_dirs), acc_logit / len(models_dirs)


def compute_accuracy_from_nested_list_models(list_ensemble, X, y, data, export_predict=False, export_mask=None, tol=1e-12):
    """
    Compute prediction for each model inside list_ensemble

    :param list_ensemble: list of list of pytorch's models
    :param X: pytorch tensor or numpy array
    :param y: pytorch tensor or numpy array
    :param data: data class instance
    :param export_predict: bool, also export a boolean vector corresponding to correct predictions of examples
    :param export_mask: tensor, also export metrics computed on examples masked by provided tensor
    :param loss: str either 'CE' or 'NLL'
    :return: tuple of 2 scalar accuracy and CE loss
    """
    #nb_models = np.sum([len(x) for x in list_ensemble])  # total nb of models
    if export_mask is not None:
        if export_mask.shape != (X.shape[0], ):
            raise ValueError('Wrong shape for export_mask')
        if not np.isin(export_mask.numpy(), [0, 1]).all():
            raise ValueError('export_mask tensor should contain only 0,1 values')
        export_mask.bool().to(DEVICE)
    torchdataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(y))
    loader = torch.utils.data.DataLoader(
        torchdataset,
        batch_size=data.batch_size, shuffle=False,
        num_workers=data.num_workers, pin_memory=USE_CUDA)
    loss = nn.NLLLoss(reduction='sum').to(DEVICE)
    correct = 0
    total = 0
    loss_sum = 0.
    predict_correct = torch.zeros((0,))
    log_probs = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            y_pred_ens_batch = torch.zeros(inputs.size(0), data.num_classes).to(DEVICE, non_blocking=True)
            size_ensemble = 0
            log_prob_models_batch = []
            for list_model in list_ensemble:
                for model in list_model:
                    output = model(inputs)
                    # more numerically stable to save log_softmax then log sum exp
                    log_prob_model_batch = torch.nn.functional.log_softmax(output, dim=1)
                    log_prob_models_batch.append(log_prob_model_batch)
            # log sum exp of log probs, over models dims for each example and classes
            log_prob_ens_batch = torch.logsumexp(torch.dstack(log_prob_models_batch), dim=2) - np.log(len(log_prob_models_batch))
            loss_sum += loss(log_prob_ens_batch, labels.long()).item()  # NLLLoss takes log prob
            _, predicted = torch.max(log_prob_ens_batch.data, 1)
            correct += (predicted == labels).sum().item()
            predict_correct = torch.cat((predict_correct, (predicted == labels).cpu()), 0)
            total += labels.size(0)
    if export_predict:
        if predict_correct.shape[0] != X.shape[0]:
            raise RuntimeError('Unexpected shape of predict_correct vector')
        return correct / total, loss_sum / total, predict_correct.bool()
    if export_mask is not None:
        correct_masked = torch.masked_select(predict_correct, mask=export_mask).sum().cpu().item()
        total_masked = export_mask.sum().cpu().item()
        if total_masked == 0:
            total_masked = np.NaN
        return correct / total, loss_sum / total, correct_masked / total_masked
    return correct / total, loss_sum / total


def save_numpy(array, path, filename):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, filename), array)

def flatten(X):
    return X.reshape((X.shape[0], -1))

def compute_norm(X_adv, X, norm=2):
    return np.linalg.norm(flatten(X_adv) - flatten(X), ord=norm, axis=1)

def project_on_sphere(X, X_adv, data, size=4., norm=2):
    """
    Project on sphere (not the ball!) of specified size
    :param X: np array
    :param X_adv: np array
    :param size:
    :param norm: Lp norm to use. Only 2 implemented
    :return:
    """
    if norm != 2:
        raise NotImplementedError('Only L2 norm implemented')
    lpnorm = compute_norm(X_adv, X, norm=2)
    X_adv_proj = X + size / lpnorm.reshape((X.shape[0], 1, 1, 1)) * (X_adv - X)
    X_adv_proj = np.clip(X_adv_proj, data.min_pixel_value, data.max_pixel_value)
    return X_adv_proj

def add_normalization_layer(model, mean, std):
    return torch.nn.Sequential(
        transforms.Normalize(mean=mean, std=std),
        model
    )

def add_resize_layer(model, size, **kwargs):
    return torch.nn.Sequential(
        transforms.Resize(size=size, **kwargs),
        model
    )

def add_random_resize_layer(model, p=0.5, min_resize=200):
    """
    Add input diversity layer as first layer

    :param model: pytorch model
    :param p: probability to apply input diversity
    :param min_resize: minimum possible resize
    :return:
    """
    return torch.nn.Sequential(
        transforms.RandomApply([
            RandomResizePad(min_resize),
        ], p=p),
        model
    )


def guess_method(path_model):
    """
    Return the name of the method to train the model
    """
    METHODS_OTHER = ['SGLD', 'cSGLD', 'pSGLD', 'HMC', 'SVI', 'VI', 'SWAG', 'FGE', 'SSE', 'SI/pca/ESS', 'SI/random/ESS', 'SWA',
                     'collected_models', 'cSGD', 'ImageNet/pretrained']
    METHODS_DNN = ['SGD', 'Adam', 'RMSprop', 'DNN', 'dnn_ensemble', 'single_model', 'deepens']  # always return 'dnn'
    candidates = [x for x in METHODS_OTHER if x in path_model]
    if len(candidates) > 1:
        # pick the longest one
        return max(candidates, key=len)
    elif len(candidates) == 1:
        return candidates[0]
    candidates = [x for x in METHODS_DNN if x in path_model]
    if len(candidates) >= 1:
        return 'dnn'
    raise ValueError('Not able to guess model training method')