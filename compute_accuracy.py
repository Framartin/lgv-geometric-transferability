import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.helpers import list_models, guess_and_load_model, guess_method
from utils.data import ImageNet


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll


# parse args
parser = argparse.ArgumentParser(description="Compute the accuracy of an ensemble of models")
parser.add_argument("dir_models", help="Path to directory containing all the models file of the ensemble model")
parser.add_argument("--data-path", default=None, help="Path of data. Only supported for ImageNet.")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Try a lower value if out of memory (especially for high values of --ensemble-inner).")
parser.add_argument("--num-workers", type=int, default=10)
parser.add_argument("--temperature", type=float, default=1, help="Apply temperature scaling.")
parser.add_argument("--validation", type=int, default=None, help="Compute on a validation dataset of provided size (subset of the train test).")
parser.add_argument("--seed", type=int, default=None, help="Random seed. Important to set with validation flag to have the same set.")
parser.add_argument("--csv-export", default=None, help="Path of CSV to export.")
args = parser.parse_args()

if not args.seed and args.validation:
    raise ValueError('Provide random seed for validation set.')

data = ImageNet(batch_size=args.batch_size, path=args.data_path)

path_ensemble = list_models(args.dir_models)

model_ensemble = []
for i, filename in enumerate(path_ensemble):
    model = guess_and_load_model(filename, data=data, load_as_ghost=False, input_diversity=False, skip_gradient=False)
    model.eval()
    model_ensemble.append(model)

num_classes = 1000

traindir = os.path.join(args.data_path, 'train')
valdir = os.path.join(args.data_path, 'val')

val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


if args.validation:
    # create a validation set from the train set
    train_dataset = datasets.ImageFolder(traindir, val_transform)
    # fix generator for reproducibility
    dataset, _ = torch.utils.data.random_split(train_dataset, lengths=[args.validation, len(train_dataset) - args.validation],
                                  generator=torch.Generator().manual_seed(args.seed))
else:
    # use the imagenet val set as test set
    dataset = datasets.ImageFolder(valdir, val_transform)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)


predictions = np.zeros((len(loader.dataset), num_classes))
targets = np.zeros(len(loader.dataset))
k = 0
for input, target in loader:
    input = input.cuda(non_blocking=True)
    # torch.manual_seed(i)
    with torch.no_grad():
        for model in model_ensemble:
            output = model(input)
            cur_preds = output.cpu().numpy() / args.temperature
            predictions[k:k+input.size()[0]] += F.softmax(output, dim=1).cpu().numpy()
        cur_targets = target.numpy()
        targets[k:(k+target.size(0))] = cur_targets
    k += input.size()[0]
predictions = predictions / len(model_ensemble)

test_acc = np.mean(np.argmax(predictions, axis=1) == targets)
test_nll = nll(predictions, targets) / predictions.shape[0]
test_ce = F.cross_entropy(predictions, targets)

print("--- VAL ---" if args.validation else "--- TEST ---")
print(f"Ensemble {args.dir_models} of {len(model_ensemble)} models")
print("  Accuracy:", test_acc)
print("  NLL:", test_nll)
print("  CE:", test_ce)

if args.csv_export:
    df_metrics = pd.DataFrame([{
                'dir_models': args.dir_models,
                'temperature': args.temperature,
                'dataset': 'val' if args.validation else 'test',
                'model_type': guess_method(args.dir_models),
                'accuracy': test_acc,
                'nll': test_nll,
                'nb_ex': len(loader.dataset),
                'n_models': len(model_ensemble),
                'args': ' '.join(sys.argv[1:]),
            },])
    os.makedirs(os.path.dirname(args.csv_export), exist_ok=True)
    df_metrics.to_csv(args.csv_export, mode='a', header=not os.path.exists(args.csv_export), index=False)

