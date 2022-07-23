import os
import logging
import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
from torchvision import transforms
from .helpers import list_models, guess_and_load_model, DEVICE


def check_args(method):
    def inner(ref, **kwargs):
        if kwargs.get('validation', False) and not ref.valset:
            raise ValueError('Trying to call validation without creating it')
        return method(ref, **kwargs)
    return inner


class DataBase:
    trainset = valset = testset = None
    transform_train = transform_test = None
    trainloader = None
    valloader = None
    testloader = None
    transform = None
    classes = None
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, batch_size=64, num_workers=0, validation=None, normalize=False, seed=None):
        """
        Base class for data

        :param batch_size: Batch size for data loader.
        :param num_workers: Number of workers.
        :param validation: Split the training set into a training set and into a validation set.
        :param normalize: False, data in [0,1] (default) for adversarial crafting; True normalize data for training.
        :param seed: Random seed.
        """
        if hasattr(self, 'min_pixel_value') + hasattr(self, 'max_pixel_value') < 2:
            self.min_pixel_value = 1e8
            self.max_pixel_value = -1e8
            for images, _ in self.testloader:
                min_pixel = torch.min(images)
                max_pixel = torch.max(images)
                if min_pixel < self.min_pixel_value:
                    self.min_pixel_value = min_pixel
                if max_pixel > self.max_pixel_value:
                    self.max_pixel_value = max_pixel
        self.classes = self.testset.classes
        self.num_classes = len(self.classes)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation = validation
        self.seed = seed
        if self.validation:
            if type(self.validation) != int:
                raise ValueError('validation should be a int of the size of the val set')
            if self.seed is None:
                logging.warning('It is recommended to provided random seed to reproducibility')
                generator = None
            else:
                generator = torch.Generator().manual_seed(self.seed)
                # create validation set of provided size
            self.trainset, self.valset = torch.utils.data.random_split(self.trainset,
                                                       lengths=[len(self.trainset) - self.validation, self.validation],
                                                       generator=generator)
            self.trainloader.data = self.trainset  # update dataset loader
            self.valset.transform = self.transform_test  # set test transform to val dataset
            self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                          shuffle=False, num_workers=num_workers)

    def get_input_shape(self):
        return tuple(self.trainset.data.shape)

    @check_args
    def to_numpy(self, train=False, validation=False, N=None, seed=None):
        """
        Return dataset as numpy array
        Becareful, data has to be able to be loaded into memory.
        :param train: bool, train set
        :param validation: bool, validation set. If train is False, test set.
        :param N: int, max number of examples to import
        :return: X, y: numpy arrays
        """
        if train:
            set = self.trainset
        elif validation:
            set = self.valset
        else:
            set = self.testset
        if N is None:
            N = len(set)
        if seed:
            torch.manual_seed(seed)
        loader = torch.utils.data.DataLoader(set, batch_size=N, shuffle=(train or N < len(set)))
        load_tmp = next(iter(loader))
        X = load_tmp[0].numpy()
        y = load_tmp[1].numpy()
        return X, y

    @check_args
    def correctly_predicted_to_numpy(self, model=None, models=None, train=False, validation=False, N=None, seed=None):
        """
        Return the examples correcty predicted by model in the dataset as numpy arrays

        :param model: pytorch model
        :param models: list of pytorch models
        :param train: bool, train or test set
        :param validation: bool, validation set. If train is False, test set.
        :param N: int, max number of examples to import
        :param seed: int, fix random seed for reproducibility and select same examples
        :return: X, y: numpy arrays
        """
        if not model and not models:
            raise ValueError('model or models should be defined')
        if model and models:
            raise ValueError('model and models cannot be both defined')
        if model:
            models = [model, ]
        if train:
            set = self.trainset
        elif validation:
            set = self.valset
        else:
            set = self.testset
        if N is None:
            N = len(set)
        if seed:
            torch.manual_seed(seed)
        loader = torch.utils.data.DataLoader(set, batch_size=self.batch_size, shuffle=(train or N < len(set)))
        X = np.zeros((N,) + self.get_input_shape()[1:], dtype='float32')
        y = np.zeros((N,), dtype='int')
        nb_images_saved = 0
        for images, labels in loader:
            images_ = images.to(DEVICE)
            idx_ok = torch.zeros(len(images))
            for model in models:
                outputs = model(images_)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu()
                idx_ok += predicted == labels
            idx_ok = idx_ok == len(models)
            images_ok = images[idx_ok,]
            labels_ok = labels[idx_ok,]
            nb_images_to_append = min(idx_ok.sum().item(), N-nb_images_saved)
            X[nb_images_saved:(nb_images_saved+nb_images_to_append),] = images_ok[0:nb_images_to_append,].numpy()
            y[nb_images_saved:nb_images_saved+nb_images_to_append] = labels_ok[0:nb_images_to_append].numpy()
            nb_images_saved += nb_images_to_append
            if nb_images_saved >= N:
                break
        X = X[0:nb_images_saved,]
        y = y[0:nb_images_saved,]
        if not (X.shape[0] == y.shape[0] <= N):
            raise RuntimeError("Array shape unexpected")
        if X.shape[0] < N < len(set):
            logging.warning('Number of examples lower than requested')
        return X, y

    @check_args
    def compute_accuracy(self, model, train=False, validation=False):
        """
        Compute the accuracy on the test or train data
        :param model: Pytorch NN
        :param train: compute on the train set
        :param train: compute on the validation set. If train is False, test set.
        :return: float
        """
        if train:
            loader = self.trainloader
        elif validation:
            loader = self.valloader
        else:
            loader = self.testloader
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total



class MNIST(DataBase):

    def __init__(self, batch_size, num_workers=0, path='data', validation=None, normalize=False, seed=None):
        if normalize:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        else:
            transform = transforms.Compose([transforms.ToTensor(),])
        self.transform_train = self.transform_test = self.transform = transform
        self.trainset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                       shuffle=True, num_workers=num_workers)
        self.testset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                      shuffle=False, num_workers=num_workers)
        super().__init__(batch_size=batch_size, num_workers=num_workers, validation=validation, seed=seed)

    def get_input_shape(self):
        return (1, 1, 28, 28)


class CIFAR10(DataBase):

    def __init__(self, batch_size, num_workers=0, path='data', validation=None, normalize=False, seed=None):
        if normalize:
            normalize_transform = transforms.Normalize(mean=(0.49139968, 0.48215841, 0.44653091),
                                             std=(0.24703223, 0.24348513, 0.26158784))
        else:
            normalize_transform = transforms.Normalize(mean=(0., 0., 0.),
                                         std=(1., 1., 1.))
        #normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
        #                                 std=(0.5, 0.5, 0.5))
        self.transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize_transform])
        self.transform_test = transforms.Compose(
            [transforms.ToTensor(),
             normalize_transform])
        self.trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                       shuffle=True, num_workers=num_workers)
        self.testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                      shuffle=False, num_workers=num_workers)
        super().__init__(batch_size=batch_size, num_workers=num_workers, validation=validation, seed=seed)

    def get_input_shape(self):
        return (1, 3, 32, 32)


class CIFAR100(DataBase):

    def __init__(self, batch_size, num_workers=0, path='data', validation=None, normalize=False, seed=None):
        if normalize:
            normalize_transform = transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
        else:
            normalize_transform = transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        self.transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize_transform])
        self.transform_test = transforms.Compose(
            [transforms.ToTensor(),
             normalize_transform])
        self.trainset = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                       shuffle=True, num_workers=num_workers)
        self.testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                      shuffle=False, num_workers=num_workers)
        super().__init__(batch_size=batch_size, num_workers=num_workers, validation=validation, seed=seed)


class ImageNet(DataBase):

    def __init__(self, batch_size, path='/work/projects/bigdata_sets/ImageNet/ILSVRC2012/raw-data/', num_workers=0,
                 validation=None, normalize=False, input_size=224, resize_size=256, seed=None):
        # default: input_size=224, resize_size=256
        # inception: input_size=299, resize_size=342
        # if DATAPATH env var is set, overright default value
        self.input_size = input_size
        self.resize_size = resize_size
        if os.environ.get('DATAPATH', False):
            path = os.environ.get('DATAPATH')
        if normalize:
            print('Loading ImageNet with normalization.')
            normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        else:
            normalize_transform = transforms.Normalize(mean=[0., 0., 0.],
                                             std=[1., 1., 1.])
        traindir = os.path.join(path, 'train')
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
        ])
        self.trainset = datasets.ImageFolder(traindir, self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=self.use_cuda)
        testdir = os.path.join(path, 'validation')
        self.transform_test = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize_transform,
        ])
        self.testset = datasets.ImageFolder(testdir, self.transform_test)
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=self.use_cuda)
        self.min_pixel_value = 0
        self.max_pixel_value = 1
        super().__init__(batch_size=batch_size, num_workers=num_workers, validation=validation, seed=seed)

    def get_input_shape(self):
        return (1, 3, self.input_size, self.input_size)
