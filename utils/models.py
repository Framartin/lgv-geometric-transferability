import torch
from torch import nn
import torch.nn.functional as F
from random import randrange, shuffle


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling.
    Code adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, temperature=1.):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = torch.tensor(temperature)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        self.temperature = self.temperature.to(*args, **kwargs)
        super(ModelWithTemperature, self).to(*args, **kwargs)


class MnistFc(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        if pretrained:
            raise NotImplementedError()
        super(MnistFc, self).__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output


class MnistSmallFc(nn.Module):
    """
    One hidden layer FC
    """
    def __init__(self, num_classes=10, pretrained=False, hidden_size=512):
        if pretrained:
            raise NotImplementedError()
        super(MnistSmallFc, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        return output


class MnistCnn(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        if pretrained:
            raise NotImplementedError()
        super(MnistCnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output


class CifarLeNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(CifarLeNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


class TorchEnsemble(nn.Module):

    def __init__(self, models, ensemble_logits=False):
        """
        :param models: list of pytorch models to ensemble
        :param ensemble_logits: True if ensemble logits, False to ensemble probabilities
        :return probablities if ensemble_logits is False, logits if True
        """
        super(TorchEnsemble, self).__init__()
        if len(models) < 1:
            raise ValueError('Empty list of models')
        self.models = nn.ModuleList(models)
        self.ensemble_logits = ensemble_logits
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # clone to make sure x is not changed by inplace methods
        if not self.ensemble_logits:
            x_list = [self.softmax(model(x.clone())) for model in self.models]  # probs
        else:
            x_list = [model(x.clone()) for model in self.models]  # logits
        x = torch.stack(x_list)  # concat on dim 0
        x = torch.mean(x, dim=0, keepdim=False)
        #for model in self.models:
        #    xi = model(x.clone())  # clone to make sure x is not changed by inplace methods

        # x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        # x1 = x1.view(x1.size(0), -1)
        # x2 = self.modelB(x)
        # x2 = x2.view(x2.size(0), -1)
        # x = torch.cat((x1, x2), dim=1)
        # x = self.classifier(F.relu(x))
        return x


class LightEnsemble(nn.Module):

    def __init__(self, models, order=None):
        """
        Perform a single forward pass to one of the models when call forward()

        :param models: list of pytorch models to ensemble
        :param random: str, 'random' draw a model with replacement, 'shuffle' draw a model w/o replacement, None cycle in provided order (Default).
        :return logits
        """
        super(LightEnsemble, self).__init__()
        self.n_models = len(models)
        if self.n_models < 1:
            raise ValueError('Empty list of models')
        if order == 'shuffle':
            shuffle(models)
        elif order in [None, 'random']:
            pass
        else:
            raise ValueError('Not supported order')
        self.models = nn.ModuleList(models)
        self.order = order
        self.f_count = 0

    def forward(self, x):
        if self.order == 'random':
            index = randrange(0, self.n_models)
        else:
            index = self.f_count % self.n_models
        x = self.models[index](x)
        self.f_count += 1
        return x


class LightNestedEnsemble(nn.Module):

    def __init__(self, list_models, order=None):
        """
        Perform ensemble a single list of models when call forward()

        :param models: nested list of pytorch models to ensemble
        :param random: str, 'random' draw a model with replacement, 'shuffle' draw a model w/o replacement, None cycle in provided order (Default).
        :return logits
        """
        super(LightNestedEnsemble, self).__init__()
        self.n_ensembles = len(list_models)
        if self.n_ensembles < 1:
            raise ValueError('Empty list of models')
        if order == 'shuffle':
            shuffle(list_models)
        elif order in [None, 'random']:
            pass
        else:
            raise ValueError('Not supported order')
        self.models = nn.ModuleList([TorchEnsemble(models=x, ensemble_logits=True) for x in list_models])
        self.order = order
        self.f_count = 0

    def forward(self, x):
        if self.order == 'random':
            index = randrange(0, self.n_ensembles)
        else:
            index = self.f_count % self.n_ensembles
        x = self.models[index](x)
        self.f_count += 1
        return x
