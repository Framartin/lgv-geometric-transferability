import torch
from sklearn.decomposition import PCA
from utils.subspace_inference.utils import flatten, bn_update


def model2vector(model):
    """
    Transform a pytorch model into its weight Tensor
    :param model: pytorch model
    :return: tensor of size (n_weights,)
    """
    w = flatten([param.detach().cpu() for param in model.parameters()])
    return w

def models2tensor(models):
    """
    Transform a list of pytorch model into its weight Tensor
    :param models: list of pytorch model
    :return: tensor of size (n_models, n_weights)
    """
    n_weights = sum(p.numel() for p in models[0].parameters())
    W = torch.empty(0, n_weights, dtype=torch.float32)
    for model in models:
        w = model2vector(model)
        W = torch.cat((W, w.view(1, -1)), dim=0)
    return W


def vector2model(w, model_cfg, update_bn=True, train_loader=None, **kwargs):
    """
    Build a pytorch model from given weight vector
    :param w: tensor of size (1, n_weights)
    :param model_cfg: model class
    :param update_bn: Update or not the BN statistics
    :param train_loader: Data loader to update BN stats
    :param kwargs: args passed to model_cfg
    :return: pytorch model
    """
    if update_bn and not train_loader:
        raise ValueError('train_loader must be provided with update_bn')
    w = w.detach().clone()
    new_model = model_cfg(**kwargs).cuda()
    offset = 0
    for param in new_model.parameters():
        param.data.copy_(w[offset:offset + param.numel()].view(param.size()).to('cuda'))
        offset += param.numel()
    if update_bn:
        bn_update(train_loader, new_model, verbose=False, subset=0.1)
    new_model.eval()
    return new_model


class PcaWeights:
    def __init__(self, model_cfg, rank=20, train_loader=None, seed=None):
        """
        PCA on pytorch models
        :param model_cfg: class of the models
        :param rank: number of PCA components
        :param train_loader: train loader to update BN of produced models
        """
        self.model_cfg = model_cfg
        self.rank = rank
        self.train_loader = train_loader
        self.pca = PCA(n_components=self.rank, svd_solver='auto', random_state=seed)

    def fit(self, models):
        """
        Fit PCA to models
        :param models: list of pytorch model
        """
        num_models = len(models)
        if self.rank > 0.8*num_models:
            print('Randomized SVD might not be ideal for rank PCA > 80% number of models')
        W = models2tensor(models)
        if W.shape[0] != num_models:
            raise RuntimeError('Wrong dimension of W')
        W = W.numpy()
        self.pca.fit(W)

    def transform(self, models=None, W=None, components=None):
        """
        Transform a model into the PCA subspace
        :param models: list of pytorch model
        :param W: pytorch tensor of the weights. Ignored if models is specified.
        :param components: list of int corresponding to the components to keep. `None` keep all `rank` components
        :return:
        """
        if not models and not torch.is_tensor(W):
            raise ValueError('models or W should be defined')
        if models:
            W = models2tensor(models)
        W = W.numpy()
        Z = self.pca.transform(W)
        if components:
            Z = Z[:, components]
        return Z

    def inverse_transform(self, Z, update_bn=True, **kwargs):
        """
        Inverse transform from latent space to model
        :param Z: Component vector
        :param update_bn: Update BN layers of models
        :param kwargs: args passed to model class
        :return: list of models
        """
        W = self.pca.inverse_transform(Z)
        W = torch.from_numpy(W)
        new_models = []
        for i in range(W.shape[0]):
            w = W[i, :]
            new_model = vector2model(w=w, model_cfg=self.model_cfg, update_bn=update_bn, train_loader=self.train_loader,
                                     **kwargs)
            new_models.append(new_model)
        return new_models
