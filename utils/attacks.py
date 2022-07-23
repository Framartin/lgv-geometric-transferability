import torch
import os
import numpy as np
import scipy.stats as st
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentPyTorch
from art.classifiers import PyTorchClassifier
from art.config import ART_NUMPY_DTYPE
from art.utils import (
    random_sphere,
    projection,
)
from tqdm import tqdm
from typing import Optional, Union, TYPE_CHECKING, Tuple, List
from utils.helpers import DEVICE, compute_accuracy_from_nested_list_models


class ExtendedFastGradientMethod(FastGradientMethod):
    attack_params = FastGradientMethod.attack_params + ['momentum', 'grad_momentum', 'translation_invariant']

    def __init__(self, momentum=None, grad_momentum=None, translation_invariant=False, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.grad_momentum = grad_momentum
        self.translation_invariant = translation_invariant
        if momentum and grad_momentum is None:
            raise ValueError("grad should be provided when using momentum")

    def _compute_perturbation(self, batch: np.ndarray, batch_labels: np.ndarray, mask: Optional[np.ndarray], batch_grad_momentum: Optional[np.ndarray]) -> \
    Tuple[np.ndarray, np.ndarray]:
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))

        if self.translation_invariant:
            # original implementation:
            #noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
            # translation_invariant kernel
            #kernel = self.gkern(15, 3).astype(np.float32)
            #stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
            #stack_kernel = np.expand_dims(stack_kernel, 3)
            kernlen = 15
            nb_channels = batch.shape[1]
            padding = int((kernlen - 1) / 2)  # same padding
            with torch.no_grad():
                kernel = torch.from_numpy(self.gkern(kernlen=kernlen).astype(np.float32))
                stack_kernel = kernel.view(1, 1, kernlen, kernlen).repeat(nb_channels, 1, 1, 1).to(DEVICE)
                grad = torch.nn.functional.conv2d(torch.from_numpy(grad).to(DEVICE), stack_kernel, padding=padding, groups=nb_channels).cpu().numpy()
            if grad.shape != batch.shape:
                raise RuntimeError('Translation invariant gradient does not have the same dimension as input')

        # Apply norm bound
        def _apply_norm(grad, norm=self.norm, object_type=False):
            if norm in [np.inf, "inf"]:
                grad = np.sign(grad)
            elif norm == 1:
                if not object_type:
                    ind = tuple(range(1, len(batch.shape)))
                else:
                    ind = None
                grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            elif norm == 2:
                if not object_type:
                    ind = tuple(range(1, len(batch.shape)))
                else:
                    ind = None
                grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
            return grad

        # momentum
        if self.momentum:
            # scale L1-norm
            if batch.dtype == np.object:
                for i_sample in range(batch.shape[0]):
                    grad[i_sample] = _apply_norm(grad[i_sample], norm=1, object_type=True)
                    assert batch[i_sample].shape == grad[i_sample].shape
            else:
                grad = _apply_norm(grad, norm=1)
            # update moving avg
            #self.grad_momentum = self.grad_momentum * self.momentum + grad
            batch_grad_momentum = batch_grad_momentum * self.momentum + grad
            grad = batch_grad_momentum.copy()
            #self.grad_momentum = self.grad_momentum.detach().clone()

        # Apply mask
        if mask is not None:
            grad = np.where(mask == 0.0, 0.0, grad)

        # Apply norm bound
        if batch.dtype == np.object:
            for i_sample in range(batch.shape[0]):
                grad[i_sample] = _apply_norm(grad[i_sample], object_type=True)
                assert batch[i_sample].shape == grad[i_sample].shape
        else:
            grad = _apply_norm(grad)

        assert batch.shape == grad.shape

        return grad, batch_grad_momentum

    @staticmethod
    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array.

        From the original implementation https://github.com/dongyp13/Translation-Invariant-Attacks/blob/master/attack_iter.py
        """
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def _compute(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray],
        eps: float,
        eps_step: float,
        project: bool,
        random_init: bool,
    ) -> np.ndarray:
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()
            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            if mask is not None:
                random_perturbation = random_perturbation * (mask.astype(ART_NUMPY_DTYPE))
            x_adv = x.astype(ART_NUMPY_DTYPE) + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            if x.dtype == np.object:
                x_adv = x.copy()
            else:
                x_adv = x.astype(ART_NUMPY_DTYPE)

            # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch_index_2 = min(batch_index_2, x.shape[0])
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]
            batch_grad_momentum = self.grad_momentum[batch_index_1:batch_index_2]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation, batch_grad_momentum = self._compute_perturbation(batch, batch_labels, mask_batch, batch_grad_momentum)

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, eps_step)
            self.grad_momentum[batch_index_1:batch_index_2] = batch_grad_momentum

            if project:
                if x_adv.dtype == np.object:
                    for i_sample in range(batch_index_1, batch_index_2):
                        perturbation = projection(x_adv[i_sample] - x_init[i_sample], eps, self.norm)
                        x_adv[i_sample] = x_init[i_sample] + perturbation

                else:
                    perturbation = projection(
                        x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], eps, self.norm
                    )
                    x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv


class ExtendedProjectedGradientDescentPyTorch(ProjectedGradientDescentPyTorch):
    """
    Implement Light Version of PGD where only a subset of models is attack at each iterations.
    Also support some test-time techniques for transferability.

    :param models_target_dict: Dictionary of target models as values with the name as key.
    :param freq_acc_target: Compute accuracy on target each provided iterations. All iterations by default.
    :param data: Data class. For now only required with models_target_dict.
    """
    attack_params = ProjectedGradientDescentPyTorch.attack_params + ['momentum', 'translation_invariant', 'grad_noise_std', 'models_target_dict', 'freq_eval_target', 'data'],

    def __init__(
            self,
            estimators: List[PyTorchClassifier],  # pass a list of classifier
            momentum: float = None,
            translation_invariant: bool = False,
            grad_noise_std: float = None,
            models_target_dict=None,
            freq_eval_target=1,
            data=None,
            **kwargs
    ):
        self.estimators = estimators
        super().__init__(estimator=estimators[0], **kwargs)
        self.momentum = momentum
        self.translation_invariant = translation_invariant
        self.grad_noise_std = grad_noise_std
        # to report target accuracy at each freq_eval_target iteration
        self.freq_eval_target = freq_eval_target
        self.data = data
        if models_target_dict and data is None:
            raise ValueError('data param should be provided if models_target_dict is set.')
        self.models_target_dict = models_target_dict
        self.stats_per_iter = {k: {'iter': [], 'acc': [], 'loss': []} for k in models_target_dict} if models_target_dict else {}

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.
        Modified to compute the successful adversarial on all the classifiers in the list. Not only one.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        # Check whether random eps is enabled
        self._random_eps()

        # Set up targets
        targets = self._set_targets(x, y)

        # Create dataset
        if mask is not None:
            # Here we need to make a distinction: if the masks are different for each input, we need to index
            # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
            if len(mask.shape) == len(x.shape):
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(mask.astype(ART_NUMPY_DTYPE)),
                )

            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])),
                )

        else:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(x.astype(ART_NUMPY_DTYPE)), torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
            )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Start to compute adversarial examples
        adv_x = x.astype(ART_NUMPY_DTYPE)

        # Compute perturbation with batching
        for (batch_id, batch_all) in enumerate(
            tqdm(data_loader, desc="PGD - Batches", leave=False, disable=not self.verbose)
        ):
            if mask is not None:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
            else:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            # Compute batch_eps and batch_eps_step
            if isinstance(self.eps, np.ndarray):
                if len(self.eps.shape) == len(x.shape) and self.eps.shape[0] == x.shape[0]:
                    batch_eps = self.eps[batch_index_1:batch_index_2]
                    batch_eps_step = self.eps_step[batch_index_1:batch_index_2]

                else:
                    batch_eps = self.eps
                    batch_eps_step = self.eps_step

            else:
                batch_eps = self.eps
                batch_eps_step = self.eps_step

            for rand_init_num in range(max(1, self.num_random_init)):
                if rand_init_num == 0:
                    # first iteration: use the adversarial examples as they are the only ones we have now
                    adv_x[batch_index_1:batch_index_2] = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )
                else:
                    adversarial_batch = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )

                    # return the successful adversarial examples
                    # modified:
                    attack_success = self.compute_success_array(
                        batch,
                        batch_labels,
                        adversarial_batch,
                        self.targeted,
                        batch_size=self.batch_size,
                    )
                    adv_x[batch_index_1:batch_index_2][attack_success] = adversarial_batch[attack_success]
        # modified:
        # logger.info(
        #     "Success rate of attack: %.2f%%",
        #     100 * self.compute_success(x, y, adv_x, self.targeted, batch_size=self.batch_size),
        # )

        return adv_x

    def compute_success_array(
            self,
            x_clean: np.ndarray,
            labels: np.ndarray,
            x_adv: np.ndarray,
            targeted: bool = False,
            batch_size: int = 1,
    ) -> float:
        """
        Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.
        Modified: use the list of estimators to compute the predictions

        :param x_clean: Original clean samples.
        :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
        :param x_adv: Adversarial samples to be evaluated.
        :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
               correct labels of the clean samples.
        :param batch_size: Batch size.
        :return: Percentage of successful adversarial samples.
        """
        adv_results = np.zeros((x_adv.shape[0], self.estimators[0].nb_classes), dtype=np.float32)
        for classifier in self.estimators:
            adv_results += classifier.predict(x_adv, batch_size=batch_size)
        adv_preds = np.argmax(adv_results, axis=1)
        if targeted:
            attack_success = adv_preds == np.argmax(labels, axis=1)
        else:
            results = np.zeros((x_clean.shape[0], self.estimators[0].nb_classes), dtype=np.float32)
            for classifier in self.estimators:
                results += classifier.predict(x_clean, batch_size=batch_size)
            preds = np.argmax(results, axis=1)
            attack_success = adv_preds != preds
        return attack_success

    def _generate_batch(
        self,
        x: "torch.Tensor",
        targets: "torch.Tensor",
        mask: "torch.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
    ) -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array. Each iteration is computed on a different estimator from estimators.

        :param x: An array with the original inputs.
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        inputs = x.to(self.estimator.device)
        targets = targets.to(self.estimator.device)
        y = torch.max(targets, dim=1).indices  # class idx
        adv_x = inputs

        if mask is not None:
            mask = mask.to(self.estimator.device)

        # init grad_momentum at beginning of a batch
        self.grad_momentum = torch.zeros(x.size()).to(self.estimator.device)

        for i_max_iter in range(self.max_iter):
            # cycle between estimators
            self._estimator = self.estimators[i_max_iter % len(self.estimators)]
            adv_x = self._compute_torch(
                adv_x, inputs, targets, mask, eps, eps_step, self.num_random_init > 0 and i_max_iter == 0,
            )
            # compute target accuracy at each freq_eval_target iteration (+ first/last iterations)
            if self.models_target_dict:
                if (i_max_iter % self.freq_eval_target == 0) or (i_max_iter == self.max_iter-1):
                    for name_target, model_target in self.models_target_dict.items():
                        acc_target, loss_target = compute_accuracy_from_nested_list_models(
                            [[model_target, ], ], X=adv_x.cpu(), y=y.cpu(), data=self.data)
                        self.stats_per_iter[name_target]['iter'].append(i_max_iter)
                        self.stats_per_iter[name_target]['acc'].append(acc_target)
                        self.stats_per_iter[name_target]['loss'].append(loss_target)

        return adv_x.cpu().detach().numpy()

    def _compute_perturbation(
        self, x: "torch.Tensor", y: "torch.Tensor", mask: Optional["torch.Tensor"]
    ) -> "torch.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :return: Perturbations.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(x=x, y=y) * (1 - 2 * int(self.targeted))

        # added:
        if self.translation_invariant:
            # original implementation:
            #noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
            # translation_invariant kernel
            #kernel = self.gkern(15, 3).astype(np.float32)
            #stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
            #stack_kernel = np.expand_dims(stack_kernel, 3)
            # kernlen = 15
            kernlen = int(os.getenv('ADV_TRANSFER_TI_KERNEL_SIZE', '15'))
            nb_channels = x.size(1)
            padding = int((kernlen - 1) / 2)  # same padding
            with torch.no_grad():
                kernel = self.gkern(kernlen=kernlen).to(self.estimator.device)
                stack_kernel = kernel.view(1, 1, kernlen, kernlen).repeat(nb_channels, 1, 1, 1)
                grad = torch.nn.functional.conv2d(grad, stack_kernel, padding=padding, groups=nb_channels)
            if grad.shape != x.shape:
                raise RuntimeError('Translation invariant gradient does not have the same dimension as input')

        # added for momentum
        if self.momentum:
            # scale L1-norm
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore
            # update moving avg
            with torch.no_grad():
                self.grad_momentum = self.grad_momentum * self.momentum + grad
            grad = self.grad_momentum

        # add gaussian noise to gradients
        if self.grad_noise_std:
            grad += torch.randn(grad.shape).to(self.estimator.device) * self.grad_noise_std

        # Apply mask
        if mask is not None:
            grad = torch.where(mask == 0.0, torch.tensor(0.0), grad)

        # Apply norm bound
        if self.norm in ["inf", np.inf]:
            grad = grad.sign()

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + tol)  # type: ignore

        assert x.shape == grad.shape

        return grad

    @staticmethod
    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array.

        From the original implementation https://github.com/dongyp13/Translation-Invariant-Attacks/blob/master/attack_iter.py
        """
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        # convert to tensor
        kernel = torch.from_numpy(kernel.astype(np.float32))
        return kernel

    def get_target_accuracy_per_iter(self, name_target):
        if not self.models_target_dict:
            raise ValueError('models_target_dict must be specified')
        return self.stats_per_iter[name_target]
