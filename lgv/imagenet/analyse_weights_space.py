import pandas as pd
import random
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torchvision import models as tmodels
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from pyhessian import hessian
from utils.data import ImageNet
from utils.helpers import list_models, guess_and_load_model, guess_model
from utils.pca_weights import PcaWeights, model2vector, models2tensor, vector2model
# from utils.subspace_inference.posteriors import SWAG
# from utils.subspace_inference.posteriors.proj_model import SubspaceModel
# from utils.subspace_inference import utils, losses
from utils.subspace_inference.utils import save_checkpoint



def parse_args():
    parser = argparse.ArgumentParser(description="Analyse weights space: PCA and shift subspace XPs")
    parser.add_argument("dir_models_pca", help="Path to directory containing all the models file of the ensemble model")
    parser.add_argument("path_original_model", help="Path to original model")
    parser.add_argument("--xp", choices=['PCA_projection', 'translate', 'hessian'], default='PCA_projection')
    parser.add_argument("--pca-rank", type=int, default=5)
    parser.add_argument("--export-dir", default=None, help="If set, export projected models on the first --export-ranks components")
    parser.add_argument("--export-ranks", nargs='*', type=int, default=[], help="If set, export projected models on the first --export-ranks components."
                                                                                  "If muliple values are provided, export values recurvively on different subfolder. Must be > 0.")
    parser.add_argument("--update-bn", action='store_true', help="Update BN after projection to original space")
    parser.add_argument("--alpha-translate", type=float, default=1., help="Multiply deviations by this constant.")
    parser.add_argument("--dir-models-translate", default=None, help="Path to directory containing the models from "
                                                                     "another local maximum, to which we translate the "
                                                                     "dir_models_pca")
    parser.add_argument("--limit-n-export-models", type=int, default=None, help="Limit the number of exported model by randomly sampling them. Default: None, no limit.")
    # parser.add_argument("--path-original-model-translate", help="Path to original model of the other local maximum, to which we translate the dir_models_pca")
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

    paths_pca = list_models(args.dir_models_pca)

    models_pca = []
    for i, filename in enumerate(paths_pca):
        model = guess_and_load_model(filename, data=data, load_as_ghost=False, input_diversity=False, skip_gradient=False)
        models_pca.append(model)
    model_original = guess_and_load_model(args.path_original_model, data=data, load_as_ghost=False, input_diversity=False, skip_gradient=False)

    model_cfg = getattr(tmodels, guess_model(paths_pca[0]))

    # build a PCA of the target models
    # swag_model = SWAG(model_cfg,
    #              subspace_type='pca',
    #              subspace_kwargs={
    #                 'max_rank': len(models_pca),  # collect all the models
    #                 'pca_rank': args.pca_rank,
    #              },
    #              num_classes=data.num_classes)
    # swag_model.cuda()
    # # emulate model collection
    # for model in models_pca:
    #     swag_model.collect_model(model)
    #
    # print('Setting SWA solution (as subspace shift vector)')
    # swag_model.set_swa()
    # utils.bn_update(loaders["train"], swag_model, verbose=True, subset=0.1)
    # print(f"  Metrics on test-set: {utils.eval(loaders['test'], swag_model, losses.cross_entropy)}")

    # print('Building subspace')
    # mean, variance, cov_factor = swag_model.get_space()
    # print(np.linalg.norm(cov_factor, axis=1))
    # subspace = SubspaceModel(mean, cov_factor)

    if args.xp == 'hessian':
        raise NotImplementedError('XP not yet implemented')

    if args.xp == 'PCA_projection':
        # build a PCA of the target models
        pca = PcaWeights(model_cfg, rank=args.pca_rank, train_loader=loaders['train'], seed=args.seed)
        pca.fit(models_pca)
        # save the origin as SWA (0 rank)
        Z0 = np.zeros((1, args.pca_rank))
        model_swa = pca.inverse_transform(Z0, update_bn=args.update_bn)[0]

        # analyse PCA
        print(f"Explained variance ratio: {pca.pca.explained_variance_ratio_}")
        total_cum_var_ratio = np.cumsum(pca.pca.explained_variance_ratio_)
        print(f"Total cumulated explained variance: {total_cum_var_ratio}")
        # print(f"Explained variance: {pca.pca.explained_variance_}")
        print(f"Singular values: {pca.pca.singular_values_}")
        df_metrics_pca = pd.DataFrame({
            'dir_models': args.dir_models_pca,
            'dim': list(range(args.pca_rank)),
            'expl_var': pca.pca.explained_variance_ratio_,
            'totcum_expl_var': total_cum_var_ratio,
            'singular_values': pca.pca.singular_values_,
        })

        # project parameters into subspace, and into original space
        Z = pca.transform(models=models_pca)
        # original model
        Z_original = pca.transform(models=[model_original])

        Z_all = np.concatenate((Z0, Z_original, Z), axis=0)
        labels_all = ['SWA'] + ['Original'] + ['Collected models'] * len(Z)
        # viz in comet:
        # experiment.log_embedding(
        #     vectors=Z_all,
        #     labels=labels_all,
        #     title="Collected models (PCA)"
        # )
        # extract models
        print('Projecting models...')
        model_original_proj = pca.inverse_transform(Z_original, update_bn=args.update_bn)[0]
        if args.export_dir:
            os.makedirs(args.export_dir, exist_ok=True)
            df_metrics_pca.to_csv(os.path.join(args.export_dir, 'metrics_pca.csv'), index=False)
            # save projected models
            export_dir = os.path.join(args.export_dir, 'dims_0')
            save_checkpoint(export_dir, name='model_swa', state_dict=model_swa.state_dict())
            export_dir = os.path.join(args.export_dir, f'original_proj')
            save_checkpoint(export_dir, name='model_original_proj', state_dict=model_original_proj.state_dict())
            for i in tqdm(args.export_ranks, desc="Export dims"):
                Z_proj = Z.copy()
                # i between 1 and pca_rank+1
                Z_proj[:, i:] = 0  # project to the first i components
                models_pca_proj = pca.inverse_transform(Z_proj, update_bn=args.update_bn)
                export_dir = os.path.join(args.export_dir, f'dims_{i}')
                for j, model in enumerate(models_pca_proj):
                    save_checkpoint(export_dir, name='models_pca_proj', sample=j, state_dict=model.state_dict())

    if args.xp == 'translate':
        if not args.dir_models_translate:
            # or not args.path_original_model_translate
            raise ValueError('dir_models_translate should de provided')
        print('Translating to new local minimum...')
        w_original = model2vector(model_original)
        # compute first SWA
        W = models2tensor(models_pca)
        w_swa = torch.mean(W, 0)
        #w_swa = model2vector(model_swa)  # can be check with: pca.transform(W=torch.reshape(w_swa, (1, w_swa.shape[0])))
        # load other collected models
        paths_new_models = list_models(args.dir_models_translate)
        new_models = []
        for i, filename in enumerate(paths_new_models):
            model = guess_and_load_model(filename, data=data, load_as_ghost=False, input_diversity=False,
                                         skip_gradient=False)
            new_models.append(model)
        W_new = models2tensor(new_models)
        w_new_swa = torch.mean(W_new, 0)

        # update BN and save new SWA
        # model_swa_new = vector2model(w_new_swa, model_cfg, update_bn=args.update_bn, train_loader=train_loader)
        # export_dir = os.path.join(args.export_dir, 'SWA_new')
        # save_checkpoint(export_dir, name='model_new_swa', state_dict=model_swa_new.state_dict())

        # save another embedding with the new original model and the new collected models translated to the first SWA
        # Z_new_original_translated = pca.transform(W=torch.reshape(w_new_original - w_new_swa + w_swa, (1, w_new_swa.shape[0])))
        # Z_new_translated = pca.transform(W=W_new - w_new_swa + w_swa)
        # Z_all = np.concatenate((Z_all, Z_new_original_translated, Z_new_translated), axis=0)
        # print(Z_new_translated.shape)
        # labels_all = labels_all + ['Original model 2 translated'] + ['Collected models 2'] * len(new_models)
        # experiment.log_embedding(
        #     vectors=Z_all,
        #     labels=labels_all,
        #     title="Collected models (PCA) + translated new models"
        # )
        # for translation_type in range(1, 4):
        #     if translation_type == 1:
        #         print('   ...Strategy 1: new_original_model + collected_models - original_model')
        #     elif translation_type == 2:
        #         print('   ...Strategy 2: new_original_model + collected_models - swa')
        #     elif translation_type == 3:
        #         print('   ...Strategy 3: new_swa + collected_models - swa')
        #     else:
        #         raise RuntimeError('Undefined translation_type')
        dot_prod_two_subspace_basis = np.zeros((len(models_pca), W_new.shape[0]))
        print('Translating: LGV-SWA_new + (LGV - LGV-SWA) ')
        index_to_export = None
        if args.limit_n_export_models:
            print(f'...Limiting Export to {args.limit_n_export_models} randomly picked models')
            random.seed(args.seed)
            index_to_export = random.sample(range(0, len(models_pca)), args.limit_n_export_models)
        for i, model_pca in enumerate(tqdm(models_pca, desc=f"Translation")):
            w_pca = model2vector(model_pca)
            # cosine similarity compute dot product of basis vectors of deviations
            for j in range(W_new.shape[0]):
                # dot product b/w unit vectors
                v1 = w_pca - w_swa
                v2 = W_new[j,:] - w_new_swa
                if not (len(v1.shape) == len(v2.shape) == 1):
                    raise RuntimeError('Should compute cosine sim on vectors')
                dot_prod_two_subspace_basis[i,j] = (torch.dot(v1, v2) / (torch.linalg.norm(v1, ord=2) * torch.linalg.norm(v2, ord=2))).cpu().numpy()
                # angles_two_subspace_basis[i,j] = np.arccos(np.clip(torch.dot(v1_u, v2_u).cpu().numpy(), -1.0, 1.0))
            # print(f'Dot product of {i} deviation LGV1 with all LGV2 deviations: {dot_prod_two_subspace_basis[i,:]}')  # debug
            if (index_to_export is None) or (i in index_to_export):
                w_pca_trans = w_new_swa + args.alpha_translate * (w_pca - w_swa)
                model_pca_trans = vector2model(w=w_pca_trans, model_cfg=model_cfg, update_bn=args.update_bn, train_loader=train_loader)
                if args.export_dir:
                    # export_dir = os.path.join(args.export_dir, f'translation_{translation_type}')
                    export_dir = os.path.join(args.export_dir, f'translation_deviations_to_new_swa')
                    save_checkpoint(export_dir, name='models_translated', sample=i, state_dict=model_pca_trans.state_dict())
        print(f'\nCosine similarity of all basis vectors from the two LGV:')
        print(f'    mean:{ dot_prod_two_subspace_basis.mean().item() }')
        print(f'    mean abs:{ np.abs(dot_prod_two_subspace_basis).mean().item() }')
        print(f'    min:{ dot_prod_two_subspace_basis.min().item() }')
        print(f'    max:{ dot_prod_two_subspace_basis.max().item() }')
        if args.export_dir:
            pd.DataFrame(dot_prod_two_subspace_basis).to_csv(os.path.join(export_dir, 'dot_prod_two_subspace_basis.csv'))

if __name__ == '__main__':
    args = parse_args()
    main(args)