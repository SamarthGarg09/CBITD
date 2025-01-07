import os
import gc
import sys
import yaml
import torch
import glob
import numpy as np
import pandas as pd
from time import sleep
from scipy.stats import ttest_ind
from captum.attr import LayerActivation
from captum._utils.gradient import compute_layer_gradients_and_eval
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import CelebA, ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchmetrics
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm, trange
import PIL
import seaborn
import random
import argparse
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
scaler = amp.GradScaler()

device = 'cuda' 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

import torch
from torch.nn import Module
from typing import Callable, Union, Tuple, List, Any, Dict

def compute_layer_gradients_and_eval_custom(
    model: torch.nn.Module,
    layer: Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    target_ind: Union[int, List[int]] = None,
    attribute_to_layer_input: bool = False,
    additional_forward_args: Any = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom function to compute gradients of the model's output with respect to a specified layer's inputs or outputs.

    Args:
        model (torch.nn.Module): The target model.
        layer (torch.nn.Module): The layer for which gradients and activations are computed.
        inputs (torch.Tensor or Tuple[torch.Tensor, ...]): Input tensors to the model's forward method.
        target_ind (int or List[int], optional): Index or list of indices of the target classes for which gradients are computed.
        attribute_to_layer_input (bool, optional): If True, compute gradients with respect to the layer's input; else, with respect to the layer's output.
        additional_forward_args (Any, optional): Additional arguments to pass to the model's forward method.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - gradients: Gradients of the outputs with respect to the specified layer's inputs or outputs.
            - activations: Activations from the specified layer.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    # Initialize variables to store gradients and activations
    gradients = []
    activations = []

    # Define forward hook to capture activations
    def forward_hook(module, input, output):
        nonlocal activations
        if attribute_to_layer_input:
            # Capture inputs to the layer
            activations.append(input[0].detach())
        else:
            # Capture outputs from the layer
            activations.append(output.detach())

    # Define backward hook to capture gradients
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        if attribute_to_layer_input:
            # Capture gradients with respect to the layer's input
            gradients.append(grad_input[0].detach())
        else:
            # Capture gradients with respect to the layer's output
            gradients.append(grad_output[0].detach())

    # Register the hooks
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    # Ensure gradients are enabled for all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Perform the forward pass
    inputs = inputs if isinstance(inputs, tuple) else (inputs,)
    if additional_forward_args is not None:
        if isinstance(additional_forward_args, tuple):
            inputs += additional_forward_args
        else:
            inputs += (additional_forward_args,)

    # Enable gradient computation
    with torch.enable_grad():
        outputs = model(*inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Handle target indices
        # target_ind = target_ind.tolist()
        if target_ind is not None:
            if isinstance(target_ind, int):
                loss = outputs[:, target_ind].sum()
            elif isinstance(target_ind, list):
                loss = outputs[:, target_ind].sum()
            else:
                print(target_ind)
                raise ValueError("target_ind must be an int or a list of ints.")
        else:
            # If no target specified, sum all outputs
            loss = outputs.sum()

        # Perform backward pass
        loss.backward()
        # print(gradients[0].shape, activations[0].shape)-->torch.Size([8, 512, 3072]) torch.Size([8, 512, 3072])
        # import sys;sys.exit(0)

    # Remove the hooks
    forward_handle.remove()
    backward_handle.remove()

    # Concatenate all captured gradients and activations
    # Assuming single layer, single device
    gradients = torch.cat(gradients, dim=0) if gradients else torch.tensor([])
    activations = torch.cat(activations, dim=0) if activations else torch.tensor([])

    return gradients, activations
     
parser = argparse.ArgumentParser(description='Tcav program')
parser.add_argument('--model_name', default='roberta-base')

def save_tcav_results(trials, tcavs, save_npz_fname=None, force=False):
    
    stacked_tcavs = np.stack([np.stack(list(tcav_.values()), axis=0) for tcav_ in tcavs], axis=0)
    stacked_accs = np.stack([np.stack(list(trial[1].values()), axis=0) for trial in trials], axis=0)

    if save_npz_fname is not None:
        if os.path.exists(save_npz_fname) and not force:
            print(f'{save_npz_fname} already exists')
        else:
            np.savez(save_npz_fname, tcavs=stacked_tcavs, accs=stacked_accs)
    
    return stacked_tcavs, stacked_accs


class ActivationHook:
    def __init__(self):
        self.activation = None
    def __call__(self, module, input, output):
        self.activation = output

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.model(x)


class TCAVScore(torchmetrics.Metric):
    def __init__(self, CAV, signed=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.CAV = CAV
        self.signed = signed
        self.add_state('all_scores', default=[], dist_reduce_fx='cat')  # Store all scores

    def update(self, grads: torch.Tensor):
        with torch.no_grad():
            # Ensure gradients are reshaped to match CAV (Batch size, Features)
            grads = grads.view(grads.size(0), 1, -1)  # Shape [16, 1, 3072]
            CAV_expanded = self.CAV.unsqueeze(0)  # Shape [1, 5, 3072] for broadcasting

            # Compute cosine similarity along the feature dimension (dim=-1)
            cos = F.cosine_similarity(grads, CAV_expanded, dim=-1)  # Result shape: [16, 5]

            if self.signed:
                # For signed TCAV, get the positive cosine similarity results
                score = (cos > 0).float()  # Binary values (0 or 1)
            else:
                # For unsigned TCAV, use raw cosine similarity
                score = cos  # Continuous values

            # Accumulate all individual scores
            self.all_scores.append(score)

    def compute(self):
        # Return all accumulated scores, concatenated into a single tensor
        return torch.cat(self.all_scores, dim=0)  # Shape [N, 5] where N is the number of batches
        
class TCAV(nn.Module):
    def __init__(self, target_model, layer_names=None, cache_dir=None):
        super().__init__()
        self.target_model = target_model.eval()
        self.random_CAVs = None
        self.CAVs = None
        self.metrics=None
        self.cache_dir = cache_dir

        assert (layer_names is not None) or (cache_dir is not None)

        if self.cache_dir is not None and \
            os.path.exists(os.path.join(self.cache_dir, 'random_CAVs.npz')) and \
            os.path.exists(os.path.join(self.cache_dir, 'CAVs.npz')) and \
            os.path.exists(os.path.join(self.cache_dir, 'metrics.npz')):

            print('Loading random_CAVs.npz, CAVs.npz and metrics.npz from cache')
            with np.load(os.path.join(self.cache_dir, 'random_CAVs.npz')) as f:
                random_CAVs = {k:v for k, v in f.items()}
            with np.load(os.path.join(self.cache_dir, 'CAVs.npz')) as f:
                CAVs = {k:v for k, v in f.items()}
            assert all([len(v) > 0 for v in random_CAVs.values()])
            assert all([len(v) > 0 for v in CAVs.values()])

            self.CAVs = CAVs
            self.random_CAVs = random_CAVs
            with np.load(os.path.join(self.cache_dir, 'metrics.npz')) as f:
                metrics = {k:v for k, v in f.items()}
            self.metrics=metrics
            assert list(self.random_CAVs.keys()) == list(self.CAVs.keys())
            assert list(self.metrics.keys()) == list(self.CAVs.keys())

            self.layer_names = list(self.random_CAVs.keys())
            print(f'Using cached layer names: {self.layer_names}')
        else:
            self.layer_names=layer_names

        self.layers = {}
        for layer_name, layer in self.target_model.named_modules():
            if layer_name in self.layer_names:
                self.layers[layer_name] = layer

        # if sorted(list(self.layers.values())) != sorted(self.layer_names):
        if sorted(self.layer_names) != sorted(list(self.layers.keys())):
            raise ValueError(f"Keys {sorted(self.layer_names)} and {sorted(list(self.layers.keys()))} don't match.")

    @staticmethod
    def get_class_balanced_sampler(ys, y_index):
        ys = ys[:, y_index]
        pos_ratio = ys.sum() / ys.shape[0]
        weights = ys * (1 - pos_ratio) + (1 - ys).abs() * pos_ratio
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        return sampler

    def _generate_CAVs(self, dset_train, dset_valid, hparams=None, verbose=True):
        default_hparams = dict(
            task='classification',
            n_epochs=100,
            lr=1e-4,
            weight_decay=1e-2,
            batch_size=32,
            patience=10,
            pos_weight=None,
            num_workers=4
        )
        hparams = {**default_hparams, **(hparams or {})}
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dl_train = DataLoader(
            dset_train,
            batch_size=hparams['batch_size'],
            shuffle=False,
            num_workers=hparams['num_workers'],
            pin_memory=True
        )
        dl_valid = DataLoader(
            dset_valid,
            batch_size=hparams['batch_size'],
            shuffle=False,
            num_workers=hparams['num_workers'],
            pin_memory=True
        )
    
        CAVs, metrics = {}, {}
        for layer_name, layer in tqdm(self.layers.items(), leave=False, desc='Layers'):
            layer_act = LayerActivation(self.target_model, layer)
    
            if verbose:
                print(f'Training Linear Model for layer: {layer_name}')
    
            # Initialize the linear model after first batch
            batch = next(iter(dl_train))
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                embeddings = self.target_model.model.get_input_embeddings()(batch['input_ids'])
                attention_mask = batch['attention_mask']
                acts = layer_act.attribute(
                    inputs=(None, attention_mask, embeddings),
                    attribute_to_layer_input=True
                )
            xs = acts[:, 0, :]
            cs = batch['labels']
    
            in_dim = xs.shape[1]
            out_dim = cs.shape[1]
    
            linear_model = torch.nn.Linear(in_dim, out_dim).to(device)
            optimizer = torch.optim.AdamW(
                linear_model.parameters(),
                lr=hparams['lr'],
                weight_decay=hparams['weight_decay']
            )
    
            if hparams['pos_weight'] is not None:
                if isinstance(hparams['pos_weight'], torch.Tensor):
                    pos_weight = hparams['pos_weight'].to(device)
                else:
                    pos_weight = hparams['pos_weight'] * torch.ones([out_dim]).to(device)
            else:
                pos_weight = None
    
            if hparams['task'] == 'classification':
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                metric = torchmetrics.Accuracy(
                    threshold=0.5,
                    task='multilabel',
                    num_labels=out_dim,
                    average=None
                ).to(device)
    
            patience = 0
            min_loss = np.inf
    
            scaler = torch.cuda.amp.GradScaler()
            linear_model.train()
    
            with trange(hparams['n_epochs'], leave=False, desc='Epochs') as tepochs:
                for epoch in tepochs:
                    losses = []
                    for batch in dl_train:
                        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                        optimizer.zero_grad(set_to_none=True)
                        with torch.no_grad():
                            embeddings = self.target_model.model.get_input_embeddings()(batch['input_ids'])
                            attention_mask = batch['attention_mask']
                            acts = layer_act.attribute(
                                inputs=(None, attention_mask, embeddings),
                                attribute_to_layer_input=True
                            )
                        xs = acts[:, 0, :]
                        cs = batch['labels']
                        with torch.cuda.amp.autocast():
                            logits = linear_model(xs)
                            loss = loss_fn(logits, cs)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        losses.append(loss.item())
    
                    epoch_loss = np.mean(losses)
                    if epoch_loss < min_loss:
                        min_loss = epoch_loss
                        patience = 0
                    else:
                        patience += 1
                    tepochs.set_postfix(loss=f'{epoch_loss:.4f} / {min_loss:.4f}')
                    if patience > hparams['patience']:
                        if verbose:
                            print(f'Early stopping at epoch {epoch}')
                        break
    
            # Evaluate on validation set
            linear_model.eval()
            with torch.no_grad():
                for batch in dl_valid:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with torch.no_grad():
                        embeddings = self.target_model.model.get_input_embeddings()(batch['input_ids'])
                        attention_mask = batch['attention_mask']
                        acts = layer_act.attribute(
                            inputs=(None, attention_mask, embeddings),
                            attribute_to_layer_input=True
                        )
                    xs = acts[:, 0, :]
                    ys = batch['labels']
    
                    logits = linear_model(xs)
                    preds = torch.sigmoid(logits) > 0.5
                    metric.update(preds, ys)
    
            CAV = linear_model.weight.detach().clone()
            CAV = CAV / torch.norm(CAV, dim=1, keepdim=True)
            CAVs[layer_name] = CAV.cpu().numpy()
            metrics[layer_name] = metric.compute().detach().cpu().numpy()
    
        return CAVs, metrics


    def generate_CAVs(self, dset_train, dset_valid, hparams=None, n_repeats=5, force_rewrite_cache=False):
        self.CAVs = {layer_name: [] for layer_name in self.layer_names}
        metrics = {layer_name: [] for layer_name in self.layer_names}

        # reload from cache
        if (self.cache_dir is not None) and (not force_rewrite_cache) and \
           (os.path.exists(os.path.join(self.cache_dir, 'CAVs.npz'))) and \
           (os.path.exists(os.path.join(self.cache_dir, 'metrics.npz'))):
            
            raise ValueError("Cached directory already exist. Use `force_rewrite_cache = True` to overwrite.")
            '''
            print("Loading from cache...")
            
            with np.load(os.path.join(self.cache_dir, 'CAVs.npz')) as f:
                self.CAVs.update({k: v for k, v in f.items()})
            with np.load(os.path.join(self.cache_dir, 'metrics.npz')) as f:
                metrics.update({k: v for k, v in f.items()})
            
            if all([len(v) > 0 for v in self.CAVs.values()]):
                return self.CAVs, metrics
            '''

        update_layer_names = [k for k, v in self.CAVs.items() if len(v) == 0]
        print(f'Generating TCAV for layers: {update_layer_names}')

        for _ in trange(n_repeats, desc='#repeats: '):
            CAVs_, metrics_ = self._generate_CAVs(dset_train, dset_valid, hparams=hparams)
            for layer in update_layer_names:
                self.CAVs[layer].append(CAVs_[layer])
                metrics[layer].append(metrics_[layer])

        for layer_name in update_layer_names:
            self.CAVs[layer_name] = np.stack(self.CAVs[layer_name], axis=0)
            metrics[layer_name] = np.stack(metrics[layer_name], axis=0)

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez_compressed(os.path.join(self.cache_dir, 'CAVs.npz'), **self.CAVs)
            np.savez_compressed(os.path.join(self.cache_dir, 'metrics.npz'), **metrics)

        return self.CAVs, metrics

    def _generate_random_CAVs(self, dset_train, dset_valid):
        dl = DataLoader(dset_train, batch_size=32, drop_last=False, 
                        num_workers=8, shuffle=False)
        device = next(self.target_model.parameters())
        
        CAVs = {}
        for layer_name, layer in tqdm(self.layers.items(), leave=False):
            layer_act = LayerActivation(self.target_model, layer)
            # get in_dim
            for batch in dl:
                input_ids = batch['input_ids'].to(device).long()
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.no_grad():
                    embeddings = self.target_model.model.get_input_embeddings()(input_ids)
                    
                embeddings.requires_grad_(True)
                attention_mask = attention_mask.float()
                attention_mask.requires_grad_(True)
                act = layer_act.attribute((None, attention_mask, embeddings), attribute_to_layer_input=True)
                in_dim, out_dim = act.shape[2], labels.shape[1]
                break
            CAV = (torch.rand(out_dim, in_dim) - 1)
            CAV = CAV / torch.norm(CAV, dim=1, keepdim=True)
            CAVs[layer_name] = CAV.cpu().numpy()
        
        return CAVs

    def generate_random_CAVs(self, dset_train, dset_valid, n_repeat=5, force_rewrite_cache=False):
        
        random_CAVs = {layer_name: [] for layer_name in self.layer_names}
        
        # reload from cache
        if (self.cache_dir is not None) and (not force_rewrite_cache) and \
           (os.path.exists(os.path.join(self.cache_dir, 'random_CAVs.npz'))):
            
            raise ValueError("Cached directory already exist. Use `force_rewrite_cache = True` to overwrite.")
            
            '''
            print("Loading from cache...")
            
            with np.load(os.path.join(self.cache_dir, 'random_CAVs.npz')) as f:
                random_CAVs.update({k: v for k, v in f.items()})
            
            if all([len(v) > 0 for v in random_CAVs.values()]):
                return random_CAVs
            '''
        
        update_layer_names = [k for k, v in random_CAVs.items() if len(v) == 0]
        print(f"Generating random TCAV for layers: {update_layer_names}")
        
        for _ in trange(n_repeat):
            random_CAVs_ = self._generate_random_CAVs(dset_train, dset_valid)
            for layer_name in update_layer_names:
                random_CAVs[layer_name].append(random_CAVs_[layer_name])
                
        for layer_name in update_layer_names:
            random_CAVs[layer_name] = np.stack(random_CAVs[layer_name], axis=0)

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True) 
            np.savez_compressed(os.path.join(self.cache_dir, 'random_CAVs.npz'), **random_CAVs)
        
        self.random_CAVs = random_CAVs
        return self.random_CAVs

    def generate_TCAVs(self, dset_valid, layer_name, target_index=None, score_signed=True, 
                   return_ttest_results=False, ttest_threshold=0.05):

        assert len(self.CAVs[layer_name]) == len(self.random_CAVs[layer_name])
        n_repeat = len(self.CAVs[layer_name])
        
        device = next(self.target_model.parameters()).device  # Ensure we get the correct device
        tcavs, random_tcavs = [], []
        
        concept_dl = DataLoader(dset_valid, batch_size=16, shuffle=False, 
                                drop_last=False, num_workers=8)
    
        # Locate the target layer in the model
        for l_name, layer in self.target_model.model.named_modules():
            if l_name == layer_name.split('model')[1][1:]:
                l = layer
    
        for i in trange(n_repeat, leave=False):
            CAV = torch.from_numpy(self.CAVs[layer_name][i]).float().to(device)
            random_CAV = torch.from_numpy(self.random_CAVs[layer_name][i]).float().to(device)
            print(f"CAV shape: {CAV.shape}")
            print(f"Random CAV shape: {random_CAV.shape}")
            # Initialize TCAVScore instances for both the CAV and random CAV
            tcavs_ = TCAVScore(CAV, signed=score_signed).to(device)
            random_tcavs_ = TCAVScore(random_CAV, signed=score_signed).to(device)
    
            # Iterate over validation set batches
            for batch in concept_dl:
                xs = batch['input_ids'].to(device)
                cs = batch['labels'].to(device)
    
                # Compute gradients
                layer_grads_, _ = compute_layer_gradients_and_eval(
                    self.target_model, l, xs, target_ind=target_index, 
                    attribute_to_layer_input=True)
                del _
                
                layer_grads_ = layer_grads_[0][:, 0, :]  # Select the gradient tensor
                # print(f"Gradients shape: {layer_grads_.shape}")
                tcavs_.update(layer_grads_)  # Update TCAVScore for CAV
                random_tcavs_.update(layer_grads_)  # Update TCAVScore for random CAV
    
            # Compute TCAV scores (without summing) and append them to lists
            tcavs.append(tcavs_.compute().detach().cpu().numpy())  # Collect all individual scores
            random_tcavs.append(random_tcavs_.compute().detach().cpu().numpy())  # Collect all individual scores
        
        # Stack the scores to prepare for T-test
        random_tcavs = np.concatenate(random_tcavs, axis=0)  # Shape [N, 5]
        tcavs = np.concatenate(tcavs, axis=0)  # Shape [N, 5]
        
        print(f'TCAVs shape: {tcavs.shape}')
        print(f'Random TCAVs shape: {random_tcavs.shape}')
        
        print('random_tcavs:', random_tcavs.mean(0))
        print('tcavs:', tcavs.mean(0))
    
        # Run two-sided T-test
        ttest_results = []
        for i in range(tcavs.shape[1]):
            # Check for zero variance before running T-test
            # if np.var(tcavs[:, i]) == 0 or np.var(random_tcavs[:, i]) == 0:
            #     print(tcavs[:, i], random_tcavs[:, i], ttest_ind(tcavs[:, i], random_tcavs[:, i]))
            #     print(f"Skipping T-test for index {i} due to zero variance.")
            #     ttest_results.append(np.nan)  # Skip this index
            # else:
            ttest_result = ttest_ind(tcavs[:, i], random_tcavs[:, i])
            ttest_results.append(ttest_result.pvalue)
    
        ttest_results = np.array(ttest_results)
        avg_tcav_scores = tcavs.mean(0)
    
        # Apply the T-test threshold
        avg_tcav_scores[~(ttest_results < ttest_threshold)] = np.nan
        
        if return_ttest_results:
            return avg_tcav_scores, ttest_results
        else:
            return avg_tcav_scores

    def attribute(self, layer_name: str, mode: str, target: Union[int, List[int]] = None,
                 abs: bool = False, use_random: bool = False, select_index: int = None, **kwargs) -> torch.Tensor:

        # print(kwargs.keys())
        assert mode in ['inner_prod', 'cosine_similarity'], "Mode must be 'inner_prod' or 'cosine_similarity'"

        # Select the appropriate CAV
        if use_random:
            assert self.random_CAVs is not None, "Random CAVs have not been generated."
            CAV_ = self.random_CAVs[layer_name].mean(0) if select_index is None else self.random_CAVs[layer_name][select_index]
        else:
            assert self.CAVs is not None, "CAVs have not been generated."
            CAV_ = self.CAVs[layer_name].mean(0) if select_index is None else self.CAVs[layer_name][select_index]

        CAV = torch.from_numpy(CAV_).float().to(device)

        # Prepare inputs as a tuple
        input_ids = kwargs.get('input_ids', None)
        attention_mask = kwargs.get('attention_mask', None)
        inputs_embeds = kwargs.get('inputs_embeds', None)
        inputs_tuple = (input_ids, attention_mask, inputs_embeds)

        # Compute gradients and activations using the custom function
        # print(target, inputs_embeds.shape)
        grads, _ = compute_layer_gradients_and_eval_custom(
            model=self.target_model,
            layer=self.layers[layer_name],
            inputs=inputs_tuple,
            target_ind=target,
            attribute_to_layer_input=True
        )

        grads = grads.squeeze()  # Remove potential singleton dimensions
        if len(grads.shape)==2:
            grads = grads.unsqueeze(0)
        grads = grads[:, 0, :]
        # print(grads.shape)
        # Ensure grads and CAV have compatible shapes
        if grads.dim() == 1:
            grads = grads.unsqueeze(0)  # Shape: [1, Features]
        if CAV.dim() == 1:
            CAV = CAV.unsqueeze(0)  # Shape: [1, Features]

        # Calculate attributions based on the selected mode
        if mode == 'inner_prod':
            attributions = torch.matmul(grads, CAV.T)  # Shape: [Batch, CAVs]
        elif mode == 'cosine_similarity':
            grads_norm = torch.norm(grads, dim=1, keepdim=True)  # Shape: [Batch, 1]
            cav_norm = torch.norm(CAV, dim=1, keepdim=True).T  # Shape: [1, CAVs]
            attributions = torch.matmul(grads, CAV.T) / (grads_norm * cav_norm + 1e-8)  # Shape: [Batch, CAVs]
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

        # Apply absolute value if specified
        if abs:
            attributions = torch.abs(attributions)

        return attributions