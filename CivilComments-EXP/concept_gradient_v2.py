from typing import Any, Callable, Sequence
from tqdm.auto import tqdm
import torch

from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
    compute_layer_gradients_and_eval
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage

class ConceptGradients(GradientAttribution):
    def __init__(self, forward_func: Callable, concept_forward_func: Callable, x2y_model, x2c_model)->None:
        GradientAttribution.__init__(self, forward_func)
        self.concept_forward_func = concept_forward_func
        self.target_model = x2y_model.model
        self.concept_model = x2c_model.model

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        mode:str,
        target:TargetType = None,
        target_concept: TargetType = None,
        n_concepts: TargetType = 1,
        target_layer_name: str = None,
        concept_layer_name: str = None,
        abs: bool = False,
        additional_forward_args: Any = None,
        additional_concept_forward_args:Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        
        assert mode in [
            'chain_rule_joint',
            'chain_rule_independent',
            'cav',
            'inner_prod',
            'cosine_similarity'
        ]
        
        _is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs) #setting requires_grad to True for all input tensors
        if target_layer_name is None:
            dydxs = self.gradient_func(
                self.forward_func, inputs, target, additional_forward_args
            )

        else:
            target_layer = self.get_named_module(self.target_model, target_layer_name)
            dydxs, acts = compute_layer_gradients_and_eval(
                self.forward_func, target_layer, inputs, target_ind=target, additional_forward_args=additional_forward_args, attribute_to_layer_input=True
            )
            del acts
        # print(dydxs[0].shape)
        dydxs = tuple(dydxs_[:, 0, :].detach() for dydxs_ in dydxs)
        # dydxs = tuple(dydxs_.detach() for dydxs_ in dydxs)

        if target_concept is None:
            dcdxs = None
            for ci in range(n_concepts):
                if concept_layer_name is None:
                    dcdxs_ = self.gradient_func(
                        self.concept_forward_func, inputs, ci, additional_concept_forward_args
                    )
                else:
                    concept_layer = self.get_named_module(self.concept_model, concept_layer_name)
                    dcdxs_, acts = compute_layer_gradients_and_eval(
                        self.concept_forward_func, concept_layer, inputs, target_ind=ci, additional_forward_args=additional_concept_forward_args, attribute_to_layer_input=True
                    )
                    del acts
                dcdxs_ = tuple(dcdxs__[:, 0, :].detach() for dcdxs__ in dcdxs_)
                # dcdxs_ = tuple(dcdxs__.detach() for dcdxs__ in dcdxs_)
                if dcdxs is None:
                    dcdxs = tuple(torch.empty([dcdxs__.shape[0], n_concepts, dcdxs__.shape[1]]).to(dcdxs__.device) 
                                  for dcdxs__ in dcdxs_)
                # print(f"dydxs shape: {[d.shape for d in dydxs]}")
                # print(f"dcdxs shape: {[d.shape for d in dcdxs]}")
                for i in range(len(dcdxs_)):
                    dcdxs[i][:, ci, :] = dcdxs_[i]
                    
        else:
            if concept_layer_name is None:
                dcdxs = self.gradient_func(
                    self.concept_forward_func, inputs, target_concept, additional_concept_forward_args
                )
            else:
                concept_layer = self.get_named_module(self.concept_model, concept_layer_name)
                dcdxs, acts = compute_layer_gradients_and_eval(
                    self.concept_forward_func, concept_layer, inputs, target_ind=target_concept,
                    additional_forward_args=additional_concept_forward_args, attribute_to_layer_input=True)
                del acts
            dcdxs = tuple(dcdxs_[:, 0, :].detach() for dcdxs_ in dcdxs)
            # dcdxs = tuple(dcdxs_.detach() for dcdxs_ in dcdxs)
        
        assert len(dydxs) == len(dcdxs)
        # print(dcdxs[0].shape, dydxs[0].shape)
        with torch.no_grad():
            if mode == 'chain_rule_joint':
                # gradients = tuple(
                #     torch.linalg.lstsq(torch.transpose(dcdx, 1, 2), dydx).solution
                #     for dydx, dcdx in zip(dydxs, dcdxs)
                # )
                gradients = tuple(
        (torch.sum(dydx * dcdx, dim=1) / (torch.norm(dcdx, dim=1) ** 2)).unsqueeze(1)
        for dydx, dcdx in zip(dydxs, dcdxs)
    )
            elif mode == 'chain_rule_independent':
                gradients = tuple(torch.bmm(dcdx, dydx.unsqueeze(-1)).squeeze(-1) / (torch.norm(dcdx, dim=2)**2)
                                  for dydx, dcdx in zip(dydxs, dcdxs))
            elif mode == 'cav':
                gradients = tuple(torch.bmm(dcdx, dydx.unsqueeze(-1)).squeeze(-1) / torch.norm(dcdx, dim=2)
                                  for dydx, dcdx in zip(dydxs, dcdxs))
            elif mode == 'inner_prod':
                gradients = tuple(torch.bmm(dcdx, dydx.unsqueeze(-1)).squeeze(-1)
                                  for dydx, dcdx in zip(dydxs, dcdxs))
            elif mode == 'cosine_similarity':
                gradients = tuple(torch.bmm(dcdx, dydx.unsqueeze(-1)).squeeze(-1) / \
                                  (torch.norm(dcdx, dim=2) * torch.norm(dydx, dim=1, keepdim=True))
                                  for dydx, dcdx in zip(dydxs, dcdxs))
            else:
                raise NotImplementedError
                
        if abs:
            attributions = tuple(torch.abs(gradient) for gradient in gradients)
        else:
            attributions = gradients
        undo_gradient_requirements(inputs, gradient_mask)
        return _format_output(_is_inputs_tuple, attributions)

    @staticmethod
    def get_named_module(model, name):
        for module_name, module in model.named_modules():
            # print(list(model.named_modules()))
            # break
            if module_name == name:
                # print(module_name)
                return module
        raise ValueError(f"{name} not found in model.")