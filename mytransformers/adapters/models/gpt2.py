from typing import Union

import torch
from torch import nn
import numpy as np

from ..composition import AdapterCompositionBlock, parse_composition
from ..heads import ClassificationHead, MultiLabelClassificationHead
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin
from .bert import (
    BertEncoderAdaptersMixin,
    BertOutputAdaptersMixin,
    BertSelfOutputAdaptersMixin,
    ModelWithFlexibleHeadsAdaptersMixin,
)


class GPT2AttentionAdaptersModule(BertSelfOutputAdaptersMixin, nn.Module):
    """Adds attention adapters to the Transformer module of DistilBert."""

    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def layer_norm(self):
        return None


class GPT2OutputAdaptersModule(BertOutputAdaptersMixin, nn.Module):
    """Adds output adapters to the Transformer module of DistilBert."""

    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def layer_norm(self):
        return None


class GPT2DecoderBlockAdaptersMixin(BertEncoderAdaptersMixin):
    """Adds adapters to the TransformerBlock module of DistilBert."""

    def _init_adapter_modules(self):
        self.attention_adapters = GPT2AttentionAdaptersModule(self)
        self.output_adapters = GPT2OutputAdaptersModule(self)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()
        self.adapter_function = []

    def add_fusion_layer(self, adapter_names):
        self.attention_adapters.add_fusion_layer(adapter_names)
        self.output_adapters.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.attention_adapters.add_adapter(adapter_name, layer_idx)
        self.output_adapters.add_adapter(adapter_name, layer_idx)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        self.attention_adapters.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)
        self.output_adapters.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)


class GPT2ModelAdapterMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        super()._init_adapter_modules()

        # add adapters specified in config; invertible adapter will only be added if required
        for adapter_name in self.config.adapters.adapters:
            self._add_adapter(adapter_name)
        # fusion
        if hasattr(self.config, "fusion_models"):
            for fusion_adapter_names in self.config.fusion_models:
                self.add_fusion_layer(fusion_adapter_names)

    def add_adapter(self, adapter_name: str, config=None):
        """
        Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        self.config.adapters.add(adapter_name, config=config)
        self._add_adapter(adapter_name)

    def add_adapter_for_mix(self, adapter_name: str, config=None):
        new_added = []
        for i, layer in enumerate(self.base_model.h):
            c_adapters = np.unique(layer.adapter_function)
            for old in c_adapters:
                new_name = adapter_name + "-" + old
                if new_name not in new_added:
                    new_added.append(new_name)
                    self.config.adapters.add(new_name, config=config)
                layer.add_adapter(new_name, i)
                layer.load_attention_weight(old, new_name)
                layer.load_output_weight(old, new_name)

            if i == 0:
                for name, para in layer.named_parameters():
                    if "adapter_down.0.bias" in name:
                        print(name)
                        print(para)

    def modify_list(self, change_layer, source_task, target_task):
        for i, layer in enumerate(self.base_model.h):
            if i == change_layer:
                print("Layer {}".format(i))
                print("previous: {}".format(layer.adapter_function))
                layer.adapter_function[target_task] = layer.adapter_function[source_task]
                print("changed: {}".format(layer.adapter_function))

    def train_adapter_for_mix(self, adapter_name):
        new_added = []
        for i, layer in enumerate(self.base_model.h):
            for c in layer.adapter_function:
                if adapter_name in c and c not in new_added:
                    new_added.append(str(c))

        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(new_added)
        print(adapter_setup)
        self.enable_adapters(adapter_setup, True, False)
        self.enable_invertible_adapters(adapter_setup.flatten())
        self.set_active_adapters(adapter_setup)

    def train_adapter_subname(self, adapter_names):
        new_added = []
        for i, layer in enumerate(self.base_model.h):
            for c in layer.adapter_function:
                for name in adapter_names:
                    if name in c and c not in new_added:
                        new_added.append(str(c))

        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(new_added)
        print(adapter_setup)
        self.enable_adapters(adapter_setup, True, False)
        self.enable_invertible_adapters(adapter_setup.flatten())
        self.set_active_adapters(adapter_setup)

    def get_path(self):
        path = []
        for i, layer in enumerate(self.base_model.h):
            path.append(layer.adapter_function)
        return path

    def _add_adapter(self, adapter_name: str):
        adapter_config = self.config.adapters.get(adapter_name)
        leave_out = adapter_config.get("leave_out", [])
        for i, layer in enumerate(self.base_model.h):
            if i not in leave_out:
                layer.add_adapter(adapter_name, i)

        self.add_invertible_adapter(adapter_name)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock], skip_layers=None):
        self.train()
        self.freeze_model(True)
        # print(adapter_setup)
        adapter_setup = parse_composition(adapter_setup)
        # print(adapter_setup)
        self.enable_adapters(adapter_setup, True, False)
        self.enable_invertible_adapters(adapter_setup.flatten())
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup, skip_layers)

    def adapter_transfer(self):
        self.train()
        self.freeze_model(True)
        adapter_list = self.enable_adapter_transfer()
        adapter_list = parse_composition(adapter_list)
        self.set_active_adapters(adapter_list)

    def enable_adapter_transfer(self):
        adapter_list = []
        for i, layer in enumerate(self.base_model.h):
            c_list = layer.enable_adapter_transfer()
            print("Layer {}, Enable {}".format(i, c_list))
            for item in c_list:
                if str(item) not in adapter_list:
                    adapter_list.append(str(item))
        print("Activate: {}".format(adapter_list))
        return adapter_list

    def train_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.enable_adapters(adapter_setup, unfreeze_adapters, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        for layer in self.base_model.h:
            layer.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)

    def setup_task_adapter(self, tid):
        res = []
        for cnt, layer in enumerate(self.base_model.h):
            c = layer.setup_task_adapter(cnt, tid)
            res.append(c)
        print(res)
        cnt_true = 0
        for i in res:
            if i:
                cnt_true += 1
        return cnt_true

    def add_adapter_by_list(self, adapter_list: list, config=None):
        names = []
        # add name to config
        for layer_adapter in adapter_list:
            for name in layer_adapter:
                if name not in names:
                    names.append(name)
                    self.config.adapters.add(name, config=config)

        for i, layer in enumerate(self.base_model.h):
            names = []
            new_list = []
            for name in adapter_list[i]:
                new_list.append(str(name))
                if name not in names:
                    names.append(name)
                    layer.add_adapter(name, i)
            layer.adapter_function = new_list
            print("Layer:{}".format(i))
            print(layer.adapter_function)
        print(adapter_list)

    def update_adapter_list(self, adapter_list: list):
        for i, layer in enumerate(self.base_model.h):
            new_list = []
            for name in adapter_list[i]:
                new_list.append(str(name))
            layer.adapter_function = new_list
            print("Layer:{}".format(i))
            print(layer.adapter_function)

    def get_adapter_list(self):
        final_list = []
        for i, layer in enumerate(self.base_model.h):
            final_list.append(layer.adapter_function)
        return final_list

    def adjust_attention_mask_for_parallel(self, hidden_states, attention_mask):
        if attention_mask is not None and hidden_states.shape[0] != attention_mask.shape[0]:
            repeats = [1] * len(attention_mask.shape)
            repeats[0] = hidden_states.shape[0] // attention_mask.shape[0]
            attention_mask = attention_mask.repeat(*repeats)
        return attention_mask

    def _add_fusion_layer(self, adapter_names):
        for layer in self.base_model.h:
            layer.add_fusion_layer(adapter_names)

    def get_fusion_regularization_loss(self):
        reg_loss = 0.0
        target = torch.zeros((self.config.hidden_size, self.config.hidden_size)).fill_diagonal_(1.0).to(self.device)
        for _, v in self.base_model.h._modules.items():

            for _, layer_fusion in v.output_adapters.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

            for _, layer_fusion in v.attention_adapters.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

        return reg_loss

    def get_adapter(self, name):
        return_adapters = {}
        for idx, layer in enumerate(self.h):
            adapters = {
                "attention": layer.attention_adapters.adapters,
                "output": layer.output_adapters.adapters,
            }
            for key, adapt in adapters.items():
                if hasattr(adapt, name):
                    if idx not in return_adapters:
                        return_adapters[idx] = {}
                    return_adapters[idx][key] = getattr(adapt, name)

        return return_adapters


class GPT2ModelHeadsMixin(ModelWithFlexibleHeadsAdaptersMixin):
    """Adds flexible heads to a GPT-2 model."""

    head_types = {
        "classification": ClassificationHead,
        "multilabel_classification": MultiLabelClassificationHead,
    }

    def add_classification_head(
        self,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        multilabel=False,
        id2label=None,
    ):
        """
        Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """

        if multilabel:
            head = MultiLabelClassificationHead(self, head_name, num_labels, layers, activation_function, id2label)
        else:
            head = ClassificationHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)
