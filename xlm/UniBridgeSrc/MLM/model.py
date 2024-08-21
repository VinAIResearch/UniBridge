import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AdapterConfig, BertAdapterModel, PreTrainedModel, XLMRobertaAdapterModel

from .configuration import MLMAdapterConfig


class MLMAdapterPreTrainedModel(PreTrainedModel):
    config_class = MLMAdapterConfig
    base_model_prefix = "mlm_adapter"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs


class MLMAdapterModel(MLMAdapterPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: MLMAdapterConfig):
        super().__init__(config)
        if config.model in ["mbert", "mbert_cased"]:
            basemodel_class = BertAdapterModel
        elif config.model == "xlm-r":
            basemodel_class = XLMRobertaAdapterModel
        else:
            assert f"parameter `model` for MLMAdapterModel must be either ['mbert', 'xlm-r'], {config.model} does not belong to that."

        if config.pretrained_ck == "":
            self.model = basemodel_class(config)
        else:
            self.model = basemodel_class.from_pretrained(config.pretrained_ck)
            config.pretrained_ck = ""

        if config.pretrained_mlm_adapter == "":
            lang_config = AdapterConfig.load(
                "pfeiffer",
                non_linearity="relu",
                reduction_factor=2,
                inv_adapter="nice",
                inv_adapter_reduction_factor=2,
                leave_out=[11],
            )
            self.model.add_adapter(config.lang, config=lang_config)
            self.model.add_masked_lm_head(f"{config.lang}_lang")
        else:
            self.model.load_adapter(config.pretrained_mlm_adapter + "/adapter", load_as=config.lang, leave_out=[11])
            # self.model.add_masked_lm_head(f'{config.lang}_lang')
            self.model.load_head(config.pretrained_mlm_adapter + "/head", load_as=f"{config.lang}_lang")

        self.model.set_active_adapters([config.lang])
        self.model.train_adapter([config.lang])

        self.config = config

        # self.post_init()

    def save_pretrained(self, path):
        if not os.path.exists(f"{path}/{self.config.lang}"):
            os.makedirs(f"{path}/{self.config.lang}")
        # save this wrap model config
        self.model.config.save_pretrained(f"{path}/{self.config.lang}")
        self.model.save_adapter(f"{path}/{self.config.lang}/adapter", self.config.lang)
        self.model.save_head(f"{path}/{self.config.lang}/head", f"{self.config.lang}_lang")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tensor:
        return self.model(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
        )["logits"]
