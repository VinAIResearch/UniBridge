import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AdapterConfig, BertAdapterModel, PreTrainedModel

from .configuration import NLIAdapterConfig
from .head import ClassificationHead, ClassificationHeadConfig


class NLIAdapterPreTrainedModel(PreTrainedModel):
    config_class = NLIAdapterConfig
    base_model_prefix = "nli_adapter"
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


class NLIAdapterModel(NLIAdapterPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: NLIAdapterConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        if config.pretrained_ck == "":
            self.model = BertAdapterModel(config)  # , add_pooling_layer=False)
        else:
            self.model = BertAdapterModel.from_pretrained(
                config.pretrained_ck
            )  # , add_pooling_layer=False)#, output_hidden_states=True)

        adapter_mode = 0
        if config.lang_adapter_ckpt == "":
            adapter_mode = 1
        else:
            # load language adapter
            self.model.load_adapter(config.lang_adapter_ckpt, load_as=config.lang, leave_out=[11])
            adapter_mode = 2

        if config.task_adapter_ckpt == "":
            # add task adapter
            task_config = AdapterConfig.load("pfeiffer", non_linearity="gelu", reduction_factor=16, leave_out=[11])
            self.model.add_adapter(f"{config.lang}_nli", config=task_config)
            # self.model.add_classification_head(f"{config.lang}_nli", num_labels=config.num_labels)
            self.head = ClassificationHead(
                ClassificationHeadConfig(
                    hidden_size=config.hidden_size,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    num_labels=config.num_labels,
                )
            )
        else:
            self.model.load_adapter(config.task_adapter_ckpt, load_as=f"{config.lang}_nli", leave_out=[11])
            self.head = ClassificationHead.from_pretrained(config.task_adapter_ckpt + "/head")

        if adapter_mode == 1:
            self.model.set_active_adapters([f"{config.lang}_nli"])
        elif adapter_mode == 2:
            self.model.set_active_adapters([config.lang, f"{config.lang}_nli"])
        else:
            assert f"Unknown adapter mode of {adapter_mode}"
        self.model.train_adapter([f"{config.lang}_nli"])

        self.config = config

    def save_pretrained(self, path):
        if not os.path.exists(f"{path}/{self.config.lang}"):
            os.makedirs(f"{path}/{self.config.lang}")
        # save this wrap model config
        self.model.config.save_pretrained(f"{path}/{self.config.lang}")
        self.model.save_adapter(f"{path}/{self.config.lang}/adapter", f"{self.config.lang}_nli")  # , with_head=True)
        self.head.save_pretrained(f"{path}/{self.config.lang}/head")

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        inputs = outputs.last_hidden_state
        outputs = self.head(inputs, attention_mask)

        # return outputs['logits']
        return outputs
