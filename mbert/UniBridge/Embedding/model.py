from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

from .configuration import UniBridgeEmbedConfig


class UniBridgeEmbedPretrainedModel(PreTrainedModel):
    config_class = UniBridgeEmbedConfig
    base_model_prefix = "UniBridge_embed"
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


class UniBridgeEmbedding(UniBridgeEmbedPretrainedModel):
    def __init__(self, config: UniBridgeEmbedConfig):
        super().__init__(config)
        if config.pretrain_ckpt == "":
            self.weight = nn.Parameter(
                data=torch.normal(
                    0.0, config.initializer_range, size=(config.vocab_size, config.hidden_size), dtype=torch.float32
                )
            )
        else:
            self.weight = nn.Parameter(torch.load(config.pretrain_ckpt).to(torch.float32))
            config.pretrain_ckpt = ""
            vocab_size, hidden_size = self.weight.shape
            config.vocab_size = vocab_size
            config.hidden_size = hidden_size

        self.config = config

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
        output = F.embedding(input_ids, weight=self.weight, padding_idx=self.config.pad_token_id)
        return output
