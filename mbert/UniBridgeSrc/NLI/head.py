from typing import Dict

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


class ClassificationHeadConfig(PretrainedConfig):
    model_type = "classification_head"

    def __init__(
        self,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.num_labels = num_labels


class ClassificationPreTrainedHead(PreTrainedModel):
    config_class = ClassificationHeadConfig
    base_model_prefix = "classification_head"
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


class ClassificationHead(ClassificationPreTrainedHead):
    def __init__(self, config: ClassificationHeadConfig):
        super().__init__(config)

        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.act_fn = nn.Tanh()
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def __mean_pooling(self, model_output: Dict, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, inputs, attention_mask, **kwargs):
        mean_token_embed = self.__mean_pooling(inputs, attention_mask)
        intermediate = self.act_fn(self.dense(self.dropout(mean_token_embed)))
        out = self.classifier(self.dropout(intermediate))
        return out
