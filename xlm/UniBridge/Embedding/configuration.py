from transformers.configuration_utils import PretrainedConfig


class UniBridgeEmbedConfig(PretrainedConfig):
    model_type = "unibridge_embed"

    def __init__(
        self,
        vocab_size=10_002,
        hidden_size=768,
        initializer_range=0.02,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        pretrain_ckpt="",
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.pretrain_ckpt = pretrain_ckpt
