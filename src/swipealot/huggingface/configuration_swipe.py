"""Configuration classes for SwipeTransformer HuggingFace models."""

from transformers import PretrainedConfig


class SwipeTransformerConfig(PretrainedConfig):
    """
    Configuration class for SwipeTransformerModel.

    This configuration stores all the parameters needed to instantiate a
    SwipeTransformerModel. This is the base configuration for the multimodal
    swipe keyboard transformer that processes path coordinates and text.

    Args:
        d_model (int, optional): Hidden dimension size. Defaults to 256.
        n_layers (int, optional): Number of transformer layers. Defaults to 4.
        n_heads (int, optional): Number of attention heads. Defaults to 4.
        d_ff (int, optional): Feedforward dimension. Defaults to 1024.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        vocab_size (int, optional): Size of vocabulary. Defaults to 100.
        max_path_len (int, optional): Maximum path sequence length. Defaults to 64.
        max_char_len (int, optional): Maximum character sequence length. Defaults to 38.
        path_input_dim (int, optional): Path feature dimension. Defaults to 6 for (x, y, dx, dy, ds, log_dt).
        predict_path (bool, optional): Whether to predict path coordinates. Defaults to True.
        predict_path_uncertainty (bool, optional): Whether to predict log sigma for path coords.
            Defaults to False.
        pad_token_id (int, optional): Padding token ID. Defaults to 0.
        cls_token_id (int, optional): CLS token ID. Defaults to 1.
        sep_token_id (int, optional): SEP token ID. Defaults to 2.
        mask_token_id (int, optional): MASK token ID. Defaults to 3.
        unk_token_id (int, optional): Unknown token ID. Defaults to 4.
        eos_token_id (int, optional): End-of-sequence token ID. Defaults to 5.
    """

    model_type = "swipe_transformer"

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        vocab_size: int = 100,
        max_path_len: int = 64,
        max_char_len: int = 38,
        path_input_dim: int = 6,
        predict_char: bool = True,
        predict_path: bool = True,
        predict_path_uncertainty: bool = False,
        predict_length: bool = True,
        pad_token_id: int = 0,
        cls_token_id: int = 1,
        sep_token_id: int = 2,
        mask_token_id: int = 3,
        unk_token_id: int = 4,
        eos_token_id: int = 5,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)

        # Model architecture parameters
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        # Vocabulary and sequence length
        self.vocab_size = vocab_size
        self.max_path_len = max_path_len
        self.max_char_len = max_char_len
        self.path_input_dim = path_input_dim

        # Model capabilities
        self.predict_char = predict_char
        self.predict_path = predict_path
        self.predict_path_uncertainty = predict_path_uncertainty
        self.predict_length = predict_length

        # Special tokens
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        self.unk_token_id = unk_token_id
