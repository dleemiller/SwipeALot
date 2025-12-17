"""HuggingFace-compatible tokenizer for SwipeTransformer."""

import json
import os

from transformers import PreTrainedTokenizer

from ..data.tokenizer import CharacterTokenizer


class SwipeTokenizer(PreTrainedTokenizer):
    """
    HuggingFace-compatible tokenizer that wraps the existing CharacterTokenizer.

    This tokenizer provides a HuggingFace-compatible interface for the custom
    character-level tokenization used in the swipe keyboard model.

    Args:
        vocab_file (str, optional): Path to vocabulary file
        unk_token (str): Unknown token. Defaults to "[UNK]"
        sep_token (str): Separator token. Defaults to "[SEP]"
        pad_token (str): Padding token. Defaults to "[PAD]"
        cls_token (str): Classification token. Defaults to "[CLS]"
        mask_token (str): Mask token. Defaults to "[MASK]"
        eos_token (str): End-of-sequence token. Defaults to "[EOS]"
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | None = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        eos_token: str = "[EOS]",
        **kwargs,
    ):
        # Initialize internal CharacterTokenizer BEFORE calling super().__init__()
        # because super().__init__() will call get_vocab() which needs self._tokenizer
        if vocab_file is not None and os.path.exists(vocab_file):
            # Load from vocab file
            with open(vocab_file, encoding="utf-8") as f:
                vocab_data = json.load(f)

            # Extract vocabulary (excluding ALL special tokens)
            # All special tokens that should NOT be passed to CharacterTokenizer
            # Convert AddedToken objects to strings
            special_tokens_to_exclude = {
                str(pad_token),
                str(cls_token),
                str(sep_token),
                str(mask_token),
                str(unk_token),
                str(eos_token),
                "[PUNC]",
            }

            if "chars" in vocab_data:
                # Filter out special tokens from the chars list
                vocab = set(c for c in vocab_data["chars"] if c not in special_tokens_to_exclude)
            elif "char_to_id" in vocab_data:
                # Get all characters except special tokens
                vocab = set(
                    c for c in vocab_data["char_to_id"].keys() if c not in special_tokens_to_exclude
                )
            else:
                vocab = None

            self._tokenizer = CharacterTokenizer(vocab=vocab)
        else:
            # Default vocab (will be built from dataset during conversion)
            self._tokenizer = CharacterTokenizer()

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary"""
        return self._tokenizer.vocab_size

    def get_vocab(self):
        """Return the vocabulary as a dict"""
        return self._tokenizer.char_to_id.copy()

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize a string into tokens (characters).

        Args:
            text (str): Text to tokenize

        Returns:
            List[str]: List of character tokens
        """
        # Convert to lowercase and split into characters
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        """
        Convert a token (character) to an id using the vocabulary.

        Args:
            token (str): Token to convert

        Returns:
            int: Token ID
        """
        return self._tokenizer.char_to_id.get(token, self._tokenizer.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        """
        Convert an index to a token using the vocabulary.

        Args:
            index (int): Token ID

        Returns:
            str: Token (character)
        """
        return self._tokenizer.id_to_char.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """
        Convert a list of tokens (characters) to a string.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            str: Concatenated string
        """
        # Filter out special tokens (must include [PUNC] which represents punctuation)
        special_tokens = {
            self.pad_token,
            self.cls_token,
            self.sep_token,
            self.mask_token,
            self.unk_token,
            self.eos_token,
            "[PUNC]",  # Punctuation token from CharacterTokenizer
        }
        filtered = [t for t in tokens if t not in special_tokens]
        return "".join(filtered)

    def save_pretrained(
        self,
        save_directory,
        legacy_format=None,
        filename_prefix=None,
        push_to_hub=False,
        **kwargs,
    ):
        """
        Save the tokenizer to a directory, ensuring auto_map is included.
        """
        # Call parent save_pretrained
        result = super().save_pretrained(
            save_directory,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
            **kwargs,
        )

        # Add auto_map to tokenizer_config.json for AutoTokenizer compatibility
        from pathlib import Path

        tokenizer_config_path = Path(save_directory) / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, "r") as f:
                config = json.load(f)

            config["auto_map"] = {
                "AutoTokenizer": ["tokenization_swipe.SwipeTokenizer", None]
            }

            with open(tokenizer_config_path, "w") as f:
                json.dump(config, f, indent=2)

        return result

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple:
        """
        Save the tokenizer vocabulary to a directory.

        Args:
            save_directory (str): Directory to save the vocabulary
            filename_prefix (str, optional): Optional prefix for the vocabulary file

        Returns:
            tuple: Tuple containing the path to the saved vocabulary file
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        # Save vocabulary and mappings
        vocab_data = {
            "chars": sorted(list(set(self._tokenizer.char_to_id.keys()))),
            "char_to_id": self._tokenizer.char_to_id,
            "special_tokens": {
                "pad_token": self.pad_token,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "mask_token": self.mask_token,
                "unk_token": self.unk_token,
                "eos_token": self.eos_token,
            },
        }

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Build model inputs from a sequence by adding special tokens.

        For swipe models, we don't add special tokens here as they are
        handled separately (CLS and SEP are managed by the model/processor).

        Args:
            token_ids_0 (List[int]): First sequence
            token_ids_1 (List[int], optional): Second sequence

        Returns:
            List[int]: Sequence with special tokens
        """
        # For swipe models, special tokens are handled by the processor
        # Just return the tokens as-is
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        """
        Retrieve sequence ids from a token list.

        Args:
            token_ids_0 (List[int]): First sequence
            token_ids_1 (List[int], optional): Second sequence
            already_has_special_tokens (bool): Whether tokens already have special tokens

        Returns:
            List[int]: Mask (1 for special tokens, 0 for normal tokens)
        """
        # All special token handling is done by the processor
        # Return all zeros
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence already has special tokens."
                )
            return [0] * len(token_ids_0)

        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * len(token_ids_0) + [0] * len(token_ids_1)
