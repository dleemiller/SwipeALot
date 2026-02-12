"""Utilities for preparing HuggingFace-compatible checkpoints."""

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_checkpoint_for_hub(output_dir: str | Path) -> None:
    """
    Prepare a checkpoint directory for HuggingFace Hub by copying necessary files
    and fixing imports for standalone loading with trust_remote_code=True.

    This function:
    1. Copies custom modeling files (modeling_swipe.py, configuration_swipe.py, etc.)
    2. Fixes imports in modeling_swipe.py to use flat file structure
    3. Copies model dependencies (embeddings.py, heads.py, tokenizer.py)
    4. Fixes imports in tokenization_swipe.py
    5. Copies preprocessing dependencies for SwipeProcessor
    6. Updates config.json with auto_map for AutoModel

    Args:
        output_dir: Path to checkpoint directory

    Note:
        Tokenizer and processor auto_maps are handled by their respective
        save_pretrained methods, not by this function.
    """
    output_dir = Path(output_dir)

    try:
        # Find the swipealot package root
        import swipealot

        package_root = Path(swipealot.__file__).parent
        src_dir = package_root / "huggingface"

        # Copy main HuggingFace integration files
        modeling_files = [
            "modeling_swipe.py",
            "configuration_swipe.py",
            "tokenization_swipe.py",
            "processing_swipe.py",
        ]

        for filename in modeling_files:
            src_file = src_dir / filename
            if src_file.exists():
                dest_file = output_dir / filename
                shutil.copy(src_file, dest_file)

                # Fix imports in modeling_swipe.py for standalone loading
                if filename == "modeling_swipe.py":
                    content = dest_file.read_text()
                    # Change relative imports to flat structure
                    content = content.replace("from ..models.embeddings", "from .embeddings")
                    content = content.replace("from ..models.heads", "from .heads")
                    dest_file.write_text(content)

        # Copy model component dependencies (embeddings, heads) as flat files
        models_dir = package_root / "models"
        dependency_files = {
            "embeddings.py": "embeddings.py",
            "heads.py": "heads.py",
            "savgol.py": "savgol.py",
        }

        for src_name, dest_name in dependency_files.items():
            src_file = models_dir / src_name
            if src_file.exists():
                shutil.copy(src_file, output_dir / dest_name)

        # Copy tokenizer.py from data directory
        data_dir = package_root / "data"
        tokenizer_file = data_dir / "tokenizer.py"
        if tokenizer_file.exists():
            shutil.copy(tokenizer_file, output_dir / "tokenizer.py")

        # Copy preprocessing helpers used by SwipeProcessor
        preprocessing_file = data_dir / "preprocessing.py"
        if preprocessing_file.exists():
            shutil.copy(preprocessing_file, output_dir / "preprocessing.py")

        # Fix imports in tokenization_swipe.py for standalone loading
        tokenization_file = output_dir / "tokenization_swipe.py"
        if tokenization_file.exists():
            content = tokenization_file.read_text()
            # Change relative imports to flat structure
            content = content.replace("from ..data.tokenizer", "from .tokenizer")
            tokenization_file.write_text(content)

        # Fix imports in processing_swipe.py for standalone loading
        processing_file = output_dir / "processing_swipe.py"
        if processing_file.exists():
            content = processing_file.read_text()
            content = content.replace("from ..data.preprocessing", "from .preprocessing")
            processing_file.write_text(content)

        # Update config.json with auto_map for AutoModel loading
        config_path = output_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)

            config_dict["auto_map"] = {
                "AutoConfig": "configuration_swipe.SwipeTransformerConfig",
                "AutoModel": "modeling_swipe.SwipeTransformerModel",
                "AutoModelForCausalLM": "modeling_swipe.SwipeTransformerModel",
            }

            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

    except Exception as e:
        # Don't fail checkpoint save if remote code copy fails
        logger.warning(f"Failed to prepare checkpoint for hub: {e}")
