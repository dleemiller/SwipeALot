## HuggingFace integration

SwipeALot provides a HuggingFace-compatible model, tokenizer, and processor so checkpoints can be
loaded via `trust_remote_code=True` and used with `AutoModel`, `AutoTokenizer`, and `AutoProcessor`.

### Loading a saved checkpoint

```python
from transformers import AutoModel, AutoProcessor

ckpt = "checkpoints/.../checkpoint-6000"  # or a Hub repo id
model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
```

### Processor inputs / model outputs

The model expects *both* `path_coords` and `input_ids`:

- `path_coords`: `[batch, max_path_len, path_input_dim]` (typically engineered `(x, y, dx, dy, ds, log_dt)`).
- `input_ids`: `[batch, max_char_len]` (character tokens including EOS + padding).
- `attention_mask`: `[batch, 1 + max_path_len + 1 + max_char_len]` for `[CLS] + path + [SEP] + text`.

The default output type is `SwipeTransformerOutput` and includes:

- `char_logits`: `[batch, char_len, vocab_size]` (text segment only; no logits for path/CLS/SEP).
- `path_logits`: `[batch, path_len, path_input_dim]` (path segment only).
- `path_log_sigma`: `[batch, path_len, path_input_dim]` (log sigma for path coords when enabled).
- `length_logits`: `[batch]` (regressed length from CLS).
- `pooler_output`: `[batch, d_model]` (SEP embedding used for contrastive/similarity).
- `attentions`: per-layer attention weights when `output_attentions=True`.
