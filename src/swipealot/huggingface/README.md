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

The processor accepts **path coordinates**, **text**, or **both**. It produces the
tensors the model expects: `path_coords`, `input_ids`, and `attention_mask`.

#### Path features (8D Savitzky-Golay)

Path coordinates are 8-dimensional:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | x | Horizontal position (normalised to [0, 1]) |
| 1 | y | Vertical position (normalised to [0, 1]) |
| 2 | dx | First derivative of x (SG window=7, poly=2) |
| 3 | dy | First derivative of y |
| 4 | d2x | Second derivative of x |
| 5 | d2y | Second derivative of y |
| 6 | speed | sqrt(dx^2 + dy^2) |
| 7 | curvature | d(atan2(dy, dx))/dt, clamped to [-2, 2] |

When you pass **raw `{"x", "y", "t"}` dict paths** (or `[N, 3]` NumPy/tensor), the processor
automatically computes these 8D features using `preprocess_raw_path_to_sg_features`.

- `path_coords`: `[batch, max_path_len, 8]`
- `input_ids`: `[batch, max_char_len]` (character tokens including EOS + padding)
- `attention_mask`: `[batch, 1 + max_path_len + 1 + max_char_len]` for `[CLS] + path + [SEP] + text`

#### Usage examples

**Path-only (e.g. swipe recognition):**
```python
raw_path = [{"x": 0.1, "y": 0.2, "t": 0}, {"x": 0.3, "y": 0.4, "t": 10}, ...]
inputs = processor(path_coords=raw_path, text=None, return_tensors="pt")
outputs = model(**inputs)
```

**Text-only (e.g. language modelling):**
```python
inputs = processor(path_coords=None, text="hello", return_tensors="pt")
outputs = model(**inputs)
```

**Both path and text:**
```python
inputs = processor(path_coords=raw_path, text="hello", return_tensors="pt")
outputs = model(**inputs)
```

#### Model outputs

The default output type is `SwipeTransformerOutput` and includes:

- `char_logits`: `[batch, char_len, vocab_size]` (text segment only; no logits for path/CLS/SEP).
- `path_logits`: `[batch, path_len, path_input_dim]` (path segment only).
- `path_log_sigma`: `[batch, path_len, path_input_dim]` (log sigma for path coords when enabled).
- `length_logits`: `[batch]` (regressed length from CLS).
- `pooler_output`: `[batch, d_model]` (SEP embedding used for contrastive/similarity).
- `attentions`: per-layer attention weights when `output_attentions=True`.

### SavgolPreprocessor (PyTorch module)

Each checkpoint includes a `savgol.py` file containing `SavgolPreprocessor`, a frozen
`nn.Module` that computes the same 8D SG features as the NumPy preprocessing but using
PyTorch Conv1d operations. This is useful for on-device or GPU-accelerated preprocessing
pipelines.

```python
from savgol import SavgolPreprocessor

sg = SavgolPreprocessor()
xy = torch.tensor(...)  # [B, 2, T] — channel 0 = x, channel 1 = y (already resampled)
features = sg(xy)       # [B, T, 8]
```

Note: `SavgolPreprocessor` expects **already-resampled** `(x, y)` positions. Resampling
(from variable-length raw paths to fixed `max_path_len`) must be done beforehand.
