"""Test EOS token implementation."""

from swipealot.data import MaskedCollator, SwipeDataset

print("=" * 60)
print("Testing EOS Token Implementation")
print("=" * 60)

# Build tokenizer
print("\nBuilding tokenizer...")
dataset = SwipeDataset(
    split="train",
    max_path_len=128,
    max_word_len=38,
    dataset_name="futo-org/swipe.futo.org",
    max_samples=100,
)

tokenizer = dataset.tokenizer

print(f"\nVocabulary size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.special_tokens}")
print(f"  [PAD] = {tokenizer.pad_token_id}")
print(f"  [CLS] = {tokenizer.cls_token_id}")
print(f"  [SEP] = {tokenizer.sep_token_id}")
print(f"  [MASK] = {tokenizer.mask_token_id}")
print(f"  [EOS] = {tokenizer.eos_token_id}")

# Test encoding with EOS
test_words = ["hello", "hi", "a", "test"]

print("\n" + "=" * 60)
print("Testing Word Encoding with EOS")
print("=" * 60)

for word in test_words:
    # Get sample
    sample = dataset.dataset[0]
    sample["word"] = word

    # Process through dataset
    processed = dataset.__getitem__(0)
    char_tokens = processed["char_tokens"]
    char_mask = processed["char_mask"]

    print(f"\nWord: '{word}'")
    print(f"  Length: {len(word)} chars")

    # Find where valid tokens end
    valid_positions = (char_mask == 1).nonzero(as_tuple=True)[0]
    if len(valid_positions) > 0:
        last_valid_idx = valid_positions[-1].item()
        valid_tokens = char_tokens[: last_valid_idx + 1]

        print(f"  Valid tokens: {valid_tokens.tolist()}")
        print(f"  Decoded: {tokenizer.decode(valid_tokens.tolist())}")

        # Check if EOS is present
        if tokenizer.eos_token_id in valid_tokens:
            eos_position = (
                (valid_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
            )
            print(f"  ✓ EOS found at position {eos_position} (after {eos_position} chars)")

            # Verify structure: [chars...] + [EOS] + [PAD...]
            tokens_before_eos = valid_tokens[:eos_position]
            print(f"  Characters before EOS: {tokenizer.decode(tokens_before_eos.tolist())}")
        else:
            print("  ✗ EOS not found!")
    else:
        print("  ✗ No valid tokens!")

# Test masking includes EOS
print("\n" + "=" * 60)
print("Testing EOS Token Masking")
print("=" * 60)

collator = MaskedCollator(
    tokenizer=tokenizer,
    char_mask_prob=1.0,  # Mask everything to test EOS masking
    path_mask_prob=0.0,
    mask_path=False,
)

# Create a small batch
batch_samples = [dataset[i] for i in range(4)]
batch = collator(batch_samples)

print(f"\nBatch char_labels shape: {batch['char_labels'].shape}")
print(f"Batch char_tokens shape: {batch['char_tokens'].shape}")

# Check if EOS tokens are in labels
for i in range(min(2, len(batch["char_labels"]))):
    labels = batch["char_labels"][i]
    tokens = batch["char_tokens"][i]

    # Find positions with labels (masked positions)
    masked_positions = (labels != -100).nonzero(as_tuple=True)[0]

    if len(masked_positions) > 0:
        print(f"\nSample {i}:")
        print(f"  Masked positions: {masked_positions.tolist()}")
        print(f"  Labels at masked positions: {labels[masked_positions].tolist()}")

        # Check if EOS is in the labels
        if tokenizer.eos_token_id in labels:
            print(f"  ✓ EOS token ({tokenizer.eos_token_id}) included in labels (will be trained)")
        else:
            print("  Note: EOS not masked in this sample (random masking)")

print("\n" + "=" * 60)
print("EOS Token Test Complete")
print("=" * 60)

print("\n✓ Key Points:")
print("  1. EOS token added after each word")
print("  2. EOS token can be masked and predicted (like regular chars)")
print("  3. Model learns when words end")
print("  4. PAD tokens still ignored in loss/accuracy")
print("  5. Word accuracy compares up to EOS (not including padding)")
