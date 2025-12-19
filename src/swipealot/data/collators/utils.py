from __future__ import annotations

import random

import torch


def mask_contiguous_blocks_1d(
    valid_mask: torch.Tensor,
    n_to_mask: int,
    max_block_len: int,
    rng: random.Random,
) -> torch.Tensor:
    """
    Create a 1D binary mask (1=masked) consisting of contiguous blocks within valid segments.

    Args:
        valid_mask: [seq_len] tensor with 1 for valid points, 0 for padding/invalid.
        n_to_mask: Target number of positions to mask (clipped to number of valid points).
        max_block_len: Absolute maximum length for any single contiguous masked block.
        rng: Random generator to sample block sizes and locations.

    Returns:
        mask_indices: [seq_len] long tensor with 1 at masked positions.
    """
    seq_len = int(valid_mask.shape[0])
    valid_mask = valid_mask.to(dtype=torch.long)
    n_valid = int(valid_mask.sum().item())
    n_to_mask = max(0, min(int(n_to_mask), n_valid))

    mask_indices = torch.zeros(seq_len, dtype=torch.long)
    if n_to_mask == 0 or n_valid == 0:
        return mask_indices
    max_block_len = max(1, int(max_block_len))

    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, v in enumerate(valid_mask.tolist()):
        if v == 1 and not in_seg:
            seg_start = i
            in_seg = True
        elif v == 0 and in_seg:
            segments.append((seg_start, i))  # end exclusive
            in_seg = False
    if in_seg:
        segments.append((seg_start, seq_len))

    masked = torch.zeros(seq_len, dtype=torch.bool)
    remaining = n_to_mask

    def _segment_available(s: int, e: int) -> int:
        return int((~masked[s:e]).sum().item())

    def _left_run(start: int) -> int:
        run = 0
        i = start - 1
        while i >= 0 and valid_mask[i].item() == 1 and masked[i].item():
            run += 1
            i -= 1
        return run

    def _right_run(end: int) -> int:
        run = 0
        i = end
        while i < seq_len and valid_mask[i].item() == 1 and masked[i].item():
            run += 1
            i += 1
        return run

    attempts = 0
    max_attempts = 50 * (n_to_mask + 1)
    while remaining > 0 and attempts < max_attempts:
        candidates: list[tuple[int, int, int]] = []
        total_avail = 0
        for s, e in segments:
            avail = _segment_available(s, e)
            if avail > 0:
                candidates.append((s, e, avail))
                total_avail += avail

        if total_avail == 0:
            break

        pick = rng.randrange(total_avail)
        chosen_s = chosen_e = 0
        chosen_avail = 0
        acc = 0
        for s, e, avail in candidates:
            acc += avail
            if pick < acc:
                chosen_s, chosen_e, chosen_avail = s, e, avail
                break

        max_len = min(remaining, chosen_avail, max_block_len)
        if max_len <= 0:
            attempts += 1
            continue

        min_len = 2 if max_len >= 2 else 1
        block_len = rng.randint(min_len, max_len)

        placed = False
        for _ in range(6):
            start_candidates: list[int] = []
            for start in range(chosen_s, chosen_e - block_len + 1):
                end = start + block_len
                if masked[start:end].any():
                    continue
                run_len = _left_run(start) + block_len + _right_run(end)
                if run_len <= max_block_len:
                    start_candidates.append(start)

            if start_candidates:
                start = rng.choice(start_candidates)
                masked[start : start + block_len] = True
                remaining -= block_len
                placed = True
                break

            if block_len <= min_len:
                break
            block_len = max(min_len, block_len // 2)

        if not placed:
            avail_positions = (~masked[chosen_s:chosen_e]).nonzero(as_tuple=True)[0]
            if avail_positions.numel() > 0:
                perm = avail_positions[torch.randperm(avail_positions.numel())]
                for rel in perm.tolist():
                    pos = chosen_s + int(rel)
                    if _left_run(pos) + 1 + _right_run(pos + 1) <= max_block_len:
                        masked[pos] = True
                        remaining -= 1
                        break

        attempts += 1

    if remaining > 0:
        avail_positions = ((valid_mask == 1) & (~masked)).nonzero(as_tuple=True)[0]
        if avail_positions.numel() > 0:
            perm = avail_positions[torch.randperm(avail_positions.numel())]

            def _max_run_with(pos: int) -> int:
                tmp = masked.clone()
                tmp[pos] = True
                m = (tmp & (valid_mask == 1)).to(torch.int64)
                max_run = 0
                cur = 0
                for v in m.tolist():
                    if v == 1:
                        cur += 1
                        if cur > max_run:
                            max_run = cur
                    else:
                        cur = 0
                return max_run

            for p in perm.tolist():
                if remaining <= 0:
                    break
                if _max_run_with(int(p)) <= max_block_len:
                    masked[int(p)] = True
                    remaining -= 1

    mask_indices[masked] = 1
    return mask_indices
