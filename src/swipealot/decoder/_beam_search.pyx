# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Trie-constrained CTC beam search — Cython port of ml-inference C++.

Parametrized for arbitrary alphabet sizes (26 for EN, 31 for RU, etc.).

Optimizations vs naive impl:
    - Flat hash map with generation-counter clear (no memset per timestep)
    - Quickselect pruning O(n) instead of selection sort O(n·k)
    - Power-of-2 hash table with bitmask instead of modulo
    - nogil inner loop
"""

import cython
from libc.math cimport INFINITY
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memset

import numpy as np

cimport numpy as np

np.import_array()


# ── Flat-array Trie ──────────────────────────────────────────────────────────

cdef struct TrieNode:
    int* children        # children[num_chars], 0 = no child
    bint is_word
    float log_frequency
    int parent
    int parent_char      # char index on edge from parent
    int depth

cdef class Trie:
    """Flat-array trie parametrized by num_chars."""

    cdef TrieNode* nodes
    cdef int capacity
    cdef int size
    cdef int num_chars
    cdef list idx_to_char  # index → character string
    cdef dict char_to_idx  # character → index

    def __cinit__(self, int num_chars, list letters):
        self.num_chars = num_chars
        self.capacity = 1024
        self.size = 1  # root is node 0
        self.nodes = <TrieNode*>malloc(self.capacity * sizeof(TrieNode))
        self._init_node(0, 0, 0, 0)

        self.idx_to_char = list(letters)
        self.char_to_idx = {ch: i for i, ch in enumerate(letters)}

    def __dealloc__(self):
        cdef int i
        if self.nodes != NULL:
            for i in range(self.size):
                if self.nodes[i].children != NULL:
                    free(self.nodes[i].children)
            free(self.nodes)

    cdef void _init_node(self, int idx, int parent, int parent_char, int depth):
        self.nodes[idx].children = <int*>malloc(self.num_chars * sizeof(int))
        memset(self.nodes[idx].children, 0, self.num_chars * sizeof(int))
        self.nodes[idx].is_word = False
        self.nodes[idx].log_frequency = -100.0
        self.nodes[idx].parent = parent
        self.nodes[idx].parent_char = parent_char
        self.nodes[idx].depth = depth

    cdef int _alloc_node(self, int parent, int parent_char, int depth):
        if self.size >= self.capacity:
            self.capacity *= 2
            self.nodes = <TrieNode*>realloc(self.nodes, self.capacity * sizeof(TrieNode))
        cdef int idx = self.size
        self.size += 1
        self._init_node(idx, parent, parent_char, depth)
        return idx

    def insert(self, str word, float log_freq=0.0):
        """Insert a word with its log frequency."""
        cdef int node = 0
        cdef int ci, child
        for ch in word.lower():
            ci_obj = self.char_to_idx.get(ch)
            if ci_obj is None:
                return  # skip unknown chars
            ci = ci_obj
            child = self.nodes[node].children[ci]
            if child == 0:
                child = self._alloc_node(node, ci, self.nodes[node].depth + 1)
                self.nodes[node].children[ci] = child
            node = child
        self.nodes[node].is_word = True
        if log_freq > self.nodes[node].log_frequency:
            self.nodes[node].log_frequency = log_freq

    def load_vocab(self, str path):
        """Load vocabulary from text file (one word per line, optional tab+freq)."""
        cdef int count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                word = parts[0].strip()
                freq = float(parts[1]) if len(parts) > 1 else 0.0
                if word:
                    self.insert(word, freq)
                    count += 1
        return count

    def reconstruct_word(self, int node_idx):
        """Reconstruct word by walking parent chain."""
        cdef list chars = []
        cdef int n = node_idx
        while n != 0:
            chars.append(self.idx_to_char[self.nodes[n].parent_char])
            n = self.nodes[n].parent
        chars.reverse()
        return "".join(chars)

    @property
    def word_count(self):
        cdef int count = 0
        for i in range(self.size):
            if self.nodes[i].is_word:
                count += 1
        return count

    def search(self, str word):
        """Check if exact word exists in trie."""
        cdef int node = 0
        for ch in word.lower():
            ci_obj = self.char_to_idx.get(ch)
            if ci_obj is None:
                return False
            if self.nodes[node].children[<int>ci_obj] == 0:
                return False
            node = self.nodes[node].children[<int>ci_obj]
        return self.nodes[node].is_word

    def __len__(self):
        return self.word_count

    def __contains__(self, str word):
        return self.search(word)


# ── Dedup hash map (generation-counter clear, power-of-2 sizing) ─────────────

cdef struct MapEntry:
    unsigned long long key
    int value
    int generation

cdef struct FlatHashMap:
    MapEntry* entries
    int capacity       # must be power of 2
    int mask           # capacity - 1
    int generation

cdef inline unsigned long long _hash64(unsigned long long x) noexcept nogil:
    x ^= x >> 33
    x *= <unsigned long long>0xFF51AFD7ED558CCD
    x ^= x >> 33
    x *= <unsigned long long>0xC4CEB9FE1A85EC53
    x ^= x >> 33
    return x

cdef inline int _next_power_of_2(int n) noexcept nogil:
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1

cdef FlatHashMap* map_create(int min_capacity):
    cdef FlatHashMap* m = <FlatHashMap*>malloc(sizeof(FlatHashMap))
    m.capacity = _next_power_of_2(min_capacity)
    m.mask = m.capacity - 1
    m.generation = 1
    m.entries = <MapEntry*>malloc(m.capacity * sizeof(MapEntry))
    memset(m.entries, 0, m.capacity * sizeof(MapEntry))
    return m

cdef inline void map_clear(FlatHashMap* m) noexcept nogil:
    m.generation += 1

cdef void map_destroy(FlatHashMap* m):
    if m != NULL:
        free(m.entries)
        free(m)

cdef inline int map_get_or_insert(FlatHashMap* m, unsigned long long key, int default_val) noexcept nogil:
    """Returns existing value if key present, else inserts default_val and returns it."""
    cdef unsigned long long h = _hash64(key)
    cdef int idx = <int>(h) & m.mask
    cdef int i
    for i in range(m.capacity):
        if m.entries[idx].generation != m.generation:
            # Empty slot
            m.entries[idx].key = key
            m.entries[idx].value = default_val
            m.entries[idx].generation = m.generation
            return default_val
        if m.entries[idx].key == key:
            return m.entries[idx].value
        idx = (idx + 1) & m.mask
    return default_val  # full — shouldn't happen with proper sizing


# ── Beam hypothesis ──────────────────────────────────────────────────────────

cdef struct BeamHyp:
    float score
    int trie_node
    bint blank_ended


# ── Quickselect (O(n) pruning, matches C++ std::nth_element) ────────────────

cdef inline void _swap(BeamHyp* a, BeamHyp* b) noexcept nogil:
    cdef BeamHyp tmp = a[0]
    a[0] = b[0]
    b[0] = tmp

cdef void _quickselect(BeamHyp* arr, int left, int right, int k) noexcept nogil:
    """Partition so that arr[0..k-1] have the k largest scores (unordered)."""
    cdef int i, store
    cdef float pivot
    while left < right:
        # Median-of-3 pivot
        pivot = arr[(left + right) >> 1].score
        _swap(&arr[(left + right) >> 1], &arr[right])
        store = left
        for i in range(left, right):
            if arr[i].score > pivot:
                _swap(&arr[i], &arr[store])
                store += 1
        _swap(&arr[store], &arr[right])
        if store == k:
            return
        elif store < k:
            left = store + 1
        else:
            right = store - 1


# ── Beam search ──────────────────────────────────────────────────────────────

def beam_search_decode(
    float[:, :] log_probs,
    Trie trie,
    int beam_width=100,
    float freq_bonus=0.4,
    float length_bonus=0.8,
    int top_k=10,
):
    """CTC beam search constrained to trie vocabulary.

    Args:
        log_probs: [T, C] numpy array of log probabilities.
                   C = num_chars + 1, last column = blank.
        trie: Trie built from vocabulary.
        beam_width: Max hypotheses per timestep.
        freq_bonus: Weight for log-frequency scoring.
        length_bonus: Bonus per character.
        top_k: Number of word candidates to return.

    Returns:
        List of (word, score) tuples, sorted by score descending.
    """
    cdef int T = log_probs.shape[0]
    cdef int C = log_probs.shape[1]
    cdef int num_chars = trie.num_chars
    cdef int blank_idx = num_chars  # last class

    # Allocate beam buffers
    cdef int max_hyps = beam_width * (num_chars + 1) * 2
    cdef BeamHyp* beam = <BeamHyp*>malloc(max_hyps * sizeof(BeamHyp))
    cdef BeamHyp* new_beam = <BeamHyp*>malloc(max_hyps * sizeof(BeamHyp))
    cdef BeamHyp* tmp_ptr
    cdef int beam_size = 0
    cdef int new_beam_size = 0

    cdef FlatHashMap* dedup = map_create(max_hyps * 4)

    # Initialize: empty hypothesis at trie root
    beam[0].score = 0.0
    beam[0].trie_node = 0
    beam[0].blank_ended = True
    beam_size = 1

    cdef int t, i, ci, child
    cdef float blank_lp, char_lp, new_score
    cdef unsigned long long hyp_key
    cdef int existing
    cdef TrieNode* trie_nodes = trie.nodes

    with nogil:
        for t in range(T):
            map_clear(dedup)
            new_beam_size = 0

            for i in range(beam_size):
                # Blank emission: stay in same trie node
                blank_lp = log_probs[t, blank_idx]
                new_score = beam[i].score + blank_lp

                hyp_key = (<unsigned long long>beam[i].trie_node << 1) | 1
                existing = map_get_or_insert(dedup, hyp_key, new_beam_size)
                if existing == new_beam_size:
                    if new_beam_size < max_hyps:
                        new_beam[new_beam_size].score = new_score
                        new_beam[new_beam_size].trie_node = beam[i].trie_node
                        new_beam[new_beam_size].blank_ended = True
                        new_beam_size += 1
                else:
                    if new_score > new_beam[existing].score:
                        new_beam[existing].score = new_score

                # Character emissions
                for ci in range(num_chars):
                    child = trie_nodes[beam[i].trie_node].children[ci]
                    if child == 0:
                        continue

                    char_lp = log_probs[t, ci]
                    new_score = beam[i].score + char_lp

                    hyp_key = (<unsigned long long>child << 1)
                    existing = map_get_or_insert(dedup, hyp_key, new_beam_size)
                    if existing == new_beam_size:
                        if new_beam_size < max_hyps:
                            new_beam[new_beam_size].score = new_score
                            new_beam[new_beam_size].trie_node = child
                            new_beam[new_beam_size].blank_ended = False
                            new_beam_size += 1
                    else:
                        if new_score > new_beam[existing].score:
                            new_beam[existing].score = new_score

            # Prune to beam_width using quickselect O(n)
            if new_beam_size > beam_width:
                _quickselect(new_beam, 0, new_beam_size - 1, beam_width)
                new_beam_size = beam_width

            # Swap buffers
            tmp_ptr = beam
            beam = new_beam
            new_beam = tmp_ptr
            beam_size = new_beam_size

    # Collect completed words (needs GIL for Python string ops)
    results = []
    for i in range(beam_size):
        if trie_nodes[beam[i].trie_node].is_word:
            word = trie.reconstruct_word(beam[i].trie_node)
            depth = trie_nodes[beam[i].trie_node].depth
            log_freq = trie_nodes[beam[i].trie_node].log_frequency

            final_score = (
                beam[i].score
                + freq_bonus * log_freq
                + length_bonus * depth
            )
            results.append((word, final_score, beam[i].score))

    # Sort by final score descending
    results.sort(key=lambda x: -x[1])

    # Cleanup
    free(beam)
    free(new_beam)
    map_destroy(dedup)

    return [(w, s) for w, s, _ in results[:top_k]]


def beam_search_decode_with_scores(
    float[:, :] log_probs,
    Trie trie,
    int beam_width=100,
    int top_k=100,
):
    """CTC beam search returning decomposed score components for tuning.

    Runs with unit freq/length bonuses so raw component scores are preserved.
    The caller can rescore: final = ctc + freq_bonus * freq + length_bonus * length.

    Args:
        log_probs: [T, C] numpy array of log probabilities.
                   C = num_chars + 1, last column = blank.
        trie: Trie built from vocabulary.
        beam_width: Max hypotheses per timestep.
        top_k: Number of word candidates to return.

    Returns:
        List of dicts: {word, ctc_score, freq_score, length_score}
        sorted by (ctc_score + freq_score + length_score) descending.
    """
    cdef int T = log_probs.shape[0]
    cdef int C = log_probs.shape[1]
    cdef int num_chars = trie.num_chars
    cdef int blank_idx = num_chars

    # Allocate beam buffers
    cdef int max_hyps = beam_width * (num_chars + 1) * 2
    cdef BeamHyp* beam = <BeamHyp*>malloc(max_hyps * sizeof(BeamHyp))
    cdef BeamHyp* new_beam = <BeamHyp*>malloc(max_hyps * sizeof(BeamHyp))
    cdef BeamHyp* tmp_ptr
    cdef int beam_size = 0
    cdef int new_beam_size = 0

    cdef FlatHashMap* dedup = map_create(max_hyps * 4)

    # Initialize: empty hypothesis at trie root
    beam[0].score = 0.0
    beam[0].trie_node = 0
    beam[0].blank_ended = True
    beam_size = 1

    cdef int t, i, ci, child
    cdef float blank_lp, char_lp, new_score
    cdef unsigned long long hyp_key
    cdef int existing
    cdef TrieNode* trie_nodes = trie.nodes

    with nogil:
        for t in range(T):
            map_clear(dedup)
            new_beam_size = 0

            for i in range(beam_size):
                # Blank emission
                blank_lp = log_probs[t, blank_idx]
                new_score = beam[i].score + blank_lp

                hyp_key = (<unsigned long long>beam[i].trie_node << 1) | 1
                existing = map_get_or_insert(dedup, hyp_key, new_beam_size)
                if existing == new_beam_size:
                    if new_beam_size < max_hyps:
                        new_beam[new_beam_size].score = new_score
                        new_beam[new_beam_size].trie_node = beam[i].trie_node
                        new_beam[new_beam_size].blank_ended = True
                        new_beam_size += 1
                else:
                    if new_score > new_beam[existing].score:
                        new_beam[existing].score = new_score

                # Character emissions
                for ci in range(num_chars):
                    child = trie_nodes[beam[i].trie_node].children[ci]
                    if child == 0:
                        continue

                    char_lp = log_probs[t, ci]
                    new_score = beam[i].score + char_lp

                    hyp_key = (<unsigned long long>child << 1)
                    existing = map_get_or_insert(dedup, hyp_key, new_beam_size)
                    if existing == new_beam_size:
                        if new_beam_size < max_hyps:
                            new_beam[new_beam_size].score = new_score
                            new_beam[new_beam_size].trie_node = child
                            new_beam[new_beam_size].blank_ended = False
                            new_beam_size += 1
                    else:
                        if new_score > new_beam[existing].score:
                            new_beam[existing].score = new_score

            # Prune
            if new_beam_size > beam_width:
                _quickselect(new_beam, 0, new_beam_size - 1, beam_width)
                new_beam_size = beam_width

            tmp_ptr = beam
            beam = new_beam
            new_beam = tmp_ptr
            beam_size = new_beam_size

    # Collect completed words with decomposed scores
    results = []
    for i in range(beam_size):
        if trie_nodes[beam[i].trie_node].is_word:
            word = trie.reconstruct_word(beam[i].trie_node)
            depth = trie_nodes[beam[i].trie_node].depth
            log_freq = trie_nodes[beam[i].trie_node].log_frequency

            results.append({
                "word": word,
                "ctc_score": beam[i].score,
                "freq_score": log_freq,
                "length_score": <float>depth,
            })

    # Sort by sum of all components (unit weights)
    results.sort(key=lambda x: -(x["ctc_score"] + x["freq_score"] + x["length_score"]))

    free(beam)
    free(new_beam)
    map_destroy(dedup)

    return results[:top_k]
