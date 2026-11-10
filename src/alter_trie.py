from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
import math

import torch
from transformers.generation.logits_process import LogitsProcessor


class BatchedTrieLogitsProcessor(LogitsProcessor):
    def __init__(self, sequences: Iterable[Iterable[int]], vocab_size: int):
        children: List[Dict[int, int]] = [{}]
        depths = [0]
        for sequence in sequences:
            node = 0
            for token in sequence:
                token = int(token)
                child = children[node].get(token)
                if child is None:
                    child = len(children)
                    children[node][token] = child
                    children.append({})
                    depths.append(depths[node] + 1)
                node = child

        transitions = sorted(
            (node * vocab_size + token, child)
            for node, node_children in enumerate(children)
            for token, child in node_children.items()
        )
        self.transition_keys = torch.tensor([x[0] for x in transitions])
        self.transition_values = torch.tensor([x[1] for x in transitions])
        offsets = [0]
        allowed_tokens = []
        self.max_children_by_depth = [0] * (max(depths) + 1)
        for node, node_children in enumerate(children):
            allowed_tokens.extend(node_children)
            offsets.append(len(allowed_tokens))
            depth = depths[node]
            self.max_children_by_depth[depth] = max(
                self.max_children_by_depth[depth], len(node_children)
            )
        self.offsets = torch.tensor(offsets)
        self.allowed_tokens = torch.tensor(allowed_tokens)
        self.vocab_size = vocab_size
        self._device_cache = {}

    def _tensors_for(self, device):
        if device not in self._device_cache:
            self._device_cache[device] = tuple(
                tensor.to(device)
                for tensor in (
                    self.transition_keys,
                    self.transition_values,
                    self.offsets,
                    self.allowed_tokens,
                )
            )
        return self._device_cache[device]

    def __call__(self, input_ids, scores):
        keys, values, offsets, allowed_tokens = self._tensors_for(scores.device)
        nodes = torch.zeros(input_ids.shape[0], dtype=torch.long, device=scores.device)
        for tokens in input_ids.unbind(dim=1):
            positions = torch.searchsorted(keys, nodes * self.vocab_size + tokens)
            nodes = values[positions]

        width = self.max_children_by_depth[input_ids.shape[1]]
        starts, ends = offsets[nodes], offsets[nodes + 1]
        positions = starts[:, None] + torch.arange(width, device=scores.device)
        valid = positions < ends[:, None]
        tokens = allowed_tokens[positions.clamp_max(allowed_tokens.numel() - 1)]
        rows = torch.arange(scores.shape[0], device=scores.device)[:, None]
        rows = rows.expand_as(tokens)
        mask = torch.full_like(scores, -math.inf)
        mask[rows[valid], tokens[valid]] = 0
        return scores + mask


class TrieNode:
    __slots__ = ("children", "is_end", "_children_keys")

    def __init__(self):
        self.children: Dict[int, TrieNode] = {}
        self.is_end: bool = False
        self._children_keys: Optional[Tuple[int, ...]] = None

    @property
    def children_keys(self) -> Tuple[int, ...]:
        if self._children_keys is None:
            self._children_keys = tuple(self.children.keys())
        return self._children_keys

    def add(self, seq: Iterable[int]) -> None:
        it = iter(seq)
        try:
            token = next(it)
        except StopIteration:
            self.is_end = True
            return

        if token not in self.children:
            self.children[token] = TrieNode()
        self.children[token].add(it)

    def get_node(self, prefix: Iterable[int]) -> Optional[TrieNode]:
        node: TrieNode = self
        for token in prefix:
            node = node.children.get(token)
            if node is None:
                return None
        return node

class Trie:
    def __init__(self, sequences: Optional[Iterable[Iterable[int]]] = None):
        self.root = TrieNode()
        self.size = 0
        self.append_trie: Optional[Trie] = None
        self.bos_token_id: Optional[int] = None

        if sequences:
            for seq in sequences:
                self.add(seq)

    def add(self, seq: Iterable[int]) -> None:
        self.root.add(seq)
        self.size += 1

    def append(self, other: "Trie", bos_token_id: int) -> None:
        self.append_trie = other
        self.bos_token_id = bos_token_id

    def _allowed_from_node(self, node: TrieNode) -> Tuple[int, ...]:
        if node is self.root and self.append_trie:
            root_keys = tuple(k for k in node.children_keys if k != self.bos_token_id)
            append_keys = self.append_trie.root.children_keys
            return root_keys + append_keys
        return node.children_keys

    def get(self, prefix: List[int]) -> Tuple[int, ...]:
        node = self.root.get_node(prefix)
        if node is None:
            if self.append_trie:
                return self.append_trie.get(prefix)
            return tuple()
        return self._allowed_from_node(node)

    __getitem__ = get
    __len__ = lambda self: self.size
    __iter__ = lambda self: self._iter_sequences()

    def _iter_sequences(self):
        stack = [(self.root, [])]  # (node, path)
        while stack:
            node, path = stack.pop()
            if node.is_end:
                yield path
            for token, child in node.children.items():
                stack.append((child, path + [token]))


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        return candidate_trie.get(sentence)
    return prefix_allowed_tokens