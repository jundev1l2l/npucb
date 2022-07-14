# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Utils Functions """


import random
import numpy as np
import torch
import warnings

warnings.filterwarnings(action='ignore')


def set_seed(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert 1 < n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def find_sublist(haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.
    https://codereview.stackexchange.com/questions/19627/finding-sub-list
    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j] != needle[-j - 1]:
                i += skip.get(haystack[i], n)
                break
        else:
            return i - n + 1
    return -1


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_random_word(vocab_words):
    i = random.randint(0, len(vocab_words) - 1)
    return vocab_words[i]


def binary2decimal(x: np.ndarray) -> np.ndarray:
    c = np.array([2 ** i for i in reversed(range(x.shape[-1]))])
    return (x * c).sum(-1).astype(int)


def decimal2binary(x: torch.float, length) -> torch.float:
    z = torch.zeros(size=(x.size() + (length,)))
    for i in reversed(range(length)):
        z[:, i] = x[:] % 2
        x = x//2
    return z
