# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch

from fairseq.data import data_utils, Dictionary
from . import BaseWrapperDataset, LRUCacheDataset


class MaskTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
        )

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            vocab: Dictionary,
            pad_idx: int,
            mask_idx: int,
            return_masked_tokens: bool = False,
            seed: int = 1,
            mask_prob: float = 0.15,
            leave_unmasked_prob: float = 0.1,
            random_token_prob: float = 0.1,
            freq_weighted_replacement: bool = False,
    ):
        # assert 0.0 < mask_prob < 1.0
        assert 0.0 <= mask_prob < 1.0  # allow all not masking
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab))
            weights[:self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            if '##' in self.vocab:
                real_bytes = \
                    torch.where(
                        (item != self.vocab.index('##')) & (item != self.vocab.bos()) & (item != self.vocab.eos()))[
                        0].cpu().numpy()
                # print(item.size())
                # exit()
                # real_bytes = torch.where(item in self.real_bytes_idx)[0].cpu().numpy()

            assert self.mask_idx not in item, \
                'Dataset contains mask_idx (={}), this is not expected!'.format(
                    self.mask_idx,
                )

            # decide elements to mask
            mask = np.full(sz, False)

            if '##' in self.vocab:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    self.mask_prob * len(real_bytes) + np.random.rand()
                )

                if num_mask > 0:
                    mask[np.random.choice(real_bytes, num_mask, replace=False)] = True
                else:
                    # all ## cases, we select only one location to make this task trivial
                    mask[np.random.choice(sz, 1, replace=False)] = True
            else:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    self.mask_prob * sz + np.random.rand()
                )
                mask[np.random.choice(sz, num_mask, replace=False)] = True

            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item)

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )

            return torch.from_numpy(new_item)
