#!/usr/bin/env python3
import random
import sys
import time
from typing import List, Optional

import torch
import torch.nn as nn
from icecream import ic

from contextlib import contextmanager

# torch.autograd.set_detect_anomaly(True)


@contextmanager
def timer(name):
    tik = time.time()
    yield
    tok = time.time()
    print(f"Time taken for {name}: {tok - tik}")


class SemiCRF(nn.Module):
    def __init__(
        self,
        num_tags: int,
        batch_first: bool = False,
        padding_idx: int = -100,
        max_segment_length: int = 1,
    ) -> None:
        # torch.autograd.set_detect_anomaly(True)
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.padding_idx = padding_idx
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.max_segment_length = max_segment_length

        self.pool_projection = nn.Linear(in_features=num_tags, out_features=num_tags)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.normal_(self.start_transitions)
        nn.init.normal_(self.end_transitions)
        nn.init.xavier_uniform_(self.transitions)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.Tensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.Tensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags=tags, mask=mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask=mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        assert reduction == "token_mean"
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.Tensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
        self, emissions: torch.Tensor, tags: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(f"emissions must have dimension of 3, got {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(f"expected last dimension of emissions is {self.num_tags}, " f"got {emissions.size(2)}")

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def pool(self, seg_emissions: torch.Tensor):
        # assert len(seg_emissions.shape) == 3
        return self.pool_projection(seg_emissions.sum(dim=0))
        # return seg_emissions.sum(dim=0)

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size): 1 if good, 0 if bad
        # lens: (n_segments, batch_size)
        # seg_ids: (seq_length, batch_size): 1 if tags[i] != tags[i-1], 0 otherwise
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        # Mask out the tags to prevent trying to index -100
        tags = tags * mask

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Indicator vector for segment starts
        seg_starts = torch.cat([torch.ones(1, batch_size, device=tags.device), tags[:-1] != tags[1:]], dim=0)

        # Ensure the max segment length is identical
        for b in range(batch_size):
            run_length = 0
            for i in range(seq_length):
                if seg_starts[i, b] == 0 and run_length == self.max_segment_length:
                    seg_starts[i, b] = 1
                    run_length = 1
                elif seg_starts[i, b] == 0:
                    run_length += 1
                else:  # seg_starts[i, b] == 1
                    run_length = 1

        seg_nums = torch.cumsum(seg_starts, dim=0, dtype=torch.long) - 1

        # Start transition score and first emission
        # first segment index = 0
        bs = torch.arange(batch_size)
        seg_emissions = emissions * (seg_nums == 0).unsqueeze(2)
        score = self.pool(seg_emissions)[bs, tags[0]] + self.start_transitions[tags[0]]

        segment_mask = mask * seg_starts
        for j in range(1, seq_length):
            seg_idx = seg_nums[j]
            # 1. Compute the emission probability of the current segment (this can be anything, right now it's the sum of first and last element)
            # shape: (batch_size, num_tags)
            # 2. Then add in the transition probability for moving between the previous and current segment
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            seg_emissions = emissions * (seg_nums == seg_idx).unsqueeze(2)
            score += (self.pool(seg_emissions)[bs, tags[j]] + self.transitions[tags[j - 1], tags[j]]) * segment_mask[j]

        # Only count the idxs where the segment starts, since otherwise we'd double count

        # Now add the end transition score; note we do this after so we don't have to worry about
        # adding the transition to the right index

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # shape: (batch_size,)
        last_tags = tags[seq_ends, bs]

        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size, _ = emissions.shape

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        # (batch_size, num_tags)
        # alpha[i, :, t] = score of segment ending at index i with tag t
        alpha = torch.zeros(seq_length, batch_size, self.num_tags, dtype=emissions.dtype, device=emissions.device)

        alpha[0, :, :] = self.start_transitions + self.pool(emissions[0].unsqueeze(0))

        for j in range(1, seq_length):
            segment_score = torch.full(
                (self.max_segment_length, batch_size, self.num_tags),
                fill_value=-10000,
                dtype=emissions.dtype,
                device=emissions.device,
            )
            for i in range(min(self.max_segment_length, j)):

                # Dynamic programming: Case i indicates a segment of length i ending at the current timestep j

                # Score up to the start of this segment:
                # shape: (batch_size, num_tags, 1)
                broadcast_score = alpha[j - i - 1, :, :].unsqueeze(2)

                # Emission score for the segment of length i, starting at j-i and ending at j (i.e., j+1 non-inclusive)
                # shape: (batch_size, 1, num_tags)
                broadcast_emissions = self.pool(emissions[j - i : j + 1]).unsqueeze(1)

                # Next score is the previous score + transition between previous and current segment + score of current segment
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emissions

                # Logsumexp over all possible segment lengths, which gets the overall score of the current tag being
                segment_score[i, :, :] = torch.logsumexp(next_score, dim=1)  # shape: (batch_size, num_tags)

            # Now, logsumexp over the previous tags (dim 0) to get the scores for each of the current tags (dim 2)
            alpha_nxt = torch.logsumexp(segment_score, dim=0)  #  shape: (batch_size, num_tags)
            alpha[j, :, :] = torch.where(mask[j].unsqueeze(1), alpha_nxt, alpha[j - 1])

        # Add in the transition score for T_j --> <STOP>
        # shape: (batch_size, num_tags)
        alpha[-1, :, :] += self.end_transitions

        # Finally, logsumexp over all possible current tags to see the final score of the sequence
        # shape: (batch_size,)
        return torch.logsumexp(alpha[-1, :, :], dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size, _ = emissions.shape

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)

        # alpha[i, :, t] = score of segment ending at index i with tag t
        alpha = torch.zeros(seq_length, batch_size, self.num_tags, dtype=emissions.dtype, device=emissions.device)
        alpha[0, :, :] = self.start_transitions + self.pool(emissions[0].unsqueeze(0))

        # tracks transitions of tags
        history = []

        for j in range(1, seq_length):
            segment_shape = (min(self.max_segment_length, j + 1), batch_size, self.num_tags)
            segment_score = torch.zeros(*segment_shape, dtype=emissions.dtype, device=emissions.device)
            segment_indices = torch.full(
                segment_shape, fill_value=self.padding_idx, dtype=torch.long, device=emissions.device
            )
            for i in range(self.max_segment_length):
                if i > j:
                    break

                # Dynamic programming: Case i indicates a segment of length i ending at the current timestep j

                # Score up to the start of this segment:
                # shape: (batch_size, num_tags, 1)
                broadcast_score = alpha[j - i - 1, :, :].unsqueeze(2)

                # Emission score for the segment of length i, starting at j-i and ending at j (i.e., j+1 non-inclusive)
                # shape: (batch_size, 1, num_tags)
                broadcast_emissions = self.pool(emissions[j - i : j + 1]).unsqueeze(1)

                # Next score is the previous score + transition between previous and current segment + score of current segment
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emissions

                # The most likely segment of length i is one of tag `indices`
                # alpha_nxt shape: (batch_size, num_tags)
                # indices shape: (batch_size, num_tags)
                alpha_nxt, indices = next_score.max(dim=1)

                # shape: (batch_size, num_tags)
                segment_score[i, :, :] = alpha_nxt
                segment_indices[i, :, :] = indices

            # What's the best score at this index? (max/argmax segment_score over segment length)
            # seg_max shape: (batch_size, num_tags)
            # seg_argmax shape: (batch_size, num_tags)
            seg_max, seg_argmax = segment_score.max(dim=0)
            alpha[j, :, :] = torch.where(mask[j].unsqueeze(1), seg_max, alpha[j - 1])
            indices = torch.gather(segment_indices, 0, seg_argmax.unsqueeze(0)).squeeze(0)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        alpha[-1, :, :] += self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        score = alpha[-1]

        for b in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[b].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[: seq_ends[b]]):
                best_last_tag = hist[b][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.Tensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.Tensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags=tags, mask=mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask=mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        assert reduction == "token_mean"
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.Tensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
        self, emissions: torch.Tensor, tags: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(f"emissions must have dimension of 3, got {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(f"expected last dimension of emissions is {self.num_tags}, " f"got {emissions.size(2)}")

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            ts_score = emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
            ts_score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


if __name__ == "__main__":
    # Reproducability
    random.seed(42)
    torch.manual_seed(100)

    # Fake setup, easy for debugging
    labels = ["a", "b", "c"]
    label2id = {l: i for i, l in enumerate(labels)}
    B = 2
    N = 10
    T = len(labels)

    # Model
    scrf = SemiCRF(num_tags=T, batch_first=True, max_segment_length=1)
    crf = CRF(num_tags=T, batch_first=True)

    # # Make the weights the same so it's easy to compare values
    scrf.start_transitions = crf.start_transitions
    scrf.end_transitions = crf.end_transitions
    scrf.transitions.data = crf.transitions
    scrf.pool_projection.weight.data = torch.eye(T)
    scrf.pool_projection.bias.data = torch.zeros(T)

    # Fake input
    emissions = torch.randn(N, B, T, dtype=torch.float32)
    tags = torch.randint(0, T, (N, B))
    tags[N - 3 :, 1] = -100
    ic(tags)
    mask = (tags != -100).long()

    if sys.argv[-1] == "scrf":
        with timer("scrf score"):
            score = scrf._compute_score(emissions, tags, mask.bool())
        with timer("scrf norm"):
            norm = scrf._compute_normalizer(emissions, mask.bool())
        with timer("scrf decode"):
            best_tags = scrf._viterbi_decode(emissions, mask.bool())
        ic(score)
        ic(norm)
        ic((score - norm).sum() / mask.sum())
        ic(best_tags)

    if sys.argv[-1] == "crf":
        with timer("crf score"):
            score = crf._compute_score(emissions, tags * mask, mask.bool())
        with timer("crf norm"):
            norm = crf._compute_normalizer(emissions, mask.bool())
        with timer("crf decode"):
            best_tags = crf._viterbi_decode(emissions, mask.bool())
        ic(score)
        ic(norm)
        ic((score - norm).sum() / mask.sum())
        ic(best_tags)
