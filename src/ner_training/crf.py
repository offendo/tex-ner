#!/usr/bin/env python3
from typing import List, Optional

import random
import torch
import torch.nn as nn
from icecream import ic
import sys


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

    def compute_segment_vector(
        self, emissions: torch.Tensor, start_idx: torch.Tensor | int, end_idx: torch.Tensor | int
    ):
        """Computes a feature vector for a sequence (x_i, ..., x_j).

        Parameters
        ----------
        emissions : torch.Tensor
            Tensor of shape [L, B, D]
        start_idx : torch.Tensor
            Start indices of the segment, of shape [B]
        end_idx : torch.Tensor
            End indices of the segment, of shape [B]

        Returns
        -------
        torch.Tensor :
            Tensor of shape [B, D]
        """
        B = emissions.size(1)
        return (emissions[start_idx, torch.arange(B)] + emissions[end_idx, torch.arange(B)]) / 2

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
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
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
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
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

    def compute_runs(self, tags):
        # Get the shape of the input tensor
        shape = tags.shape
        if len(shape) == 1:
            # If it's a 1D tensor, reshape to (1, -1) to make it 2D
            tags = tags.unsqueeze(0)

        # List to hold the result tensors for each row
        runs_list = []
        for row in tags.T:
            # Ignore the mask
            row = row[row != self.padding_idx]
            # Identify changes in labels
            changes = torch.cat([torch.tensor([True]), row[1:] != row[:-1], torch.tensor([True])])
            run_lengths = torch.diff(torch.where(changes)[0])
            run_values = row[torch.where(changes[:-1])[0]]

            # Apply max_run_length if provided
            extended_runs = []
            for length, value in zip(run_lengths, run_values):
                # Split runs longer than max_run_length
                while length > self.max_segment_length:
                    extended_runs.append(self.max_segment_length)
                    length -= self.max_segment_length
                if length > 0:
                    extended_runs.append(length)
            run_lengths = torch.tensor(extended_runs)
            runs_list.append(run_lengths)

        # Find the maximum length of runs across all rows
        max_len = max(len(r) for r in runs_list)

        # Pad each run list with 1s (unit segments) to make them the same length
        padded_runs = [torch.cat([r, torch.ones(max_len - len(r), dtype=torch.long)]) for r in runs_list]

        # Stack the results into a tensor
        result = torch.stack(padded_runs)

        # If the input was 1D, return a 1D tensor as output
        if len(shape) == 1:
            return result.squeeze(0)
        return result.T

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
        # seg_ids: (seq_length, batch_size) - 1 if tags[i] != tags[i-1], 0 otherwise
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        # Calculate run lengths
        lens = self.compute_runs(tags)

        # Mask out the tags to prevent trying to index -100
        tags = tags * mask

        seq_length, batch_size = tags.shape
        mask = mask.float()
        n_seg = int(lens.size(0))

        max_idx = torch.minimum(mask.long().sum(dim=0), torch.tensor(seq_length - 1))
        # start_idx = torch.zeros(batch_size, dtype=torch.int32, device=emissions.device)
        # end_idx = torch.minimum(start_idx + lens[0], max_idx)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        seg_emissions = self.compute_segment_vector(emissions, 0, lens[0] - 1)
        score += seg_emissions[torch.arange(batch_size), tags[0]]

        for seg in range(1, n_seg):
            seg_length = lens[seg]
            start_idx = torch.minimum(torch.sum(lens[:seg], dim=0), max_idx)
            end_idx = start_idx + seg_length
            # 1. Compute the emission probability of the current segment (this can be anything, right now it's the sum of first and last element)
            # shape: (batch_size, num_tags)
            seg_emissions = self.compute_segment_vector(emissions, start_idx, end_idx - 1)

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            start_tag = tags[start_idx, torch.arange(batch_size)]
            score += seg_emissions[torch.arange(batch_size), start_tag] * mask[start_idx, torch.arange(batch_size)]

            # 2. Now add in the transition probability for moving between the previous and current segment
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            start_tag = tags[start_idx - 1, torch.arange(batch_size)]
            end_tag = tags[end_idx - 1, torch.arange(batch_size)]
            score += self.transitions[start_tag, end_tag] * mask[start_idx, torch.arange(batch_size)]

            # Update the start index to the next segment
            # start_idx = torch.minimum(start_idx + seg_length, max_idx)

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

        seq_length, batch_size, _ = emissions.shape

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        broadcast_emissions = self.compute_segment_vector(emissions, 0, 0)

        # alpha[i, :, t] = score of segment ending at index i with tag t
        alpha = torch.full(
            (seq_length, batch_size, self.num_tags),
            0,
            dtype=emissions.dtype,
            device=emissions.device,
        )

        alpha[0, :, :] = self.start_transitions + broadcast_emissions

        for j in range(1, seq_length):
            segment_score = torch.zeros(
                min(self.max_segment_length, j + 1),
                batch_size,
                self.num_tags,
                dtype=emissions.dtype,
                device=emissions.device,
            )
            for i in range(self.max_segment_length):
                if i > j:
                    break

                # Broadcast score for every possible next tag
                # shape: (batch_size, num_tags, 1)
                broadcast_score = alpha[j - i - 1, :, :].unsqueeze(2)

                # Broadcast emission score for every possible current tag
                # shape: (batch_size, 1, num_tags)
                broadcast_emissions = self.compute_segment_vector(emissions, j - i, j).unsqueeze(1)

                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emissions

                # shape: (batch_size, num_tags)
                segment_score[i, :, :] = torch.logsumexp(next_score, dim=1)

            # segment_score shape: (seg_length, batch_size, num_tags)
            # shape: (batch_size, num_tags)
            alpha_nxt = torch.logsumexp(segment_score, dim=0)
            alpha[j, :, :] = torch.where(mask[j].unsqueeze(1), alpha_nxt, alpha[j - 1])

        # End transition score
        # shape: (batch_size, num_tags)
        alpha[-1, :, :] += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
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
        broadcast_emissions = self.compute_segment_vector(emissions, 0, 0)

        # alpha[i, :, t] = score of segment ending at index i with tag t
        alpha = torch.full((seq_length, batch_size, self.num_tags), 0, dtype=emissions.dtype, device=emissions.device)
        alpha[0, :, :] = self.start_transitions + broadcast_emissions

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

                # Broadcast score for every possible next tag
                # shape: (batch_size, num_tags, 1)
                broadcast_score = alpha[j - i - 1, :, :].unsqueeze(2)

                # Broadcast emission score for every possible current tag
                # shape: (batch_size, 1, num_tags)
                broadcast_emissions = self.compute_segment_vector(emissions, j - i, j).unsqueeze(1)

                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emissions

                # vit_max shape: (batch_size, num_tags)
                # vit_argmax shape: (batch_size, num_tags)
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
        tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
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
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask, tags)
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

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
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
        self, emissions: torch.Tensor, tags: Optional[torch.LongTensor] = None, mask: Optional[torch.ByteTensor] = None
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

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
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

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
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

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor) -> List[List[int]]:
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
    N = 20
    T = len(labels)

    # Model
    scrf = SemiCRF(num_tags=T, batch_first=True, max_segment_length=1)
    crf = CRF(num_tags=T, batch_first=True)

    # Make the weights the same so it's easy to compare values
    scrf.start_transitions = crf.start_transitions
    scrf.transitions = crf.transitions
    scrf.end_transitions = crf.end_transitions

    # Fake input
    emissions = torch.randn(N, B, T, dtype=torch.float32)
    tags = torch.randint(0, T, (N, B))
    # tags = torch.tensor([[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]).T
    # ic(tags)
    # ic(emissions)
    # tags[N - 3 :, 1] = -100
    mask = (tags != -100).long()

    if sys.argv[-1] == "scrf":
        score = scrf._compute_score(emissions, tags, mask.bool())
        ic(score)

    if sys.argv[-1] == "crf":
        score = crf._compute_score(emissions, tags * mask, mask.bool())
        ic(score)

    if sys.argv[-1] == "scrf":
        # score = crf._compute_score(emissions, tags, mask)
        norm = scrf._compute_normalizer(emissions, mask.bool())
        ic(norm)

    if sys.argv[-1] == "crf":
        norm = crf._compute_normalizer(emissions, mask.bool())
        ic(norm)

    if sys.argv[-1] == "scrf":
        best_tags = scrf._viterbi_decode(emissions, mask.bool())
        ic(best_tags)

    if sys.argv[-1] == "crf":
        best_tags = crf._viterbi_decode(emissions, mask.bool())
        ic(best_tags)
