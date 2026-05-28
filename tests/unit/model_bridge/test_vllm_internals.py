"""Unit tests for the vLLM internals chokepoint.

``segment_by_request``'s primary path reads ``model_runner.query_start_loc`` and
needs no vLLM install (backend-agnostic). The attn-metadata fallback imports
vLLM's forward context, so it's exercised on GPU via the Colab notebook instead.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from transformer_lens.model_bridge.sources.vllm.internals import segment_by_request


def _model_runner(query_start_loc, req_ids):
    return SimpleNamespace(
        query_start_loc=query_start_loc,
        input_batch=SimpleNamespace(req_ids=req_ids),
    )


class TestSegmentByRequest:
    """Per-request query offsets come from model_runner.query_start_loc."""

    def test_plain_tensor_sliced_to_num_reqs_plus_one(self):
        # The buffer is padded past the active batch; only the first n+1 entries
        # are this step's offsets ([0, len0, len0+len1, ...]).
        mr = _model_runner(torch.tensor([0, 6, 14, 16, 99, 99]), ["21", "22", "23"])
        offsets, req_ids = segment_by_request(mr)
        assert req_ids == ["21", "22", "23"]
        assert offsets.tolist() == [0, 6, 14, 16]

    def test_cpu_gpu_buffer_via_cpu_accessor(self):
        """vLLM's CpuGpuBuffer exposes the data tensor as ``.cpu``."""
        buf = SimpleNamespace(cpu=torch.tensor([0, 3, 5, 100]))
        offsets, _ = segment_by_request(_model_runner(buf, ["a", "b"]))
        assert offsets.tolist() == [0, 3, 5]

    def test_cpu_gpu_buffer_via_np_accessor(self):
        """Some buffers surface a numpy view as ``.np``."""
        buf = SimpleNamespace(np=np.array([0, 2, 7, 9, 50]))
        offsets, _ = segment_by_request(_model_runner(buf, ["a", "b", "c"]))
        assert offsets.tolist() == [0, 2, 7, 9]

    def test_single_request(self):
        mr = _model_runner(torch.tensor([0, 5]), ["only"])
        offsets, req_ids = segment_by_request(mr)
        assert offsets.tolist() == [0, 5]
        assert req_ids == ["only"]
