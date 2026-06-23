# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import List, Tuple

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "google/gemma-3-4b-it"


def _check_correctness(test_name: str, baseline_outputs: list,
                       test_outputs: list):
    """Verify generated token ids match the baseline run."""
    assert len(baseline_outputs) == len(test_outputs)

    for i, (baseline,
            test_result) in enumerate(zip(baseline_outputs, test_outputs)):
        baseline_completion = baseline.outputs[0]
        test_completion = test_result.outputs[0]
        baseline_token_ids = tuple(baseline_completion.token_ids)
        test_token_ids = tuple(test_completion.token_ids)

        assert baseline_token_ids == test_token_ids, (
            f"{test_name} token mismatch in prompt {i}:\n"
            f"  Baseline text: {baseline_completion.text!r}\n"
            f"  {test_name} text: {test_completion.text!r}\n"
            f"  Baseline token ids: {baseline_token_ids}\n"
            f"  {test_name} token ids: {test_token_ids}")

    print(f"{test_name} generated token ids match baseline outputs.")


def _reset_engine_prefix_cache(llm: LLM) -> None:
    if hasattr(llm, "reset_prefix_cache"):
        llm.reset_prefix_cache()
        return
    llm.llm_engine.engine_core.reset_prefix_cache()


@pytest.fixture
def sampling_params():
    return SamplingParams(
        temperature=0.0,
        max_tokens=16,
        seed=42,
        ignore_eos=True,
    )


@pytest.fixture
def shared_prefix_prompts():
    shared_prefix = (
        "This is a shared prefix for prefix cache testing. "
        "The model should reuse the same cached tokens for each prompt. ")
    return [
        shared_prefix + "Write one short sentence about the weather.",
        shared_prefix + "Write one short sentence about the weekend.",
    ]


def _run_prefix_cache_sequence(
    prompts: List[str],
    sampling_params: SamplingParams,
    disable_hybrid_kv_cache_manager: bool,
) -> Tuple[list, list, list]:
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=192,
        tensor_parallel_size=8,
        max_num_batched_tokens=2048,
        max_num_seqs=64,
        enable_prefix_caching=True,
        disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager,
    )

    try:
        # Step 1: Warm up and populate the in-memory prefix cache.
        initial_outputs = llm.generate(prompts, sampling_params)

        # Step 2: Repeat generations to exercise prefix cache hits.
        cached_outputs = llm.generate(prompts, sampling_params)

        # Step 3: Clear the in-memory prefix cache index, perturb the cache
        # with unrelated prompts, then force recomputation for the target prompts.
        _reset_engine_prefix_cache(llm)
        time.sleep(1)

        filler_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a short note about deterministic sampling.",
        ]
        llm.generate(filler_prompts, sampling_params)

        recomputed_outputs = llm.generate(prompts, sampling_params)
        return initial_outputs, cached_outputs, recomputed_outputs
    finally:
        if hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        time.sleep(5)


def test_kv_cache_prefix_caching_with_hybrid_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
    sampling_params: SamplingParams,
    shared_prefix_prompts: List[str],
):
    """
    Exercise TPU Prefix Caching combined with Hybrid KV Cache (HMA).
    """
    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "0")

    baseline_outputs, baseline_cached_outputs, baseline_recomputed_outputs = (
        _run_prefix_cache_sequence(
            shared_prefix_prompts,
            sampling_params,
            disable_hybrid_kv_cache_manager=True,
        ))

    hybrid_outputs, hybrid_cached_outputs, hybrid_recomputed_outputs = (
        _run_prefix_cache_sequence(
            shared_prefix_prompts,
            sampling_params,
            disable_hybrid_kv_cache_manager=False,
        ))

    _check_correctness(
        "Baseline prefix cache hit",
        baseline_outputs,
        baseline_cached_outputs,
    )
    _check_correctness(
        "Baseline prefix cache recomputation",
        baseline_outputs,
        baseline_recomputed_outputs,
    )
    _check_correctness(
        "Hybrid KV cache with prefix caching",
        baseline_outputs,
        hybrid_outputs,
    )
    _check_correctness(
        "Hybrid KV cache prefix cache hit",
        baseline_outputs,
        hybrid_cached_outputs,
    )
    _check_correctness(
        "Hybrid KV cache prefix cache recomputation",
        baseline_outputs,
        hybrid_recomputed_outputs,
    )
