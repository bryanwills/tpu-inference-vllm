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

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_conv1d_gdn import (configs, conv1d, gdn,
                                                    ldst_helper, ref_classes)


def inner_kernel(
    # Inputs (from in_specs)
    qkv_slot_ref: jax.Array,
    b_slot_ref: jax.Array,
    a_slot_ref: jax.Array,
    conv_state_slot_ref: jax.Array,
    recurrent_slot_ref: jax.Array,
    # Outputs (from out_specs)
    out_slot_ref: jax.Array,
    # Scratches (from scratches=...)
    metadata_ref: ref_classes.MetadataRef,
    weights_ref: ref_classes.WeightRefs,
    prev_conv_scratch_ref: jax.Array | None,
    prev_recurrent_scratch_ref: jax.Array | None,
    *,
    cfgs: configs.GDNConfigs,
):
    p_id = pl.program_id(0)

    # Prepare states.
    real_size, prev_conv, prev_recurrent = ldst_helper.load_and_mask_states(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_slot_ref=conv_state_slot_ref,
        recurrent_slot_ref=recurrent_slot_ref,
        prev_conv_scratch_ref=prev_conv_scratch_ref,
        prev_recurrent_scratch_ref=prev_recurrent_scratch_ref,
        cfgs=cfgs,
    )

    # Step 1: Conv1D.
    # NOTE: Conv1D requires performing sliding window where inputs are slided
    # across rows. If typical 2D layout was used, multiple rows are stored in a
    # single register which necessitate costly shuffling for every sliding.
    # Therefore, it is extremely important to leverage compact layout that
    # ensures 1 register only stores data from 1 row.
    qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
    qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

    target_val_list = []
    for idx in range(cfgs.seq_tile_size):
        target_s = qkv_in_compact[idx, 1:cfgs.kernel_size]
        for row_start in range(2, cfgs.chunk_size + 1):
            row_end = row_start + cfgs.prev_kernel_size
            target_s = jnp.where(
                row_start == real_size[idx],
                qkv_in_compact[idx, row_start:row_end],
                target_s,
            )
        target_val_list.append(target_s)
    target_val = jnp.stack(target_val_list, axis=0)
    conv_state_slot_ref[...] = target_val
    if prev_conv_scratch_ref is not None:
        prev_conv_scratch_ref[...] = target_val

    qkv_out_compact = conv1d.causal_conv1d(
        lhs=qkv_in_compact,
        conv_weights_ref=weights_ref.conv,
        cfgs=cfgs,
    )

    # Apply activation function.
    qkv_out_compact = jax.nn.silu(qkv_out_compact)

    # Step 2: GDN.
    if cfgs.chunk_size == 1:
        q_compact, k_compact, v_compact, b_compact, a_compact = (
            ldst_helper.load_activation_as_compact(
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfgs,
            ))

        out, new_recurrent_state = gdn.recurrent_gdn(
            q_compact=q_compact,
            k_compact=k_compact,
            v_compact=v_compact,
            b_compact=b_compact,
            a_compact=a_compact,
            state_prev=prev_recurrent,
            gdn_weights_ref=weights_ref.gdn,
            cfgs=cfgs,
            real_size=real_size,
        )

    else:
        q_large, k_large, v_large, b_large, a_large = (
            ldst_helper.load_activation_as_large(
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfgs,
            ))

        out, new_recurrent_state = gdn.chunked_gdn(
            q_large=q_large,
            k_large=k_large,
            v_large=v_large,
            b_large=b_large,
            a_large=a_large,
            state_prev=prev_recurrent,
            gdn_weights_ref=weights_ref.gdn,
            cfgs=cfgs,
            real_size=real_size,
        )

    # Store output and recurrent to vmem.
    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
    recurrent_slot_ref[...] = new_recurrent_state.astype(
        recurrent_slot_ref.dtype)

    if prev_recurrent_scratch_ref is not None:
        prev_recurrent_scratch_ref[...] = new_recurrent_state


def main_kernel(
    # Inputs.
    metadata_ref: ref_classes.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    _: jax.Array,
    weights_ref: ref_classes.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    # Scratch
    prev_conv_scratch_ref: jax.Array | None,
    prev_recurrent_scratch_ref: jax.Array | None,
    *,
    cfgs: configs.GDNConfigs,
):
    del conv_state_out_ref, recurrent_state_out_ref

    qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc = (
        ref_classes.create_allocs(
            metadata_ref=metadata_ref,
            qkv_ref=qkv_ref,
            b_ref=b_ref,
            a_ref=a_ref,
            out_ref=out_ref,
            conv_state_ref=conv_state_ref,
            recurrent_state_ref=recurrent_state_ref,
            cfgs=cfgs,
        ))

    num_tiles = metadata_ref.num_tiles[...]

    pipeline_func = pltpu.emit_pipeline(
        body=functools.partial(
            inner_kernel,
            cfgs=cfgs,
        ),
        grid=(num_tiles, ),
        in_specs=(
            qkv_alloc.spec,
            b_alloc.spec,
            a_alloc.spec,
            conv_alloc.spec,
            recurrent_alloc.spec,
        ),
        out_specs=(out_alloc.spec, ),
    )

    @pl.with_scoped(final_allocs=(
        qkv_alloc,
        b_alloc,
        a_alloc,
        conv_alloc,
        recurrent_alloc,
        out_alloc,
    ), )
    def _run(final_allocs):
        pipeline_func(
            qkv_ref,
            b_ref,
            a_ref,
            conv_state_ref,
            recurrent_state_ref,
            out_ref,
            scratches=(
                metadata_ref,
                weights_ref,
                prev_conv_scratch_ref,
                prev_recurrent_scratch_ref,
            ),
            allocations=final_allocs,
        )

    _run()


def compute_batched_seq_metadata(
    cfgs: configs.GDNConfigs,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    end_seq: jax.Array,
) -> ref_classes.MetadataRef:
    """Metadata for computing multiple sequences per tile."""

    max_seqs = seq_lens.size
    all_seqs = jnp.arange(max_seqs)

    # NOTE: Only supports use case where query_lens[i] = 1 for where i < end_seq.
    # This must be guaranteed by the function caller.
    # TODO(kyuyeunk): Add error handling when above condition is not met.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_valid_st_idx = jnp.where(all_seqs < end_seq, True, False)
    has_initial_state = (seq_lens - query_lens) > 0
    all_valid_seqs = jnp.where(is_valid_st_idx, all_seqs, 0)

    return ref_classes.MetadataRef.create(
        cfgs=cfgs,
        num_tiles=pl.cdiv(end_seq, cfgs.tile_size),
        st_idx_to_s_idx=all_valid_seqs,
        st_idx_to_b_idx=all_valid_seqs,
        st_idx_to_b_size=jnp.where(is_valid_st_idx, 1, 0),
        st_idx_is_first_tile=is_valid_st_idx,
        st_idx_is_last_tile=is_valid_st_idx,
        s_idx_has_initial_state=has_initial_state,
        s_idx_to_state_indices=state_indices,
    )


def compute_per_seq_metadata(
    cfgs: configs.GDNConfigs,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    start_seq: jax.Array,
    end_seq: jax.Array,
) -> ref_classes.MetadataRef:
    """Metadata for computing single sequence per tile."""

    max_seqs = seq_lens.size
    max_tokens = cfgs.batch_size
    all_seqs = jnp.arange(max_seqs)
    all_tokens = jnp.arange(max_tokens)

    # Shift to ensure first element is for start_seq.
    query_start_loc = jnp.roll(query_start_loc, shift=-start_seq)
    seq_lens = jnp.roll(seq_lens, shift=-start_seq)
    state_indices = jnp.roll(state_indices, shift=-start_seq)

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    # NOTE: query_lens is used for calculating num_tiles. Defensive programming
    # that masks out all the other values (seq_lens, state_indices) are not needed
    # since they will not be visited as long as num_tiles is correct.
    num_seqs = end_seq - start_seq
    query_lens = jnp.where(all_seqs < num_seqs, query_lens, 0)

    s_idx_to_num_tiles = pl.cdiv(query_lens, cfgs.chunk_size)
    s_idx_to_start_st_idx = jnp.cumulative_sum(s_idx_to_num_tiles,
                                               include_initial=True)
    st_idx_to_s_idx = jnp.repeat(all_seqs,
                                 s_idx_to_num_tiles,
                                 total_repeat_length=max_tokens)
    num_tiles = s_idx_to_num_tiles.sum()
    st_idx_to_t_idx = all_tokens - s_idx_to_start_st_idx[st_idx_to_s_idx]
    st_idx_to_b_idx = (query_start_loc[st_idx_to_s_idx] +
                       st_idx_to_t_idx * cfgs.chunk_size)

    st_idx_to_b_size = jnp.minimum(
        query_lens[st_idx_to_s_idx] - st_idx_to_t_idx * cfgs.chunk_size,
        cfgs.tile_size,
    )

    has_initial_state = (seq_lens - query_lens) > 0
    st_idx_is_first_tile = st_idx_to_t_idx == 0
    st_idx_is_last_tile = st_idx_to_t_idx == (
        s_idx_to_num_tiles[st_idx_to_s_idx] - 1)

    return ref_classes.MetadataRef.create(
        cfgs=cfgs,
        num_tiles=num_tiles,
        st_idx_to_s_idx=st_idx_to_s_idx,
        st_idx_to_b_idx=st_idx_to_b_idx,
        st_idx_to_b_size=st_idx_to_b_size,
        st_idx_is_first_tile=st_idx_is_first_tile,
        st_idx_is_last_tile=st_idx_is_last_tile,
        s_idx_has_initial_state=has_initial_state,
        s_idx_to_state_indices=state_indices,
    )


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "decode_tile_size",
        "mixed_tile_size",
        "zero_initialize_out",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    *,
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
    zero_initialize_out: bool = True,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    # TODO(kyuyeunk): Support bf16
    act_out_dtype = qkv.dtype
    conv_out_dtype = conv_state.dtype
    recurrent_out_dtype = recurrent_state.dtype

    qkv = qkv.astype(jnp.float32)
    b = b.astype(jnp.float32)
    a = a.astype(jnp.float32)
    conv_state = conv_state.astype(jnp.float32)

    # Step 1: Validate inputs.
    num_seqs = state_indices.size
    batch_size, dim = qkv.shape
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim, )
    assert query_start_loc.shape == (num_seqs + 1, )
    assert state_indices.shape == (num_seqs, )
    assert distribution.shape == (3, )
    act_in_dtype = qkv.dtype
    assert a.dtype == b.dtype == qkv.dtype == act_in_dtype

    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    decode_tile_size = min(decode_tile_size, batch_size)
    mixed_tile_size = min(mixed_tile_size, batch_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = aligned_num_v_heads - n_v
    qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size)))
    # a_log = jnp.pad(a_log, ((0, num_v_padding_size)))
    # dt_bias = jnp.pad(dt_bias, ((0, num_v_padding_size)))

    qkv = qkv.reshape(padded_batch_size, 1, -1)
    b = b.reshape(padded_batch_size, 1, -1)
    a = a.reshape(padded_batch_size, 1, -1)

    # Step 3: States and weights pre-processing.
    # TODO(kyuyeunk): To eliminate runtime cost, move this logic into model
    # loading stage.
    conv_state_shape = conv_state.shape
    conv_state = conv_state.reshape(-1, kernel_size - 1, 1, dim)
    conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = conv_bias.astype(
        jnp.float32) if conv_bias is not None else None

    # Step 4: Wrap inputs for the kernel.
    conv_weights = ref_classes.ConvWeightsRef(weight=conv_weight,
                                              bias=conv_bias)
    gdn_weights = ref_classes.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
    weights = ref_classes.WeightRefs(conv=conv_weights, gdn=gdn_weights)

    # Step 5: Create specs.
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

    def call_kernel(
        in_conv_state: jax.Array,
        in_recurrent_state: jax.Array,
        in_act: jax.Array | None,
        mode: configs.GDNMode,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if mode == configs.GDNMode.BATCHED:
            tile_size = decode_tile_size
        else:
            tile_size = mixed_tile_size

        cfgs = configs.GDNConfigs(
            mode=mode,
            batch_size=padded_batch_size,
            kernel_size=kernel_size,
            tile_size=tile_size,
            dim_size=dim,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d_k,
            v_head_dim=d_v,
            dtypes=configs.Dtypes(
                act_in=act_in_dtype,
                act_out=act_out_dtype,
                compute=jnp.bfloat16.dtype,
                # compute=jnp.float32.dtype,
                recurrent_state=in_recurrent_state.dtype,
                conv_state=in_conv_state.dtype,
            ),
        )

        # Step 6: Metadata preprocessing. Will be executed multiple times per-layer
        # but will be CSEed by compiler.
        if mode == configs.GDNMode.BATCHED:
            metadata = compute_batched_seq_metadata(
                cfgs=cfgs,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                end_seq=distribution[0],
            )
        else:
            metadata = compute_per_seq_metadata(
                cfgs=cfgs,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                start_seq=distribution[0],
                end_seq=distribution[-1],
            )

        metadata_spec = jax.tree.map(lambda _: smem_spec, metadata)

        # Step 7: Handle case where write needs to be done in existing out.
        in_out_spec = None
        input_output_aliases = {len(metadata) + 3: 1, len(metadata) + 4: 2}
        out_shape = cfgs.get_out_shape()

        if in_act is None and zero_initialize_out:
            in_act = jnp.zeros_like(out_shape)
        if in_act is not None:
            out_shape = in_act
            in_out_spec = hbm_spec
            input_output_aliases[len(metadata) + 5] = 0

        return pl.pallas_call(
            functools.partial(main_kernel, cfgs=cfgs),
            out_shape=(out_shape, in_conv_state, in_recurrent_state),
            in_specs=(
                metadata_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                in_out_spec,
                weights_spec,
            ),
            out_specs=(hbm_spec, hbm_spec, hbm_spec),
            scratch_shapes=cfgs.get_scratch_shape_dict(),
            input_output_aliases=input_output_aliases,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=cfgs.get_vmem_limit_bytes(),
            ),
            name=cfgs.get_kernel_name(),
            metadata=cfgs.get_metadata(),
        )(metadata, qkv, b, a, in_conv_state, in_recurrent_state, in_act,
          weights)

    out_act, out_conv_state, out_recurrent_state = call_kernel(
        conv_state, recurrent_state, None, configs.GDNMode.BATCHED)
    out_act, out_conv_state, out_recurrent_state = call_kernel(
        out_conv_state, out_recurrent_state, out_act, configs.GDNMode.PER_SEQ)

    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
    out_conv_state = out_conv_state.astype(conv_out_dtype)
    out_conv_state = out_conv_state.reshape(conv_state_shape)
    out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

    return (out_conv_state, out_recurrent_state), out_act
