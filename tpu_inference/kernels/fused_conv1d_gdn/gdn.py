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

import jax
import jax.numpy as jnp

from tpu_inference.kernels.fused_conv1d_gdn import configs, ref_classes


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    norm = jnp.sqrt(
        jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps)
    return x / norm


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
    match jnp.dtype(dtype).itemsize:
        case 4:
            return jnp.int32
        case 2:
            return jnp.int16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


# NOTE: Fork of recurrent_scan_v2.py but applied various optimizations.
def invert_triangular_matrix(t: jax.Array, block_size=16) -> jax.Array:
    """Compute invert matrix of a given triauglar matrix."""

    # NOTE: if chunk_size=1, compiler will perform DCE.
    out_dtype = t.dtype
    chunk = t.shape[-1]
    block_size = min(block_size, chunk)
    num_blocks = chunk // block_size

    def local_forward_sub(t_mat, b_mat):
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_t = t_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_t[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    iota_r = jax.lax.broadcasted_iota(jnp.int32, t.shape, 1)
    iota_c = jax.lax.broadcasted_iota(jnp.int32, t.shape, 2)
    identity_mask = jnp.where(iota_r == iota_c, 1.0, 0.0)
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = identity_mask[:, start:end, :]

        if i == 0:
            target_b = e_block
        else:
            interaction_t = t[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jax.lax.dot(
                interaction_t,
                solved_x,
                dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            )
            target_b = e_block - prev_sum

        # NOTE: Utilize fp32 to minimize cost of sublane rolling.
        local_t = t[:, start:end, start:end].astype(jnp.float32)
        x_block = local_forward_sub(local_t, target_b)
        x_blocks.append(x_block.astype(out_dtype))

    return jnp.concatenate(x_blocks, axis=1)


def fused_transpose_broadcast(x: jax.Array, src_dim: int,
                              dst_dim: int) -> jax.Array:
    """Perform 1D transpose where results are broadcasted along src_dim."""
    assert x.shape[dst_dim] == 1

    dtype = x.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(x.shape)
    mask_size = mask_shape[src_dim]
    mask_shape[dst_dim] = mask_size
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


def chunked_gdn_per_seq(
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    gating_log: jax.Array,
    beta: jax.Array,
    state_prev: jax.Array,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array]:

    # NOTE: Repeat along non lane/sublane dim is free.
    q_repeat = jnp.repeat(q_large, cfgs.v_per_kq_head, axis=0)
    k_repeat = jnp.repeat(k_large, cfgs.v_per_kq_head, axis=0)

    # Compute cumulative sum of decay.
    # (1, 1, num_v_heads)
    g_cum_sum_list = [gating_log[:, :1]]
    for row in range(1, cfgs.chunk_size):
        g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[:, row:row + 1])
    # (1, chunk, num_v_heads)
    g_cum_sum_log = jnp.concat(g_cum_sum_list, axis=1)

    # (num_v_heads, chunk, 1)
    g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log,
                                              src_dim=2,
                                              dst_dim=0)
    g_cum_sum_log = g_cum_sum_log[:cfgs.num_v_heads]
    beta = fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)
    beta_large = beta[:cfgs.num_v_heads]

    # (num_v_heads, 1, chunk)
    g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log,
                                                src_dim=1,
                                                dst_dim=2)
    # (num_v_heads, chunk, chunk)
    g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
    gating_map = jnp.exp(g_cum_sum_diff_log)
    # (num_v_heads, chunk, 1)
    gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
    # (num_v_heads, chunk, 1)
    gating_forward = jnp.exp(g_cum_sum_log)
    # (num_v_heads, 1, 1)
    gating_last = gating_forward[:, -1:]

    mask_dtype = get_mask_dtype(cfgs.dtypes.compute)
    iota_r = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 1)
    iota_c = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 2)
    identity_mask = iota_r == iota_c
    strictly_lower_mask = iota_r > iota_c
    lower_mask = iota_r >= iota_c
    # (num_v_heads, chunk, chunk)
    gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)

    # (num_v_heads, chunk, kq_head_dim)
    k_beta_repeat = k_repeat * beta_large

    # (num_v_heads, chunk, chunk)
    beta_k_k_t = jax.lax.dot(
        k_beta_repeat,
        k_repeat,
        dimension_numbers=(((2, ), (2, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)
    gating_beta_k_k_t = gating_map_masked * beta_k_k_t
    t = jnp.where(identity_mask, 1, gating_beta_k_k_t)

    # (num_v_heads, chunk, chunk)
    t_inv = invert_triangular_matrix(t)

    # (num_v_heads, chunk, v_head_dim)
    v_beta_large = v_large * beta_large
    k_beta_gating = k_beta_repeat * gating_forward
    # (num_v_heads, chunk, 2 * v_head_dim)
    merged_v_k = jnp.concat([v_beta_large, k_beta_gating], axis=-1)
    merged_uw = jax.lax.dot(
        t_inv,
        merged_v_k,
        dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)

    # (num_v_heads, chunk, v_head_dim)
    u, w = jnp.split(merged_uw, 2, axis=-1)

    # (num_v_heads, chunk, kq_head_dim)
    q_large_gating = q_repeat * gating_forward
    # NOTE: Concatenate lhs with same rhs to leverage weight
    # stationary architecture.
    # (num_v_heads, 2 * chunk, kq_head_dim)
    merged_w_q = jnp.concat([w, q_large_gating], axis=1)
    # (num_v_heads, 2 * chunk, v_head_dim)
    merged_ws_out_updated = jax.lax.dot(
        merged_w_q,
        state_prev,
        dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )

    # NOTE: Splitting along non sublane/lane dim is free.
    ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
    ws = ws.astype(cfgs.dtypes.compute)

    # (num_v_heads, chunk, v_head_dim)
    u_ws = u - ws

    # (num_v_heads, chunk, kq_head_dim)
    k_repeat_gating = k_repeat * gating_backward

    # (num_v_heads, kq_head_dim, v_head_dim)
    state_new = jax.lax.dot(
        k_repeat_gating,
        u_ws,
        dimension_numbers=(((1, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)

    # (num_v_heads, kq_head_dim, v_head_dim)
    state_updated = state_prev * gating_last
    state = state_updated + state_new

    # (num_kq_heads, chunk, chunk)
    out_qk = jax.lax.dot(
        q_large,
        k_large,
        dimension_numbers=(((2, ), (2, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)
    # NOTE: must perform repeat after matmul to reduce required compute.
    # (num_v_heads, chunk, chunk)
    out_qk = jnp.repeat(out_qk, cfgs.v_per_kq_head, axis=0)
    out_qk *= gating_map
    out_qk = jnp.where(lower_mask, out_qk, 0)

    # (num_v_heads, chunk, v_head_dim)
    out_new = jax.lax.dot(
        out_qk,
        u_ws,
        dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )
    out = out_updated + out_new

    return out, state


def chunked_gdn(
    real_size: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    gdn_weights_ref: ref_classes.GDNWeightsRef,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array]:
    mask_dtype = get_mask_dtype(cfgs.dtypes.compute)
    mask_list = []
    iota = jax.lax.broadcasted_iota(mask_dtype, (1, cfgs.chunk_size, 1), 1)
    for idx in range(cfgs.seq_tile_size):
        mask_list.append(iota < real_size[idx])
    mask = jnp.stack(mask_list, axis=0)

    # (seqs, num_kq_heads, chunk, kq_head_dim)
    q_large = jnp.where(mask, q_large.astype(cfgs.dtypes.compute), 0)
    k_large = jnp.where(mask, k_large.astype(cfgs.dtypes.compute), 0)
    # (seqs, num_v_heads, chunk, v_head_dim)
    v_large = jnp.where(mask, v_large.astype(cfgs.dtypes.compute), 0)

    b_large = b_large.astype(cfgs.dtypes.compute)
    a_large = a_large.astype(cfgs.dtypes.compute)

    a_log = gdn_weights_ref.a_log[...].reshape(1, 1, 1, -1)
    a_log = a_log.astype(cfgs.dtypes.compute)
    dt_bias = gdn_weights_ref.dt_bias[...].reshape(1, 1, 1, -1)
    dt_bias = dt_bias.astype(cfgs.dtypes.compute)

    # NOTE: Any element-wise computations should occur before repeat.
    q_large = l2_norm(q_large)
    q_scale = cfgs.kq_head_dim**-0.5
    q_large *= q_scale
    k_large = l2_norm(k_large)

    # (seqs, 1, chunk, num_v_heads)
    beta = jax.nn.sigmoid(b_large)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_large + dt_bias)

    # NOTE: Maskout invalid rows to ensure they do not contribute to intra-chunk
    # computation.
    gating_log = jnp.where(mask, gating_log, 0)
    beta = jnp.where(mask, beta, 0)

    out_list = []
    state_list = []
    for idx in range(cfgs.seq_tile_size):
        out, state = chunked_gdn_per_seq(
            q_large[idx],
            k_large[idx],
            v_large[idx],
            gating_log[idx],
            beta[idx],
            state_prev[idx],
            cfgs,
        )
        out_list.append(out.swapaxes(0, 1))
        state_list.append(state)
    out = jnp.stack(out_list, axis=0)
    state = jnp.stack(state_list, axis=0)
    return out, state


def recurrent_gdn_per_seq(
    q_compact: jax.Array,
    k_compact: jax.Array,
    k_compact_t: jax.Array,
    v_compact: jax.Array,
    gating_log: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array]:

    out_list = []
    for c_idx in range(cfgs.chunk_size):
        # (num_v_heads, 1, kq_head_dim)
        q_curr = q_compact[:, c_idx]
        q_curr = jnp.repeat(q_curr, cfgs.v_per_kq_head, axis=0)
        k_curr = k_compact[:, c_idx]
        k_curr = jnp.repeat(k_curr, cfgs.v_per_kq_head, axis=0)

        # (num_v_heads, 1, v_head_dim)
        v_curr = v_compact[:, c_idx]

        # (num_v_heads, kq_head_dim, 1)
        k_curr_t = k_compact_t[:, c_idx]
        k_curr_t = jnp.repeat(k_curr_t, cfgs.v_per_kq_head, axis=0)

        # (num_v_heads, 1, 1)
        beta_curr = beta[:, c_idx]
        gating_curr = gating_log[:, c_idx]

        # (num_v_heads, kq_head_dim, v_head_dim)
        state_updated = state * gating_curr

        # (num_v_heads, 1, v_head_dim)
        v_updated = jax.lax.dot(
            k_curr,
            state_updated,
            dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        ).astype(cfgs.dtypes.compute)

        # (num_v_heads, 1, v_head_dim)
        v_diff = v_curr - v_updated
        v_new = beta_curr * v_diff

        # (num_v_heads, kq_head_dim, v_head_dim)
        # NOTE: Multiplication with k_curr_t needs to be deferred as much as
        # possible as it expands the dimension size by kq_head_dim.
        state_new = k_curr_t * v_new
        # (num_v_heads, kq_head_dim, v_head_dim)
        state = state_updated + state_new

        # (num_v_heads, 1, v_head_dim)
        out = jax.lax.dot(
            q_curr,
            state,
            dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        ).astype(cfgs.dtypes.compute)

        out_list.append(out[:, 0, :])

    return jnp.stack(out_list, axis=0), state


def recurrent_gdn(
    real_size: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    gdn_weights_ref: ref_classes.GDNWeightsRef,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array]:
    mask_dtype = get_mask_dtype(cfgs.dtypes.compute)
    mask_list = []
    iota = jax.lax.broadcasted_iota(mask_dtype, (1, cfgs.chunk_size, 1, 1), 1)
    for idx in range(cfgs.seq_tile_size):
        mask_list.append(iota < real_size[idx])
    mask = jnp.stack(mask_list, axis=0)

    # (seqs, num_kq_heads, chunk, kq_head_dim)
    q_compact = jnp.where(mask, q_compact.astype(cfgs.dtypes.compute), 0)
    k_compact = jnp.where(mask, k_compact.astype(cfgs.dtypes.compute), 0)
    # (seqs, num_v_heads, chunk, v_head_dim)
    v_compact = jnp.where(mask, v_compact.astype(cfgs.dtypes.compute), 0)

    a_log = gdn_weights_ref.a_log[...].reshape(1, 1, 1, 1, -1)
    a_log = a_log.astype(cfgs.dtypes.compute)
    dt_bias = gdn_weights_ref.dt_bias[...].reshape(1, 1, 1, 1, -1)
    dt_bias = dt_bias.astype(cfgs.dtypes.compute)

    # (seqs, num_kq_heads, toks, 1, kq_head_dim)
    q_compact = l2_norm(q_compact)
    q_scale = cfgs.kq_head_dim**-0.5
    q_compact *= q_scale
    k_compact = l2_norm(k_compact)
    k_compact_t = fused_transpose_broadcast(k_compact, src_dim=4, dst_dim=3)

    beta = jax.nn.sigmoid(b_compact)
    beta = jnp.where(mask, beta, 0)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_compact + dt_bias)
    gating_log = jnp.where(mask, gating_log, 0)
    gating_log = jnp.exp(gating_log)

    beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
    beta = beta[:, :cfgs.num_v_heads]
    gating_log = fused_transpose_broadcast(gating_log, src_dim=4, dst_dim=1)
    gating_log = gating_log[:, :cfgs.num_v_heads]

    out_list = []
    new_state_list = []

    for idx in range(cfgs.seq_tile_size):
        out, state = recurrent_gdn_per_seq(
            q_compact[idx],
            k_compact[idx],
            k_compact_t[idx],
            v_compact[idx],
            gating_log[idx],
            beta[idx],
            state_prev[idx],
            cfgs,
        )
        out_list.append(out)
        new_state_list.append(state)

    out = jnp.stack(out_list, axis=0)
    new_recurrent_state = jnp.stack(new_state_list, axis=0)

    return out, new_recurrent_state
