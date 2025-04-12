import triton
import triton.language as tl
import torch
from .utils import find_spec
import os, json


with open(os.path.join(os.path.dirname(__file__), 'bf16_ffn_fwd.json')) as f:
    fwd_config = list(json.load(f).values())

with open(os.path.join(os.path.dirname(__file__), 'bf16_ffn_bwd.json')) as f:
    bwd_config = list(json.load(f).values())


@triton.jit
def _masked_ffn_infer(
        a_ptr, 
        w1_ptr, 
        w3_ptr, 
        u1_ptr, 
        u3_ptr, 
        c_ptr, 
        mask_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_mm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_mm = tl.max_contiguous(tl.multiple_of(offs_mm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    u1_ptrs = u1_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    u3_ptrs = u3_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    mask_ptrs = mask_ptr + offs_mm

    mask_mask = offs_mm < M
    mask_val = tl.load(mask_ptrs, mask=mask_mask, other=0).to(tl.int1)

    accum1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    accum2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        base_mask_a = offs_k[None, :] < K - k * BLOCK_SIZE_K
        base_mask_b = offs_k[:, None] < K - k * BLOCK_SIZE_K

        if tl.max(mask_val) == 0:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0).to(tl.float16)
            u1 = tl.load(u1_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            u3 = tl.load(u3_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            accum1 = tl.dot(a, u1, accum1, out_dtype=tl.float16)
            accum2 = tl.dot(a, u3, accum2, out_dtype=tl.float16)

        elif tl.min(mask_val) == 1:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0).to(tl.float16)
            w1 = tl.load(w1_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            w3 = tl.load(w3_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            accum1 = tl.dot(a, w1, accum1, out_dtype=tl.float16)
            accum2 = tl.dot(a, w3, accum2, out_dtype=tl.float16)
        
        else:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0).to(tl.float16)
            w1 = tl.load(w1_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            w3 = tl.load(w3_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            u1 = tl.load(u1_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            u3 = tl.load(u3_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            accum1 += tl.where(
                mask_val[:, None], 
                tl.dot(a, w1, out_dtype=tl.float16),
                tl.dot(a, u1, out_dtype=tl.float16))
            accum2 += tl.where(
                mask_val[:, None],
                tl.dot(a, w3, out_dtype=tl.float16), 
                tl.dot(a, u3, out_dtype=tl.float16))

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_wk
        w3_ptrs += BLOCK_SIZE_K * stride_wk
        u1_ptrs += BLOCK_SIZE_K * stride_wk
        u3_ptrs += BLOCK_SIZE_K * stride_wk

    accum1 = accum1 * tl.sigmoid(accum1.to(tl.float32)).to(tl.float16) * accum2

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accum1.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _masked_ffn_fwd(
        a_ptr, 
        w1_ptr, 
        w3_ptr, 
        u1_ptr, 
        u3_ptr, 
        c_ptr, 
        mask_ptr, 
        t1_ptr, 
        t3_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        stride_tm, stride_tn,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_mm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_mm = tl.multiple_of(offs_mm, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    u1_ptrs = u1_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    u3_ptrs = u3_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    mask_ptrs = mask_ptr + offs_mm

    mask_mask = offs_mm < M
    mask_val = tl.load(mask_ptrs, mask=mask_mask, other=0).to(tl.int1)

    accum1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accum2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        base_mask_a = offs_k[None, :] < K - k * BLOCK_SIZE_K
        base_mask_b = offs_k[:, None] < K - k * BLOCK_SIZE_K

        if tl.max(mask_val) == 0:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0)
            u1 = tl.load(u1_ptrs, mask=base_mask_b, other=0.0)
            u3 = tl.load(u3_ptrs, mask=base_mask_b, other=0.0)
            accum1 = tl.dot(a, u1, accum1)
            accum2 = tl.dot(a, u3, accum2)

        elif tl.min(mask_val) == 1:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0)
            w1 = tl.load(w1_ptrs, mask=base_mask_b, other=0.0)
            w3 = tl.load(w3_ptrs, mask=base_mask_b, other=0.0)
            accum1 = tl.dot(a, w1, accum1)
            accum2 = tl.dot(a, w3, accum2)
        
        else:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0)
            w1 = tl.load(w1_ptrs, mask=base_mask_b, other=0.0)
            w3 = tl.load(w3_ptrs, mask=base_mask_b, other=0.0)
            u1 = tl.load(u1_ptrs, mask=base_mask_b, other=0.0)
            u3 = tl.load(u3_ptrs, mask=base_mask_b, other=0.0)
            accum1 += tl.where(
                mask_val[:, None], 
                tl.dot(a, w1),
                tl.dot(a, u1))
            accum2 += tl.where(
                mask_val[:, None],
                tl.dot(a, w3), 
                tl.dot(a, u3))

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_wk
        w3_ptrs += BLOCK_SIZE_K * stride_wk
        u1_ptrs += BLOCK_SIZE_K * stride_wk
        u3_ptrs += BLOCK_SIZE_K * stride_wk

    output = accum1 * tl.sigmoid(accum1) * accum2

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out_offs = stride_cm * offs_out_m[:, None] + stride_cn * offs_out_n[None, :]
    out_mask = (offs_out_m[:, None] < M) & (offs_out_n[None, :] < N)

    tl.store(c_ptr + out_offs, output.to(tl.bfloat16), mask=out_mask)
    tl.store(t1_ptr + out_offs, accum1.to(tl.bfloat16), mask=out_mask)
    tl.store(t3_ptr + out_offs, accum2.to(tl.bfloat16), mask=out_mask)


@triton.jit
def _masked_ffn_bwd_step1_kernel(
        g_ptr,
        t1_ptr,
        t3_ptr,
        s1_ptr,
        s3_ptr,
        mask_ptr,
        stride_m, stride_n,
        M, N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    m_idx, n_idx = pid // BLOCK_SIZE_N, pid % BLOCK_SIZE_N

    m_rng = tl.arange(BLOCK_SIZE_M)
    m_off = m_idx * BLOCK_SIZE_M + m_rng

    n_rng = tl.arange(BLOCK_SIZE_N)
    n_off = n_idx * BLOCK_SIZE_N + n_rng

    data_off = m_off * stride_m + n_off * stride_n
    data_msk = (m_off[:, None] < M) &  (n_off[None, :] < N)

    mask_off = m_idx + m_rng
    mask_msk = mask_off < M
    mask_data = tl.load(mask_ptr + mask_off, mask_msk)

    g_data = tl.load(g_ptr + data_off, data_msk, other=0.0)

    if tl.min(mask_data) == 1:
        t1_data = tl.load(t1_ptr + data_off, data_msk, other=0.0)
        t3_data = tl.load(t3_ptr + data_off, data_msk, other=0.0)

        tmp = t1_data * tl.sigmoid(t1_data.to(tl.float32).to(tl.bfloat16))
        path1 = g_data * t3_data * (t1_data + tmp - t1_data * tmp)
        path3 = g_data * tmp

        tl.store(t1_ptr + data_off, path1, data_msk)
        tl.store(t3_ptr + data_off, path3, data_msk)
    
    elif tl.max(mask_data) == 0:
        s1_data = tl.load(s1_ptr + data_off, data_msk, other=0.0)
        s3_data = tl.load(s3_ptr + data_off, data_msk, other=0.0)

        tmp = s1_data * tl.sigmoid(s1_data.to(tl.float32).to(tl.bfloat16))
        path1 = g_data * s3_data * (s1_data + tmp - s1_data * tmp)
        path3 = g_data * tmp

        tl.store(s1_ptr + data_off, path1, data_msk)
        tl.store(s3_ptr + data_off, path3, data_msk)

    else:
        t1_data = tl.load(t1_ptr + data_off, data_msk, other=0.0)
        t3_data = tl.load(t3_ptr + data_off, data_msk, other=0.0)
        s1_data = tl.load(s1_ptr + data_off, data_msk, other=0.0)
        s3_data = tl.load(s3_ptr + data_off, data_msk, other=0.0)

        tmp = t1_data * tl.sigmoid(t1_data.to(tl.float32).to(tl.bfloat16))
        path1 = g_data * t3_data * (t1_data + tmp - t1_data * tmp)
        path3 = g_data * tmp

        tl.store(t1_ptr + data_off, path1, data_msk & mask_data[:, None])
        tl.store(t3_ptr + data_off, path3, data_msk & mask_data[:, None])

        tmp = s1_data * tl.sigmoid(s1_data.to(tl.float32).to(tl.bfloat16))
        path1 = g_data * s3_data * (s1_data + tmp - s1_data * tmp)
        path3 = g_data * tmp

        tl.store(s1_ptr + data_off, path1, data_msk & ~mask_data[:, None])
        tl.store(s3_ptr + data_off, path3, data_msk & ~mask_data[:, None])


@triton.jit
def _masked_ffn_bwd_step2_kernel_dx(
        thread_idx,
        a_ptr,
        mask_ptr,
        t1_ptr,
        t3_ptr,
        s1_ptr,
        s3_ptr,
        dw1_ptr,
        dw3_ptr,
        du1_ptr,
        du3_ptr,
        stride_ak, stride_am,
        stride_tm, stride_tn,
        stride_dwk, stride_dwn,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_K: tl.constexpr
):
    pid = thread_idx
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_K * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE_K
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_K)
    pid_k = first_pid_k + (pid % group_size_k)
    pid_n = (pid % num_pid_in_group) // group_size_k

    start_k = pid_k * BLOCK_SIZE_K
    start_n = pid_n * BLOCK_SIZE_N

    offs_ak = start_k + tl.arange(0, BLOCK_SIZE_K)
    offs_rn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_mask = tl.arange(0, BLOCK_SIZE_M)

    offs_ak = tl.where(offs_ak < K, offs_ak, 0)
    offs_rn = tl.where(offs_rn < N, offs_rn, 0)

    offs_ak = tl.max_contiguous(tl.multiple_of(offs_ak, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_rn = tl.max_contiguous(tl.multiple_of(offs_rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_mask = tl.multiple_of(offs_mask, BLOCK_SIZE_M)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = a_ptr + offs_ak[:, None] * stride_ak + offs_m[None, :] * stride_am
    t1_ptrs = r_ptr + offs_m[:, None] * stride_rm + offs_rn[None, :] * stride_rn
    mask_ptrs = mask_ptr + offs_m

    accum1 = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    accum2 = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    for m in range(tl.cdiv(M, BLOCK_SIZE_M)):
        mask_a = offs_m[None, :] < M - m * BLOCK_SIZE_M
        mask_r = offs_m[:, None] < M - m * BLOCK_SIZE_M

        mask_val = tl.load(mask_ptrs, offs_m < M - m * BLOCK_SIZE_M)
        a = tl.load(a_ptrs, mask_a, other=0.0)
        r = tl.load(r_ptrs, mask_r, other=0.0)

        if tl.min(mask_val) == 1:
            accum1 = tl.dot(a, r, accum1)
        
        elif tl.max(mask_val) == 0:
            accum2 = tl.dot(a, r, accum2)

        else:
            r1 = tl.where(mask_val[None, :], r, 0)
            r2 = tl.where(~mask_val[None, :], r, 0)
            accum1 = tl.dot(a, r1, accum1)
            accum2 = tl.dot(a, r2, accum2)

    offs_out_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out_offs = stride_dwk * offs_out_k + stride_dwn * offs_out_n
    out_mask = (offs_out_k[:, None] < K) & (offs_out_n[None, :] < N)

    tl.store(dw1_ptr + out_offs, accum1.to(tl.bfloat16), out_mask)
    tl.store(dw3_ptr + out_offs, accum2.to(tl.bfloat16), out_mask)


@triton.jit
def _masked_ffn_bwd_step2_kernel_dwdu(
        thread_idx,
        r_ptr,
        w1_ptr,
        w3_ptr,
        u1_ptr,
        u3_ptr,
        mask_ptr,
        dx_ptr,
        stride_rm, stride_rn,
        stride_wn, stride_wk,
        stride_un, stride_uk,
        stride_xm, stride_xk,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_N: tl.constexpr
):
    pid = thread_idx
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_N * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_N
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_N)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_k = pid_k * BLOCK_SIZE_K

    offs_rm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_wk = start_k + tl.arange(0, BLOCK_SIZE_K)
    offs_mask = start_m + tl.arange(0, BLOCK_SIZE_M)

    offs_rm = tl.where(offs_rm < M, offs_rm, 0)
    offs_wk = tl.where(offs_wk < K, offs_wk, 0)

    offs_rm = tl.max_contiguous(tl.multiple_of(offs_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_wk = tl.max_contiguous(tl.multiple_of(offs_wk, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_mask = tl.multiple_of(offs_mask, BLOCK_SIZE_M)

    offs_n = tl.arange(0, BLOCK_SIZE_N)
    r_ptrs = r_ptr + offs_rm[:, None] * stride_rm + offs_n[None, :] * stride_rn
    w1_ptrs = w1_ptr + offs_n[:, None] * stride_rn + offs_wk[None, :] * stride_wk
    w3_ptrs = w3_ptr + offs_n[:, None] * stride_rn + offs_wk[None, :] * stride_wk
    mask_ptrs = mask_ptr + offs_mask
    mask_val = tl.load(mask_ptrs, offs_mask < M)

    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        mask_r = offs_n[None, :] < N - n * BLOCK_SIZE_N
        mask_w = offs_n[:, None] < N - n * BLOCK_SIZE_N

        r = tl.load(r_ptrs, mask_r, other=0.0)

        if tl.min(mask_val) == 1:
            w = tl.load(w1_ptrs, mask_w, other=0.0)
            accum = tl.dot(r, w, accum)
        
        elif tl.max(mask_val) == 0:
            w = tl.load(w3_ptrs, mask_w, other=0.0)
            accum = tl.dot(r, w, accum)

        else:
            w1 = tl.load(w1_ptrs, mask_w, other=0.0)
            w3 = tl.load(w3_ptrs, mask_w, other=0.0)
            accum += tl.where(
                mask_val[:, None],
                tl.dot(r, w1),
                tl.dot(r, w3))

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    out_offs = stride_xm * offs_out_m + stride_xk * offs_out_k
    out_mask = (offs_out_m[:, None] < M) & (offs_out_k[None, :] < K)

    tl.store(dx_ptr + out_offs, accum.to(tl.bfloat16), out_mask)


@triton.jit
def _masked_ffn_bwd_step2_fused_kernel(
        a_ptr,
        w1_ptr,
        w3_ptr,
        mask_ptr,
        r_ptr,
        dx_ptr,
        dw1_ptr,
        dw3_ptr,
        stride_ak, stride_am,
        stride_wn, stride_wk,
        stride_rm, stride_rn,
        stride_dxm, stride_dxk,
        stride_dwk, stride_dwn,
        M, N, K,
        DX_BLOCK_SIZE_M: tl.constexpr,
        DX_BLOCK_SIZE_N: tl.constexpr,
        DX_BLOCK_SIZE_K: tl.constexpr,
        DX_GROUP_SIZE: tl.constexpr,
        DW_BLOCK_SIZE_M: tl.constexpr,
        DW_BLOCK_SIZE_N: tl.constexpr,
        DW_BLOCK_SIZE_K: tl.constexpr,
        DW_GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid < K * N:
        _masked_ffn_bwd_step2_kernel_dx(
            pid, 
            a_ptr,
            mask_ptr,
            r_ptr,
            dw1_ptr,
            dw3_ptr,
            stride_ak, stride_am,
            stride_rm, stride_rn,
            stride_dwk, stride_dwn,
            M, N, K,
            DX_BLOCK_SIZE_M,
            DX_BLOCK_SIZE_N,
            DX_BLOCK_SIZE_K,
            DX_GROUP_SIZE)
    else:
        pid -= K * N
        _masked_ffn_bwd_step2_kernel_dw(
            pid,
            r_ptr,
            w1_ptr,
            w3_ptr,
            mask_ptr,
            dx_ptr,
            stride_rm, stride_rn,
            stride_wn, stride_wk,
            stride_dxm, stride_dxk,
            M, N, K,
            DW_BLOCK_SIZE_M,
            DW_BLOCK_SIZE_N,
            DW_BLOCK_SIZE_K,
            DW_GROUP_SIZE)
        

def _bf16_ffn_forward(x, w1, w3, u1, u3, mask):
    M, K = x.shape
    _, N = w1.shape

    cfg = find_spec(fwd_config, M // 1024)
    blk_m, blk_n, blk_k, group_m, num_warps, num_stages = cfg
    x = x.contiguous()
    
    grid = (
        triton.cdiv(M, blk_m) * 
        triton.cdiv(N, blk_n),)

    c = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    _masked_ffn_infer[grid](
        x, 
        w1, 
        w3, 
        u1, 
        u3, 
        c, 
        mask,
        M, N, K,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        c.stride(0), c.stride(1),
        blk_m,
        blk_n, 
        blk_k,
        group_m,
        num_warps=num_warps,
        num_stages=num_stages)

    return c
