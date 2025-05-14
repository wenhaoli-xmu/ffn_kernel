import triton
import triton.language as tl
import torch
import json, os
from .utils import find_spec


with open(os.path.join(os.path.dirname(__file__), 'bf16_linear_fwd.json')) as f:
    fwd_config = list(json.load(f).values())

with open(os.path.join(os.path.dirname(__file__), 'bf16_linear_bwd.json')) as f:
    bwd_config = list(json.load(f).values())


@triton.jit
def _fused_bwd_kernel(
    g_ptr, a_ptr, b1_ptr, b2_ptr, da_ptr, db1_ptr, db2_ptr, mask_ptr,
    M, K, N,
    stride_gm, stride_gn,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_dam, stride_dak,
    stride_dbk, stride_dbn,
    DA_BLOCK_SIZE_M: tl.constexpr,
    DA_BLOCK_SIZE_K: tl.constexpr,
    DA_BLOCK_SIZE_N: tl.constexpr,
    DB_BLOCK_SIZE_M: tl.constexpr,
    DB_BLOCK_SIZE_K: tl.constexpr,
    DB_BLOCK_SIZE_N: tl.constexpr,
    DA_GROUP_SIZE: tl.constexpr,
    DB_GROUP_SIZE: tl.constexpr,
):
    """
    Intro
    -----
    Fused kernel for computing da and db simutanously.
    """
    pid = tl.program_id(axis=0)
    thresh = tl.cdiv(M, DA_BLOCK_SIZE_M) * tl.cdiv(K, DA_BLOCK_SIZE_K)
    if pid < thresh:
        _masked_matmul_bwd_da(
            pid, 
            g_ptr,
            b1_ptr,
            b2_ptr,
            da_ptr,
            mask_ptr,
            M, N, K,
            stride_gm, stride_gn,
            stride_bk, stride_bn,
            stride_dam, stride_dak,
            DA_BLOCK_SIZE_M,
            DA_BLOCK_SIZE_N,
            DA_BLOCK_SIZE_K,
            DA_GROUP_SIZE)
    else:
        pid -= thresh
        _masked_matmul_bwd_db(
            pid,
            g_ptr,
            a_ptr,
            db1_ptr,
            db2_ptr,
            mask_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_gm, stride_gn,
            stride_dbk, stride_dbn,
            DB_BLOCK_SIZE_M,
            DB_BLOCK_SIZE_N,
            DB_BLOCK_SIZE_K,
            DB_GROUP_SIZE)



@triton.jit
def _masked_matmul_infer(
        a_ptr, b1_ptr, b2_ptr, c_ptr, mask_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_b1k, stride_b1n,
        stride_b2k, stride_b2n,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    """
    Intro
    -----
    Forward propagation in inference mode
    This method is different with `_masked_matmul_forward`, replacing tf32 matmul to fp16 to speed up.
    """
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
    b1_ptrs = b1_ptr + (offs_k[:, None] * stride_b1k + offs_bn[None, :] * stride_b1n)
    b2_ptrs = b2_ptr + (offs_k[:, None] * stride_b2k + offs_bn[None, :] * stride_b2n)
    mask_ptrs = mask_ptr + offs_mm

    mask_mask = offs_mm < M
    mask_val = tl.load(mask_ptrs, mask=mask_mask, other=0).to(tl.int1)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        base_mask_a = offs_k[None, :] < K - k * BLOCK_SIZE_K
        base_mask_b = offs_k[:, None] < K - k * BLOCK_SIZE_K
        
        if tl.max(mask_val) == 0:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0).to(tl.float16)
            b = tl.load(b2_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float16)

        elif tl.min(mask_val) == 1:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0).to(tl.float16)
            b = tl.load(b1_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float16)

        else:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0).to(tl.float16)
            b1 = tl.load(b1_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            b2 = tl.load(b2_ptrs, mask=base_mask_b, other=0.0).to(tl.float16)
            accumulator += tl.where(
                mask_val[:, None], 
                tl.dot(a, b1, out_dtype=tl.float16), 
                tl.dot(a, b2, out_dtype=tl.float16))
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b1_ptrs += BLOCK_SIZE_K * stride_b1k
        b2_ptrs += BLOCK_SIZE_K * stride_b2k

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _masked_matmul_fwd(
        a_ptr, b1_ptr, b2_ptr, c_ptr, mask_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_b1k, stride_b1n,
        stride_b2k, stride_b2n,
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
    offs_mm = tl.multiple_of(offs_mm, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b1_ptrs = b1_ptr + (offs_k[:, None] * stride_b1k + offs_bn[None, :] * stride_b1n)
    b2_ptrs = b2_ptr + (offs_k[:, None] * stride_b2k + offs_bn[None, :] * stride_b2n)
    mask_ptrs = mask_ptr + offs_mm

    mask_mask = offs_mm < M
    mask_val = tl.load(mask_ptrs, mask=mask_mask, other=0).to(tl.int1)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        base_mask_a = offs_k[None, :] < K - k * BLOCK_SIZE_K
        base_mask_b = offs_k[:, None] < K - k * BLOCK_SIZE_K
        
        if tl.max(mask_val) == 0:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0)
            b = tl.load(b2_ptrs, mask=base_mask_b, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        elif tl.min(mask_val) == 1:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0)
            b = tl.load(b1_ptrs, mask=base_mask_b, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        else:
            a = tl.load(a_ptrs, mask=base_mask_a, other=0.0)
            b1 = tl.load(b1_ptrs, mask=base_mask_b, other=0.0)
            b2 = tl.load(b2_ptrs, mask=base_mask_b, other=0.0)
            accumulator += tl.where(
                mask_val[:, None], 
                tl.dot(a, b1), 
                tl.dot(a, b2))
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b1_ptrs += BLOCK_SIZE_K * stride_b1k
        b2_ptrs += BLOCK_SIZE_K * stride_b2k

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _masked_matmul_bwd_da(
        thread_idx,
        g_ptr, b1_ptr, b2_ptr, da_ptr, mask_ptr,
        M, N, K,
        stride_gm, stride_gn,
        stride_bk, stride_bn,
        stride_dam, stride_dak,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    """
    Intro
    -----
    Backward gradient toward activation.
    """
    pid = thread_idx
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_k = pid_k * BLOCK_SIZE_K

    offs_gm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bk = start_k + tl.arange(0, BLOCK_SIZE_K)
    offs_mask = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_gm = tl.where(offs_gm < M, offs_gm, 0)
    offs_bk = tl.where(offs_bk < K, offs_bk, 0)

    offs_gm = tl.max_contiguous(tl.multiple_of(offs_gm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bk = tl.max_contiguous(tl.multiple_of(offs_bk, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_mask = tl.max_contiguous(tl.multiple_of(offs_mask, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    g_ptrs = g_ptr + (offs_gm[:, None] * stride_gm + offs_n[None, :] * stride_gn)
    b1_ptrs = b1_ptr + (offs_bk[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    b2_ptrs = b2_ptr + (offs_bk[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    mask_ptrs = mask_ptr + offs_mask

    mask_mask = offs_mask < M
    mask_val = tl.load(mask_ptrs, mask=mask_mask, other=0).to(tl.int1)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):

        mask = offs_n[None, :] < N - n * BLOCK_SIZE_N
        
        if tl.min(mask_val) == 1:
            g = tl.load(g_ptrs, mask=mask, other=0.0)
            b = tl.load(b1_ptrs, mask=mask, other=0.0)
            accumulator = tl.dot(g, b.T, accumulator)
        
        elif tl.max(mask_val) == 0:
            g = tl.load(g_ptrs, mask=mask, other=0.0)
            b = tl.load(b2_ptrs, mask=mask, other=0.0)
            accumulator = tl.dot(g, b.T, accumulator)

        else:
            g = tl.load(g_ptrs, mask=mask, other=0.0)
            b1 = tl.load(b1_ptrs, mask=mask, other=0.0)
            b2 = tl.load(b2_ptrs, mask=mask, other=0.0)
            accumulator += tl.where(
                mask_val[:, None], 
                tl.dot(g, b1.T), 
                tl.dot(g, b2.T))
        
        g_ptrs += BLOCK_SIZE_N * stride_gn
        b1_ptrs += BLOCK_SIZE_N * stride_bn
        b2_ptrs += BLOCK_SIZE_N * stride_bn

    offs_dam = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dak = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    da_ptrs = da_ptr + stride_dam * offs_dam[:, None] + stride_dak * offs_dak[None, :]
    da_mask = (offs_dam[:, None] < M) & (offs_dak[None, :] < K)
    tl.store(da_ptrs, accumulator.to(tl.bfloat16), mask=da_mask)


@triton.jit
def _masked_matmul_bwd_db(
        thread_idx,
        g_ptr, 
        a_ptr, 
        db1_ptr, 
        db2_ptr, 
        mask_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_gm, stride_gn,
        stride_dbk, stride_dbn,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_K: tl.constexpr
):
    """
    Intro
    -----
    Backward gradient toward model weights.
    """
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
    offs_gn = start_n + tl.arange(0, BLOCK_SIZE_N)

    offs_ak = tl.where(offs_ak < K, offs_ak, 0)
    offs_gn = tl.where(offs_gn < N, offs_gn, 0)

    offs_ak = tl.max_contiguous(tl.multiple_of(offs_ak, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_gn = tl.max_contiguous(tl.multiple_of(offs_gn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_ak[None, :] * stride_ak
    g_ptrs = g_ptr + offs_m[:, None] * stride_gm + offs_gn[None, :] * stride_gn
    mask_ptrs = mask_ptr + offs_m

    accum_db1 = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    accum_db2 = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):

        mask = offs_m[:, None] < M - m * BLOCK_SIZE_M

        mask_mask = offs_m < M - m * BLOCK_SIZE_M
        mask_val = tl.load(mask_ptrs, mask=mask_mask).to(tl.int1)

        a = tl.load(a_ptrs, mask=mask, other=0.0)
        g = tl.load(g_ptrs, mask=mask, other=0.0) 

        if tl.min(mask_val) == 1:
            accum_db1 = tl.dot(a.T, g, accum_db1)

        elif tl.max(mask_val) == 0:
            accum_db2 = tl.dot(a.T, g, accum_db2)

        else:
            a_trans = a.T
            a1 = tl.where(mask_val[None, :], a_trans, 0.0)
            a2 = tl.where(~mask_val[None, :], a_trans, 0.0)
            accum_db1 = tl.dot(a1, g, accum_db1)
            accum_db2 = tl.dot(a2, g, accum_db2)
        
        a_ptrs += BLOCK_SIZE_M * stride_am
        g_ptrs += BLOCK_SIZE_M * stride_gm
        mask_ptrs += BLOCK_SIZE_M

    offs_dbk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_dbn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    db1_ptrs = db1_ptr + stride_dbk * offs_dbk[:, None] + stride_dbn * offs_dbn[None, :]
    db2_ptrs = db2_ptr + stride_dbk * offs_dbk[:, None] + stride_dbn * offs_dbn[None, :]
    db_mask = (offs_dbk[:, None] < K) & (offs_dbn[None, :] < N)
    tl.store(db1_ptrs, accum_db1.to(tl.bfloat16), mask=db_mask)
    tl.store(db2_ptrs, accum_db2.to(tl.bfloat16), mask=db_mask)


def _bf16_linear_forward(a, b1, b2, mask):
    """
    Intro
    -----
    Forward kernel launcher.
    Automatically choose tf32/fp16 computation according to enable gradient flag.
    """
    M, K = a.shape
    _, N = b1.shape
    
    cfg = find_spec(fwd_config, M // 1024)
    blk_m, blk_n, blk_k, group_m, num_warps, num_stages = cfg
    a = a.contiguous()

    grid = (
        triton.cdiv(M, blk_m) *
        triton.cdiv(N, blk_n),)

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    (_masked_matmul_fwd if torch.is_grad_enabled() else _masked_matmul_infer)[grid](
        a, b1, b2, c, mask,
        M, N, K,
        a.stride(0), a.stride(1),
        b1.stride(0), b1.stride(1),
        b2.stride(0), b2.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages)

    return c


def _bf16_linear_backward(gd, a, b1, b2, mask):
    """
    Intro
    -----
    Backward kernel launcher.
    """
    M, K = a.shape
    _, N = b1.shape
    
    gd = gd.contiguous()
    cfg = find_spec(bwd_config, M // 1024)
    da_m, da_k, da_n, db_m, db_k, db_n, da_group, db_group, num_warps, num_stages = cfg

    grid = (
        triton.cdiv(M, da_m) *
        triton.cdiv(K, da_k) + 
        triton.cdiv(K, db_k) * 
        triton.cdiv(N, db_n),
    )

    da, db1, db2 = torch.empty_like(a), torch.empty_like(b1), torch.empty_like(b2)

    _fused_bwd_kernel[grid](
        gd, a, b1, b2, da, db1, db2, mask,
        M, N, K,
        gd.stride(0), gd.stride(1),
        a.stride(0), a.stride(1),
        b1.stride(0), b1.stride(1),
        da.stride(0), da.stride(1),
        db1.stride(0), db1.stride(1),
        DA_BLOCK_SIZE_M=da_m,
        DA_BLOCK_SIZE_K=da_k,
        DA_BLOCK_SIZE_N=da_n,
        DB_BLOCK_SIZE_M=db_m,
        DB_BLOCK_SIZE_K=db_k,
        DB_BLOCK_SIZE_N=db_n,
        DA_GROUP_SIZE=da_group,
        DB_GROUP_SIZE=db_group,
        num_warps=num_warps,
        num_stages=num_stages)

    return da, db1, db2
