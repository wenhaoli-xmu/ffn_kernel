import triton
import triton.language as tl
import torch


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


@triton.jit
def two_way_matmul(
        a_ptr, b1_ptr, b2_ptr, c_ptr, mask_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_b1k, stride_b1n,
        stride_b2k, stride_b2n,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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
def ffn_kernel(
        a_ptr, w1_ptr, w3_ptr, u1_ptr, u3_ptr, c_ptr, mask_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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


def last_matmul(a, b1, b2, mask):
    M, K = a.shape
    _, N = b1.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    two_way_matmul[grid](
        a, b1, b2, c, mask,
        M, N, K,
        a.stride(0), a.stride(1),
        b1.stride(0), b1.stride(1),
        b2.stride(0), b2.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=4)
    return c


def ffn_fp16(x, w1, w2, w3, u1, u2, u3, mask):
    M, K = x.shape
    _, N = w1.shape
    assert w1.shape == w3.shape
    c = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    ffn_kernel[grid](
        x, w1, w3, u1, u3, c, mask,
        M, N, K,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=4)
    return last_matmul(c, w2, u2, mask)
