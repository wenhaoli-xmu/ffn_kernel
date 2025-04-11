import triton
import triton.language as tl
import torch


@triton.jit
def _masked_ffn_infer(
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
def _masked_ffn_bwd_elem_kernel(
        a_ptr,
        w1_ptr
)


# def ffn_bf16_forward(x, w1, w2, w3, u1, u2, u3, mask):
#     M, K = x.shape
#     _, N = w1.shape
#     assert w1.shape == w3.shape
#     c = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
#     grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
#     ffn_kernel[grid](
#         x, w1, w3, u1, u3, c, mask,
#         M, N, K,
#         x.stride(0), x.stride(1),
#         w1.stride(0), w1.stride(1),
#         c.stride(0), c.stride(1),
#         BLOCK_SIZE_M=128,
#         BLOCK_SIZE_N=64,
#         BLOCK_SIZE_K=64,
#         GROUP_SIZE_M=8,
#         num_warps=4,
#         num_stages=4)
#     return linear_bf16(c, w2, u2, mask)
