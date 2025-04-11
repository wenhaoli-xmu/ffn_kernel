import torch
from ffn_kernel.linear import _masked_matmul_bwd_da
import triton
import random
from random import randint
from find import BaseSearch
import json
random.seed(0)
torch.cuda.manual_seed(0)


def _bf16_linear_bwd_da(grad, a, b1, b2, mask, blk_m, blk_n, blk_k, grp_m, num_warps, num_stages):
    M, K = a.shape
    _, N = b1.shape
    assert isinstance(b1, torch.Tensor)
    b1 = b1.t().contiguous()
    b2 = b2.t().contiguous()
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']), )
    da = torch.empty_like(a)
    _masked_matmul_bwd_da[grid](
        grad, b1, b2, da, mask,
        M, N, K,
        grad.stride(0), grad.stride(1),
        b1.stride(0), b1.stride(1),
        da.stride(0), da.stride(1),
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=grp_m,
        num_warps=num_warps,
        num_stages=num_stages)
    return da


def generate_mask(batch_size, seq_len, image_size):
    visual_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device='cuda')
    mid = randint(a=image_size // 2, b=visual_mask.numel() - image_size // 2)
    low = mid - image_size // 2
    high = mid + image_size // 2
    visual_mask[:, low:high] = True
    return visual_mask


class Search(BaseSearch):
    def get_configs(self):
        return {
            "blk_m": [32,64,128],
            "blk_n": [32,64,128],
            "blk_k": [32,64,128],
            "grp_m": [1,2,4,8,16],
            "warps": [4],
            "stages": [4]
        }
    
    def benchmark_object(self, inputs):
        return _bf16_linear_bwd_da(*inputs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=str, default="[1024 * (i + 1) for i in range(128)]")
    parser.add_argument('--d', type=int, default=1024, help='embed dim')
    parser.add_argument('--i', type=int, default=1024, help='numebr of image tokens')
    args = parser.parse_args()
    args.s = eval(args.s)

    configs = {}

    for s in args.s:
        # prepare inputs
        proj_1 = torch.randn((args.d, args.d), dtype=torch.bfloat16, device='cuda')
        proj_2 = torch.randn((args.d, args.d), dtype=torch.bfloat16, device='cuda')
        x = torch.randn((s, args.d), dtype=torch.bfloat16, device='cuda')
        grad = torch.randn((s, args.d), dtype=torch.bfloat16, device='cuda')
        mask = generate_mask(1, s, args.i).view(-1)
        inputs = (grad, x, proj_1, proj_2, mask)

        print(f"seq-len: {s}", flush=True, end = '\t')

        # search for optimum config
        search = Search(inputs)
        cfg = search.search()
        configs.update({s: cfg})
    
    file = json.dumps(configs, indent=4)
    with open(f"bf16_linear_bwd_da.json", 'w') as f:
        f.write(file)
