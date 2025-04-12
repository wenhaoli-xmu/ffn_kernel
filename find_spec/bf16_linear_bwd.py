import torch
from ffn_kernel.linear import _fused_bwd_kernel
import triton
import random
from random import randint
from find import BaseSearch
import json
random.seed(0)
torch.cuda.manual_seed(0)


def _bf16_linear_backward(gd, a, b1, b2, mask, da_m, da_k, da_n, db_m, db_k, db_n, da_group, db_group, num_warps, num_stages):
    M, K = a.shape
    _, N = b1.shape

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
            "da_m": [16,32,64,128],
            "da_k": [16,32,64,128],
            "da_n": [16,32,64,128],
            "db_m": [16,32,64,128],
            "db_k": [16,32,64,128],
            "db_n": [16,32,64,128],
            "da_group": [8,16],
            "db_group": [8,16],
            "num_warps": [4],
            "num_stages": [4]
        }
    
    def benchmark_object(self, *inputs):
        return _bf16_linear_backward(*inputs)


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

        search = Search(inputs)
        cfg = search.search()
        configs.update({s: cfg})
    
    file = json.dumps(configs, indent=4)
    with open(f"bf16_linear_bwd.json", 'w') as f:
        f.write(file)
