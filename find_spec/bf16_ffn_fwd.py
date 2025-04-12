import torch
from ffn_kernel.ffn import _masked_ffn_infer
import triton
import random
from random import randint
from find import BaseSearch
import json
random.seed(0)
torch.cuda.manual_seed(0)


def _bf16_ffn_forward(x, w1, w3, u1, u3, mask, blk_m, blk_n, blk_k, group_m, num_warps, num_stages):
    M, K = x.shape
    _, N = w1.shape
    
    grid = (
        triton.cdiv(M, blk_m) * 
        triton.cdiv(N, blk_n),)

    if torch.is_grad_enabled():
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
    
    def benchmark_object(self, *inputs):
        return _bf16_ffn_forward(*inputs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=str, default="[1024 * (i + 1) for i in range(128)]")
    parser.add_argument('--d', type=int, default=1024, help='embed dim')
    parser.add_argument('--h', type=int, default=2048, help='intermediate dim')
    parser.add_argument('--i', type=int, default=1024, help='numebr of image tokens')
    args = parser.parse_args()
    args.s = eval(args.s)

    configs = {}

    for s in args.s:
        # prepare inputs
        w1 = torch.randn((args.d, args.h), dtype=torch.bfloat16, device='cuda')
        w3 = torch.randn((args.d, args.h), dtype=torch.bfloat16, device='cuda')
        u1 = torch.randn((args.d, args.h), dtype=torch.bfloat16, device='cuda')
        u3 = torch.randn((args.d, args.h), dtype=torch.bfloat16, device='cuda')
        x = torch.randn((s, args.d), dtype=torch.bfloat16, device='cuda')
        mask = generate_mask(1, s, args.i).view(-1)
        inputs = (x, w1, w3, u1, u3, mask)

        print(f"seq-len: {s}", flush=True, end = '\t')

        search = Search(inputs)
        cfg = search.search()
        configs.update({s: cfg})
    
    file = json.dumps(configs, indent=4)
    with open(f"bf16_ffn_fwd.json", 'w') as f:
        f.write(file)
