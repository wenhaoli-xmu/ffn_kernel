import torch
import torch.nn as nn
from profiler import WallTime
from random import randint
import random
from ffn_kernel import linear_bf16, linear_fp32
import IPython
random.seed(0)
torch.random.manual_seed(0)


def generate_mask(batch_size, seq_len, image_size):
    visual_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device='cuda')
    mid = randint(a=image_size // 2 + 1, b=visual_mask.numel() - image_size // 2 - 1)
    low = mid - image_size // 2
    high = mid + image_size // 2
    visual_mask[:, low:high] = True
    return visual_mask


def profile_module(name, module, inputs, num_trials=10):
    profile = WallTime(f"{name}", cuda=0)
    outs = []
    for _ in range(num_trials):
        with profile:
            out = module(*inputs)
            outs.append(out)
    profile.result(detail=True)
    return outs
    

class Torch(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_v = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_t = nn.Linear(in_dim, out_dim, bias=False)

    @torch.no_grad()
    def forward(self, x, visual_mask):
        visual_mask = visual_mask[:, :, None]
        return self.linear_v(x) * visual_mask + self.linear_t(x) * (~visual_mask)


class Triton(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_v = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_t = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, visual_mask):
        batch_size = x.shape[0]
        return (linear_bf16 if x.dtype == torch.bfloat16 else linear_fp32)(
            x.flatten(0,1),
            self.linear_v.weight.data.T,
            self.linear_t.weight.data.T,
            visual_mask.flatten(),
        ).unflatten(0, (batch_size, -1))


if __name__ == "__main__":
    import argparse
    from pygments.console import colorize

    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--check", action='store_true')
    parser.add_argument("--autotune", action='store_true')
    parser.add_argument("--fp32", action='store_true')
    args = parser.parse_args()

    assert ~(args.check & args.autotune)

    torch.cuda.set_device(0)

    batch_size = args.bsz
    embed_dim = 1024
    intermediate_dim = 2048
    image_size = 1024

    # create modules
    pytorch_module = Torch(embed_dim, intermediate_dim).cuda()
    triton_module = Triton(embed_dim, intermediate_dim).cuda()
    if not args.fp32:
        pytorch_module = pytorch_module.to(torch.bfloat16)
        triton_module = triton_module.to(torch.bfloat16)

    # clone weights
    triton_module.linear_v.weight.data = pytorch_module.linear_v.weight.data.clone()
    triton_module.linear_t.weight.data = pytorch_module.linear_t.weight.data.clone()

    if args.autotune:
        import os
        os.environ['TRITON_PRINT_AUTOTUNING'] = "1"

    for seq_len in [2048 * i for i in range(1, 32)]:
        x = torch.rand((batch_size, seq_len, embed_dim), dtype=torch.bfloat16 if not args.fp32 else torch.float32, device='cuda')
        mask = generate_mask(batch_size, seq_len, image_size)
        inputs = (x, mask)

        outs1 = profile_module(f"pytorch-{seq_len}", pytorch_module, inputs)
        outs2 = profile_module(f"triton-{seq_len}", triton_module, inputs)

        if args.check:
            print(colorize('green', 'pytorch:'), outs1[0])
            print(colorize('green', 'triton:'), outs2[0])
            print(colorize('green', 'l2-norm distance: ') + f"{torch.dist(outs1[0], outs2[0])}")

        if args.autotune or args.check:
            IPython.embed()

        print("="*10)
