import torch
import torch.nn as nn
from profiler import WallTime
from random import randint
import random
from ffn_kernel import MaskedLinear
import IPython
import argparse
from pygments.console import colorize
from contextlib import nullcontext
random.seed(0)
torch.random.manual_seed(0)


def generate_mask(batch_size, seq_len, image_size):
    visual_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device='cuda')
    mid = randint(a=image_size // 2, b=visual_mask.numel() - image_size // 2)
    low = mid - image_size // 2
    high = mid + image_size // 2
    visual_mask[:, low:high] = True
    return visual_mask


def zero_grad(module):
    for param in module.parameters():
        param.grad = None


def profile_module(name, module, inputs, bwd=False, num_trials=10):
    profile = WallTime(f"{name}", cuda=0)
    for _ in range(num_trials):
        zero_grad(module)
        with profile:
            out = module(*inputs)
            if bwd:
                out = out.sum()
                out.backward()
    profile.result(detail=True)
    

class Torch(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_v = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_t = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, visual_mask):
        visual_mask = visual_mask[:, :, None]
        return self.linear_v(x) * visual_mask + self.linear_t(x) * (~visual_mask)


class Triton(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_v = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_t = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, visual_mask):
        batch_size, seq_len = x.shape[:2]
        return MaskedLinear.apply(
            x.view(batch_size * seq_len, -1),
            self.linear_v.weight.T,
            self.linear_t.weight.T,
            visual_mask.view(-1),
        ).view(batch_size, seq_len, -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--check", action='store_true')
    parser.add_argument("--bwd", action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(0)

    batch_size = args.bsz
    embed_dim = 1024
    image_size = 1024

    # create modules
    pytorch_module = Torch(embed_dim, embed_dim).cuda().bfloat16()
    triton_module = Triton(embed_dim, embed_dim).cuda().bfloat16()

    # clone weights
    triton_module.linear_v.weight.data = pytorch_module.linear_v.weight.data.clone()
    triton_module.linear_t.weight.data = pytorch_module.linear_t.weight.data.clone()

    for seq_len in [2048 * i for i in range(1, 32)]:
        x = torch.rand((batch_size, seq_len, embed_dim), dtype=torch.bfloat16, device='cuda')
        mask = generate_mask(batch_size, seq_len, image_size)

        if args.check:
            x.requires_grad_(True)
            out1 = pytorch_module(x, mask)
            out1.sum().backward()
            ref_dx, ref_db1, ref_db2 = x.grad, pytorch_module.linear_v.weight.grad, pytorch_module.linear_t.weight.grad
            x.grad, pytorch_module.linear_v.weight.grad, pytorch_module.linear_t.weight.grad = None, None, None

            out2 = triton_module(x, mask)
            out2.sum().backward()
            my_dx, my_db1, my_db2 = x.grad, triton_module.linear_v.weight.grad, triton_module.linear_t.weight.grad
            x.grad, triton_module.linear_v.weight.grad, triton_module.linear_t.weight.grad = None, None, None

            print(colorize('green', 'pytorch-') + colorize('red', 'output value'), out1)
            print(colorize('green', 'triton-') + colorize('red', 'output value'), out2)
            print(colorize('green', 'l2-norm distance: ') + f"{torch.dist(out1, out2)}")
            print('=' * 64)

            print(colorize('green', 'pytorch-') + colorize('red', 'gd x'), ref_dx)
            print(colorize('green', 'triton-') + colorize('red', 'gd x'), my_dx)
            print(colorize('green', 'l2-norm distance: ') + f"{torch.dist(ref_dx, my_dx)}")
            print('=' * 64)
            
            print(colorize('green', 'pytorch-') + colorize('red', 'gd db1'), ref_db1)
            print(colorize('green', 'triton-') + colorize('red', 'gd db1'), my_db1)
            print(colorize('green', 'l2-norm distance: ') + f"{torch.dist(ref_db1, my_db1)}")
            print('=' * 64)

            print(colorize('green', 'pytorch-') + colorize('red', 'gd db2'), ref_db2)
            print(colorize('green', 'triton-') + colorize('red', 'gd db2'), my_db2)
            print(colorize('green', 'l2-norm distance: ') + f"{torch.dist(ref_db2, my_db2)}")
            print('=' * 64)

            IPython.embed(header='check')
            
        else:
            inputs = (x, mask)
            with nullcontext() if args.bwd else torch.no_grad():
                profile_module(f"pytorch-{seq_len}", pytorch_module, inputs, args.bwd)
                profile_module(f"triton-{seq_len}", triton_module, inputs, args.bwd)

        print("="*10)
