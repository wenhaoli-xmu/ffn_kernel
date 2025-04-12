from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from .linear import _bf16_linear_forward, _bf16_linear_backward
from .ffn import _bf16_ffn_forward


class MaskedLinear(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, a, b1, b2, mask):
        ctx.save_for_backward(a, mask)
        ctx.b1, ctx.b2 = b1, b2
        return _bf16_linear_forward(a, b1, b2, mask)

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        a, mask = ctx.saved_tensors
        b1, b2 = ctx.b1, ctx.b2
        da, db1, db2 = _bf16_linear_backward(grad, a, b1, b2, mask)
        return da, db1, db2, None


class MaskedFFN(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, a, w1, w2, w3, u1, u2, u3, mask):
        a = _bf16_ffn_forward(a, w1, w3, u1, u3, mask)
        return _bf16_linear_forward(a, w2, u2, mask)

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        raise NotImplementedError
