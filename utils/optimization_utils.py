import math
import torch
from transformers import AdamW
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(tensor1=grad, tensor2=grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(other=grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(alpha=-group['weight_decay'] * group['lr'], other=p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(value=-step_size * group['lr'], tensor1=exp_avg, tensor2=denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(alpha=-group['weight_decay'] * group['lr'], other=p_data_fp32)
                    p_data_fp32.add_(alpha=-step_size * group['lr'], other=exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


OPTIMIZER_CLASSES = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
    'radam': RAdam,
}


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    https://github.com/allenai/allennlp/blob/9d88f8c5d5f5eb638d94241fcbb88f7273674be4/allennlp/nn/util.py#L2002-L2016
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def masked_log_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    """
    https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/allennlp/nn/util.py#L290-L318
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a log_softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

#
# def run_test():
#     import torch.nn as nn
#     model = nn.Sequential(*[nn.Linear(100, 10), nn.ReLU(), nn.Linear(10, 2)])
#     x = torch.randn(10, 100).repeat(100, 1)
#     y = torch.randint(0, 2, (10,)).repeat(100)
#     crit = nn.CrossEntropyLoss()
#     optim = RAdam(model.parameters(), lr=1e-2, weight_decay=0.01)
#     model.train()
#     for a in range(0, 1000, 10):
#         b = a + 10
#         loss = crit(model(x[a:b]), y[a:b])
#         loss.backward()
#         optim.step()
#         print('| loss: {:.4f} |'.format(loss.item()))
#
#
# if __name__ == '__main__':
#     run_test()
