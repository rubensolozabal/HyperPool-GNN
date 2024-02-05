"""Math utils functions."""

import torch


def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5
    



def logsumexp(inputs, dim=None, keepdim=False):
    """ Compute the log of the sum of exponentials of input elements. """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    max_input = torch.max(inputs, dim=dim, keepdim=True).values
    return max_input + torch.log(torch.sum(torch.exp(inputs - max_input), dim=dim, keepdim=keepdim))

def logminexp(inputs, dim=None, keepdim=False):
    """ Compute the log of the sum of exponentials of the negative of input elements,
        and then takes the negative of the result. """
    return -logsumexp(-inputs, dim, keepdim)


def temperature_scaled_logsumexp(inputs, tau, dim=None, keepdim=False):
    """ Compute the negative temperature-scaled logsumexp of input elements.
    Args:
        inputs (Tensor): The input tensor.
        tau (float): The temperature scaling factor.
        dim (int or tuple of ints, optional): The dimension or dimensions to reduce.
        keepdim (bool, optional): Whether to retain the reduced dimension(s) in the output.
    
    Returns:
        Tensor: The result tensor.
    """
    scaled_inputs = inputs / tau
    if dim is None:
        scaled_inputs = scaled_inputs.view(-1)
        dim = 0
    max_input = torch.max(scaled_inputs, dim=dim, keepdim=True).values
    sum_exp = torch.sum(torch.exp(scaled_inputs - max_input), dim=dim, keepdim=keepdim)
    log_sum_exp = max_input + torch.log(sum_exp) - torch.log(torch.tensor(inputs.shape[dim], dtype=inputs.dtype))
    return tau * log_sum_exp


def temperature_scaled_logminexp(inputs, tau, dim=None, keepdim=False):
    """ Compute the log of the sum of exponentials of the negative of input elements,
        and then takes the negative of the result. """
    return -temperature_scaled_logsumexp(-inputs, tau, dim, keepdim)


if __name__ == "__main__":
    # Example usage
    inputs = torch.tensor([1.0, 2.0, 3.0])
    tau = 0.1
    result = temperature_scaled_logminexp(inputs, tau)
    print(result)


