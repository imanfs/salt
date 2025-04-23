import torch
from torch import nn


class Gradients(nn.Module):
    def __init__(self):
        super().__init__()

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.model.parameters():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        for count, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[: (count + 1)])
                grad[beg:end] = param.grad.data.view(-1)
        return grad

    def _combine_grads(self, grad):
        for i, (g, param) in enumerate(zip(grad, self.model.parameters(), strict=False)):
            if g is None:
                grad[i] = torch.zeros_like(param)  # Replace None gradient with a tensor of zeros
        return torch.cat([g.view(-1) for g in grad])

    def _compute_grad(self, losses, mode):
        """mode: backward, autograd."""
        grads = torch.zeros(self.task_num, self.grad_dim)
        for tn, task in enumerate(self.task_name):
            if mode == "backward":
                losses[task].backward(retain_graph=True) if (tn + 1) != self.task_num else losses[
                    task
                ].backward()
                grads[tn] = self._grad2vec()
            elif mode == "autograd":
                grad = list(
                    torch.autograd.grad(
                        losses[task],
                        self.model.parameters(),
                        retain_graph=True,
                        allow_unused=True,
                    )
                )
                grads[tn] = self._combine_grads(grad)
            elif mode == "no_grad":
                with torch.no_grad():
                    grad = list(
                        torch.autograd.grad(
                            losses[task],
                            self.model.parameters(),
                            retain_graph=True,
                            allow_unused=True,
                        )
                    )
                grads[tn] = self._combine_grads(grad)
            else:
                raise ValueError("No support {} mode for gradient computation")
            if not self.automatic_optimization:
                self.model.zero_grad(set_to_none=False)
        return grads

    def _get_grads(self, losses, mode="backward"):
        r"""Returns the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        self._compute_grad_dim()
        return self._compute_grad(losses, mode)

    def _reset_grad(self, new_grads):
        for count, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[: (count + 1)])
                param.grad.data = (
                    new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
                )

    def _backward_new_grads(self, batch_weight, grads=None):
        r"""Reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): needed if ``rep_grad`` True. gradients of the representations.
            grads (torch.Tensor): needed if ``rep_grad`` False. gradients of the shared parameters.
        """
        self.new_grads = [batch_weight[i] * grads[i] for i in range(self.task_num)]
        self._reset_grad(sum(self.new_grads))
