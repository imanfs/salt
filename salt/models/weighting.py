import torch


class Weighting:
    def __init__(self, rep_grad=False):
        self.rep_grad = rep_grad
        self.grad_dim = 0
        self.grad_index = []

    def init_param(self):
        r"""Define and initialize trainable parameters required by specific weighting methods."""
        if self.weighting == "UW":
            self.weight = torch.nn.Parameter([-0.5] * self.task_num)
        elif self.weighting == "DB_MTL":
            self.step = 0
            self._compute_grad_dim()
            self.grad_buffer = torch.zeros(self.task_num, self.grad_dim)

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        for count, param in enumerate(self.get_share_params()):
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[: (count + 1)])
                grad[beg:end] = param.grad.data.view(-1)
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        """mode: backward, autograd."""
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim)
            for tn in range(self.task_num):
                if mode == "backward":
                    losses[tn].backward(retain_graph=True) if (tn + 1) != self.task_num else losses[
                        tn
                    ].backward()
                    grads[tn] = self._grad2vec()
                elif mode == "autograd":
                    grad = list(
                        torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True)
                    )
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError("No support {} mode for gradient computation")
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size())
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == "backward":
                    losses[tn].backward(retain_graph=True) if (tn + 1) != self.task_num else losses[
                        tn
                    ].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        for count, param in enumerate(self.get_share_params()):
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[: (count + 1)])
                param.grad.data = (
                    new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
                )

    def _get_grads(self, losses, mode="backward"):
        r"""Returns the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:  # noqa: E722
                    raise ValueError(
                        "The representation dimensions of different tasks must be consistent"
                    ) from None
            return [per_grads, grads]
        self._compute_grad_dim()
        return self._compute_grad(losses, mode)

    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""Reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): needed if ``rep_grad`` True. gradients of the representations.
            grads (torch.Tensor): needed if ``rep_grad`` False. gradients of the shared parameters.
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([
                    batch_weight[i] * per_grads[i] for i in range(self.task_num)
                ])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = (tn + 1) != self.task_num
                    self.rep[task].backward(batch_weight[tn] * per_grads[tn], retain_graph=rg)
        else:
            # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)

    # @property
    # def backward(self, losses, **kwargs):
    #     r"""Args:
    #     losses (list): A list of losses of each task.
    #     kwargs (dict): A dictionary of hyperparameters of weighting methods.
    #     """
    #     pass
