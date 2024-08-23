import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch import nn

try:
    import cvxpy as cp
except ModuleNotFoundError:
    import pip

    pip.main(["install", "--user", "cvxpy"])
    import cvxpy as cp


class Weighting:
    def __init__(self, task_num, **kwargs):
        self.task_num = task_num

    def weight_loss(self, losses: dict):
        return losses

    def on_fit_start(self):
        if not self.automatic_optimization:
            #  IMTL has torch.linalg.inv operations which needs 32bit or bfloat16 precision
            assert (
                self.trainer.precision != "16-mixed"
            ), f"{self.weighting} requires 32-bit or bfloat16 precision for manual optimization. "

    def get_share_params(self):
        """Return the shared parameters of the model."""
        return self.model.parameters()

    def zero_grad_share_params(self):
        """Set gradients of the shared parameters to zero."""
        self.model.zero_grad(set_to_none=False)

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
            for tn, task in enumerate(self.task_name):
                if mode == "backward":
                    losses[task].backward(retain_graph=True) if (
                        tn + 1
                    ) != self.task_num else losses[task].backward()
                    grads[tn] = self._grad2vec()
                elif mode == "autograd":
                    grad = list(
                        torch.autograd.grad(
                            losses[task],
                            self.get_share_params(),
                            retain_graph=True,
                            allow_unused=True,
                        )
                    )

                    # grad = [g if g is not None else torch.Tensor(0).to("cuda") for g in grad]
                    for i, (g, param) in enumerate(
                        zip(grad, self.get_share_params(), strict=False)
                    ):
                        if g is None:
                            grad[i] = torch.zeros_like(
                                param
                            )  # Replace None gradient with a tensor of zeros
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
                    losses[task].backward(retain_graph=True) if (
                        tn + 1
                    ) != self.task_num else losses[task].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
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

    def _reset_grad(self, new_grads):
        for count, param in enumerate(self.get_share_params()):
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[: (count + 1)])
                param.grad.data = (
                    new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
                )

    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""Reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): needed if ``rep_grad`` True. gradients of the representations.
            grads (torch.Tensor): needed if ``rep_grad`` False. gradients of the shared parameters.
        """
        # if self.rep_grad: implement
        # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
        new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
        self._reset_grad(new_grads)


class StaticWeighting(Weighting):
    def __init__(self, loss_weights: dict):
        self.loss_weights = loss_weights

    def weight_loss(self, losses: dict) -> dict:
        return {k: v * self.loss_weights[k] for k, v in losses.items()}


class RLW(Weighting):
    def __init__(self, task_num: int):
        self.task_num = task_num

    def weight_loss(self, losses: dict) -> dict:
        weights = F.softmax(torch.randn(self.task_num), dim=-1)
        return {k: v * weights[i] for i, (k, v) in enumerate(losses.items())}


class DWA(Weighting):
    def __init__(self, task_num: int):
        self.task_num = task_num
        self.avg_losses = {}
        self.batch_losses = {}
        self.current_epoch = 0

    def on_fit_start(self):
        self.train_loss_buffer = torch.zeros([6, self.trainer.max_epochs])

    def weight_loss(self, losses: dict) -> dict:
        T = 2.0
        if self.current_epoch > 1:
            w_i = torch.Tensor(
                self.train_loss_buffer[:, self.current_epoch - 1]
                / self.train_loss_buffer[:, self.current_epoch - 2]
            )
            weights = self.task_num * F.softmax(w_i / T, dim=-1)
        else:
            weights = torch.ones(self.task_num)
        return {k: v * weights[i] for i, (k, v) in enumerate(losses.items())}

    def on_epoch_start(self):
        self.batch_losses = {}

    def on_epoch_end(self):
        for task_name, losses in self.batch_losses.items():
            if self.avg_losses.get(task_name) is None:
                self.avg_losses[task_name] = torch.zeros(len(self.train_loss_buffer[0]))
            self.avg_losses[task_name][self.current_epoch] = sum(losses) / len(losses)
        for task_idx, task_name in enumerate(self.batch_losses.keys()):
            self.train_loss_buffer[task_idx, self.current_epoch] = self.avg_losses[task_name][
                self.current_epoch
            ]
        self.current_epoch += 1


class UW(Weighting):
    def __init__(self, task_num: int):
        self.loss_scale = torch.nn.Parameter(torch.tensor([-0.5] * self.task_num))

    def weight_loss(self, losses: dict) -> dict:
        return {
            key: value / (2 * self.loss_scale[i].exp()) + self.loss_scale[i] / 2
            for i, (key, value) in enumerate(losses.items())
        }


class IMTL(Weighting):
    def __init__(self):
        self.automatic_optimization = False
        self.loss_scale = nn.Parameter(torch.tensor([0.0] * self.task_num))

    def weight_loss(self, losses: dict) -> dict:
        return {
            key: self.loss_scale[i].exp() * value - self.loss_scale[i]
            for i, (key, value) in enumerate(losses.items())
        }

    def manual_backward(self, losses):
        grads = self._get_grads(losses, mode="backward").to("cuda")
        grads_unit = grads / torch.norm(grads + 1e-8, p=2, dim=-1, keepdim=True)

        D = grads[0:1].repeat(self.task_num - 1, 1) - grads[1:]
        U = grads_unit[0:1].repeat(self.task_num - 1, 1) - grads_unit[1:]
        if self.trainer.precision != "32-true":
            # Convert to fp32 for the matrix operations
            D_fp32 = D.to(torch.float32)
            U_fp32 = U.to(torch.float32)
            grads_0_fp32 = grads[0].to(torch.float32)

            DU_t = torch.matmul(D_fp32, U_fp32.t()).to(torch.float32)
            DU_t_inv = torch.linalg.inv(DU_t)  # Using linalg.inv instead of inverse

            alpha_fp32 = torch.matmul(torch.matmul(grads_0_fp32, U_fp32.t()), DU_t_inv)
            alpha = alpha_fp32.to(grads.dtype)
        else:
            alpha = torch.matmul(
                torch.matmul(grads[0], U.t()), torch.linalg.inv(torch.matmul(D, U.t()))
            )
        alpha = torch.cat((1 - alpha.sum().unsqueeze(0), alpha), dim=0)
        self._backward_new_grads(alpha, grads=grads)


class AlignedMTL(Weighting):
    def __init__(self, task_num: int):
        self.automatic_optimization = False

    def manual_backward(self, losses):
        grads = self._get_grads(losses, mode="backward").to("cuda")

        M = torch.matmul(grads, grads.t())  # [num_tasks, num_tasks]
        lmbda, V = torch.linalg.eigh(M.to(torch.float32))  # , UPLO="U" if upper else "L")
        tol = torch.max(lmbda) * max(M.shape[-2:]) * torch.finfo().eps
        rank = sum(lmbda > tol)

        order = torch.argsort(lmbda, dim=-1, descending=True)
        lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]

        sigma = torch.diag(1 / lmbda.sqrt())
        B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())
        alpha = B.sum(0).to(torch.bfloat16)

        self._backward_new_grads(alpha, grads=grads)


class NashMTL(Weighting):
    def __init__(self, task_num: int):
        self.automatic_optimization = False
        self.step = 0
        self.prvs_alpha_param = None
        self.init_gtg = np.eye(self.task_num)
        self.prvs_alpha = np.ones(self.task_num, dtype=np.float32)
        self.normalization_factor = np.ones((1,))

    def manual_backward(self, losses):
        self.update_weights_every = 1
        self.optim_niter = 20
        self.max_norm = 1.0

        if self.step == 0:
            self._init_optim_problem()
        if (self.step % self.update_weights_every) == 0:
            self.step += 1
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode="autograd")

            GTG = torch.mm(grads, grads.t())
            self.normalization_factor = (
                torch.norm(GTG, dtype=torch.float32)
                .detach()
                .to(torch.float32)
                .cpu()
                .numpy()
                .reshape((1,))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.to(torch.float32).cpu().detach().numpy())
        else:
            self.step += 1
            alpha = self.prvs_alpha

        alpha = torch.from_numpy(alpha).to(torch.bfloat16).to("cuda")
        losses = {task: alpha[tn] * losses[task] for tn, task in enumerate(self.task_name)}
        sum(subloss for subloss in losses.values()).backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.get_share_params(), self.max_norm)

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
        )

    def solve_optimization(self, gtg):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            alpha_t = np.abs(alpha_t)  # ensures that alpha_t is non-negative in edge cases
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:  # noqa: E722
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        return prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)  # phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.task_num,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.task_num,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.task_num, self.task_num), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.task_num):
            constraint.extend([
                -cp.log(self.alpha_param[i] * self.normalization_factor_param) - cp.log(G_alpha[i])
                <= 0
                for i in range(self.task_num)
            ])
        obj = cp.Minimize(cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param)
        self.prob = cp.Problem(obj, constraint)


class MoCo(Weighting):
    def __init__(self, task_num: int):
        self.automatic_optimization = False
        self._compute_grad_dim()
        self.step = 0
        self.y = torch.zeros(self.task_num, self.grad_dim).to("cuda")
        self.lambd = (torch.ones([self.task_num]) / self.task_num).to("cuda")
        self.task_name = list(self.loss_weights.keys())

    def manual_backward(self, losses: dict):
        self.step += 1
        beta, beta_sigma = 0.5, 0.5  # kwargs['MoCo_beta'], kwargs['MoCo_beta_sigma']
        gamma, gamma_sigma = 0.1, 0.5  # kwargs['MoCo_gamma'], kwargs['MoCo_gamma_sigma']
        rho = 0  # kwargs['MoCo_rho']
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode="backward").to("cuda")
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                grads[tn] = grads[tn] / (grads[tn].norm() + 1e-8) * losses[task]
        self.y = self.y - (beta / self.step**beta_sigma) * (self.y - grads)
        self.lambd = F.softmax(
            self.lambd
            - (gamma / self.step**gamma_sigma)
            * (self.y @ self.y.t() + rho * torch.eye(self.task_num).to("cuda"))
            @ self.lambd,
            -1,
        )
        new_grads = self.y.t() @ self.lambd
        self._reset_grad(new_grads.to(torch.float32))  # .to(torch.bfloat16))


class DBMTL(Weighting):
    def __init__(self, task_num: int):
        self.automatic_optimization = False
        self.step = 0
        self._compute_grad_dim()
        self.grad_buffer = torch.zeros(self.task_num, self.grad_dim).to("cuda")

    def manual_backward(self, losses: dict):
        self.step += 1
        beta = 0.9  # kwargs['DB_beta']
        beta_sigma = 0  # kwargs['DB_beta_sigma']

        self._compute_grad_dim()
        log_losses = {k: torch.log(v + 1e-8) for k, v in losses.items()}
        batch_grads = self._compute_grad(log_losses, mode="backward").to("cuda")
        # [task_num, grad_dim]

        self.grad_buffer = batch_grads + (beta / self.step**beta_sigma) * (
            self.grad_buffer - batch_grads
        )

        u_grad = self.grad_buffer.norm(dim=-1)
        alpha = u_grad.max() / (u_grad + 1e-8)
        new_grads = sum([alpha[i] * self.grad_buffer[i] for i in range(self.task_num)])
        self._reset_grad(new_grads)  # .to(torch.bfloat16))


class PCGrad(Weighting):
    def __init__(self, task_num: int):
        self.automatic_optimization = False

    def manual_backward(self, losses: dict):
        batch_weight = np.ones(len(losses))
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode="backward").to("cuda")  # [task_num, grad_dim]
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = torch.randperm(self.task_num, device="cuda")
            for tn_j in task_index:
                g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                if g_ij < 0:
                    pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2) + 1e-8)
                    batch_weight[tn_j] -= (g_ij / (grads[tn_j].norm().pow(2) + 1e-8)).item()
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)


class CAGrad(Weighting):
    def __init__(self, task_num: int):
        self.automatic_optimization = False

    def manual_backward(self, losses: dict):
        calpha, rescale = 0.5, 1  # kwargs["calpha"], kwargs["rescale"]

        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode="backward").to("cuda")

        GG = torch.matmul(grads, grads.t()).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.task_num) / self.task_num
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (calpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
                + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to("cuda")
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(0) + lmbda * gw
        if rescale == 0:
            new_grads = g
        elif rescale == 1:
            new_grads = g / (1 + calpha**2)
        elif rescale == 2:
            new_grads = g / (1 + calpha)
        else:
            raise ValueError(f"No support rescale type {rescale}")
        self._reset_grad(new_grads)


class GradVac(Weighting):
    def __init__(self, task_num: int):
        self.automatic_optimization = False
        self.step = 0

    def manual_backward(self, losses: dict):
        beta = 0.5  # kwargs['GradVac_beta']
        group_type = 0  # kwargs['GradVac_group_type']
        if self.step == 0:
            self._init_rho(group_type)

        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode="backward").to("cuda")  # [task_num, grad_dim]

        batch_weight = np.ones(len(losses))
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                for k in range(len(self.k_idx)):
                    beg, end = sum(self.k_idx[:k]), sum(self.k_idx[: k + 1])
                    if end == -1:
                        end = grads.size()[-1]
                    rho_ijk = torch.dot(pc_grads[tn_i, beg:end], grads[tn_j, beg:end]) / (
                        pc_grads[tn_i, beg:end].norm() * grads[tn_j, beg:end].norm() + 1e-8
                    )
                    if rho_ijk < self.rho_T[tn_i, tn_j, k]:
                        w = (
                            pc_grads[tn_i, beg:end].norm()
                            * (
                                self.rho_T[tn_i, tn_j, k] * (1 - rho_ijk**2).sqrt()
                                - rho_ijk * (1 - self.rho_T[tn_i, tn_j, k] ** 2).sqrt()
                            )
                            / (
                                grads[tn_j, beg:end].norm()
                                * (1 - self.rho_T[tn_i, tn_j, k] ** 2).sqrt()
                                + 1e-8
                            )
                        )
                        pc_grads[tn_i, beg:end] += grads[tn_j, beg:end] * w
                    self.rho_T[tn_i, tn_j, k] = (1 - beta) * self.rho_T[
                        tn_i, tn_j, k
                    ] + beta * rho_ijk
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        self.step += 1

    def _init_rho(self, group_type):
        if group_type == 0:  # whole_model
            self.k_idx = [-1]
        elif group_type == 1:  # all_layer
            self.k_idx = []
            for module in self.encoder.modules():
                # if len(module._modules.items()) == 0 and len(module._parameters) > 0:
                self.k_idx.append(sum([w.data.numel() for w in module.parameters()]))
        elif group_type == 2:  # all_matrix
            self._compute_grad_dim()
            self.k_idx = self.grad_index
        else:
            raise ValueError
        self.rho_T = torch.zeros(self.task_num, self.task_num, len(self.k_idx)).to("cuda")
