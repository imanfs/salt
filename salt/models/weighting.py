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
    def __init__(self, task_names: list | None = None, auto_opt: bool = True):
        self.task_names = (
            task_names
            if task_names is not None
            else [
                "object_class_ce",
                "mask_ce",
                "mask_dice",
                "regression",
                "jets_classification",
                "track_origin",
            ]
        )
        self.task_num = len(self.task_names)
        self.auto_opt = auto_opt
        self.name = self.__class__.__name__
        print(
            "-----------------------------------------------------------------------------------"
            "-----------------\n"
            f"Model is being trained with {self.name} weighting."
        )

    def set_model(self, model: nn.Module):
        """Sets the model for parameter access."""
        self.model = model

    def weight_loss(self, losses: dict):
        """Weights the losses based on the weighting method."""
        return losses

    def on_fit_start(self, trainer):
        self.max_epochs = trainer.max_epochs
        self.calc_cos_sim = False
        if not self.auto_opt:
            #  IMTL has torch.linalg.inv operations which needs 32bit or bfloat16 precision
            assert (
                trainer.precision != "16-mixed"
            ), f"{self.name} requires 32-bit or bfloat16 precision for manual optimization."
            self.calc_cos_sim = True

    def on_train_start(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def manual_backward(self, loss):
        loss["loss"].backward()

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
        """Filter out or zero out gradients for parameters that have None gradients."""
        for i, (g, param) in enumerate(zip(grad, self.model.parameters(), strict=False)):
            if g is None:
                grad[i] = torch.zeros_like(param)  # Replace None gradient with a tensor of zeros
        return torch.cat([g.view(-1) for g in grad])

    def compute_grad(self, losses, mode):
        """mode: backward, autograd."""
        self._compute_grad_dim()
        grads = torch.zeros(self.task_num, self.grad_dim)
        for tn, task in enumerate(self.task_names):
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
            else:
                raise ValueError("No support {} mode for gradient computation")
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
        return self.compute_grad(losses, mode)

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
        self.new_grads = torch.stack(self.new_grads)

    def _get_grad_cos_sim(self, grad1, grad2):
        """Computes cosine similarity of gradients after flattening of tensors."""
        self.cos_sim = nn.CosineSimilarity(dim=0)
        cosine = self.cos_sim(grad1, grad2)
        return torch.clamp(cosine, -1, 1)

    def compute_pairwise_cossim(self, grads):
        self.cos_sims = []
        self.task_pairs = []

        for a, task_a in enumerate(self.task_names):
            for b, task_b in enumerate(self.task_names[a + 1 :], start=a + 1):
                cos_sim = self._get_grad_cos_sim(grads[a], grads[b])
                self.cos_sims.append(cos_sim.item())
                self.task_pairs.append((task_a, task_b))


class Static(Weighting):
    """Manual weighting which can be modified by the user using the config file."""

    def __init__(self, task_names=None, loss_weights: dict | None = None):
        super().__init__(task_names=task_names, auto_opt=True)
        self.loss_weights = (
            loss_weights if loss_weights is not None else dict.fromkeys(task_names, 1.0)
        )
        print("Weights are: ", self.loss_weights)

    def weight_loss(self, losses: dict) -> dict:
        return {k: v * self.loss_weights[k] for k, v in losses.items()}


class RLW(Weighting):
    """Random Loss Weighting (RLW).

    Proposed in
    `Reasonable Effectiveness of Random Weighting: Litmus Test for Multi-Task Learning (TMLR 2022):
      <https://openreview.net/forum?id=jjtFD8A1Wx>`_ \
    and implemented by us.

    """

    def __init__(self, task_names=None):
        super().__init__(task_names=task_names, auto_opt=True)

    def weight_loss(self, losses: dict) -> dict:
        weights = F.softmax(torch.randn(self.task_num), dim=-1)
        return {k: v * weights[i] for i, (k, v) in enumerate(losses.items())}


class DWA(Weighting):
    """Dynamic Weight Average (DWA).

    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019):
      <https://arxiv.org/abs/1803.10704>`_ \
    and implemented by modifying from the `official PyTorch implementation:
      <https://github.com/lorenmt/mtan>`_.

    Args:
        T (float, default=2.0): The softmax temperature.

    """

    def __init__(self, task_names=None):
        super().__init__(task_names=task_names, auto_opt=True)
        self.avg_losses = {}
        self.batch_losses = {}
        self.current_epoch = 0

    def on_train_start(self):
        self.train_loss_buffer = torch.zeros([6, self.max_epochs])

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
        self.loss_weights = {k: weights[i] for i, k in enumerate(self.task_names)}
        return {k: v * weights[i] for i, (k, v) in enumerate(losses.items())}

    def on_train_epoch_start(self):
        self.batch_losses = {}

    def on_train_epoch_end(self):
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
    """Uncertainty Weights (UW).

    This method is proposed in
    `MTL Using Uncertainty to Weigh Losses for Scene Geometry & Semantics (CVPR 2018):
      <https://arxiv.org/abs/1705.07115>`_ \
    and implemented by us.
    """

    def __init__(self, task_names=None):
        super().__init__(task_names=task_names, auto_opt=True)
        self.loss_scale = torch.nn.Parameter(torch.tensor([-0.5] * self.task_num))

    def weight_loss(self, losses: dict) -> dict:
        self.loss_weights = {
            k: 1 / (2 * self.loss_scale[i].exp()) for i, k in enumerate(self.task_names)
        }
        return {
            key: value / (2 * self.loss_scale[i].exp()) + self.loss_scale[i] / 2
            for i, (key, value) in enumerate(losses.items())
        }


class IMTL(Weighting):
    """Impartial Multi-task Learning (IMTL).

    This method is proposed in `Towards Impartial Multi-task Learning (ICLR 2021):
      <https://openreview.net/forum?id=IMPnRXEWpvr>`_ \
    and implemented by us.

    """

    def __init__(self, task_names=None):
        super().__init__(task_names=task_names, auto_opt=False)
        self.loss_scale = nn.Parameter(torch.tensor([0.0] * self.task_num))

    def weight_loss(self, losses: dict) -> dict:
        self.loss_weights = {k: self.loss_scale[i].exp() for i, k in enumerate(self.task_names)}
        return {
            task: self.loss_scale[tn].exp() * losses[task] - self.loss_scale[tn]
            for tn, task in enumerate(self.task_names)
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
        alpha = alpha.cpu() if alpha.device == torch.device("cuda") else alpha


class AlignedMTL(Weighting):
    """Aligned-MTL.

    This method is proposed in `Independent Component Alignment for Multi-Task Learning (CVPR 2023):
      <https://arxiv.org/abs/2305.19000>
    and implemented by modifying from the official PyTorch implementation in:
      <https://github.com/SamsungLabs/MTL>.

    """

    def __init__(self, task_names=None):
        super().__init__(task_names=task_names, auto_opt=False)

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
        alpha = B.sum(0).to(grads.dtype)

        self._backward_new_grads(alpha, grads=grads)
        print(self.new_grads.shape)
        # record alpha for weight logging
        alpha = alpha.cpu() if alpha.device == torch.device("cuda") else alpha
        self.alpha = {task: alpha[tn] for tn, task in enumerate(self.task_names)}


class NashMTL(Weighting):
    """Nash-MTL.

    This method is proposed in `Multi-Task Learning as a Bargaining Game (ICML 2022):
      <https://proceedings.mlr.press/v162/navon22a/navon22a.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation:
      <https://github.com/AvivNavon/nash-mtl>`_.

    Args:
        update_weights_every (int, default=1): Period of weights update.
        optim_niter (int, default=20): The max iteration of optimization solver.
        max_norm (float, default=1.0): The max norm of the gradients.
    """

    def __init__(self, task_names=None, update_weights_every=1, optim_niter=20, max_norm=1.0):
        super().__init__(task_names=task_names, auto_opt=False)
        self.update_weights_every = update_weights_every
        self.optim_niter = optim_niter
        self.max_norm = max_norm

        self.step = 0
        self.prvs_alpha_param = None
        self.init_gtg = np.eye(self.task_num)
        self.prvs_alpha = np.ones(self.task_num, dtype=np.float32)
        self.normalization_factor = np.ones((1,))

    def manual_backward(self, losses):
        if self.step == 0:
            self._init_optim_problem()
        if (self.step % self.update_weights_every) == 0:
            self.step += 1
            grads = self.compute_grad(losses, mode="autograd")

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
            self.new_grads = grads
        else:
            self.step += 1
            alpha = self.prvs_alpha

        alpha = torch.from_numpy(alpha).to(torch.bfloat16).to("cuda")
        losses = {task: alpha[tn] * losses[task] for tn, task in enumerate(self.task_names)}
        sum(subloss for subloss in losses.values()).backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        # record alpha for weight logging
        alpha = alpha.cpu() if alpha.device == torch.device("cuda") else alpha
        self.alpha = {task: alpha[tn] for tn, task in enumerate(self.task_names)}

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
    """MoCo.

    This method is proposed in
    `Mitigating Gradient Bias in Multi-objective Learning: Provably Convergent Approach (ICLR 2023):
      <https://openreview.net/forum?id=dLAYGdKTi2>`_ \
    and implemented based on the author sharing code (Heshan Fernando: fernah@rpi.edu).

    Args:
        MoCo_beta (float, default=0.5): The learning rate of y.
        MoCo_beta_sigma (float, default=0.5): The decay rate of MoCo_beta.
        MoCo_gamma (float, default=0.1): The learning rate of lambd.
        MoCo_gamma_sigma (float, default=0.5): The decay rate of MoCo_gamma.
        MoCo_rho (float, default=0): The L2 regularization parameter of lambda's update.
    """

    def __init__(
        self,
        task_names=None,
        MoCo_beta=0.5,
        MoCo_beta_sigma=0.5,
        MoCo_gamma=0.1,
        MoCo_gamma_sigma=0.5,
        MoCo_rho=0,
    ):
        super().__init__(task_names=task_names, auto_opt=False)
        self.beta, self.beta_sigma = MoCo_beta, MoCo_beta_sigma
        self.gamma, self.gamma_sigma = MoCo_gamma, MoCo_gamma_sigma
        self.rho = MoCo_rho

        self.step = 0
        self._compute_grad_dim()
        self.y = torch.zeros(self.task_num, self.grad_dim).to("cuda")
        self.lambd = (torch.ones([self.task_num]) / self.task_num).to("cuda")

    def manual_backward(self, losses: dict):
        grads = self.compute_grad(losses, mode="backward").to("cuda")
        with torch.no_grad():
            for tn, task in enumerate(self.task_names):
                grads[tn] = grads[tn] / (grads[tn].norm() + 1e-6) * losses[task]
        self.y = self.y - (self.beta / self.step**self.beta_sigma) * (self.y - grads)
        self.lambd = F.softmax(
            self.lambd
            - (self.gamma / self.step**self.gamma_sigma)
            * (
                self.y @ self.y.t()
                + self.rho * torch.eye(self.task_num, dtype=torch.float32).to("cuda")
            )
            @ self.lambd,
            -1,
        )
        self.new_grads = self.y.t() @ self.lambd
        self._reset_grad(self.new_grads.to(grads.dtype))
        # record alpha for weight logging
        alpha = self.lambd.cpu() if self.lambd.device == torch.device("cuda") else self.lambd
        self.alpha = {task: alpha[tn] for tn, task in enumerate(self.task_names)}


class DBMTL(Weighting):
    def __init__(self, task_names=None, DB_beta=0.9, DB_beta_sigma=0):
        super().__init__(task_names=task_names, auto_opt=False)
        self.beta, self.beta_sigma = DB_beta, DB_beta_sigma
        self.step = 0

    def set_model(self, model):
        super().set_model(model)
        self._compute_grad_dim()
        self.grad_buffer = torch.zeros(self.task_num, self.grad_dim).to("cuda")

    def manual_backward(self, losses: dict):
        self.step += 1

        log_loss = {k: torch.log(v + 1e-8) for k, v in losses.items()}
        grads = self.compute_grad(log_loss, mode="backward").to("cuda")
        # [task_num, grad_dim]

        self.grad_buffer = grads + (self.beta / self.step**self.beta_sigma) * (
            self.grad_buffer - grads
        )

        u_grad = self.grad_buffer.norm(dim=-1)
        alpha = u_grad.max() / (u_grad + 1e-8)
        self.new_grads = [alpha[i] * self.grad_buffer[i] for i in range(self.task_num)]
        self._reset_grad(sum(self.new_grads).to(grads.dtype))
        # record alpha for weight logging
        alpha = alpha.cpu() if alpha.device == torch.device("cuda") else alpha
        self.alpha = {task: alpha[tn] for tn, task in enumerate(self.task_names)}


class PCGrad(Weighting):
    """Project Conflicting Gradients (PCGrad).

    This method is proposed in `Gradient Surgery for Multi-Task Learning (NeurIPS 2020):
      <https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>`_ \
    and implemented by us.
    """

    def __init__(self, task_names=None):
        super().__init__(task_names=task_names, auto_opt=False)

    def manual_backward(self, losses: dict):
        batch_weight = np.ones(len(losses))
        grads = self.compute_grad(losses, mode="backward").to("cuda")  # [task_num, grad_dim]
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            # task_index = torch.randperm(self.task_num, device="cuda")
            task_index = list(range(self.task_num))
            random.shuffle(task_index)
            for tn_j in task_index:
                g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                if g_ij < 0:
                    pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2) + 1e-8)
                    batch_weight[tn_j] -= (g_ij / (grads[tn_j].norm().pow(2) + 1e-8)).item()
        self.new_grads = pc_grads
        self._reset_grad(self.new_grads.sum(0))
        # record alpha for weight logging
        alpha = torch.Tensor(batch_weight).cpu()
        self.alpha = {task: alpha[tn] for tn, task in enumerate(self.task_names)}


class CAGrad(Weighting):
    """Conflict-Averse Gradient descent (CAGrad).

    Proposed in `Conflict-Averse Gradient Descent for Multi-task learning (NeurIPS 2021):
      <https://openreview.net/forum?id=_61Qh8tULj_>`
    and implemented by modifying from the `official PyTorch implementation:
      <https://github.com/Cranial-XIX/CAGrad>`_.

    Args:
        calpha (float, default=0.5): A hyperparameter that controls the convergence rate.
        rescale ({0, 1, 2}, default=1): The type of the gradient rescaling.
    """

    def __init__(self, task_names=None, calpha=0.5, rescale=1):
        super().__init__(task_names=task_names, auto_opt=False)

        self.calpha = calpha
        self.rescale = rescale

    def manual_backward(self, losses: dict):
        grads = self.compute_grad(losses, mode="backward").to("cuda")

        GG = torch.matmul(grads, grads.t()).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.task_num) / self.task_num
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.to(torch.float32).numpy()
        b = x_start.copy()
        c = (self.calpha * g0_norm + 1e-8).item()

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
        if self.rescale == 0:
            new_grads = g
        elif self.rescale == 1:
            new_grads = g / (1 + self.calpha**2)
        elif self.rescale == 2:
            new_grads = g / (1 + self.calpha)
        else:
            raise ValueError(f"No support rescale type {self.rescale}")
        self.new_grads = new_grads
        self._reset_grad(new_grads)
        # record alpha for weight logging
        alpha = ww.cpu() if ww.device == torch.device("cuda") else ww
        self.alpha = {task: alpha[tn] for tn, task in enumerate(self.task_names)}


class GradVac(Weighting):
    """Gradient Vaccine (GradVac).

    Proposed in `Gradient Vaccine: Investigating and Improving Multi-task Optimization
    in Massively Multilingual Models (ICLR 2021 Spotlight):
      <https://openreview.net/forum?id=F1vEjWK-lH_>`_ \
    and implemented by us.

    Args:
        GradVac_beta (float, default=0.5):
            The exponential moving average (EMA) decay parameter.
        GradVac_group_type (int, default=0):
            Parameter granularity (0: whole_model; 1: all_layer; 2: all_matrix).
    """

    def __init__(self, task_names=None, GradVac_beta=0.5, GradVac_group_type=0):
        super().__init__(task_names=task_names, auto_opt=False)
        self.beta = GradVac_beta
        self.group_type = GradVac_group_type

        self.step = 0

    def manual_backward(self, losses: dict):
        if self.step == 0:
            self._init_rho(self.group_type)

        grads = self.compute_grad(losses, mode="backward").to("cuda")  # [task_num, grad_dim]

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
                        batch_weight[tn_j] += w.item()
                    self.rho_T[tn_i, tn_j, k] = (1 - self.beta) * self.rho_T[
                        tn_i, tn_j, k
                    ] + self.beta * rho_ijk
        self.new_grads = pc_grads
        self._reset_grad(self.new_grads.sum(0))
        self.step += 1
        # record alpha for weight logging
        alpha = torch.Tensor(batch_weight).cpu()
        self.alpha = {task: alpha[tn] for tn, task in enumerate(self.task_names)}

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
