import random
import warnings
from collections.abc import Mapping

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch import nn

from salt.models import InputNorm

try:
    import cvxpy as cp
except ModuleNotFoundError:
    import pip

    pip.main(["install", "--user", "cvxpy"])
    import cvxpy as cp


def check_unique(modules: nn.ModuleList, attr_name: str) -> None:
    assert len({getattr(m, attr_name) for m in modules}) == len(
        modules
    ), f"Attribute '{attr_name}' must be unique for class {modules[0].__class__.__name__}"


class ModelWrapperIman(L.LightningModule):  # noqa: PLR0904
    def __init__(
        self,
        model: nn.Module,
        lrs_config: Mapping[str, float],
        global_object: str,
        norm_config: dict | None = None,
        name: str = "salt",
        muP_config: dict | None = None,
    ):
        """A wrapper class for any model implemented in Salt.

        This wrapper class allows is as generic as possible. It wraps
        [`SaltModel`][salt.models.SaltModel], but could also be used to
        wrap any other model if you want to do train something that doesn't
        fit into the [`SaltModel`][salt.models.SaltModel] architecture.

        This class is responsible for containing things that are common to all
        salt models. These are:

        - A generic forward pass, including input normalisation
        - Training, validation and test steps, which include logging
        - Some sanity checks on the model configuration

        Parameters
        ----------
        model : nn.Module
            Model to be wrapped
        lrs_config: Mapping
            LRS config which has to be set manually for now
            https://github.com/omni-us/jsonargparse/issues/170#issuecomment-1288167674
        global_object : str
            Name of the global input object, as opposed to the constituent-level
            inputs. This argument is set automatically by the framework.
        norm_config : dict, optional
            Keyword arguments for [`salt.models.InputNorm`][salt.models.InputNorm].
        name: str, optional
            Name of the model, used for logging and inference output names
        muP_config: dict, optional
            The muP configuration.
        """
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.save_hyperparameters(logger=False)

        self.model = model
        self.lrs_config = lrs_config
        self.global_object = global_object
        self.name = name
        self.muP = muP_config if muP_config else {}
        self.last_val_batch_outs = None
        # Here the model should pick it up
        if self.muP:
            from salt.utils.muP_utils.configuration_muP import instantiate_mup

            load_path = None
            if "shape_path" in self.muP:
                load_path = self.muP["shape_path"]
            instantiate_mup(model, load_path)

        # all tasks should inherit the global object type
        self.model.global_object = self.global_object
        for task in self.model.tasks:
            task.global_object = self.global_object

        # ensure unique names for init_nets and tasks
        check_unique(self.model.init_nets, "input_name")
        check_unique(self.model.tasks, "name")

        # check that the model has the same output size for all init nets
        assert len({t.net.output_size for t in self.model.init_nets if t.input_name != "EDGE"}) == 1

        # create input normaliser
        assert norm_config is not None
        self.norm = InputNorm(**norm_config)

        # weighting params
        self.weighting = self.model.mask_decoder.mask_loss.weighting
        self.loss_weights = self.model.mask_decoder.mask_loss.loss_weights
        self.task_num = len(self.loss_weights)
        self.task_name = list(self.loss_weights.keys())
        self.calc_cos_sim = True
        self.init_param()
        self.rep_grad = False
        self.cos_sim = nn.CosineSimilarity(dim=0)

    def init_param(self):
        r"""Define and initialize trainable parameters required by specific weighting methods."""
        if self.weighting == "static":
            self._compute_grad_dim()
        if self.weighting == "DWA":
            self.avg_losses = {}  # type: dict[str, torch.Tensor]
            self.batch_losses = {}  # type: dict[str, list[float]]

        elif self.weighting == "UW":
            self.loss_scale = torch.nn.Parameter(torch.tensor([-0.5] * self.task_num))

        elif self.weighting in {"Aligned_MTL", "PCGrad", "CAGrad"}:
            self.automatic_optimization = False

        elif self.weighting == "IMTL":
            self.automatic_optimization = False
            self.loss_scale = nn.Parameter(torch.tensor([0.0] * self.task_num))

        elif self.weighting == "Nash_MTL":
            self.automatic_optimization = False
            self.step = 0
            self.prvs_alpha_param = None
            self.init_gtg = np.eye(self.task_num)
            self.prvs_alpha = np.ones(self.task_num, dtype=np.float32)
            self.normalization_factor = np.ones((1,))

        elif self.weighting == "MoCo":
            self.automatic_optimization = False
            self._compute_grad_dim()
            self.step = 0
            self.y = torch.zeros(self.task_num, self.grad_dim).to("cuda")
            self.lambd = (torch.ones([self.task_num]) / self.task_num).to("cuda")

        elif self.weighting == "DB_MTL":
            self.automatic_optimization = False
            self.step = 0
            self._compute_grad_dim()
            self.grad_buffer = torch.zeros(self.task_num, self.grad_dim).to("cuda")

        elif self.weighting == "GradVac":
            self.automatic_optimization = False
            self.step = 0

    def on_fit_start(self):
        if not self.automatic_optimization:
            [self.opt], [self.sch] = self.configure_optimizers()  # Access the optimizer
            assert (
                self.trainer.precision != "16-mixed"
            ), f"{self.weighting} requires 32-bit or bfloat16 precision for manual optimization. "
            self.calc_cos_sim = True
        if self.weighting == "DWA":
            self.train_loss_buffer = torch.zeros([6, self.trainer.max_epochs])

    def on_train_epoch_start(self):
        # Reset the list of losses and batch sizes at the start of each epoch
        self.batch_losses = {}

    def on_train_epoch_end(self):
        if self.weighting == "DWA":
            # Calculate average loss for each task for the epoch
            for task_name, losses in self.batch_losses.items():
                if self.avg_losses.get(task_name) is None:
                    self.avg_losses[task_name] = torch.zeros(self.trainer.max_epochs)

                self.avg_losses[task_name][self.current_epoch] = sum(losses) / len(losses)
            for task_idx, task_name in enumerate(self.batch_losses.keys()):
                self.train_loss_buffer[task_idx, self.current_epoch] = self.avg_losses[task_name][
                    self.current_epoch
                ]

    def weight_loss(self, losses: dict):
        # """Apply the loss weights to the loss dict."""

        if self.weighting == "static":
            for k in list(losses.keys()):
                losses[k] *= self.loss_weights[k]

        elif self.weighting == "GLS":
            loss_values = torch.tensor(list(losses.values()))  # Extract loss values as a tensor
            lossprod = loss_values.prod()  # Compute the product of all loss values

            self.loss_weights = {k: loss / (self.task_num * lossprod) for k, loss in losses.items()}

        elif self.weighting == "RLW":
            weights = F.softmax(torch.randn(self.task_num), dim=-1)
            # Apply weights to each loss
            for i, k in enumerate(list(losses.keys())):
                losses[k] *= weights[i]
            self.loss_weights = {k: weights[i] for i, k in enumerate(self.task_name)}

        elif self.weighting == "DWA":
            T = 2.0
            if self.current_epoch > 1:
                w_i = torch.Tensor(
                    self.train_loss_buffer[:, self.current_epoch - 1]
                    / self.train_loss_buffer[:, self.current_epoch - 2]
                )
                weights = self.task_num * F.softmax(w_i / T, dim=-1)

            else:
                weights = torch.ones(self.task_num)
            for i, k in enumerate(list(losses.keys())):
                losses[k] *= weights[i]
            self.loss_weights = {k: weights[i] for i, k in enumerate(self.task_name)}

        elif self.weighting == "UW":
            losses = {
                key: value / (2 * self.loss_scale[i].exp()) + self.loss_scale[i] / 2
                for i, (key, value) in enumerate(losses.items())
            }
            self.loss_weights = {
                k: 1 / (2 * self.loss_scale[i].exp()) for i, k in enumerate(self.task_name)
            }

        elif self.weighting == "IMTL":
            self.loss_weights = {k: self.loss_scale[i].exp() for i, k in enumerate(self.task_name)}
        return losses

    def total_loss(self, loss: dict):
        if self.weighting == "GLS":
            # Calculate the geometric mean of the losses
            loss_prod = 1.0
            for subloss in loss.values():
                loss_prod *= subloss
            geometric_mean_loss = torch.pow(loss_prod, 1.0 / self.task_num)
            loss["loss"] = geometric_mean_loss
        else:
            # combine the task losses through summation
            loss["loss"] = sum(subloss for subloss in loss.values())
        return loss

    def forward(self, inputs, pad_masks=None, labels=None):
        """Generic forward pass through any salt-compatible model.

        This function performs input normalisation and then calls the `self.model`'s
        forward pass. Don't call this method directy, instead use `__call__`.

        Parameters
        ----------
        inputs
            Any generic input to the model.
        pad_masks
            Input padding masks.
        labels
            Training targets. If not specified, assume we are running model inference
            (i.e. no loss computation).

        Returns
        -------
        Whatever is returned by `self.model`'s forward pass.
        """
        x = self.norm(inputs)
        return self.model(x, pad_masks, labels)

    def shared_step(self, batch, evaluation=False):
        """Function used to unpack the batch, run the forward pass, and compute
        losses, used by training, validation and test steps.

        Parameters
        ----------
        batch : tuple
            A single batch of inputs, pad_masks and labels
        evaluation : bool
            If true, don't compute the losses and return early

        Returns
        -------
        preds
            Model predictions
        labels
            True labels
        loss
            Reduced loss over the input batch
        """
        # unpack the batch
        inputs, pad_masks, labels = batch

        # forward pass through model
        preds, loss = self(inputs, pad_masks, labels)

        if evaluation:
            return preds, labels, pad_masks, None

        # calculate batch losses
        if self.weighting == "DWA":
            for task_name, loss_item in loss.items():
                if task_name not in self.batch_losses:
                    self.batch_losses[task_name] = []
                self.batch_losses[task_name].append(loss_item.detach())

        return preds, labels, pad_masks, loss

    def log_losses(self, loss, stage):
        kwargs = {"sync_dist": len(self.trainer.device_ids) > 1}
        self.log(f"{stage}_loss", loss["loss"], **kwargs)
        for t, loss_value in loss.items():
            n = f"{stage}_{t}_loss" if "loss" not in t else f"{stage}_{t}"
            self.log(n, loss_value, **kwargs)

    def log_weights(self, weights, stage):
        kwargs = {"sync_dist": len(self.trainer.device_ids) > 1}
        for t, weight in weights.items():
            n = f"{stage}_{t}_weights"
            self.log(n, weight, **kwargs)

    def log_cossim(self, task_pairs, cos_sims):
        kwargs = {"sync_dist": len(self.trainer.device_ids) > 1}
        for (task_a, task_b), cos_sim in zip(task_pairs, cos_sims, strict=False):
            n = f"cos_sim_{task_a}_{task_b}"
            self.log(n, cos_sim, **kwargs)

    def log_grads(self, grads):
        kwargs = {"sync_dist": len(self.trainer.device_ids) > 1}
        if not isinstance(grads, dict):
            grads = {task: grads[tn] for tn, task in enumerate(self.task_name)}
        for t, grad in grads.items():
            n = f"{t}_grad"
            L1_norm = torch.norm(grad, p=1)
            L2_norm = torch.norm(grad, p=2)
            self.log(n + "_L1", L1_norm, **kwargs)
            self.log(n + "_L2", L2_norm, **kwargs)

    def training_step(self, batch):
        # forward pass
        preds, labels, pad_masks, loss = self.shared_step(batch)

        log_loss = loss.copy()
        log_loss = self.total_loss(log_loss)
        self.log_losses(log_loss, stage="train")
        # weight and combine losses
        loss = self.total_loss(self.weight_loss(loss))
        if loss["loss"].isnan():
            raise RuntimeError(
                "Loss is NaN - this indicates something significant has gone wrong."
                "Check for any NaNs or infs in the input dataset. If nothing is found here, "
                "check 'docs/training.md - NaNs' for more information"
            )

        outputs = {
            "preds": preds,
            "labels": labels,
            "pad_masks": pad_masks,
        }

        if self.automatic_optimization:
            self.log_weights(self.loss_weights, stage="train")
            if self.calc_cos_sim:
                self.new_grads = self._compute_grad(loss, mode="autograd")  # for cossim calculation
                self.log_grads(self.new_grads)

        else:
            opt = self.optimizers()
            opt.zero_grad()  # Clear previous gradients
            self.manual_backward(loss)  # Manually perform backward pass
            opt.step()  # Update model parameters
            self.lr_schedulers().step()
            self.log_weights(self.alpha, stage="train")
            self.log_grads(self.new_grads)

        if self.calc_cos_sim:
            self.compute_pairwise_cossim(self.new_grads)
            self.log_cossim(self.task_pairs, self.cos_sims)
        return {**loss, "outputs": outputs}

    def validation_step(self, batch):
        # foward pass
        preds, labels, pad_masks, loss = self.shared_step(batch)

        # log raw losses
        log_loss = loss.copy()
        log_loss = self.total_loss(log_loss)
        self.log_losses(log_loss, stage="val")

        # weight and combine losses
        loss = self.total_loss(self.weight_loss(loss))
        # Store outputs to be used by the MaskformerMetrics callback
        outputs = {
            "preds": preds,
            "labels": labels,
            "pad_masks": pad_masks,
        }

        return {**loss, "outputs": outputs}

    def manual_backward(self, loss):
        if self.weighting == "IMTL":
            loss = {
                task: self.loss_scale[tn].exp() * loss[task] - self.loss_scale[tn]
                for tn, task in enumerate(self.task_name)
            }
            grads = self._get_grads(loss, mode="backward").to("cuda")
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

        elif self.weighting == "Aligned_MTL":
            grads = self._get_grads(loss, mode="backward").to("cuda")

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

        elif self.weighting == "Nash_MTL":
            self.update_weights_every = 1
            self.optim_niter = 20
            self.max_norm = 1.0

            if self.step == 0:
                self._init_optim_problem()
            if (self.step % self.update_weights_every) == 0:
                self.step += 1

                self._compute_grad_dim()
                grads = self._compute_grad(loss, mode="autograd")

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
            # record alpha for weight logging
            alpha = torch.from_numpy(alpha).to(torch.bfloat16).to(self.device)
            loss = {task: alpha[tn] * loss[task] for tn, task in enumerate(self.task_name)}
            sum(subloss for subloss in loss.values()).backward()

            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        elif self.weighting == "MoCo":
            self.step += 1
            beta, beta_sigma = 0.5, 0.5  # kwargs['MoCo_beta'], kwargs['MoCo_beta_sigma']
            gamma, gamma_sigma = 0.1, 0.5  # kwargs['MoCo_gamma'], kwargs['MoCo_gamma_sigma']
            rho = 0  # kwargs['MoCo_rho']

            self._compute_grad_dim()
            grads = self._compute_grad(loss, mode="backward").to("cuda")
            with torch.no_grad():
                for tn, task in enumerate(self.task_name):
                    grads[tn] = grads[tn] / (grads[tn].norm() + 1e-6) * loss[task]
            self.y = self.y - (beta / self.step**beta_sigma) * (self.y - grads)
            self.lambd = F.softmax(
                self.lambd
                - (gamma / self.step**gamma_sigma)
                * (
                    self.y @ self.y.t()
                    + rho * torch.eye(self.task_num, dtype=torch.float32).to("cuda")
                )
                @ self.lambd,
                -1,
            )
            self.new_grads = self.y.t() @ self.lambd
            self._reset_grad(self.new_grads.to(grads.dtype))
            # record alpha for weight logging
            alpha = self.lambd

        elif self.weighting == "DB_MTL":
            self.step += 1
            beta = 0.9  # kwargs['DB_beta']
            beta_sigma = 0  # kwargs['DB_beta_sigma']

            self._compute_grad_dim()
            log_loss = {k: torch.log(v + 1e-8) for k, v in loss.items()}
            grads = self._compute_grad(log_loss, mode="backward").to("cuda")
            # [task_num, grad_dim]

            self.grad_buffer = grads + (beta / self.step**beta_sigma) * (self.grad_buffer - grads)

            u_grad = self.grad_buffer.norm(dim=-1)
            alpha = u_grad.max() / (u_grad + 1e-8)  # record alpha for weight logging
            self.new_grads = [alpha[i] * self.grad_buffer[i] for i in range(self.task_num)]
            self._reset_grad(sum(self.new_grads).to(grads.dtype))

        elif self.weighting == "PCGrad":
            batch_weight = np.ones(len(loss))
            self._compute_grad_dim()
            grads = self._compute_grad(loss, mode="backward").to("cuda")  # [task_num, grad_dim]
            pc_grads = grads.clone()
            for tn_i in range(self.task_num):
                task_index = torch.randperm(self.task_num, device="cuda")
                for tn_j in task_index:
                    g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                    if g_ij < 0:
                        pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2) + 1e-8)
                        batch_weight[tn_j] -= (g_ij / (grads[tn_j].norm().pow(2) + 1e-8)).item()
            self.new_grads = pc_grads
            self._reset_grad(self.new_grads.sum(0))
            # record alpha for weight logging
            alpha = torch.Tensor(batch_weight)

        elif self.weighting == "CAGrad":
            calpha, rescale = 0.5, 1  # kwargs["calpha"], kwargs["rescale"]

            self._compute_grad_dim()
            grads = self._compute_grad(loss, mode="backward").to("cuda")

            GG = torch.matmul(grads, grads.t()).cpu()  # [num_tasks, num_tasks]
            g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

            x_start = np.ones(self.task_num) / self.task_num
            bnds = tuple((0, 1) for x in x_start)
            cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
            A = GG.to(torch.float32).numpy()
            b = x_start.copy()
            c = (calpha * g0_norm + 1e-8).item()

            def objfn(x):
                return (
                    x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
                    + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)
                ).sum()

            res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
            w_cpu = res.x
            ww = torch.Tensor(w_cpu).to(self.device)
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
            self.new_grads = new_grads
            self._reset_grad(new_grads)
            # record alpha for weight logging
            alpha = ww

        elif self.weighting == "GradVac":
            beta = 0.5  # kwargs['GradVac_beta']
            group_type = 0  # kwargs['GradVac_group_type']
            if self.step == 0:
                self._init_rho(group_type)

            self._compute_grad_dim()
            grads = self._compute_grad(loss, mode="backward").to("cuda")  # [task_num, grad_dim]

            batch_weight = np.ones(len(loss))
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
                        self.rho_T[tn_i, tn_j, k] = (1 - beta) * self.rho_T[
                            tn_i, tn_j, k
                        ] + beta * rho_ijk
            self.new_grads = pc_grads
            self._reset_grad(self.new_grads.sum(0))
            self.step += 1
            # record alpha for weight logging
            alpha = torch.Tensor(batch_weight)
        else:
            loss["loss"].backward()
        alpha = alpha.cpu() if alpha.device == torch.device("cuda") else alpha
        self.alpha = {task: alpha[tn] for tn, task in enumerate(self.task_name)}

    def test_step(self, batch):
        inputs, pad_masks, _ = batch
        batch = (inputs, pad_masks, None)
        return self.shared_step(batch, evaluation=True)[0]

    def configure_optimizers(self):
        if self.muP:
            from mup import MuAdamW as AdamW
        else:
            from torch.optim import AdamW
        opt = AdamW(
            self.parameters(),
            lr=self.lrs_config["initial"],
            weight_decay=self.lrs_config.get("weight_decay", 1e-5),
        )

        # 1cycle
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lrs_config["max"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
            final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
            pct_start=float(self.lrs_config["pct_start"]),
            last_epoch=int(self.lrs_config.get("last_epoch", -1)),
        )
        sch = {"scheduler": sch, "interval": "step"}

        return [opt], [sch]

    @property
    def input_dims(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.norm.variables.items()}

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

    # IMTL ####
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

    # GRADVAC ####
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

    def _get_grad_cos_sim(self, grad1, grad2):
        """Computes cosine similarity of gradients after flattening of tensors."""
        cosine = self.cos_sim(grad1, grad2)
        return torch.clamp(cosine, -1, 1)

    def compute_pairwise_cossim(self, grads):
        self.cos_sims = []
        self.task_pairs = []

        for a, task_a in enumerate(self.task_name):
            for b, task_b in enumerate(self.task_name[a + 1 :], start=a + 1):
                cos_sim = self._get_grad_cos_sim(grads[a], grads[b])
                self.cos_sims.append(cos_sim.item())
                self.task_pairs.append((task_a, task_b))
