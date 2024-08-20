import warnings
from collections.abc import Mapping

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from salt.models import InputNorm


def check_unique(modules: nn.ModuleList, attr_name: str) -> None:
    assert len({getattr(m, attr_name) for m in modules}) == len(
        modules
    ), f"Attribute '{attr_name}' must be unique for class {modules[0].__class__.__name__}"


class ModelWrapperIman(L.LightningModule):
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

        self.init_param()
        self.rep_grad = False
        # if self.rep_grad:
        #     self.rep_tasks = {}  # type: dict[str, torch.Tensor]
        #     self.rep = {}  # type: dict[str, torch.Tensor]

    def init_param(self):
        r"""Define and initialize trainable parameters required by specific weighting methods."""
        if self.weighting == "UW":
            self.loss_scale = torch.nn.Parameter(torch.tensor([-0.5] * self.task_num))
        if self.weighting == "DWA":
            self.avg_losses = {}  # type: dict[str, torch.Tensor]
            self.batch_losses = {}  # type: dict[str, list[float]]
        elif self.weighting == "IMTL":
            self.loss_scale = nn.Parameter(torch.tensor([0.0] * self.task_num))
            self.automatic_optimization = False

    def on_fit_start(self):
        if self.weighting == "DWA":
            self.train_loss_buffer = torch.zeros([6, self.trainer.max_epochs])
        elif not self.automatic_optimization:
            assert self.trainer.precision in {
                "32-true",
                "bf16-true",
                "bf16-mixed",
                "bf16",
            }, "IMTL requires 32-bit or bfloat16 precision."
            if self.trainer.precision != "32-true":
                torch.set_default_dtype(torch.bfloat16)

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

        elif self.weighting == "RLW":
            weights = F.softmax(torch.randn(self.task_num), dim=-1)
            # Apply weights to each loss
            for i, k in enumerate(list(losses.keys())):
                losses[k] *= weights[i]

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

        elif self.weighting == "UW":
            weighted_losses = {
                key: value / (2 * self.loss_scale[i].exp()) + self.loss_scale[i] / 2
                for i, (key, value) in enumerate(losses.items())
            }
            losses = weighted_losses

        elif self.weighting == "IMTL":
            losses = {
                key: self.loss_scale[i].exp() * value - self.loss_scale[i]
                for i, (key, value) in enumerate(losses.items())
            }
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

    def training_step(self, batch):
        # forward pass
        preds, labels, pad_masks, loss = self.shared_step(batch)

        # log raw losses
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
        # print("train step loss", {**loss})
        if self.weighting == "IMTL":
            opt = self.optimizers()  # Access the optimizer
            opt.zero_grad()  # Clear previous gradients
            loss = {key: value.to(torch.bfloat16) for key, value in loss.items()}
            # print("train step loss", {**loss}, loss["loss"].dtype)
            self.manual_backward(loss)  # Manually perform backward pass
            opt.step()  # Update model parameters
        return {**loss, "outputs": outputs}

    def validation_step(self, batch):
        # if self.weighting == "IMTL":
        #     self.eval()
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
        if self.weighting == "IMTL":
            loss = {key: value.to(torch.bfloat16) for key, value in loss.items()}
        # print("val step loss", {**loss})
        return {**loss, "outputs": outputs}

    def manual_backward(self, loss):
        if self.weighting == "IMTL":
            grads = self._get_grads(loss, mode="backward").to("cuda")
            # print(grads, grads.dtype, grads.device)
            grads_unit = grads / torch.norm(grads, p=2, dim=-1, keepdim=True)

            D = grads[0:1].repeat(self.task_num - 1, 1) - grads[1:]
            U = grads_unit[0:1].repeat(self.task_num - 1, 1) - grads_unit[1:]
            if self.trainer.precision != "32-true":
                # Convert to fp32 for the matrix operations
                D_fp32 = D.to(torch.float32)
                U_fp32 = U.to(torch.float32)
                grads_0_fp32 = grads[0].to(torch.float32)

                # Perform matrix operations in fp32
                DU_t = torch.matmul(D_fp32, U_fp32.t())
                DU_t_inv = torch.linalg.inv(DU_t)  # Using linalg.inv instead of inverse
                alpha_fp32 = torch.matmul(torch.matmul(grads_0_fp32, U_fp32.t()), DU_t_inv)

                # Convert the result back to the original dtype (presumably bf16)
                alpha = alpha_fp32.to(grads.dtype)
            else:
                alpha = torch.matmul(
                    torch.matmul(grads[0], U.t()), torch.linalg.inv(torch.matmul(D, U.t()))
                )
            alpha = torch.cat((1 - alpha.sum().unsqueeze(0), alpha), dim=0)
            if self.rep_grad:
                # self._backward_new_grads(alpha, per_grads=per_grads)
                pass
            else:
                self._backward_new_grads(alpha, grads=grads)
                # print(grads, grads.dtype, grads.device)
        else:
            loss["loss"].backward()

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
                            losses[task], self.get_share_params(), retain_graph=True
                        )
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

    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        return rep
