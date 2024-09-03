import math
import warnings
from collections.abc import Mapping

import lightning as L
import torch
from torch import nn

from salt.models import InputNorm
from salt.models.weighting import Static, Weighting


def check_unique(modules: nn.ModuleList, attr_name: str) -> None:
    assert len({getattr(m, attr_name) for m in modules}) == len(
        modules
    ), f"Attribute '{attr_name}' must be unique for class {modules[0].__class__.__name__}"


class ModelWrapper(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lrs_config: Mapping[str, float],
        global_object: str,
        norm_config: dict | None = None,
        name: str = "salt",
        muP_config: dict | None = None,
        loss_weighting: Weighting | None = None,
        loss_mode: str = "wsum",
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
        loss_mode: str, optional
            The loss mode to use. Default is "wsum" (weighted sum).
            Other options are
            - 'GLS' : arxiv.org/1904.08492
        loss_weighting: str, optional
            The loss weighting to use. Default is "Static", which requires setting manual weights.
            Other options are
            - DWA, UW, DWA, RLW, NashMTL, IMTL, AlignedMTL, DBMTL, MoCo, PCGrad, CAGrad, GradVac
            Refer to salt.models.weighting for more information on each strategy
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
        allowed_loss_modes = ["wsum", "GLS"]
        assert loss_mode in allowed_loss_modes, f"Loss mode must be one of {allowed_loss_modes}"
        self.loss_mode = loss_mode
        if loss_mode == "GLS":
            assert all(
                task.weight == 1.0 for task in self.model.tasks
            ), "GLS does not utilise task weights - remove all/set to 1"

        self.weighting = loss_weighting if loss_weighting else Static()
        self.weighting.set_model(self.model)
        # self.calc_cos_cim = self.weighting.calc_cos_sim # override to True if desired for auto opt
        self.calc_cos_sim = False
        self.automatic_optimization = self.weighting.auto_opt
        self.task_names = self.weighting.task_names

    def on_fit_start(self):
        self.weighting.on_fit_start(trainer=self.trainer)

    def on_train_start(self):
        self.weighting.on_train_start()

    def on_train_epoch_start(self):
        self.weighting.on_train_epoch_start()

    def on_train_epoch_end(self):
        self.weighting.on_train_epoch_end()

    def manual_backward(self, loss):
        self.weighting.manual_backward(loss)

    def total_loss(self, loss: dict):
        """Computes the final loss based on the loss mode."""
        if self.loss_mode == "GLS":
            # Calculate the geometric mean of the losses
            loss_prod = math.prod(subloss for subloss in loss.values())
            return torch.pow(loss_prod, 1.0 / len(loss))

        # Return the default weighted sum
        return sum(subloss for subloss in loss.values())

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

        # compute total loss
        # loss["loss"] = self.total_loss(loss) ## i weight them in the train step

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
            grads = {task: grads[tn] for tn, task in enumerate(self.task_names)}
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
        log_loss["loss"] = sum(subloss for subloss in log_loss.values())
        self.log_losses(log_loss, stage="train")

        # weight and combine losses
        loss = self.weighting.weight_loss(loss)
        loss["loss"] = self.weighting.total_loss(loss)
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
            # log weights (calculated in weighting class)
            self.log_weights(self.weighting.loss_weights, stage="train")
            if self.calc_cos_sim:
                # log explicitly calculated gradients
                self.grads = self.weighting.compute_grad(loss, mode="autograd").to("cuda")
                self.log_grads(self.grads)

                # log cos similarities
                self.weighting.compute_pairwise_cossim(self.grads)
                self.log_cossim(self.weighting.task_pairs, self.weighting.cos_sims)
        else:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            self.lr_schedulers().step()

            # log weights (gradients or loss weights for NashMTL, calculated in weighting class)
            self.log_weights(self.weighting.alpha, stage="train")

            if self.calc_cos_sim:
                # log transformed gradients from weighting methods (calculated in weighting class)
                self.log_grads(self.weighting.new_grads)
                # log cos similarities
                self.weighting.compute_pairwise_cossim(self.weighting.new_grads)
                self.log_cossim(self.weighting.task_pairs, self.weighting.cos_sims)

        return {**loss, "outputs": outputs}

    def validation_step(self, batch):
        # foward pass
        preds, labels, pad_masks, loss = self.shared_step(batch)

        # log raw losses
        log_loss = loss.copy()
        log_loss["loss"] = self.total_loss(log_loss)
        self.log_losses(log_loss, stage="val")

        # weight and combine losses
        loss = self.weighting.weight_loss(loss)
        loss["loss"] = self.total_loss(loss)

        # Store outputs to be used by the MaskformerMetrics callback
        outputs = {
            "preds": preds,
            "labels": labels,
            "pad_masks": pad_masks,
        }

        return {**loss, "outputs": outputs}

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
