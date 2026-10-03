from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
from lightning.pytorch import LightningDataModule
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from anndata import AnnData
from annbatch import DatasetCollection, Loader
from scipy import sparse as sp_sparse
from scipy.stats import rankdata
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as kl

import scvi
from scvi import REGISTRY_KEYS
from scvi._types import MinifiedDataType
from scvi.data import AnnDataManager
from scvi.data._constants import ADATA_MINIFY_TYPE, _ADATA_MINIFY_TYPE_UNS_KEY, _SETUP_ARGS_KEY
from scvi.data._utils import _get_adata_minify_type, get_anndata_attribute
from scvi.data.fields import (
    BaseAnnDataField,
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
    StringUnsField,
)
from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.model._scanvi import (
    _SCANVI_LATENT_QZM,
    _SCANVI_LATENT_QZV,
    _SCANVI_OBSERVED_LIB_SIZE,
)
from scvi.model._scvi import SCVI
from scvi.model._utils import _init_library_size, get_max_epochs_heuristic
from scvi.model.base import ArchesMixin, BaseMinifiedModeModelClass, RNASeqMixin, VAEMixin
from scvi.module import SCANVAE
from scvi.module.base import LossOutput, auto_move_data
from scvi.train import SemiSupervisedTrainingPlan, TrainRunner
from scvi.train._callbacks import SubSampleLabels
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp


HIERARCHY_CLASS_KEY = "hierarchy_class"
HIERARCHY_SUBCLASS_KEY = "hierarchy_subclass"


@dataclass
class TrainingPhase:
    name: str
    max_epochs: int
    lambda_sparse_start: float
    lambda_sparse_end: float
    tau_start: float
    tau_end: float
    classification_ratio: float | None = None
    freeze_encoder: bool = False
    freeze_gene_gate: bool = False


class MLPClassifier(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 256,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.logits = True
        layers: list[nn.Module] = []

        if n_layers <= 0:
            layers.append(nn.Linear(n_input, n_output))
        else:
            in_features = n_input
            for _ in range(n_layers):
                layers.append(nn.Linear(in_features, n_hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                in_features = n_hidden
            layers.append(nn.Linear(in_features, n_output))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GeneGatingLayer(nn.Module):
    def __init__(
        self,
        n_genes: int,
        init_prob: float = 0.5,
        tau: float = 1.5,
        min_tau: float = 0.1,
    ) -> None:
        super().__init__()
        init_prob = float(np.clip(init_prob, 1e-4, 1.0 - 1e-4))
        init_logit = np.log(init_prob / (1.0 - init_prob))
        self.logits = nn.Parameter(torch.full((n_genes,), init_logit, dtype=torch.float32))
        self.register_buffer("tau", torch.tensor(float(tau), dtype=torch.float32))
        self.min_tau = float(min_tau)

    def set_temperature(self, tau: float) -> None:
        self.tau.fill_(max(float(tau), self.min_tau))

    def gate(self) -> torch.Tensor:
        temperature = torch.clamp(self.tau, min=self.min_tau)
        scaled_logits = torch.clamp(self.logits / temperature, min=-20.0, max=20.0)
        return torch.sigmoid(scaled_logits)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.gate()
        return x * mask, mask


class SparseHierarchicalSCANVAE(SCANVAE):
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_subclass_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 64,
        n_layers: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: str = "gene-batch",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        linear_classifier: bool = False,
        classifier_hidden: int = 128,
        classifier_layers: int = 1,
        gate_init_prob: float = 0.5,
        initial_tau: float = 1.5,
        min_tau: float = 0.1,
        lambda_sparse: float = 0.0,
        lambda_binary: float = 1e-2,
        lambda_panel_size: float = 1.0,
        target_panel_size: int | None = None,
        gamma_redundancy: float = 0.0,
        redundancy_top_k: int = 256,
        redundancy_eps: float = 1e-4,
        cluster_separation_weight: float = 1.0,
        use_batch_norm: str = "none",
        use_layer_norm: str = "both",
        **vae_kwargs: Any,
    ) -> None:
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            linear_classifier=linear_classifier,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **vae_kwargs,
        )
        cluster_layers = 0 if linear_classifier else classifier_layers
        cluster_hidden = n_latent if linear_classifier else classifier_hidden

        self.gene_gate = GeneGatingLayer(
            n_genes=n_input,
            init_prob=gate_init_prob,
            tau=initial_tau,
            min_tau=min_tau,
        )
        self.register_buffer("eligible_gene_mask", torch.ones(n_input, dtype=torch.float32))
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_binary = float(lambda_binary)
        self.lambda_panel_size = float(lambda_panel_size)
        self.target_panel_size = None if target_panel_size is None else int(target_panel_size)
        self.gamma_redundancy = float(gamma_redundancy)
        self.redundancy_top_k = int(redundancy_top_k)
        self.redundancy_eps = float(redundancy_eps)
        self.cluster_separation_weight = float(cluster_separation_weight)

        self.cluster_classifier = MLPClassifier(
            n_input=n_latent,
            n_output=n_labels,
            n_hidden=cluster_hidden,
            n_layers=cluster_layers,
            dropout_rate=dropout_rate,
        )
        self.classifier = self.cluster_classifier
        self.subclass_head = MLPClassifier(
            n_input=n_latent,
            n_output=n_subclass_labels,
            n_hidden=classifier_hidden,
            n_layers=classifier_layers,
            dropout_rate=dropout_rate,
        )

    def set_temperature(self, tau: float) -> None:
        self.gene_gate.set_temperature(tau)

    def set_sparsity_weight(self, lambda_sparse: float) -> None:
        self.lambda_sparse = float(lambda_sparse)

    def freeze_encoder(self, freeze: bool = True) -> None:
        encoder_modules = [self.z_encoder, self.l_encoder, self.encoder_z2_z1]
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = not freeze

    def freeze_gene_gate(self, freeze: bool = True) -> None:
        self.gene_gate.logits.requires_grad = not freeze

    def set_eligible_gene_mask(self, mask: torch.Tensor) -> None:
        self.eligible_gene_mask.copy_(mask.float())

    def get_gene_mask(self, detached: bool = True) -> torch.Tensor:
        mask = self.gene_gate.gate() * self.eligible_gene_mask
        return mask.detach() if detached else mask

    def _apply_gene_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated_x, mask = self.gene_gate(x)
        mask = mask * self.eligible_gene_mask
        return x * mask, mask

    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, Any]:
        x_gated, mask = self._apply_gene_gate(x)
        encoder_x = x_gated

        if self.use_observed_lib_size:
            library = torch.log(x.sum(1).clamp_min(1.0)).unsqueeze(1)
        if self.log_variational:
            encoder_x = torch.log1p(encoder_x)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((encoder_x, cont_covs), dim=-1)
        else:
            encoder_input = encoder_x

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))
            else:
                library = ql.sample((n_samples,))

        return {
            "z": z,
            "qz": qz,
            "ql": ql,
            "library": library,
            "gene_mask": mask,
            "x_gated": x_gated,
        }

    def _hierarchy_logits_from_latent(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "subclass": self.subclass_head(z),
            "cluster": self.cluster_classifier(z),
        }

    @auto_move_data
    def classify(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
    ) -> torch.Tensor:
        logits = self.classify_hierarchy(
            x=x,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            use_posterior_mean=use_posterior_mean,
        )
        return logits["cluster"]

    @auto_move_data
    def classify_hierarchy(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
    ) -> dict[str, torch.Tensor]:
        if batch_index is None:
            batch_index = torch.zeros(x.size(0), 1, dtype=torch.int64, device=x.device)

        inference_outputs = self._regular_inference(
            x=x,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        qz = inference_outputs["qz"]
        z = qz.loc if use_posterior_mean else inference_outputs["z"]
        return self._hierarchy_logits_from_latent(z)

    @auto_move_data
    def forward_hierarchy(
        self,
        tensors: dict[str, torch.Tensor],
        use_posterior_mean: bool = True,
    ) -> dict[str, Any]:
        inference_outputs, generative_outputs = self.forward(tensors, compute_loss=False)
        qz = inference_outputs["qz"]
        z = qz.loc if use_posterior_mean and qz is not None else inference_outputs["z"]
        return {
            "reconstruction": generative_outputs["px"],
            "latent_z": z,
            "predictions": self._hierarchy_logits_from_latent(z),
            "gene_mask": inference_outputs["gene_mask"],
        }

    def _hierarchical_classification_metrics(
        self,
        labelled_tensors: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x = labelled_tensors[REGISTRY_KEYS.X_KEY]
        batch_index = labelled_tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_covs = labelled_tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = labelled_tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)

        logits = self.classify_hierarchy(
            x=x,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            use_posterior_mean=True,
        )
        labels = {
            "subclass": labelled_tensors[HIERARCHY_SUBCLASS_KEY].view(-1).long(),
            "cluster": labelled_tensors[REGISTRY_KEYS.LABELS_KEY].view(-1).long(),
        }

        losses = {
            key: F.cross_entropy(logits[key], labels[key]) for key in ("subclass", "cluster")
        }
        accuracies = {
            f"{key}_accuracy": (torch.argmax(logits[key], dim=-1) == labels[key]).float().mean()
            for key in ("subclass", "cluster")
        }
        return losses, logits, {**labels, **accuracies}

    def _broadcast_cluster_labels(
        self,
        y: torch.Tensor | None,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if y is None:
            labels = torch.arange(self.n_labels, device=z.device, dtype=torch.long)
            labels = labels.repeat_interleave(z.size(0))
            ys = F.one_hot(labels, num_classes=self.n_labels).float()
            z_broadcast = z.repeat(self.n_labels, 1)
            return ys, z_broadcast

        y = y.view(-1).long()
        ys = F.one_hot(y, num_classes=self.n_labels).float()
        return ys, z

    def _redundancy_penalty(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        k = min(self.redundancy_top_k, x.size(1))
        if self.gamma_redundancy <= 0.0 or k < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        _, top_idx = torch.topk(mask, k=k, largest=True, sorted=False)
        x_sub = x[:, top_idx]
        weights = mask[top_idx]

        x_sub = torch.log1p(x_sub)
        x_sub = x_sub - x_sub.mean(dim=0, keepdim=True)
        std = x_sub.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.redundancy_eps)
        x_sub = x_sub / std

        corr = (x_sub.transpose(0, 1) @ x_sub) / max(x_sub.size(0), 1)
        off_diag = corr - torch.diag_embed(torch.diagonal(corr))
        weight_matrix = torch.outer(weights, weights)
        normalizer = weight_matrix.sum().clamp_min(self.redundancy_eps)
        return ((off_diag.square() * weight_matrix).sum()) / normalizer

    def _cluster_mean_separation_penalty(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.cluster_separation_weight <= 0.0 or z.ndim != 2:
            return torch.zeros((), device=z.device, dtype=z.dtype)

        labels = labels.view(-1).long()
        valid = labels < self.n_labels
        if valid.sum() < 2:
            return torch.zeros((), device=z.device, dtype=z.dtype)

        z_valid = z[valid]
        labels_valid = labels[valid]
        unique_labels, inverse = torch.unique(labels_valid, sorted=False, return_inverse=True)
        if unique_labels.numel() < 2:
            return torch.zeros((), device=z.device, dtype=z.dtype)

        n_groups = unique_labels.numel()
        centroids = torch.zeros((n_groups, z_valid.size(1)), device=z.device, dtype=z.dtype)
        centroids.index_add_(0, inverse, z_valid)
        counts = torch.bincount(inverse, minlength=n_groups).to(z.dtype).unsqueeze(1).clamp_min(1.0)
        centroids = centroids / counts

        pairwise_sqdist = torch.cdist(centroids, centroids, p=2).pow(2)
        off_diag = ~torch.eye(n_groups, device=z.device, dtype=torch.bool)
        if off_diag.sum() == 0:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        # Penalize similar cluster centroids; this decreases as centroids separate.
        return torch.exp(-pairwise_sqdist[off_diag]).mean()

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, Any],
        generative_outputs: dict[str, Any],
        feed_labels: bool = False,
        kl_weight: float = 1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float | None = None,
    ) -> LossOutput:
        px = generative_outputs["px"]
        qz1 = inference_outputs["qz"]
        z1 = inference_outputs["z"]
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        gene_mask = inference_outputs["gene_mask"]

        observed_y = tensors[REGISTRY_KEYS.LABELS_KEY].view(-1, 1)
        has_only_labeled_cells = bool(torch.all(observed_y.view(-1) < self.n_labels).item())
        if feed_labels or has_only_labeled_cells:
            y = observed_y
        else:
            y = None
        is_labelled = y is not None
        classification_ratio = 1.0 if classification_ratio is None else float(classification_ratio)

        ys, z1s = self._broadcast_cluster_labels(y, z1)
        qz2, z2 = self.encoder_z2_z1(z1s, ys)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)
        reconst_loss = -px.log_prob(x).sum(-1)

        mean = torch.zeros_like(qz2.loc)
        scale = torch.ones_like(qz2.scale)
        kl_divergence_z2 = kl(qz2, Normal(mean, scale)).sum(dim=1)
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = qz1.log_prob(z1).sum(dim=-1)

        if not self.use_observed_lib_size:
            ql = inference_outputs["ql"]
            library_log_means, library_log_vars = self._compute_local_library_params(batch_index)
            kl_divergence_l = kl(
                ql,
                Normal(library_log_means, torch.sqrt(library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(reconst_loss)

        if is_labelled:
            base_per_cell = (
                loss_z1_weight
                + loss_z1_unweight
                + kl_divergence_z2
                + kl_divergence_l
            )
        else:
            probs = self.classifier(z1)
            if self.classifier.logits:
                probs = F.softmax(probs, dim=-1)

            base_per_cell = loss_z1_weight
            base_per_cell = base_per_cell + (
                loss_z1_unweight.view(self.n_labels, -1).t() * probs
            ).sum(dim=1)
            kl_divergence = (kl_divergence_z2.view(self.n_labels, -1).t() * probs).sum(dim=1)
            kl_divergence = kl_divergence + kl(
                Categorical(probs=probs),
                Categorical(probs=self.y_prior.repeat(probs.size(0), 1)),
            )
            kl_divergence = kl_divergence + kl_divergence_l
            base_per_cell = base_per_cell + kl_weight * kl_divergence

        sparsity_loss_raw = gene_mask.abs().sum()
        if self.target_panel_size is None:
            sparsity_loss = sparsity_loss_raw
            target_panel_size = None
        else:
            target_panel_size = torch.tensor(
                float(self.target_panel_size),
                device=x.device,
                dtype=x.dtype,
            ).clamp_min(1.0)
            sparsity_loss = sparsity_loss_raw / target_panel_size
        binary_penalty = (gene_mask * (1.0 - gene_mask)).mean()
        if target_panel_size is None:
            panel_size_penalty = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            active_mass_ratio = gene_mask.sum() / target_panel_size
            under_target = F.relu(1.0 - active_mass_ratio).pow(2)
            over_target = F.relu(active_mass_ratio - 1.0).pow(2)
            panel_size_penalty = 4.0 * under_target + over_target
        redundancy_penalty = self._redundancy_penalty(x, gene_mask)
        cluster_separation_penalty = self._cluster_mean_separation_penalty(z1, observed_y)

        extra_metrics: dict[str, torch.Tensor] = {
            "gene_mask_l1": sparsity_loss_raw.detach(),
            "gene_mask_l1_normalized": sparsity_loss.detach(),
            "gene_mask_mean": gene_mask.mean().detach(),
            "selected_genes_over_0p5": (gene_mask > 0.5).float().sum().detach(),
            "tau": self.gene_gate.tau.detach().clone(),
            "lambda_sparse": torch.tensor(self.lambda_sparse, device=x.device),
            "lambda_binary": torch.tensor(self.lambda_binary, device=x.device),
            "lambda_panel_size": torch.tensor(self.lambda_panel_size, device=x.device),
            "binary_penalty": binary_penalty.detach(),
            "panel_size_penalty": panel_size_penalty.detach(),
            "redundancy_penalty": redundancy_penalty.detach(),
            "cluster_separation_penalty": cluster_separation_penalty.detach(),
            "cluster_separation_weight": torch.tensor(self.cluster_separation_weight, device=x.device),
        }

        total_loss = base_per_cell.mean()
        classification_loss = None
        logits = None
        true_labels = None

        if labelled_tensors is not None:
            hierarchy_losses, hierarchy_logits, hierarchy_metrics = self._hierarchical_classification_metrics(
                labelled_tensors
            )
            classification_total = sum(hierarchy_losses.values())
            total_loss = total_loss + classification_ratio * classification_total

            classification_loss = hierarchy_losses["cluster"]
            logits = hierarchy_logits["cluster"]
            true_labels = hierarchy_metrics["cluster"]
            extra_metrics.update(
                {
                    "classification_loss_subclass": hierarchy_losses["subclass"].detach(),
                    "classification_loss_cluster": hierarchy_losses["cluster"].detach(),
                    "accuracy_subclass": hierarchy_metrics["subclass_accuracy"].detach(),
                    "accuracy_cluster": hierarchy_metrics["cluster_accuracy"].detach(),
                }
            )

        total_loss = (
            total_loss
            + self.lambda_sparse * sparsity_loss
            + self.lambda_binary * binary_penalty
            + self.lambda_panel_size * panel_size_penalty
            + self.gamma_redundancy * redundancy_penalty
            + self.cluster_separation_weight * cluster_separation_penalty
        )

        kl_local = {
            "kl_divergence_z2": kl_divergence_z2,
            "kl_divergence_l": kl_divergence_l,
        }
        return LossOutput(
            loss=total_loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            classification_loss=classification_loss,
            logits=logits,
            true_labels=true_labels,
            extra_metrics=extra_metrics,
        )


class HierarchicalSemiSupervisedTrainingPlan(SemiSupervisedTrainingPlan):
    def __init__(
        self,
        module: SparseHierarchicalSCANVAE,
        n_classes: int,
        *,
        lambda_sparse_start: float = 0.0,
        lambda_sparse_end: float = 0.0,
        tau_start: float = 1.5,
        tau_end: float = 1.5,
        freeze_encoder: bool = False,
        freeze_gene_gate: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(module=module, n_classes=n_classes, **kwargs)
        self.lambda_sparse_start = float(lambda_sparse_start)
        self.lambda_sparse_end = float(lambda_sparse_end)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.freeze_encoder_flag = bool(freeze_encoder)
        self.freeze_gene_gate_flag = bool(freeze_gene_gate)
        self._current_log_batch_size: int | None = None

    @staticmethod
    def _infer_batch_size_from_batch(batch: Any) -> int | None:
        if isinstance(batch, torch.Tensor) and batch.ndim > 0:
            return int(batch.shape[0])
        if isinstance(batch, dict):
            x = batch.get(REGISTRY_KEYS.X_KEY, batch.get("X"))
            if isinstance(x, torch.Tensor) and x.ndim > 0:
                return int(x.shape[0])
            for value in batch.values():
                inferred = HierarchicalSemiSupervisedTrainingPlan._infer_batch_size_from_batch(value)
                if inferred is not None:
                    return inferred
            return None
        if isinstance(batch, (list, tuple)):
            for value in batch:
                inferred = HierarchicalSemiSupervisedTrainingPlan._infer_batch_size_from_batch(value)
                if inferred is not None:
                    return inferred
        return None

    def training_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> Any:
        self._current_log_batch_size = self._infer_batch_size_from_batch(batch)
        try:
            return super().training_step(batch, batch_idx, *args, **kwargs)
        finally:
            self._current_log_batch_size = None

    def validation_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> Any:
        self._current_log_batch_size = self._infer_batch_size_from_batch(batch)
        try:
            return super().validation_step(batch, batch_idx, *args, **kwargs)
        finally:
            self._current_log_batch_size = None

    def log(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("batch_size") is None and self._current_log_batch_size is not None:
            kwargs["batch_size"] = self._current_log_batch_size
        super().log(name, value, *args, **kwargs)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.module.train()
        self.module.freeze_encoder(self.freeze_encoder_flag)
        self.module.freeze_gene_gate(self.freeze_gene_gate_flag)
        if self.freeze_encoder_flag:
            for submodule in (self.module.z_encoder, self.module.l_encoder, self.module.encoder_z2_z1):
                submodule.eval()

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.module.train()
        if self.freeze_encoder_flag:
            for submodule in (self.module.z_encoder, self.module.l_encoder, self.module.encoder_z2_z1):
                submodule.eval()
        max_epochs = max(int(self.trainer.max_epochs), 1)
        if max_epochs == 1:
            progress = 1.0
        else:
            progress = float(self.current_epoch) / float(max_epochs - 1)
        lambda_sparse = self.lambda_sparse_start + progress * (
            self.lambda_sparse_end - self.lambda_sparse_start
        )
        tau = self.tau_start + progress * (self.tau_end - self.tau_start)
        self.module.set_sparsity_weight(lambda_sparse)
        self.module.set_temperature(tau)


class HierarchicalSCANVI(RNASeqMixin, VAEMixin, ArchesMixin, BaseMinifiedModeModelClass):
    _module_cls = SparseHierarchicalSCANVAE
    _training_plan_cls = HierarchicalSemiSupervisedTrainingPlan

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 16,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        gene_likelihood: str = "zinb",
        linear_classifier: bool = False,
        classifier_hidden: int = 128,
        classifier_layers: int = 1,
        gate_init_prob: float = 0.5,
        initial_tau: float = 1.5,
        min_tau: float = 0.1,
        lambda_sparse: float = 0.0,
        lambda_binary: float = 1e-2,
        lambda_panel_size: float = 1.0,
        target_panel_size: int | None = None,
        gamma_redundancy: float = 0.0,
        redundancy_top_k: int = 256,
        cluster_separation_weight: float = 1.0,
        class_key: str = "class",
        subclass_key: str = "subclass",
        cluster_key: str = "cluster",
        **model_kwargs: Any,
    ) -> None:
        super().__init__(adata)
        self.class_key = class_key
        self.subclass_key = subclass_key
        self.cluster_key = cluster_key

        self._set_indices_and_labels()

        n_cluster_labels = self.summary_stats.n_labels - 1
        subclass_registry = self.adata_manager.get_state_registry(HIERARCHY_SUBCLASS_KEY)
        n_subclass_labels = len(subclass_registry.categorical_mapping)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        precomputed_library_log_means = model_kwargs.pop("library_log_means", None)
        precomputed_library_log_vars = model_kwargs.pop("library_log_vars", None)
        library_log_means, library_log_vars = (
            precomputed_library_log_means,
            precomputed_library_log_vars,
        )
        if (
            (library_log_means is None or library_log_vars is None)
            and not use_size_factor_key
            and self.minified_data_type is None
        ):
            try:
                library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)
            except TypeError as exc:
                if "keepdims" not in str(exc):
                    raise
                # Lazy dask+sparse matrices can fail inside _init_library_size meta reductions.
                # Fall back to observed library size and let pretrained weights overwrite priors.
                model_kwargs.setdefault("use_observed_lib_size", True)
                library_log_means = np.zeros((1, n_batch), dtype=np.float32)
                library_log_vars = np.ones((1, n_batch), dtype=np.float32)

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=n_cluster_labels,
            n_subclass_labels=n_subclass_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            linear_classifier=linear_classifier,
            classifier_hidden=classifier_hidden,
            classifier_layers=classifier_layers,
            gate_init_prob=gate_init_prob,
            initial_tau=initial_tau,
            min_tau=min_tau,
            lambda_sparse=lambda_sparse,
            lambda_binary=lambda_binary,
            lambda_panel_size=lambda_panel_size,
            target_panel_size=target_panel_size,
            gamma_redundancy=gamma_redundancy,
            redundancy_top_k=redundancy_top_k,
            cluster_separation_weight=cluster_separation_weight,
            **model_kwargs,
        )
        self.module.minified_data_type = self.minified_data_type
        self.n_labels = n_cluster_labels
        self.n_subclass_labels = n_subclass_labels
        self._model_summary_string = (
            "Hierarchical sparse scANVI with "
            f"n_hidden={n_hidden}, n_latent={n_latent}, n_layers={n_layers}, "
            f"dropout_rate={dropout_rate}, dispersion={dispersion}, "
            f"gene_likelihood={gene_likelihood}, lambda_sparse={lambda_sparse}, "
            f"gamma_redundancy={gamma_redundancy}, "
            f"cluster_separation_weight={cluster_separation_weight}"
        )
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False

    @classmethod
    def from_scvi_model(
        cls,
        scvi_model: SCVI,
        unlabeled_category: str,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        class_key: str = "class",
        subclass_key: str = "subclass",
        cluster_key: str = "cluster",
        **scanvi_kwargs: Any,
    ) -> "HierarchicalSCANVI":
        scvi_model._check_if_trained(message="Passed in scvi model has not been trained yet.")
        if adata is None:
            adata = scvi_model.adata
        elif scvi_model.adata_manager is not None:
            scvi_model._validate_anndata(adata)

        if "dispersion" not in scanvi_kwargs:
            scanvi_kwargs["dispersion"] = scvi_model.module.dispersion
        if "gene_likelihood" not in scanvi_kwargs:
            scanvi_kwargs["gene_likelihood"] = scvi_model.module.gene_likelihood
        if "use_observed_lib_size" not in scanvi_kwargs and hasattr(
            scvi_model.module,
            "use_observed_lib_size",
        ):
            scanvi_kwargs["use_observed_lib_size"] = bool(scvi_model.module.use_observed_lib_size)
        if "library_log_means" not in scanvi_kwargs and hasattr(
            scvi_model.module,
            "library_log_means",
        ):
            scanvi_kwargs["library_log_means"] = (
                scvi_model.module.library_log_means.detach().cpu().numpy()
            )
        if "library_log_vars" not in scanvi_kwargs and hasattr(
            scvi_model.module,
            "library_log_vars",
        ):
            scanvi_kwargs["library_log_vars"] = scvi_model.module.library_log_vars.detach().cpu().numpy()
        labels_key = labels_key or cluster_key
        if scvi_model.adata_manager is not None:
            setup_args = scvi_model.adata_manager.registry[_SETUP_ARGS_KEY]
        else:
            setup_args = scvi_model.registry_[_SETUP_ARGS_KEY]
        cls.setup_anndata(
            adata,
            batch_key=setup_args.get("batch_key"),
            class_key=class_key,
            subclass_key=subclass_key,
            cluster_key=labels_key,
            unlabeled_category=unlabeled_category,
            layer=setup_args.get("layer"),
            size_factor_key=setup_args.get("size_factor_key"),
            categorical_covariate_keys=setup_args.get("categorical_covariate_keys"),
            continuous_covariate_keys=setup_args.get("continuous_covariate_keys"),
        )
        model = cls(
            adata=adata,
            class_key=class_key,
            subclass_key=subclass_key,
            cluster_key=labels_key,
            **scanvi_kwargs,
        )
        state_dict = scvi_model.module.state_dict()
        model.module.load_state_dict(state_dict, strict=False)
        model.was_pretrained = True
        return model

    def _set_indices_and_labels(self) -> None:
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self.unlabeled_category_ = labels_state_registry.unlabeled_category

        labels = get_anndata_attribute(
            self.adata,
            self.adata_manager.data_registry.labels.attr_name,
            self.original_label_key,
        ).ravel()
        self._label_mapping = labels_state_registry.categorical_mapping
        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category_).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category_).ravel()
        self._code_to_label = dict(enumerate(self._label_mapping))

    @torch.inference_mode()
    def predict_hierarchy(
        self,
        adata: AnnData | None = None,
        indices: list[int] | np.ndarray | None = None,
        batch_size: int | None = None,
        soft: bool = False,
        use_posterior_mean: bool = True,
    ) -> dict[str, np.ndarray | pd.DataFrame]:
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)

        loader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        outputs = {"subclass": [], "cluster": []}
        label_maps = {
            "subclass": list(
                self.adata_manager.get_state_registry(HIERARCHY_SUBCLASS_KEY).categorical_mapping
            ),
            "cluster": list(self._label_mapping[:-1]),
        }
        for tensors in loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch = tensors[REGISTRY_KEYS.BATCH_KEY]
            cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
            cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
            logits = self.module.classify_hierarchy(
                x=x,
                batch_index=batch,
                cont_covs=cont_covs,
                cat_covs=cat_covs,
                use_posterior_mean=use_posterior_mean,
            )
            for key, value in logits.items():
                outputs[key].append(value.detach().cpu())

        merged = {key: torch.cat(parts, dim=0).numpy() for key, parts in outputs.items()}
        if soft:
            return {
                key: pd.DataFrame(
                    F.softmax(torch.from_numpy(value), dim=-1).numpy(),
                    index=adata.obs_names[indices],
                    columns=label_maps[key],
                )
                for key, value in merged.items()
            }
        return {
            key: np.asarray(label_maps[key], dtype=object)[value.argmax(axis=1)]
            for key, value in merged.items()
        }

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        class_key: str = "class",
        subclass_key: str = "subclass",
        cluster_key: str = "cluster",
        unlabeled_category: str = "Unknown",
        layer: str | None = None,
        batch_key: str | None = "batch",
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register an AnnData object for hierarchical sparse scANVI training.

        Parameters
        ----------
        %(param_adata)s
        class_key
            Key in ``adata.obs`` containing top-level class labels.
        subclass_key
            Key in ``adata.obs`` containing subclass labels.
        cluster_key
            Key in ``adata.obs`` containing cluster labels used as the semi-supervised target.
        unlabeled_category
            Category in ``cluster_key`` that identifies unlabeled observations.
        %(param_layer)s
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        for key in (class_key, subclass_key, cluster_key):
            if key not in adata.obs:
                raise KeyError(f"Missing required obs column: {key}")

        cluster_series = adata.obs[cluster_key].astype("category")
        if unlabeled_category not in cluster_series.cat.categories:
            cluster_series = cluster_series.cat.add_categories([unlabeled_category])
        adata.obs[cluster_key] = cluster_series
        adata.obs[class_key] = adata.obs[class_key].astype("category")
        adata.obs[subclass_key] = adata.obs[subclass_key].astype("category")

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            LabelsWithUnlabeledObsField(REGISTRY_KEYS.LABELS_KEY, cluster_key, unlabeled_category),
            CategoricalObsField(HIERARCHY_CLASS_KEY, class_key),
            CategoricalObsField(HIERARCHY_SUBCLASS_KEY, subclass_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @staticmethod
    def _get_fields_for_adata_minification(
        minified_data_type: MinifiedDataType,
    ) -> list[BaseAnnDataField]:
        if minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            return [
                ObsmField(REGISTRY_KEYS.LATENT_QZM_KEY, _SCANVI_LATENT_QZM),
                ObsmField(REGISTRY_KEYS.LATENT_QZV_KEY, _SCANVI_LATENT_QZV),
                NumericalObsField(REGISTRY_KEYS.OBSERVED_LIB_SIZE, _SCANVI_OBSERVED_LIB_SIZE),
                StringUnsField(REGISTRY_KEYS.MINIFY_TYPE_KEY, _ADATA_MINIFY_TYPE_UNS_KEY),
            ]
        raise NotImplementedError(f"Unsupported minified data type: {minified_data_type}")

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        n_samples_per_label: float | None = None,
        check_val_every_n_epoch: int | None = None,
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 256,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        datasplitter_kwargs: dict[str, Any] | None = None,
        plan_kwargs: dict[str, Any] | None = None,
        datamodule: LightningDataModule | None = None,
        **trainer_kwargs: Any,
    ) -> Any:
        """Train the hierarchical sparse scANVI model.

        Parameters
        ----------
        max_epochs
            Number of epochs to train for in the current phase.
        n_samples_per_label
            Number of labelled samples per class to subsample per epoch.
        check_val_every_n_epoch
            Validation frequency.
        train_size
            Fraction of cells used for training.
        validation_size
            Fraction of cells used for validation.
        shuffle_set_split
            Whether to shuffle before splitting train and validation sets.
        batch_size
            Minibatch size.
        %(param_accelerator)s
        %(param_devices)s
        datasplitter_kwargs
            Extra keyword arguments passed to ``SemiSupervisedDataSplitter``.
        plan_kwargs
            Extra keyword arguments passed to the custom semi-supervised training plan.
        datamodule
            Optional Lightning datamodule used instead of ``SemiSupervisedDataSplitter``.
        **trainer_kwargs
            Additional keyword arguments forwarded to the scvi-tools trainer.
        """
        if max_epochs is None:
            if datamodule is None:
                max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
            elif hasattr(datamodule, "n_obs"):
                max_epochs = get_max_epochs_heuristic(datamodule.n_obs)
            else:
                raise ValueError(
                    "If `datamodule` does not expose `n_obs`, `max_epochs` must be provided."
                )
            if self.was_pretrained:
                max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))

        plan_kwargs = {} if plan_kwargs is None else dict(plan_kwargs)
        datasplitter_kwargs = {} if datasplitter_kwargs is None else dict(datasplitter_kwargs)
        if datamodule is None:
            sampler_callback = [SubSampleLabels()] if len(self._labeled_indices) != 0 else []
            train_datamodule: LightningDataModule = SemiSupervisedDataSplitter(
                adata_manager=self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                shuffle_set_split=shuffle_set_split,
                n_samples_per_label=n_samples_per_label,
                batch_size=batch_size,
                **datasplitter_kwargs,
            )
        else:
            sampler_callback = []
            train_datamodule = datamodule

        training_plan = self._training_plan_cls(self.module, self.n_labels, **plan_kwargs)
        callbacks = list(trainer_kwargs.pop("callbacks", []))
        callbacks.extend(sampler_callback)

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=train_datamodule,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=callbacks,
            **trainer_kwargs,
        )
        return runner()


def default_training_phases(
    warmup_epochs: int = 5,
    sparsify_epochs: int = 40,
    finetune_epochs: int = 10,
    final_lambda_sparse: float = 5e-3,
    start_tau: float = 2.0,
    end_tau: float = 0.25,
    finetune_tau: float = 0.25,
    base_classification_ratio: float = 10.0,
    freeze_encoder_in_finetune: bool = True,
) -> list[TrainingPhase]:
    return [
        TrainingPhase(
            name="warmup",
            max_epochs=warmup_epochs,
            lambda_sparse_start=0.0,
            lambda_sparse_end=0.0,
            tau_start=start_tau,
            tau_end=start_tau,
            classification_ratio=base_classification_ratio,
        ),
        TrainingPhase(
            name="sparsification",
            max_epochs=sparsify_epochs,
            lambda_sparse_start=0.0,
            lambda_sparse_end=final_lambda_sparse,
            tau_start=start_tau,
            tau_end=end_tau,
            classification_ratio=base_classification_ratio * 0.25,
        ),
        TrainingPhase(
            name="finetune",
            max_epochs=finetune_epochs,
            lambda_sparse_start=final_lambda_sparse,
            lambda_sparse_end=final_lambda_sparse,
            tau_start=finetune_tau,
            tau_end=finetune_tau,
            classification_ratio=base_classification_ratio * 0.5,
            freeze_encoder=freeze_encoder_in_finetune,
        ),
    ]


def _resolve_training_device(
    accelerator: str | None = None,
    devices: int | list[int] | str | None = None,
    precision: str | None = None,
) -> tuple[str, int | list[int] | str, str]:
    cuda_available = torch.cuda.is_available()
    accelerator = accelerator or ("gpu" if cuda_available else "cpu")
    devices = devices if devices is not None else 1
    precision = precision or ("16-mixed" if cuda_available else "32-true")
    return accelerator, devices, precision


def _uses_multi_gpu(accelerator: str, devices: int | list[int] | str) -> bool:
    if accelerator != "gpu":
        return False
    if isinstance(devices, int):
        return devices > 1 or devices == -1
    if isinstance(devices, list):
        return len(devices) > 1
    if isinstance(devices, str):
        if devices == "auto":
            return False
        if devices == "-1":
            return True
        if devices.isdigit():
            return int(devices) > 1
    return False


def _chunk_to_csr_float32(x_chunk: Any) -> sp_sparse.csr_matrix:
    compute_fn = getattr(x_chunk, "compute", None)
    if callable(compute_fn):
        x_chunk = compute_fn()

    if sp_sparse.issparse(x_chunk):
        return x_chunk.tocsr().astype(np.float32)

    to_scipy_sparse_fn = getattr(x_chunk, "to_scipy_sparse", None)
    if callable(to_scipy_sparse_fn):
        return to_scipy_sparse_fn().tocsr().astype(np.float32)

    try:
        return sp_sparse.csr_matrix(np.asarray(x_chunk, dtype=np.float32))
    except (TypeError, ValueError) as exc:
        flat_values = np.asarray(x_chunk, dtype=object).ravel()
        if flat_values.size > 0 and all(sp_sparse.issparse(value) for value in flat_values):
            return sp_sparse.vstack([value.tocsr() for value in flat_values], format="csr").astype(np.float32)
        raise TypeError("Unable to convert expression chunk to CSR float32 matrix.") from exc


def select_top_k_genes(
    model: HierarchicalSCANVI,
    adata: AnnData | None = None,
    k: int = 1000,
    strategy: str = "greedy_spearman",
    candidate_pool_size: int | None = None,
    diversity_strength: float = 0.25,
) -> tuple[list[str], pd.DataFrame]:
    if adata is None:
        adata = model.adata
    mask = model.module.get_gene_mask(detached=True).cpu().numpy()
    eligible = model.module.eligible_gene_mask.detach().cpu().numpy().astype(bool)
    scores = pd.DataFrame(
        {
            "gene": adata.var_names.astype(str),
            "gate_value": mask,
            "eligible": eligible,
        }
    ).sort_values("gate_value", ascending=False, kind="stable")
    scores["selected_top_k"] = False
    scores["selection_score"] = np.nan

    eligible_scores = scores.loc[scores["eligible"]].copy()
    if eligible_scores.empty:
        return [], scores

    if strategy == "topk":
        selected = eligible_scores["gene"].head(k).tolist()
        scores.loc[scores["gene"].isin(selected), "selected_top_k"] = True
        scores.loc[scores["gene"].isin(selected), "selection_score"] = scores.loc[
            scores["gene"].isin(selected), "gate_value"
        ]
        return selected, scores

    if strategy != "greedy_spearman":
        raise ValueError(f"Unsupported selection strategy: {strategy}")

    candidate_pool_size = candidate_pool_size or max(3 * k, 2000)
    candidate_pool_size = min(candidate_pool_size, len(eligible_scores))
    candidates = eligible_scores.head(candidate_pool_size).copy()
    candidate_gene_names = candidates["gene"].tolist()
    candidate_gene_indices = adata.var_names.get_indexer(candidate_gene_names)

    cluster_labels = adata.obs[model.cluster_key].astype(str)
    cluster_codes, cluster_names = pd.factorize(cluster_labels, sort=True)
    n_clusters = len(cluster_names)
    if n_clusters < 2:
        raise ValueError("Need at least two clusters for greedy Spearman panel selection.")

    cluster_sums = np.zeros((n_clusters, candidate_pool_size), dtype=np.float32)
    aggregation_chunk_size = 16384
    for start in range(0, adata.n_obs, aggregation_chunk_size):
        stop = min(start + aggregation_chunk_size, adata.n_obs)
        x_chunk = _chunk_to_csr_float32(adata[start:stop, candidate_gene_indices].X)

        chunk_cluster_codes = cluster_codes[start:stop]
        chunk_indicator = sp_sparse.csr_matrix(
            (
                np.ones(stop - start, dtype=np.float32),
                (chunk_cluster_codes, np.arange(stop - start, dtype=np.int64)),
            ),
            shape=(n_clusters, stop - start),
        )
        chunk_sums = chunk_indicator @ x_chunk
        if sp_sparse.issparse(chunk_sums):
            chunk_sums = chunk_sums.toarray()
        cluster_sums += np.asarray(chunk_sums, dtype=np.float32)

    cluster_counts = np.bincount(cluster_codes, minlength=n_clusters).astype(np.float32)
    cluster_means = cluster_sums / np.clip(cluster_counts[:, None], a_min=1.0, a_max=None)
    cluster_means = np.clip(cluster_means, a_min=0.0, a_max=None)

    ranked = rankdata(cluster_means, axis=0, method="average").astype(np.float32, copy=False)
    ranked -= ranked.mean(axis=0, keepdims=True)
    ranked_std = ranked.std(axis=0, keepdims=True)
    ranked_std[ranked_std < 1e-6] = 1.0
    ranked /= ranked_std

    similarity = (ranked.T @ ranked) / max(ranked.shape[0], 1)
    similarity = np.abs(similarity).astype(np.float32, copy=False)
    np.fill_diagonal(similarity, 0.0)

    relevance = candidates["gate_value"].to_numpy(dtype=np.float32)
    selected_indices: list[int] = []
    selected_scores: list[float] = []
    selected_mask = np.zeros(candidate_pool_size, dtype=bool)
    redundancy_penalty = np.zeros(candidate_pool_size, dtype=np.float32)

    for _ in range(min(k, candidate_pool_size)):
        objective = relevance - diversity_strength * redundancy_penalty
        objective[selected_mask] = -np.inf
        best_idx = int(np.argmax(objective))
        if not np.isfinite(objective[best_idx]):
            break
        selected_indices.append(best_idx)
        selected_scores.append(float(objective[best_idx]))
        selected_mask[best_idx] = True
        redundancy_penalty = np.maximum(redundancy_penalty, similarity[:, best_idx])

    selected = [candidate_gene_names[i] for i in selected_indices]
    selected_score_map = {candidate_gene_names[i]: selected_scores[pos] for pos, i in enumerate(selected_indices)}
    scores.loc[scores["gene"].isin(selected), "selected_top_k"] = True
    scores.loc[scores["gene"].isin(selected), "selection_score"] = scores.loc[
        scores["gene"].isin(selected), "gene"
    ].map(selected_score_map)
    return selected, scores


def _history_value_to_series(value: Any) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 1:
            return value.iloc[:, 0].reset_index(drop=True)
        if "value" in value.columns:
            return value["value"].reset_index(drop=True)
        return value.iloc[:, -1].reset_index(drop=True)
    if isinstance(value, pd.Series):
        return value.reset_index(drop=True)
    if isinstance(value, np.ndarray):
        return pd.Series(value.reshape(-1))
    if torch.is_tensor(value):
        return pd.Series(value.detach().cpu().reshape(-1).numpy())
    return pd.Series(value)


def extract_elbo_history(model: Any, start_epoch: int = 0) -> pd.DataFrame:
    history = getattr(model, "history", None)
    if callable(history):
        history = history()
    if history is None:
        return pd.DataFrame(columns=["epoch", "elbo_train", "elbo_validation"])

    train_series = None
    val_series = None
    if "elbo_train" in history:
        train_series = _history_value_to_series(history["elbo_train"])
    if "elbo_validation" in history:
        val_series = _history_value_to_series(history["elbo_validation"])

    max_len = max(len(train_series) if train_series is not None else 0, len(val_series) if val_series is not None else 0)
    if max_len == 0:
        return pd.DataFrame(columns=["epoch", "elbo_train", "elbo_validation"])

    frame = pd.DataFrame(
        {
            "epoch": np.arange(max_len, dtype=int),
            "elbo_train": np.full(max_len, np.nan, dtype=np.float64),
            "elbo_validation": np.full(max_len, np.nan, dtype=np.float64),
        }
    )
    if train_series is not None:
        train_values = pd.to_numeric(train_series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        if len(train_values) > 0:
            frame.iloc[: len(train_values), frame.columns.get_loc("elbo_train")] = train_values
    if val_series is not None:
        val_values = pd.to_numeric(val_series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        if len(val_values) > 0:
            frame.iloc[: len(val_values), frame.columns.get_loc("elbo_validation")] = val_values
    if start_epoch > 0:
        frame = frame.iloc[start_epoch:].reset_index(drop=True)
        frame["epoch"] = np.arange(len(frame), dtype=int)
    return frame


def plot_elbo_history(elbo_history: pd.DataFrame, output_path: str | os.PathLike[str]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

    if "elbo_train" in elbo_history.columns:
        train = elbo_history.dropna(subset=["elbo_train"])
        if not train.empty:
            ax.plot(train["epoch"], train["elbo_train"], label="elbo_train", linewidth=2.2)

    if "elbo_validation" in elbo_history.columns:
        val = elbo_history.dropna(subset=["elbo_validation"])
        if not val.empty:
            ax.plot(
                val["epoch"],
                val["elbo_validation"],
                label="elbo_validation",
                linewidth=2.2,
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("ELBO")
    ax.set_title("Train and Validation ELBO")
    ax.legend()
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank() -> int:
    return torch.distributed.get_rank() if _dist_is_initialized() else 0


def _dist_barrier() -> None:
    if _dist_is_initialized():
        torch.distributed.barrier()


class ZarrSparseDataModule(LightningDataModule):
    """Ori-annbatch-style adapter that makes annbatch.Loader trainable in scvi-tools."""

    class _IterableLoaderWrapper:
        """Expose only iteration to avoid Lightning misusing loader length semantics."""

        def __init__(self, loader: Loader):
            self._loader = loader

        def __iter__(self):
            return iter(self._loader)

    def __init__(
        self,
        dataset: Loader,
        *,
        batch_key: str | None = None,
        label_key: str | None = None,
        subclass_key: str | None = None,
        class_key: str | None = None,
        sample_key: str | None = None,
        unlabeled_category: str | None = "Unknown",
        model_name: str = "SCVI",
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        dataset_val: Loader | None = None,
        var_names: list[str] | np.ndarray | pd.Index | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self._validset = dataset_val
        self._batch_key = batch_key
        self._label_key = label_key
        self._subclass_key = subclass_key
        self._class_key = class_key
        self._sample_key = sample_key
        self._categorical_covariate_keys = categorical_covariate_keys
        self._continuous_covariate_keys = continuous_covariate_keys
        self.model_name = model_name
        self.unlabeled_category = unlabeled_category

        self._n_obs = int(getattr(dataset, "n_obs", 0))
        self._n_vars = int(getattr(dataset, "n_var", getattr(dataset, "n_vars", 0)))
        if self._n_vars <= 0 and var_names is not None:
            self._n_vars = int(len(var_names))

        if var_names is not None:
            self._var_names = np.asarray(var_names, dtype=object)
        else:
            self._var_names = np.asarray([f"gene_{i}" for i in range(self._n_vars)], dtype=object)

        self._obs_frames = list(getattr(dataset, "_obs", []) or [])

        self.batch_mapping, self.batch_labels = self._fit_obs_encoder(self._batch_key)
        self.label_mapping, self._label_classes = self._fit_obs_encoder(self._label_key)
        self.subclass_mapping, self.subclass_labels = self._fit_obs_encoder(self._subclass_key)
        self.class_mapping, self.class_labels = self._fit_obs_encoder(self._class_key)
        self.sample_mapping, self.sample_labels = self._fit_obs_encoder(self._sample_key)

        self.labels_ = (
            self._concat_obs_col(self._label_key)
            if self._label_key is not None and self._obs_has_key(self._label_key)
            else np.array([], dtype=object)
        )

        if self.sample_mapping and self._sample_key is not None and self._obs_has_key(self._sample_key):
            sample_values = self._concat_obs_col(self._sample_key)
            sample_indices = np.asarray(
                [self.sample_mapping.get(value, 0) for value in sample_values],
                dtype=np.int64,
            )
            counts = np.bincount(sample_indices, minlength=len(self.sample_labels)).astype(np.float32)
            self._n_obs_per_sample = torch.tensor(counts, dtype=torch.float32)
        else:
            self._n_obs_per_sample = torch.tensor([])

        self.categ_cov_mappings: dict[str, dict[str, int]] = {}
        self.categ_cov_labels: dict[str, np.ndarray] = {}
        if self._categorical_covariate_keys is not None:
            for key in self._categorical_covariate_keys:
                mapping, labels = self._fit_obs_encoder(key)
                if mapping:
                    self.categ_cov_mappings[key] = mapping
                    self.categ_cov_labels[key] = labels

        self.n_train = self.n_obs
        self.n_val = int(getattr(dataset_val, "n_obs", 0)) if dataset_val is not None else 0
        self.data_loader_kwargs: dict[str, Any] = {}

    def _obs_has_key(self, key: str | None) -> bool:
        return (
            key is not None
            and len(self._obs_frames) > 0
            and all(key in obs.columns for obs in self._obs_frames)
        )

    def _concat_obs_col(self, key: str) -> np.ndarray:
        return np.concatenate([obs[key].astype(str).to_numpy() for obs in self._obs_frames]).astype(str)

    def _fit_obs_encoder(self, key: str | None) -> tuple[dict[str, int], np.ndarray]:
        if key is None or not self._obs_has_key(key):
            return {}, np.array([], dtype=object)
        values = self._concat_obs_col(key)
        labels = pd.Index(pd.unique(values)).astype(str).to_numpy(dtype=object)
        mapping = {value: idx for idx, value in enumerate(labels.tolist())}
        return mapping, labels

    def _encode_obs_column(
        self,
        obs: pd.DataFrame | None,
        key: str | None,
        mapping: dict[str, int],
        n_cells: int,
    ) -> torch.Tensor:
        if obs is not None and key is not None and key in obs.columns and mapping:
            values = obs[key].astype(str).to_numpy()
            codes = np.asarray([mapping.get(value, 0) for value in values], dtype=np.int64)
            return torch.from_numpy(codes).unsqueeze(1)
        return torch.zeros((n_cells, 1), dtype=torch.long)

    @staticmethod
    def _to_dense_float_tensor(x_np: Any) -> torch.Tensor:
        if isinstance(x_np, torch.Tensor):
            if x_np.layout in (torch.sparse_csr, torch.sparse_csc, torch.sparse_coo):
                # Sparse UInt32 tensors cannot be densified directly in PyTorch.
                if not x_np.is_floating_point():
                    x_np = x_np.to(dtype=torch.float32)
                return x_np.to_dense().float()
            return x_np.float()
        if sp_sparse.issparse(x_np):
            return torch.from_numpy(x_np.toarray()).float()
        return torch.from_numpy(np.asarray(x_np)).float()

    def train_dataloader(self):
        return self._IterableLoaderWrapper(self.dataset)

    def val_dataloader(self):
        if self._validset is not None:
            return self._IterableLoaderWrapper(self._validset)
        return []

    def inference_dataloader(self):
        return self._InferenceDataloader(self.train_dataloader(), self.on_before_batch_transfer)

    class _InferenceDataloader:
        def __init__(self, dataloader, transform_fn):
            self.dataloader = dataloader
            self.transform_fn = transform_fn

        def __iter__(self):
            for batch in self.dataloader:
                yield self.transform_fn(batch, dataloader_idx=None)

    def on_before_batch_transfer(self, batch: dict[str, Any], dataloader_idx):
        x = self._to_dense_float_tensor(batch["X"])
        obs = batch.get("obs")
        n_cells = int(x.shape[0])

        batch_tensor = self._encode_obs_column(obs, self._batch_key, self.batch_mapping, n_cells)
        labels_tensor = self._encode_obs_column(obs, self._label_key, self.label_mapping, n_cells)
        subclass_tensor = self._encode_obs_column(obs, self._subclass_key, self.subclass_mapping, n_cells)
        class_tensor = self._encode_obs_column(obs, self._class_key, self.class_mapping, n_cells)
        sample_tensor = self._encode_obs_column(obs, self._sample_key, self.sample_mapping, n_cells)

        extra_cat_covs = None
        if self._categorical_covariate_keys is not None and obs is not None:
            encoded_covs = [
                self._encode_obs_column(obs, key, self.categ_cov_mappings.get(key, {}), n_cells)
                for key in self._categorical_covariate_keys
                if key in self.categ_cov_mappings
            ]
            if len(encoded_covs) > 0:
                extra_cat_covs = torch.cat(encoded_covs, dim=1)

        extra_cont_covs = None
        if self._continuous_covariate_keys is not None and obs is not None:
            available_keys = [key for key in self._continuous_covariate_keys if key in obs.columns]
            if len(available_keys) > 0:
                extra_cont_covs = torch.cat(
                    [torch.from_numpy(obs[key].to_numpy()).float().unsqueeze(1) for key in available_keys],
                    dim=1,
                )

        return {
            REGISTRY_KEYS.X_KEY: x,
            REGISTRY_KEYS.BATCH_KEY: batch_tensor,
            REGISTRY_KEYS.LABELS_KEY: labels_tensor,
            REGISTRY_KEYS.CAT_COVS_KEY: extra_cat_covs,
            REGISTRY_KEYS.CONT_COVS_KEY: extra_cont_covs,
            REGISTRY_KEYS.INDICES_KEY: torch.arange(n_cells, dtype=torch.long).unsqueeze(1),
            REGISTRY_KEYS.SAMPLE_KEY: sample_tensor,
            HIERARCHY_SUBCLASS_KEY: subclass_tensor,
            HIERARCHY_CLASS_KEY: class_tensor,
        }

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def n_batch(self) -> int:
        return len(self.batch_labels) if len(self.batch_labels) > 0 else 1

    @property
    def n_labels(self) -> int:
        return len(self.labels_mapping) if len(self.label_mapping) > 0 else 0

    @property
    def batch_keys(self) -> list[str]:
        return [self._batch_key] if self._batch_key is not None else []

    @property
    def label_keys(self) -> list[str]:
        return [self._label_key] if self._label_key is not None else []

    @property
    def labels_key(self) -> str | None:
        return self._label_key

    @property
    def categorical_covariate_keys(self) -> list[str]:
        return list(self._categorical_covariate_keys or [])

    @property
    def continuous_covariate_keys(self) -> list[str]:
        return list(self._continuous_covariate_keys or [])

    @property
    def n_samples(self) -> int:
        return len(self.sample_labels) if len(self.sample_labels) > 0 else 0

    @property
    def labels_mapping(self) -> np.ndarray:
        if len(self._label_classes) > 0:
            combined = np.concatenate(
                (self._label_classes, [self.unlabeled_category or "Unknown"]),
            )
            unique_values, idx = np.unique(combined, return_index=True)
            return unique_values[np.argsort(idx)].astype(object)
        return np.array([self.unlabeled_category or "Unknown"], dtype=object)

    @property
    def n_obs_per_sample(self) -> torch.Tensor:
        return self._n_obs_per_sample

    @property
    def extra_categorical_covs(self) -> dict[str, Any]:
        if self._categorical_covariate_keys is None:
            return {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_categorical_covs": 0},
            }
        mapping = {
            key: self.categ_cov_labels[key]
            for key in self._categorical_covariate_keys
            if key in self.categ_cov_labels
        }
        return {
            "data_registry": {"attr_key": "_scvi_extra_categorical_covs", "attr_name": "obsm"},
            "state_registry": {
                "field_keys": self._categorical_covariate_keys,
                "mapping": mapping,
                "n_cats_per_key": [len(mapping[key]) for key in mapping],
            },
            "summary_stats": {"n_extra_categorical_covs": len(self._categorical_covariate_keys)},
        }

    @property
    def extra_continuous_covs(self) -> dict[str, Any]:
        if self._continuous_covariate_keys is None:
            return {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_continuous_covs": 0},
            }
        return {
            "data_registry": {"attr_key": "_scvi_extra_continuous_covs", "attr_name": "obsm"},
            "state_registry": {"columns": np.array(self._continuous_covariate_keys, dtype=object)},
            "summary_stats": {"n_extra_continuous_covs": len(self._continuous_covariate_keys)},
        }

    @property
    def registry(self) -> dict[str, Any]:
        return {
            "scvi_version": scvi.__version__,
            "model_name": self.model_name,
            "setup_args": {
                "layer": None,
                "batch_key": self._batch_key,
                "labels_key": self._label_key,
                "samples_key": self._sample_key,
                "size_factor_key": None,
                "categorical_covariate_keys": self._categorical_covariate_keys,
                "continuous_covariate_keys": self._continuous_covariate_keys,
            },
            "field_registries": {
                "X": {
                    "data_registry": {"attr_name": "X", "attr_key": None},
                    "state_registry": {
                        "n_obs": self.n_obs,
                        "n_vars": self.n_vars,
                        "column_names": self._var_names.astype(str),
                    },
                    "summary_stats": {"n_vars": self.n_vars, "n_cells": self.n_obs},
                },
                REGISTRY_KEYS.BATCH_KEY: {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_batch"},
                    "state_registry": {
                        "categorical_mapping": self.batch_labels,
                        "original_key": self._batch_key,
                    },
                    "summary_stats": {"n_batch": self.n_batch},
                },
                REGISTRY_KEYS.LABELS_KEY: {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_labels"},
                    "state_registry": {
                        "categorical_mapping": self.labels_mapping,
                        "original_key": self._label_key,
                        "unlabeled_category": self.unlabeled_category,
                    },
                    "summary_stats": {"n_labels": self.n_labels},
                },
                HIERARCHY_SUBCLASS_KEY: {
                    "data_registry": {"attr_name": "obs", "attr_key": HIERARCHY_SUBCLASS_KEY},
                    "state_registry": {
                        "categorical_mapping": self.subclass_labels,
                        "original_key": self._subclass_key,
                    },
                    "summary_stats": {"n_subclass_labels": len(self.subclass_labels)},
                },
                HIERARCHY_CLASS_KEY: {
                    "data_registry": {"attr_name": "obs", "attr_key": HIERARCHY_CLASS_KEY},
                    "state_registry": {
                        "categorical_mapping": self.class_labels,
                        "original_key": self._class_key,
                    },
                    "summary_stats": {"n_class_labels": len(self.class_labels)},
                },
                REGISTRY_KEYS.INDICES_KEY: {
                    "data_registry": {"attr_name": "obs", "attr_key": "_indices"},
                    "state_registry": {},
                    "summary_stats": {},
                },
                REGISTRY_KEYS.SAMPLE_KEY: {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_sample"},
                    "state_registry": {
                        "categorical_mapping": self.sample_labels,
                        "original_key": self._sample_key,
                    },
                    "n_obs_per_sample": {"n_obs_per_sample": self.n_obs_per_sample},
                    "summary_stats": {"n_sample": self.n_samples},
                },
                REGISTRY_KEYS.SIZE_FACTOR_KEY: {
                    "data_registry": {},
                    "state_registry": {},
                    "summary_stats": {},
                },
                REGISTRY_KEYS.CAT_COVS_KEY: self.extra_categorical_covs,
                REGISTRY_KEYS.CONT_COVS_KEY: self.extra_continuous_covs,
            },
            "setup_method_name": "setup_datamodule",
        }


def build_eligible_gene_mask(
    adata: AnnData,
    *,
    exclude_gene_name_patterns: tuple[str, ...] = (
        r"^Gm\d+",
        r"(?i)^mt-",
        r"^Rps",
        r"^Rpl",
        r"^Rik$",
        r"^[A-Z][a-z]*\d*Rik$",
    ),
) -> np.ndarray:
    names = adata.var_names.astype(str).to_numpy()
    eligible = np.ones(len(names), dtype=bool)
    for pattern in exclude_gene_name_patterns:
        eligible &= ~pd.Series(names).str.match(pattern, na=False).to_numpy()
    return eligible


def build_annbatch_collection(
    adata: AnnData,
    collection_path: str,
    layer: str | None = None,
    obs_keys: list[str] | None = None,
    backing_zarr_path: str | None = None,
) -> DatasetCollection:
    del layer, obs_keys
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

    collection_root = Path(collection_path)
    if collection_root.suffix != ".zarr":
        collection_root = collection_root.with_suffix(".zarr")
    collection_exists = collection_root.exists() and any(collection_root.iterdir())
    collection_root.mkdir(parents=True, exist_ok=True)

    if backing_zarr_path is not None:
        zarr_path = Path(backing_zarr_path)
        if not zarr_path.exists():
            raise FileNotFoundError(f"backing_zarr_path does not exist: {zarr_path}")
    else:
        zarr_path = collection_root if collection_root.suffix == ".zarr" else collection_root / "adata.zarr"
        if not zarr_path.exists():
            zarr_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write_zarr(str(zarr_path))

    collection = DatasetCollection(str(collection_root))
    if not collection_exists:
        collection.add_adatas([str(zarr_path)], shuffle=True)
    return collection


def make_annbatch_loader(
    collection: DatasetCollection,
    batch_size: int,
    obs_keys: list[str],
    layer: str | None = "counts",
    chunk_size: int = 32,
    preload_nchunks: int = 256,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    return_index: bool = False,
) -> Loader:
    def custom_load_func(g: zarr.Group) -> ad.AnnData:
        x_key = "X" if layer is None else f"layers/{layer}"
        x = ad.io.sparse_dataset(g[x_key])
        obs = ad.io.read_elem(g["obs"])
        selected_obs_keys = [key for key in obs_keys if key in obs.columns]
        if selected_obs_keys:
            obs = obs.loc[:, selected_obs_keys].copy()
        var = ad.io.read_elem(g["var"])
        return ad.AnnData(X=x, obs=obs, var=var)

    resolved_chunk_size = max(int(chunk_size), 1)
    resolved_preload_nchunks = max(int(preload_nchunks), 1)
    max_supported_batch_size = resolved_chunk_size * resolved_preload_nchunks
    if int(batch_size) > max_supported_batch_size:
        adjusted_chunk_size = int(np.ceil(float(batch_size) / float(resolved_preload_nchunks)))
        print(
            "[checkpoint] annbatch loader adjusted chunk_size from "
            f"{resolved_chunk_size} to {adjusted_chunk_size} to satisfy "
            f"batch_size={batch_size} with preload_nchunks={resolved_preload_nchunks}",
            flush=True,
        )
        resolved_chunk_size = adjusted_chunk_size

    with ad.settings.override(remove_unused_categories=False):
        return Loader(
            batch_size=batch_size,
            chunk_size=resolved_chunk_size,
            preload_nchunks=resolved_preload_nchunks,
            shuffle=shuffle,
            drop_last=drop_last,
            return_index=return_index,
            to_torch=True,
        ).use_collection(collection, load_adata=custom_load_func)


def make_annbatch_train_val_loaders(
    collection: DatasetCollection,
    *,
    batch_size: int,
    obs_keys: list[str],
    layer: str | None,
    chunk_size: int,
    preload_nchunks: int,
    train_size: float,
    validation_size: float | None,
    return_index: bool = False,
) -> tuple[Loader, Loader | None]:
    collection_groups = list(collection)
    if len(collection_groups) == 0:
        raise ValueError("annbatch collection is empty")

    resolved_validation_size = float(validation_size) if validation_size is not None else float(1.0 - train_size)
    resolved_validation_size = float(np.clip(resolved_validation_size, 0.0, 1.0))

    if resolved_validation_size <= 0.0 or len(collection_groups) < 2:
        train_loader = make_annbatch_loader(
            collection,
            batch_size=batch_size,
            obs_keys=obs_keys,
            layer=layer,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            shuffle=True,
            drop_last=True,
            return_index=return_index,
        )
        return train_loader, None

    n_total_groups = len(collection_groups)
    n_val_groups = int(round(n_total_groups * resolved_validation_size))
    n_val_groups = min(max(n_val_groups, 1), n_total_groups - 1)
    n_train_groups = n_total_groups - n_val_groups

    train_groups = collection_groups[:n_train_groups]
    val_groups = collection_groups[n_train_groups:]

    print(
        "[checkpoint] annbatch split: "
        f"train_groups={len(train_groups)} val_groups={len(val_groups)} "
        f"(requested validation_size={resolved_validation_size:.3f})",
        flush=True,
    )

    def custom_load_func(g: zarr.Group) -> ad.AnnData:
        x_key = "X" if layer is None else f"layers/{layer}"
        x = ad.io.sparse_dataset(g[x_key])
        obs = ad.io.read_elem(g["obs"])
        selected_obs_keys = [key for key in obs_keys if key in obs.columns]
        if selected_obs_keys:
            obs = obs.loc[:, selected_obs_keys].copy()
        var = ad.io.read_elem(g["var"])
        return ad.AnnData(X=x, obs=obs, var=var)

    resolved_chunk_size = max(int(chunk_size), 1)
    resolved_preload_nchunks = max(int(preload_nchunks), 1)
    max_supported_batch_size = resolved_chunk_size * resolved_preload_nchunks
    if int(batch_size) > max_supported_batch_size:
        adjusted_chunk_size = int(np.ceil(float(batch_size) / float(resolved_preload_nchunks)))
        print(
            "[checkpoint] annbatch loader adjusted chunk_size from "
            f"{resolved_chunk_size} to {adjusted_chunk_size} to satisfy "
            f"batch_size={batch_size} with preload_nchunks={resolved_preload_nchunks}",
            flush=True,
        )
        resolved_chunk_size = adjusted_chunk_size

    with ad.settings.override(remove_unused_categories=False):
        train_loader = Loader(
            batch_size=batch_size,
            chunk_size=resolved_chunk_size,
            preload_nchunks=resolved_preload_nchunks,
            shuffle=True,
            drop_last=True,
            return_index=return_index,
            to_torch=True,
        )
        for group in train_groups:
            train_loader.add_adata(custom_load_func(group))

        val_loader = Loader(
            batch_size=batch_size,
            chunk_size=resolved_chunk_size,
            preload_nchunks=resolved_preload_nchunks,
            shuffle=False,
            drop_last=False,
            return_index=return_index,
            to_torch=True,
        )
        for group in val_groups:
            val_loader.add_adata(custom_load_func(group))

    return train_loader, val_loader


def _infer_annbatch_obs_keys(
    adata: AnnData,
    adata_manager: AnnDataManager | None,
    *,
    batch_key: str | None,
    cluster_key: str,
    subclass_key: str,
    class_key: str,
) -> list[str]:
    obs_keys: list[str] = []
    registry_keys = (
        REGISTRY_KEYS.BATCH_KEY,
        REGISTRY_KEYS.LABELS_KEY,
        HIERARCHY_SUBCLASS_KEY,
        HIERARCHY_CLASS_KEY,
    )
    if adata_manager is not None:
        for registry_key in registry_keys:
            try:
                state_registry = adata_manager.get_state_registry(registry_key)
            except Exception:
                continue
            obs_key = getattr(state_registry, "original_key", None)
            if obs_key is None:
                data_registry = adata_manager.data_registry.get(registry_key)
                obs_key = getattr(data_registry, "attr_key", None)
            if isinstance(obs_key, str) and obs_key in adata.obs and obs_key not in obs_keys:
                obs_keys.append(obs_key)

    fallback_keys = [key for key in (batch_key, cluster_key, subclass_key, class_key) if key is not None]
    for key in fallback_keys:
        if key in adata.obs and key not in obs_keys:
            obs_keys.append(key)
    return obs_keys


def _get_registered_adata_manager(
    model_cls: type[BaseMinifiedModeModelClass],
    adata: AnnData,
) -> AnnDataManager | None:
    getter = getattr(model_cls, "get_anndata_manager", None)
    if callable(getter):
        for kwargs in ({}, {"required": False}):
            try:
                manager = getter(adata, **kwargs)
            except TypeError:
                continue
            except Exception:
                return None
            if manager is not None:
                return manager
    return None


def train_sparse_hierarchical_scanvi(
    adata: AnnData,
    *,
    batch_key: str = "batch",
    scvi_pretrain_batch_key: str | None = None,
    class_key: str = "class",
    subclass_key: str = "subclass",
    cluster_key: str = "cluster",
    unlabeled_category: str = "Unknown",
    top_k: int = 1000,
    selection_strategy: str = "greedy_spearman",
    selection_candidate_pool_size: int | None = None,
    selection_diversity_strength: float = 0.25,
    layer: str | None = None,
    n_hidden: int = 128,
    n_latent: int = 16,
    n_layers: int = 2,
    dropout_rate: float = 0.1,
    classifier_hidden: int = 128,
    classifier_layers: int = 1,
    gate_init_prob: float = 0.5,
    lambda_binary: float = 1e-2,
    lambda_panel_size: float = 1.0,
    gamma_redundancy: float = 1e-4,
    redundancy_top_k: int = 256,
    cluster_separation_weight: float = 1.0,
    classification_ratio: float = 10.0,
    batch_size: int = 8192,
    scvi_pretrain_epochs: int = 100,
    scvi_handoff_dir: str | None = None,
    train_size: float = 0.9,
    validation_size: float | None = None,
    check_val_every_n_epoch: int | None = 1,
    exclude_gene_name_patterns: tuple[str, ...] = (
        r"^Gm\d+",
        r"(?i)^mt-",
        r"^Rps",
        r"^Rpl",
        r"^[A-Z][a-z]*\d*Rik$",
    ),
    phases: list[TrainingPhase] | None = None,
    accelerator: str | None = None,
    devices: int | list[int] | str | None = None,
    precision: str | None = None,
    plan_kwargs: dict[str, Any] | None = None,
    trainer_kwargs: dict[str, Any] | None = None,
    datasplitter_kwargs: dict[str, Any] | None = None,
    collection_path: str | None = None,
    backing_zarr_path: str | None = None,
    annbatch_chunk_size: int = 32,
    annbatch_preload_nchunks: int = 256,
) -> dict[str, Any]:
    """Train the hierarchical sparse scANVI workflow and select a gene panel.

    Parameters
    ----------
    adata
        Input AnnData object.
    layer
        Layer used as the count matrix during training. When ``None``, ``adata.X`` is used.
    batch_size
        Per-step batch size used for training.
    datasplitter_kwargs
        Extra keyword arguments forwarded to scvi-tools data splitting utilities.
    collection_path
        Optional annbatch collection directory for out-of-core loading.
        Note: the backing ``.zarr`` store must live on local NVMe scratch rather than NFS for
        ``zarrs`` to reach full throughput. On Linux, ``to_torch=True`` with multiprocessing
        requires setting ``multiprocessing_context='spawn'`` in ``datasplitter_kwargs``.
    annbatch_chunk_size
        Number of annbatch chunks loaded per fetch.
    annbatch_preload_nchunks
        Number of annbatch chunks prefetched in the background.
    """
    phases = phases or default_training_phases(base_classification_ratio=classification_ratio)
    trainer_kwargs = {} if trainer_kwargs is None else dict(trainer_kwargs)
    trainer_kwargs.setdefault("enable_progress_bar", True)
    datasplitter_kwargs = {} if datasplitter_kwargs is None else dict(datasplitter_kwargs)
    datasplitter_kwargs.setdefault("num_workers", 4)
    datasplitter_kwargs.setdefault("pin_memory", False)
    base_plan_kwargs = {} if plan_kwargs is None else dict(plan_kwargs)
    accelerator, devices, precision = _resolve_training_device(accelerator, devices, precision)
    torch.set_float32_matmul_precision("high")
    if _uses_multi_gpu(accelerator, devices):
        trainer_kwargs.setdefault("strategy", "ddp_find_unused_parameters_true")
    scvi_pretrain_batch_size = batch_size * 2
    scvi_pretrain_batch_key = batch_key
    annbatch_collection = None
    annbatch_obs_keys: list[str] = []
    scvi_pretrain_datamodule = None
    # Keep the scVI handoff model on the canonical setup_anndata contract so
    # save/load remains architecture-stable even when training via annbatch datamodule.
    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=scvi_pretrain_batch_key,
        layer=layer,
    )
    if collection_path is not None:
        HierarchicalSCANVI.setup_anndata(
            adata,
            batch_key=batch_key,
            class_key=class_key,
            subclass_key=subclass_key,
            cluster_key=cluster_key,
            unlabeled_category=unlabeled_category,
            layer=layer,
        )
        annbatch_collection = build_annbatch_collection(
            adata,
            collection_path=collection_path,
            layer=layer,
            backing_zarr_path=backing_zarr_path,
        )
        annbatch_obs_keys = _infer_annbatch_obs_keys(
            adata,
            _get_registered_adata_manager(HierarchicalSCANVI, adata),
            batch_key=batch_key,
            cluster_key=cluster_key,
            subclass_key=subclass_key,
            class_key=class_key,
        )
        pretrain_loader, pretrain_val_loader = make_annbatch_train_val_loaders(
            annbatch_collection,
            batch_size=scvi_pretrain_batch_size,
            obs_keys=annbatch_obs_keys,
            layer=layer,
            chunk_size=annbatch_chunk_size,
            preload_nchunks=annbatch_preload_nchunks,
            train_size=train_size,
            validation_size=validation_size,
        )
        scvi_pretrain_datamodule = ZarrSparseDataModule(
            pretrain_loader,
            dataset_val=pretrain_val_loader,
            batch_key=scvi_pretrain_batch_key,
            label_key=cluster_key,
            subclass_key=subclass_key,
            class_key=class_key,
            unlabeled_category=unlabeled_category,
            var_names=adata.var_names,
        )

    if scvi_handoff_dir is None:
        scvi_handoff_path = tempfile.mkdtemp(prefix="scvi_handoff_")
    else:
        scvi_handoff_path = scvi_handoff_dir
    scvi_model_path = f"{scvi_handoff_path}/model.pt"
    scvi_elbo_plot_path = f"{scvi_handoff_path}/scvi_pretrain_elbo.png"
    model_artifact_exists = os.path.exists(scvi_model_path)
    reuse_existing_scvi = scvi_handoff_dir is not None and model_artifact_exists
    pretrain_elbo = pd.DataFrame(columns=["epoch", "elbo_train", "elbo_validation"])
    scvi_model: scvi.model.SCVI | None = None

    if reuse_existing_scvi:
        if _dist_rank() == 0:
            print(
                f"[checkpoint] Found existing scVI handoff model at {scvi_model_path}. "
                "Skipping scVI pretraining.",
                flush=True,
            )
    else:
        print(
            f"[checkpoint] Starting scVI pretraining (epochs={scvi_pretrain_epochs}, "
            f"batch_size={scvi_pretrain_batch_size}, "
            f"batch_key={scvi_pretrain_batch_key})",
            flush=True,
        )
        if collection_path is not None and scvi_pretrain_datamodule is not None:
            scvi_model = scvi.model.SCVI(
                adata=adata,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                gene_likelihood="nb",
                dispersion="gene-batch",
                use_batch_norm="none",
                use_layer_norm="both",
            )
            scvi_model.train(
                max_epochs=scvi_pretrain_epochs,
                check_val_every_n_epoch=check_val_every_n_epoch,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                datamodule=scvi_pretrain_datamodule,
                **trainer_kwargs,
            )
        else:
            scvi_model = scvi.model.SCVI(
                adata=adata,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                gene_likelihood="nb",
                dispersion="gene-batch",
                use_batch_norm="none",
                use_layer_norm="both",
            )
            scvi_model.train(
                max_epochs=scvi_pretrain_epochs,
                check_val_every_n_epoch=check_val_every_n_epoch,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=scvi_pretrain_batch_size,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                datasplitter_kwargs=datasplitter_kwargs,
                **trainer_kwargs,
            )
        if _dist_rank() == 0:
            print("[checkpoint] Finished scVI pretraining", flush=True)
        pretrain_elbo = extract_elbo_history(scvi_model)

        if _dist_rank() == 0:
            print(f"[checkpoint] Saving scVI handoff model to {scvi_handoff_path}", flush=True)
            if os.path.exists(scvi_model_path):
                # Remove stale model artifact so a failed save cannot be mistaken as success.
                os.remove(scvi_model_path)

            scvi_model.save(
                scvi_handoff_path,
                overwrite=True,
                save_anndata=False,
            )

            if not os.path.exists(scvi_model_path):
                raise FileNotFoundError(
                    f"Expected scVI handoff model file at {scvi_model_path}, but it was not written."
                )
            if not pretrain_elbo.empty:
                print(f"[checkpoint] Plotting scVI ELBO history to {scvi_elbo_plot_path}", flush=True)
                plot_elbo_history(pretrain_elbo, scvi_elbo_plot_path)
    _dist_barrier()
    model_file_exists = os.path.exists(scvi_model_path)
    if model_file_exists:
        if _dist_rank() == 0:
            print("[checkpoint] Reloading scVI handoff model from disk", flush=True)
        reloaded_scvi_model = scvi.model.SCVI.load(
            scvi_handoff_path,
            adata=adata,
            accelerator="cpu",
            device=1,
        )
    else:
        if scvi_model is None:
            raise FileNotFoundError(
                f"Expected handoff model at {scvi_model_path}, but it does not exist and no in-memory scVI model is available."
            )
        if _dist_rank() == 0:
            print("[checkpoint] Using in-memory scVI handoff model", flush=True)
        reloaded_scvi_model = scvi_model

    model = HierarchicalSCANVI.from_scvi_model(
        scvi_model=reloaded_scvi_model,
        unlabeled_category=unlabeled_category,
        labels_key=cluster_key,
        adata=adata,
        class_key=class_key,
        subclass_key=subclass_key,
        cluster_key=cluster_key,
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        classifier_hidden=classifier_hidden,
        classifier_layers=classifier_layers,
        gate_init_prob=gate_init_prob,
        lambda_binary=lambda_binary,
        lambda_panel_size=lambda_panel_size,
        target_panel_size=top_k,
        gamma_redundancy=gamma_redundancy,
        redundancy_top_k=redundancy_top_k,
        cluster_separation_weight=cluster_separation_weight,
    )
    eligible_gene_mask = torch.tensor(
        build_eligible_gene_mask(adata, exclude_gene_name_patterns=exclude_gene_name_patterns),
        device=model.module.device,
        dtype=torch.float32,
    )
    model.module.set_eligible_gene_mask(eligible_gene_mask)

    phase_summaries: list[dict[str, Any]] = []
    elbo_frames: list[pd.DataFrame] = []
    epoch_offset = 0
    history_cursor = 0
    if not pretrain_elbo.empty:
        pretrain_elbo = pretrain_elbo.copy()
        pretrain_elbo["phase"] = "scvi_pretrain"
        pretrain_elbo["phase_epoch"] = pretrain_elbo["epoch"]
        elbo_frames.append(pretrain_elbo)
        epoch_offset += len(pretrain_elbo)
    else:
        epoch_offset += scvi_pretrain_epochs
    print(
        f"[checkpoint] Data loader settings: num_workers={datasplitter_kwargs['num_workers']}, "
        f"pin_memory={datasplitter_kwargs['pin_memory']}",
        flush=True,
    )
    for phase in phases:
        print(
            f"[checkpoint] Starting phase '{phase.name}' "
            f"(epochs={phase.max_epochs}, lambda_sparse={phase.lambda_sparse_start}->{phase.lambda_sparse_end}, "
            f"tau={phase.tau_start}->{phase.tau_end}, batch_size={batch_size})",
            flush=True,
        )
        current_plan_kwargs = dict(base_plan_kwargs)
        current_plan_kwargs.update(
            {
                "classification_ratio": (
                    phase.classification_ratio
                    if phase.classification_ratio is not None
                    else classification_ratio
                ),
                "lambda_sparse_start": phase.lambda_sparse_start,
                "lambda_sparse_end": phase.lambda_sparse_end,
                "tau_start": phase.tau_start,
                "tau_end": phase.tau_end,
                "freeze_encoder": phase.freeze_encoder,
                "freeze_gene_gate": phase.freeze_gene_gate,
            }
        )
        if collection_path is not None:
            phase_obs_keys = _infer_annbatch_obs_keys(
                adata,
                model.adata_manager,
                batch_key=batch_key,
                cluster_key=cluster_key,
                subclass_key=subclass_key,
                class_key=class_key,
            )
            phase_loader, phase_val_loader = make_annbatch_train_val_loaders(
                annbatch_collection,
                batch_size=batch_size,
                obs_keys=phase_obs_keys,
                layer=layer,
                chunk_size=annbatch_chunk_size,
                preload_nchunks=annbatch_preload_nchunks,
                train_size=train_size,
                validation_size=validation_size,
            )
            phase_datamodule = ZarrSparseDataModule(
                phase_loader,
                dataset_val=phase_val_loader,
                batch_key=batch_key,
                label_key=cluster_key,
                subclass_key=subclass_key,
                class_key=class_key,
                unlabeled_category=unlabeled_category,
                var_names=adata.var_names,
            )
            model.train(
                max_epochs=phase.max_epochs,
                check_val_every_n_epoch=check_val_every_n_epoch,
                batch_size=batch_size,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                plan_kwargs=current_plan_kwargs,
                datamodule=phase_datamodule,
                **trainer_kwargs,
            )
        else:
            model.train(
                max_epochs=phase.max_epochs,
                check_val_every_n_epoch=check_val_every_n_epoch,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                plan_kwargs=current_plan_kwargs,
                datasplitter_kwargs=datasplitter_kwargs,
                **trainer_kwargs,
            )
        phase_elbo = extract_elbo_history(model, start_epoch=history_cursor)
        if not phase_elbo.empty:
            phase_elbo = phase_elbo.copy()
            phase_elbo["phase"] = phase.name
            phase_elbo["phase_epoch"] = phase_elbo["epoch"]
            phase_elbo["epoch"] = phase_elbo["epoch"] + epoch_offset
            elbo_frames.append(phase_elbo)
            epoch_offset += len(phase_elbo)
            history_cursor += len(phase_elbo)
        else:
            epoch_offset += phase.max_epochs
        print(f"[checkpoint] Finished phase '{phase.name}'", flush=True)
        phase_summaries.append(
            {
                "phase": phase.name,
                "epochs": phase.max_epochs,
                "lambda_sparse_end": phase.lambda_sparse_end,
                "tau_end": phase.tau_end,
            }
        )

    selected_genes, importance_scores = select_top_k_genes(
        model,
        adata=adata,
        k=top_k,
        strategy=selection_strategy,
        candidate_pool_size=selection_candidate_pool_size,
        diversity_strength=selection_diversity_strength,
    )
    elbo_history = (
        pd.concat(elbo_frames, ignore_index=True)
        if len(elbo_frames) > 0
        else pd.DataFrame(columns=["epoch", "phase", "phase_epoch", "elbo_train", "elbo_validation"])
    )
    print(
        f"[checkpoint] Training complete. Selected top {len(selected_genes)} genes.",
        flush=True,
    )
    return {
        "selected_genes": selected_genes,
        "model": model,
        "importance_scores": importance_scores,
        "elbo_history": elbo_history,
        "phase_summaries": phase_summaries,
        "scvi_handoff_dir": scvi_handoff_path,
        "scvi_elbo_plot": scvi_elbo_plot_path,
        "selection_strategy": selection_strategy,
    }


def example_usage(adata: AnnData) -> dict[str, Any]:
    return train_sparse_hierarchical_scanvi(
        adata=adata,
        batch_key="batch",
        class_key="class",
        subclass_key="subclass",
        cluster_key="cluster",
        unlabeled_category="Unknown",
        top_k=1000,
        n_hidden=256,
        n_latent=32,
        n_layers=2,
        batch_size=2048,
        gamma_redundancy=5e-5,
    )


__all__ = [
    "HierarchicalSCANVI",
    "SparseHierarchicalSCANVAE",
    "HierarchicalSemiSupervisedTrainingPlan",
    "TrainingPhase",
    "default_training_phases",
    "extract_elbo_history",
    "build_eligible_gene_mask",
    "train_sparse_hierarchical_scanvi",
    "select_top_k_genes",
    "example_usage",
]
