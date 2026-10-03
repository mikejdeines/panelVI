from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anndata as ad


def write_gene_list(genes: list[str], output_path: Path, quoted: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for gene in genes:
            line = f'"{gene}"\n' if quoted else f"{gene}\n"
            handle.write(line)


def plot_elbo_history(elbo_history, output_path: Path) -> None:
    import matplotlib.pyplot as plt

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
    ax.set_ylabel("elbo")
    ax.set_title("Train and Validation ELBO")
    ax.legend()
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_gate_histogram(gene_scores, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    sns.histplot(data=gene_scores, x="gate_value", bins=60, color="black", ax=ax)
    ax.set_xlabel("gate value")
    ax.set_ylabel("count")
    ax.set_title("Distribution of Gene Gate Values")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close(fig)


def _looks_like_zarr_store(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    markers = ("zarr.json", ".zgroup", ".zattrs")
    return any((path / marker).exists() for marker in markers)


def _looks_like_annbatch_collection(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        return any(child.is_dir() and child.name.startswith("dataset_") for child in path.iterdir())
    except OSError:
        return False


def _has_annbatch_preshuffled_encoding(path: Path) -> bool:
    zarr_json = path / "zarr.json"
    if not zarr_json.exists():
        return False
    try:
        payload = json.loads(zarr_json.read_text())
    except Exception:
        return False
    attributes = payload.get("attributes", {})
    return attributes.get("encoding-type") == "annbatch-preshuffled"


def _looks_like_annbatch_root(path: Path) -> bool:
    return _looks_like_annbatch_collection(path) or _has_annbatch_preshuffled_encoding(path)


def _recommended_collection_path_for_zarr(zarr_path: Path) -> Path:
    if zarr_path.name == "adata.zarr":
        parent_path = zarr_path.parent
        if parent_path.suffix != ".zarr":
            return parent_path.with_suffix(".zarr")
        return parent_path.with_name(f"{parent_path.stem}_collection.zarr")
    return zarr_path.with_name(f"{zarr_path.stem}_collection.zarr")


def _resolve_backing_zarr_for_annbatch_root(collection_root: Path) -> Path:
    candidates: list[Path] = []
    candidates.append(collection_root / "adata.zarr")
    if collection_root.suffix == ".zarr":
        candidates.append(collection_root.with_suffix("") / "adata.zarr")
        candidates.append(collection_root.with_name(f"{collection_root.stem}_adata.zarr"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and _looks_like_zarr_store(candidate):
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Detected an annbatch collection root but could not find a compatible backing "
        f"AnnData zarr store. collection={collection_root}; searched=[{searched}]"
    )


def resolve_collection_and_zarr_paths(input_path: Path) -> tuple[Path, Path]:
    resolved_input = input_path
    if not resolved_input.exists() and resolved_input.suffix == ".zarr":
        suffixless_input = resolved_input.with_suffix("")
        if suffixless_input.exists():
            print(
                f"[checkpoint] Input path {resolved_input} not found; "
                f"using existing path {suffixless_input}.",
                flush=True,
            )
            resolved_input = suffixless_input

    if not resolved_input.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

    if _looks_like_annbatch_root(resolved_input):
        backing_zarr_path = _resolve_backing_zarr_for_annbatch_root(resolved_input)
        if resolved_input.suffix == ".zarr":
            collection_path = resolved_input
        else:
            collection_path = _recommended_collection_path_for_zarr(backing_zarr_path)
        return collection_path, backing_zarr_path

    collection_zarr_path = resolved_input / "adata.zarr"
    if collection_zarr_path.exists():
        return _recommended_collection_path_for_zarr(collection_zarr_path), collection_zarr_path

    if resolved_input.name == "adata.zarr" and _looks_like_zarr_store(resolved_input):
        return _recommended_collection_path_for_zarr(resolved_input), resolved_input

    if _looks_like_zarr_store(resolved_input):
        return _recommended_collection_path_for_zarr(resolved_input), resolved_input

    raise FileNotFoundError(
        "Could not resolve input as either an annbatch collection root with adata.zarr "
        "or a direct AnnData zarr store. "
        f"Input={resolved_input}"
    )


def _get_obs_columns(obs: object) -> list[str]:
    columns = list(getattr(obs, "columns", []))
    if columns:
        return columns
    keys = getattr(obs, "keys", None)
    if callable(keys):
        try:
            return [str(key) for key in keys()]
        except Exception:
            pass
    data_vars = getattr(obs, "data_vars", None)
    if data_vars is not None:
        try:
            return [str(key) for key in data_vars]
        except Exception:
            pass
    return []


def _subset_obs_columns(adata: ad.AnnData, obs_keys: list[str] | None) -> bool:
    if obs_keys is None:
        return False
    obs = adata.obs
    obs_columns = _get_obs_columns(obs)
    selected_keys = [key for key in obs_keys if key in obs_columns]
    if not selected_keys:
        return False
    try:
        obs_subset = obs[selected_keys]
    except Exception:
        obs_subset = obs.loc[:, selected_keys]
    if hasattr(obs_subset, "to_memory"):
        obs_subset = obs_subset.to_memory()
    else:
        obs_subset = obs_subset.copy()
    adata.obs = obs_subset
    return True


def load_collection_adata(zarr_path: Path, obs_keys: list[str] | None = None) -> ad.AnnData:
    import anndata as ad

    experimental_module = getattr(ad, "experimental", None)
    read_lazy = getattr(experimental_module, "read_lazy", None)
    if callable(read_lazy):
        try:
            adata = read_lazy(str(zarr_path))
        except Exception as exc:
            if "annbatch-preshuffled" not in str(exc):
                raise
            fallback_candidates: list[Path] = []
            fallback_candidates.append(zarr_path / "adata.zarr")
            if zarr_path.suffix == ".zarr":
                fallback_candidates.append(zarr_path.with_suffix("") / "adata.zarr")
                fallback_candidates.append(zarr_path.with_name(f"{zarr_path.stem}_adata.zarr"))

            fallback_zarr = next(
                (
                    candidate
                    for candidate in fallback_candidates
                    if candidate.exists() and candidate.is_dir() and _looks_like_zarr_store(candidate)
                ),
                None,
            )
            if fallback_zarr is None:
                raise
            print(
                "[checkpoint] Resolved annbatch collection root to backing AnnData zarr at "
                f"{fallback_zarr}",
                flush=True,
            )
            adata = read_lazy(str(fallback_zarr))
        has_subset = _subset_obs_columns(adata, obs_keys)
        if obs_keys is not None and not has_subset:
            available_obs_keys = _get_obs_columns(adata.obs)
            raise KeyError(
                "Could not materialize required obs keys from lazy AnnData. "
                f"required={obs_keys}; available={available_obs_keys[:20]}"
            )
        if hasattr(adata.obs, "to_memory"):
            adata.obs = adata.obs.to_memory()
        if hasattr(adata.var, "to_memory"):
            adata.var = adata.var.to_memory()
        return adata
    adata = ad.read_zarr(str(zarr_path))
    _subset_obs_columns(adata, obs_keys)
    return adata


def run_panel_selection(
    collection_path: str | Path,
    *,
    output: str | Path | None = None,
    elbo_output: str | Path | None = None,
    elbo_plot: str | Path | None = None,
    gene_scores_output: str | Path | None = None,
    gate_hist_output: str | Path | None = None,
    top_k: int = 1000,
    selection_strategy: str = "greedy_spearman",
    selection_candidate_pool_size: int = 8000,
    selection_diversity_strength: float = 0.25,
    batch_key: str = "feature_matrix_label",
    class_key: str = "class",
    subclass_key: str = "subclass",
    cluster_key: str = "cluster",
    unlabeled_category: str = "Unknown",
    layer: str | None = None,
    batch_size: int = 8192,
    scvi_pretrain_epochs: int = 100,
    warmup_epochs: int = 5,
    sparsify_epochs: int = 40,
    finetune_epochs: int = 10,
    initial_phase_tau: float = 2.0,
    lambda_sparse_start: float = 0.0,
    lambda_sparse_end: float = 5e-3,
    tau_start: float | None = None,
    tau_end: float = 0.25,
    scvi_handoff_dir: str | Path | None = None,
    annbatch_chunk_size: int = 32,
    annbatch_preload_nchunks: int = 256,
    gate_init_prob: float = 0.5,
    classification_ratio: float = 10.0,
    lambda_binary: float = 1e-2,
    lambda_panel_size: float = 1.0,
    gamma_redundancy: float = 1e-4,
    cluster_separation_weight: float = 1.0,
    train_size: float = 0.9,
    validation_size: float | None = None,
    check_val_every_n_epoch: int = 1,
    accelerator: str | None = None,
    devices: int | list[int] | str | None = None,
    precision: str | None = None,
    num_workers: int = 4,
    pin_memory: bool = False,
    multiprocessing_context: str | None = None,
    quoted: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train from an annbatch collection or AnnData Zarr store and save a panel.

    Paths accept strings or Path objects. Existing result files require
    ``overwrite=True``. ``devices`` accepts Lightning's integer, list, or string
    device specification. Training and selection defaults match the source script.

    Return the training result dictionary, including ``selected_genes``,
    ``importance_scores``, ``model``, ``elbo_history``, ``phase_summaries``, and
    scVI handoff paths. Tables and plots are written beside ``output`` unless
    their individual paths are supplied. The default output name is
    ``<collection_stem>_top<top_k>_genes.txt``.
    """
    input_path = Path(collection_path).expanduser().resolve()
    collection_path, zarr_path = resolve_collection_and_zarr_paths(input_path)
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else collection_path.with_name(f"{collection_path.stem}_top{top_k}_genes.txt")
    )
    elbo_output_path = (
        Path(elbo_output).expanduser().resolve()
        if elbo_output is not None
        else output_path.with_name(f"{output_path.stem}_elbo.tsv")
    )
    elbo_plot_path = (
        Path(elbo_plot).expanduser().resolve()
        if elbo_plot is not None
        else output_path.with_name(f"{output_path.stem}_elbo.png")
    )
    gene_scores_output_path = (
        Path(gene_scores_output).expanduser().resolve()
        if gene_scores_output is not None
        else output_path.with_name(f"{output_path.stem}_gene_scores.csv")
    )
    gate_hist_output_path = (
        Path(gate_hist_output).expanduser().resolve()
        if gate_hist_output is not None
        else output_path.with_name(f"{output_path.stem}_gate_hist.png")
    )

    for artifact_path in (
        output_path, elbo_output_path, elbo_plot_path,
        gene_scores_output_path, gate_hist_output_path,
    ):
        if artifact_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {artifact_path}. Use overwrite=True to replace it.")

    from .hierarchical_scanvi_panel import default_training_phases, train_sparse_hierarchical_scanvi

    print(f"[checkpoint] Input path: {input_path}", flush=True)
    print(f"[checkpoint] Collection path: {collection_path}", flush=True)
    print(f"[checkpoint] Backing zarr: {zarr_path}", flush=True)
    print(f"[checkpoint] Output gene list: {output_path}", flush=True)
    print(f"[checkpoint] Output ELBO table: {elbo_output_path}", flush=True)
    print(f"[checkpoint] Output ELBO plot: {elbo_plot_path}", flush=True)
    print(f"[checkpoint] Output gene scores CSV: {gene_scores_output_path}", flush=True)
    print(f"[checkpoint] Output gate histogram: {gate_hist_output_path}", flush=True)
    print(
        f"[checkpoint] annbatch tuning: chunk_size={annbatch_chunk_size}, "
        f"preload_nchunks={annbatch_preload_nchunks}",
        flush=True,
    )
    print(
        f"[checkpoint] Configuration: top_k={top_k}, batch_size={batch_size}, "
        f"selection_strategy={selection_strategy}, "
        f"selection_candidate_pool_size={selection_candidate_pool_size}, "
        f"selection_diversity_strength={selection_diversity_strength}, "
        f"scvi_pretrain_epochs={scvi_pretrain_epochs}, "
        f"warmup_epochs={warmup_epochs}, "
        f"sparsify_epochs={sparsify_epochs}, "
        f"finetune_epochs={finetune_epochs}, "
        f"initial_phase_tau={initial_phase_tau}, "
        f"lambda_sparse_start={lambda_sparse_start}, "
        f"lambda_sparse_end={lambda_sparse_end}, "
        f"tau_start={tau_start if tau_start is not None else initial_phase_tau}, "
        f"tau_end={tau_end}, "
        f"scvi_pretrain_batch_size={batch_size * 2}, "
        f"gate_init_prob={gate_init_prob}, classification_ratio={classification_ratio}, "
        f"lambda_binary={lambda_binary}, lambda_panel_size={lambda_panel_size}, "
        f"gamma_redundancy={gamma_redundancy}, "
        f"cluster_separation_weight={cluster_separation_weight}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}, "
        f"multiprocessing_context={multiprocessing_context}, "
        f"batch_key={batch_key}, class_key={class_key}, "
        f"subclass_key={subclass_key}, cluster_key={cluster_key}",
        flush=True,
    )
    print(
        "[checkpoint] Runtime defaults: dataloader num_workers and pin_memory are "
        "set automatically inside the training function unless overridden.",
        flush=True,
    )
    if os.name == "posix" and num_workers > 0 and multiprocessing_context is None:
        print(
            "[checkpoint] annbatch note: consider multiprocessing_context='spawn' when "
            "using num_workers > 0 on Linux.",
            flush=True,
        )
    required_obs_keys = [batch_key, class_key, subclass_key, cluster_key]
    print("[checkpoint] Loading AnnData lazily from collection zarr...", flush=True)
    adata = load_collection_adata(zarr_path, obs_keys=required_obs_keys)
    print(
        f"[checkpoint] Finished loading AnnData with {adata.n_obs} cells and {adata.n_vars} genes.",
        flush=True,
    )
    print("[checkpoint] Starting panel selection training...", flush=True)
    datasplitter_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if multiprocessing_context is not None:
        datasplitter_kwargs["multiprocessing_context"] = multiprocessing_context

    sparsify_tau_start = initial_phase_tau if tau_start is None else tau_start
    phases = default_training_phases(
        warmup_epochs=warmup_epochs,
        sparsify_epochs=sparsify_epochs,
        finetune_epochs=finetune_epochs,
        final_lambda_sparse=lambda_sparse_end,
        start_tau=initial_phase_tau,
        end_tau=tau_end,
        finetune_tau=tau_end,
        base_classification_ratio=classification_ratio,
    )
    phases[1].lambda_sparse_start = lambda_sparse_start
    phases[1].tau_start = sparsify_tau_start

    result = train_sparse_hierarchical_scanvi(
        adata=adata,
        batch_key=batch_key,
        class_key=class_key,
        subclass_key=subclass_key,
        cluster_key=cluster_key,
        unlabeled_category=unlabeled_category,
        top_k=top_k,
        selection_strategy=selection_strategy,
        selection_candidate_pool_size=selection_candidate_pool_size,
        selection_diversity_strength=selection_diversity_strength,
        layer=layer,
        batch_size=batch_size,
        scvi_pretrain_epochs=scvi_pretrain_epochs,
        scvi_handoff_dir=(
            str(Path(scvi_handoff_dir).expanduser().resolve())
            if scvi_handoff_dir is not None
            else None
        ),
        train_size=train_size,
        validation_size=validation_size,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        datasplitter_kwargs=datasplitter_kwargs,
        collection_path=str(collection_path),
        backing_zarr_path=str(zarr_path),
        annbatch_chunk_size=annbatch_chunk_size,
        annbatch_preload_nchunks=annbatch_preload_nchunks,
        gate_init_prob=gate_init_prob,
        classification_ratio=classification_ratio,
        lambda_binary=lambda_binary,
        lambda_panel_size=lambda_panel_size,
        gamma_redundancy=gamma_redundancy,
        cluster_separation_weight=cluster_separation_weight,
        phases=phases,
        n_hidden=256,
        n_latent=32,
        n_layers=2
    )
    print("[checkpoint] Writing ELBO history...", flush=True)
    elbo_history = result["elbo_history"]
    print(f"[checkpoint] scVI handoff directory: {result['scvi_handoff_dir']}", flush=True)
    elbo_output_path.parent.mkdir(parents=True, exist_ok=True)
    elbo_history.to_csv(elbo_output_path, sep="\t", index=False)
    print("[checkpoint] Plotting ELBO history...", flush=True)
    plot_elbo_history(elbo_history, elbo_plot_path)
    print("[checkpoint] Writing gene score table...", flush=True)
    gene_scores = result["importance_scores"]
    gene_scores_output_path.parent.mkdir(parents=True, exist_ok=True)
    gene_scores.to_csv(gene_scores_output_path, index=False)
    print("[checkpoint] Plotting gate histogram...", flush=True)
    plot_gate_histogram(gene_scores, gate_hist_output_path)
    print("[checkpoint] Writing selected genes...", flush=True)
    selected_genes = result["selected_genes"]
    write_gene_list(selected_genes, output_path, quoted=quoted)
    print(f"[checkpoint] Wrote {len(selected_genes)} genes to {output_path}", flush=True)
    print(f"[checkpoint] Wrote ELBO history to {elbo_output_path}", flush=True)
    print(f"[checkpoint] Wrote ELBO plot to {elbo_plot_path}", flush=True)
    print(f"[checkpoint] Wrote gene score table to {gene_scores_output_path}", flush=True)
    print(f"[checkpoint] Wrote gate histogram to {gate_hist_output_path}", flush=True)
    return result
