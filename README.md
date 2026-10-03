# panelVI

iST gene panel selection with sparse hierarchical scANVI. The workflow
pretrains scVI, learns gene gates with subclass and cluster supervision, and
selects a panel using gate scores or a greedy Spearman redundancy penalty.

The distribution is named `panelVI`; the Python import package is `panelvi`.
The API uses functions and keyword arguments. There is no command-line parser.

## Installation

Use Python 3.12 or newer.

```bash
python -m pip install .
```

For development, use `python -m pip install -e .`. To include Jupyter, install
`python -m pip install '.[notebook]'`.

The package uses scRefine's flat layout (`panelvi/` beside `pyproject.toml`),
explicit setuptools package list, and author contact information. The BSD-3-Clause
license is included in `LICENSE`. The version, Python requirement, and main
dependencies come from `panel_selection.def`. Direct dependencies imported by the scripts are also
listed. Versions remain unpinned, as in the definition file; this is not a
lockfile. The model uses private scvi-tools APIs, so compatibility should be
checked when upgrading that dependency.

For the Linux CUDA 12.9 PyTorch build used by the definition file, install Torch
before installing panelVI:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu129 torch
python -m pip install .
```

CUDA wheel selection is an installation step, not a portable project dependency.
The Rocky Linux system packages and Conda setup belong to the container
configuration. Training retains the original CPU fallback when CUDA is unavailable.

## Prepare a collection and select a panel

```python
from panelvi import create_annbatch_collection, run_panel_selection

collection_path, backing_path = create_annbatch_collection(
    "reference.h5ad",
    "reference.zarr",
)

result = run_panel_selection(
    collection_path,
    output="results/panel.txt",
    top_k=1000,
    batch_key="feature_matrix_label",
    class_key="class",
    subclass_key="subclass",
    cluster_key="cluster",
    scvi_handoff_dir="results/scvi_handoff",
)

genes = result["selected_genes"]
scores = result["importance_scores"]
model = result["model"]
```

Collection preparation loads the input h5ad into memory and creates annbatch
metadata in `reference.zarr` with a backing AnnData store in
`reference_adata.zarr`. Keep both artifacts together. Set `shuffle=False` to
disable shuffling or `overwrite=True` to replace existing artifacts. If an
explicit `zarr_path` does not follow the default naming scheme, pass the returned
`backing_path` directly to `run_panel_selection`.

The count matrix is read from `adata.X`, or from the layer specified by `layer`.
By default, `adata.obs` must contain `feature_matrix_label`, `class`, `subclass`,
and `cluster`; adapt the keyword arguments to your data. Gene names come from
`adata.var_names`.

For `results/panel.txt`, selection also writes `panel_elbo.tsv`, `panel_elbo.png`,
`panel_gene_scores.csv`, and `panel_gate_hist.png` in `results/`. Existing result
files require `overwrite=True`. Plots use library defaults without font settings.
An existing scVI handoff model is reused; use a new handoff directory for a new
pretraining run or different input data.

Training and selection parameters are explicit keyword arguments. For example,
use `selection_strategy="topk"` to select solely by gate score, or
`accelerator="cpu", precision="32-true"` to explicitly train on CPU. For Linux
multiprocessing, the original workflow recommends `multiprocessing_context="spawn"`.
Use `help(run_panel_selection)` to inspect the complete function signature.

## Train directly from AnnData

The original model and training API is preserved in
`panelvi.hierarchical_scanvi_panel`:

```python
import anndata as ad
from panelvi.hierarchical_scanvi_panel import (
    HierarchicalSCANVI,
    TrainingPhase,
    default_training_phases,
    select_top_k_genes,
    train_sparse_hierarchical_scanvi,
)

adata = ad.read_h5ad("reference.h5ad")
result = train_sparse_hierarchical_scanvi(
    adata,
    batch_key="feature_matrix_label",
    top_k=1000,
    scvi_handoff_dir="results/scvi_handoff",
)
```

The training function returns the selected genes, scores, trained model, ELBO
history, phase summaries, and scVI handoff paths. Use `run_panel_selection` when
you also want result tables and plots written automatically. Importing `panelvi`
defers scientific dependencies until they are needed.

## Lightweight checks

```bash
python -m unittest discover -s tests -v
```

These checks cover public imports, collection path resolution, output protection,
and forwarding of selection arguments with training mocked out. They do not
convert datasets or launch model training.
