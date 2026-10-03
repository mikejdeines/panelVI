from __future__ import annotations

import shutil
from pathlib import Path


def resolve_zarr_path(collection_path: Path, explicit_zarr_path: Path | None) -> Path:
    if explicit_zarr_path is not None:
        return explicit_zarr_path
    return collection_path.with_name(f"{collection_path.stem}_adata.zarr")


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def create_annbatch_collection(
    input_h5ad: str | Path,
    collection_path: str | Path,
    *,
    zarr_path: str | Path | None = None,
    overwrite: bool = False,
    shuffle: bool = True,
) -> tuple[Path, Path]:
    """Convert an h5ad file and return the collection and backing Zarr paths.

    The input is loaded into memory. Existing artifacts require ``overwrite=True``.
    """
    input_h5ad = Path(input_h5ad).expanduser().resolve()
    requested_collection_path = Path(collection_path).expanduser().resolve()
    if requested_collection_path.suffix == ".zarr":
        collection_path = requested_collection_path
    else:
        collection_path = requested_collection_path.with_suffix(".zarr")
        print(
            f"[checkpoint] Normalized collection path from {requested_collection_path} "
            f"to {collection_path}.",
            flush=True,
        )
    zarr_path = resolve_zarr_path(
        collection_path,
        Path(zarr_path).expanduser().resolve() if zarr_path is not None else None,
    )

    if not input_h5ad.is_file():
        raise FileNotFoundError(f"Input h5ad does not exist or is not a file: {input_h5ad}")
    if collection_path == zarr_path or collection_path.is_relative_to(zarr_path):
        raise ValueError("The collection must be separate from, and not inside, its backing Zarr store.")
    for output_path in (collection_path, zarr_path):
        if input_h5ad == output_path or input_h5ad.is_relative_to(output_path):
            raise ValueError(f"Output path would overwrite the input h5ad: {output_path}")
    if not overwrite and zarr_path.exists():
        raise FileExistsError(f"Backing Zarr store already exists: {zarr_path}. Use overwrite=True to rebuild it.")

    print(f"[checkpoint] Input h5ad: {input_h5ad}", flush=True)
    print(f"[checkpoint] Collection path: {collection_path}", flush=True)
    print(f"[checkpoint] Zarr path: {zarr_path}", flush=True)
    print(
        "[checkpoint] Note: for best throughput, write the collection to local NVMe scratch "
        "rather than NFS.",
        flush=True,
    )

    if not overwrite and collection_path.exists():
        try:
            collection_has_contents = any(collection_path.iterdir())
        except NotADirectoryError as exc:
            raise ValueError(f"Collection path is not a directory: {collection_path}") from exc
        if collection_has_contents:
            raise FileExistsError(
                f"Collection path already exists and is not empty: {collection_path}. "
                "Use overwrite=True to rebuild it."
            )

    import anndata as ad
    import zarr
    from annbatch import DatasetCollection

    if overwrite:
        print("[checkpoint] Removing existing collection artifacts...", flush=True)
        remove_path(zarr_path)
        remove_path(collection_path)

    collection_path.mkdir(parents=True, exist_ok=True)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

    print("[checkpoint] Loading h5ad into memory...", flush=True)
    adata = ad.read_h5ad(input_h5ad)
    print(
        f"[checkpoint] Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes.",
        flush=True,
    )

    print("[checkpoint] Writing zarr store...", flush=True)
    adata.write_zarr(str(zarr_path))

    print("[checkpoint] Building annbatch DatasetCollection...", flush=True)
    collection = DatasetCollection(str(collection_path))
    collection.add_adatas([str(zarr_path)], shuffle=shuffle)
    print("[checkpoint] Finished building annbatch collection.", flush=True)
    return collection_path, zarr_path
