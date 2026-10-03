"""Dependency-free checks of the public functions and workflow integration."""

import contextlib
import importlib
import io
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import panelvi

prepare = importlib.import_module("panelvi.create_annbatch_collection")
select = importlib.import_module("panelvi.run_panel_selection")


class PackageTests(unittest.TestCase):
    def test_metadata_and_public_functions(self):
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        self.assertEqual(metadata["version"], panelvi.__version__)
        self.assertEqual(metadata["requires-python"], ">=3.12")
        self.assertNotIn("scripts", metadata)
        self.assertIs(panelvi.create_annbatch_collection, prepare.create_annbatch_collection)
        self.assertIs(panelvi.run_panel_selection, select.run_panel_selection)

    def test_default_collection_layout_and_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            collection = root / "reference.zarr"
            backing = prepare.resolve_zarr_path(collection, None)
            self.assertEqual(backing, root / "reference_adata.zarr")
            (collection / "dataset_0").mkdir(parents=True)
            backing.mkdir()
            (backing / "zarr.json").write_text('{}')
            self.assertEqual(
                select.resolve_collection_and_zarr_paths(collection), (collection, backing)
            )
            output = root / "panel.txt"
            output.write_text("keep this panel\n")
            with self.assertRaises(FileExistsError):
                panelvi.run_panel_selection(str(collection), output=str(output))
            self.assertEqual(output.read_text(), "keep this panel\n")

    def test_collection_rejects_existing_backing_and_input_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "input.h5ad"
            source.write_bytes(b"input sentinel")
            collection = root / "reference.zarr"
            backing = root / "reference_adata.zarr"
            backing.mkdir()
            sentinel = backing / "keep"
            sentinel.write_text("existing data")
            with self.assertRaises(FileExistsError):
                panelvi.create_annbatch_collection(source, collection)
            with self.assertRaises(ValueError):
                panelvi.create_annbatch_collection(
                    source, collection, zarr_path=source, overwrite=True
                )
            with self.assertRaises(ValueError):
                panelvi.create_annbatch_collection(
                    source, collection, zarr_path=collection, overwrite=True
                )
            self.assertEqual(source.read_bytes(), b"input sentinel")
            self.assertEqual(sentinel.read_text(), "existing data")
            self.assertFalse(collection.exists())

    def test_selection_forwards_arguments_writes_genes_and_returns_result(self):
        phases = [SimpleNamespace() for _ in range(3)]
        result = {
            "selected_genes": ["Gad1", "Slc17a7"],
            "importance_scores": Mock(),
            "elbo_history": Mock(),
            "scvi_handoff_dir": "handoff",
        }
        training = SimpleNamespace(
            default_training_phases=Mock(return_value=phases),
            train_sparse_hierarchical_scanvi=Mock(return_value=result),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            backing = root / "reference_adata.zarr"
            backing.mkdir()
            (backing / "zarr.json").write_text('{}')
            output = root / "results" / "panel.txt"
            with (
                patch.dict("sys.modules", {"panelvi.hierarchical_scanvi_panel": training}),
                patch.object(select, "load_collection_adata", return_value=SimpleNamespace(n_obs=3, n_vars=2)),
                patch.object(select, "plot_elbo_history"),
                patch.object(select, "plot_gate_histogram"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                actual = panelvi.run_panel_selection(
                    str(backing), output=str(output), top_k=2,
                    devices=[0, 1], tau_start=0.7, lambda_sparse_start=0.002,
                    multiprocessing_context="spawn", quoted=True,
                )
            self.assertIs(actual, result)
            self.assertEqual(output.read_text(), '"Gad1"\n"Slc17a7"\n')
            kwargs = training.train_sparse_hierarchical_scanvi.call_args.kwargs
            self.assertEqual(kwargs["top_k"], 2)
            self.assertEqual(kwargs["devices"], [0, 1])
            self.assertEqual(kwargs["backing_zarr_path"], str(backing))
            self.assertEqual(kwargs["datasplitter_kwargs"]["multiprocessing_context"], "spawn")
            self.assertEqual(phases[1].tau_start, 0.7)
            self.assertEqual(phases[1].lambda_sparse_start, 0.002)
            result["elbo_history"].to_csv.assert_called_once_with(
                output.with_name("panel_elbo.tsv"), sep="\t", index=False
            )
            result["importance_scores"].to_csv.assert_called_once_with(
                output.with_name("panel_gene_scores.csv"), index=False
            )


if __name__ == "__main__":
    unittest.main()
