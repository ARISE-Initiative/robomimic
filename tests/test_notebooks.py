"""
Regression tests for the example notebooks. Excludes stdout output by
default (pass --verbose to see stdout output).

These tests execute the code cells of the notebooks with network access
stubbed out, to catch bitrot such as the dataset download cell in
datasets.ipynb passing DATASET_REGISTRY urls (which are Hugging Face
repo-relative paths as of the hosting migration) to FileUtils.download_url,
which expects absolute urls.
"""
import argparse
import json
import os
import sys
import traceback
from termcolor import colored

import robomimic
import robomimic.utils.file_utils as FileUtils


def get_code_cells(notebook_name):
    """
    Return the source of all code cells in the notebook at
    examples/notebooks/@notebook_name as a list of strings.
    """
    notebook_path = os.path.join(robomimic.__path__[0], "../examples/notebooks/{}".format(notebook_name))
    with open(notebook_path, "r") as f:
        notebook = json.load(f)
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_datasets_notebook_download_cell():
    """
    Executes the dataset download cell of examples/notebooks/datasets.ipynb
    with FileUtils.download_file_from_hf replaced by a stub (no network).

    The DATASET_REGISTRY urls for sim datasets are repo-relative Hugging Face
    paths, so the cell must go through FileUtils.download_file_from_hf (with
    robomimic.HF_REPO_ID); passing them to FileUtils.download_url raises
    "ValueError: unknown url type". Also verifies that the file name the cell
    asserts on matches what the registry download actually produces.
    """
    # find the download cell (the only code cell that indexes DATASET_REGISTRY)
    cells = get_code_cells("datasets.ipynb")
    download_cells = [cell for cell in cells if 'DATASET_REGISTRY[task][dataset_type][hdf5_type]' in cell]
    assert len(download_cells) == 1, "expected exactly one download cell in datasets.ipynb, got {}".format(len(download_cells))
    cell = download_cells[0]

    # stub out the HF download: record the call and materialize the file that
    # download_file_from_hf would write (<download_dir>/<basename of filename>)
    hf_calls = []

    def fake_download_file_from_hf(repo_id, filename, download_dir, check_overwrite=True):
        hf_calls.append(dict(repo_id=repo_id, filename=filename, download_dir=download_dir))
        file_to_write = os.path.join(download_dir, os.path.basename(filename))
        with open(file_to_write, "w") as f:
            f.write("stub\n")

    from robomimic import DATASET_REGISTRY
    saved_download_file_from_hf = FileUtils.download_file_from_hf
    FileUtils.download_file_from_hf = fake_download_file_from_hf
    try:
        # executing the cell asserts that the downloaded dataset exists
        exec(compile(cell, "<datasets.ipynb download cell>", "exec"), {})
    finally:
        FileUtils.download_file_from_hf = saved_download_file_from_hf

    # the cell must use the Hugging Face downloader with the registry entry
    assert len(hf_calls) == 1, "download cell did not call FileUtils.download_file_from_hf exactly once (got {} calls)".format(len(hf_calls))
    task, dataset_type, hdf5_type = "lift", "ph", "low_dim"
    expected_filename = DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"]
    assert hf_calls[0]["repo_id"] == robomimic.HF_REPO_ID, "download cell used repo_id {} instead of robomimic.HF_REPO_ID".format(hf_calls[0]["repo_id"])
    assert hf_calls[0]["filename"] == expected_filename, "download cell used filename {} instead of the registry entry {}".format(hf_calls[0]["filename"], expected_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="don't suppress stdout during tests",
    )
    args = parser.parse_args()

    failed = False
    try:
        test_datasets_notebook_download_cell()
        print("test-datasets-notebook-download-cell: {}".format(colored("passed", "green")))
    except Exception:
        failed = True
        print("test-datasets-notebook-download-cell: {}".format(colored("failed - traceback below", "red")))
        traceback.print_exc()
    sys.exit(1 if failed else 0)
