"""End-to-end tests for psse_model_util.util.contingency_util.

These exercise the full create_area_con_files() pipeline against the synthetic
Model_1.raw network and tests/data/synthetic_contingencies.con, with BOTH the
RAW and the .con file placed in a single input folder (the supported layout).

Coverage of the recent fixes:
  * contingency_name is the first column of both output CSVs.
  * The name is trimmed of surrounding whitespace and quotes (", ').
  * KV_FILTER default is (115, 765); mid-voltage branch-only contingencies are
    removed while MACHINE/LOAD contingencies bypass the kV filter.
  * The RAW file sharing the input folder is NOT parsed as a contingency file.
  * The trailing raw_*.csv exports run (the create_area_con_files tail that
    previously referenced the __main__-only `args` global).
  * "Bad" contingencies are validated with two ISOLATED failure modes:
      - C_BAD_BUS references a bus number (999999) absent from the model,
        with no branch involved (undefined_branches stays empty).
      - C_BAD_BRANCH references a branch (151-153) between two buses that
        both exist in the model but are several hops apart with no direct
        line between them (undefined_buses stays empty).
    This distinguishes "missing bus" detection from "missing branch"
    detection instead of conflating them via a single nonexistent bus.
"""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from psse_model_util.util import contingency_util as cu

DATA_DIR = Path(__file__).resolve().parent / "data"

EXPECTED_NAMES = {
    "C_GOOD_HV", "C_LOAD_LV", "C_UNQUOTED_GOOD",
    "C_KV_REMOVED", "C_BAD_BUS", "C_BAD_BRANCH",
}


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    """A single input folder holding BOTH the RAW and the .con file (fix 4)."""
    d = tmp_path / "inputs"
    d.mkdir()
    shutil.copy(DATA_DIR / "Model_1.raw", d / "Model_1.raw")
    shutil.copy(DATA_DIR / "synthetic_contingencies.con", d / "synthetic_contingencies.con")
    return d


@pytest.fixture
def output_dir(input_dir: Path, tmp_path: Path) -> Path:
    """Run the pipeline once; return the populated output folder."""
    out = tmp_path / "output"
    cu.create_area_con_files(
        raw_file=None,            # discover the RAW inside input_dir (fix 4)
        input_folder=input_dir,
        output_folder=out,
        kv_filter=(115, 765),
        delete_old_output=True,
    )
    return out


def _read(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def test_all_input_csv_name_is_first_column(output_dir: Path):
    df = _read(output_dir / "all_input_contingencies.csv")
    assert df.columns[0] == "contingency_name"
    assert set(df["contingency_name"]) == EXPECTED_NAMES


def test_name_quote_trimming(output_dir: Path):
    """Double-quoted, single-quoted, and unquoted names all arrive clean."""
    names = set(_read(output_dir / "all_input_contingencies.csv")["contingency_name"])
    assert "C_GOOD_HV" in names        # source: "C_GOOD_HV"
    assert "C_LOAD_LV" in names        # source: 'C_LOAD_LV'
    assert "C_UNQUOTED_GOOD" in names  # source: C_UNQUOTED_GOOD (no quotes)


def test_raw_not_parsed_as_contingency(output_dir: Path):
    """The RAW shares the folder; it must not add spurious contingency rows."""
    df = _read(output_dir / "all_input_contingencies.csv")
    assert len(df) == len(EXPECTED_NAMES)


def test_kv_removed_csv_has_name_and_only_midvoltage(output_dir: Path):
    df = _read(output_dir / "kv_removed_input_contingencies.csv")
    assert df.columns[0] == "contingency_name"
    # Only the 230 kV branch-only contingency is dropped at 115/765; the
    # MACHINE/LOAD contingencies bypass the kV filter.
    assert set(df["contingency_name"]) == {"C_KV_REMOVED"}


def test_undefined_bus_isolated_from_undefined_branch(output_dir: Path):
    """C_BAD_BUS: a bus number absent from the model, no branch involved."""
    df = _read(output_dir / "all_input_contingencies.csv").set_index("contingency_name")
    row = df.loc["C_BAD_BUS"]
    assert row["undefined_buses"] == "(999999,)"
    assert row["undefined_branches"] == "()"


def test_undefined_branch_between_two_real_distant_buses(output_dir: Path):
    """C_BAD_BRANCH: buses 151 and 153 both exist (several hops apart, no
    direct line between them) so undefined_buses stays empty while the
    branch itself is flagged undefined."""
    df = _read(output_dir / "all_input_contingencies.csv").set_index("contingency_name")
    row = df.loc["C_BAD_BRANCH"]
    assert row["undefined_buses"] == "()"
    assert "151" in row["undefined_branches"]
    assert "153" in row["undefined_branches"]


def test_good_bad_split(output_dir: Path):
    good = (output_dir / "CENTRAL.con").read_text()
    bad = (output_dir / "CENTRAL_bad.con").read_text()

    assert "C_GOOD_HV" in good
    assert "C_LOAD_LV" in good
    assert "C_UNQUOTED_GOOD" in good
    assert "C_BAD_BUS" not in good
    assert "C_BAD_BRANCH" not in good

    assert "C_BAD_BUS" in bad
    assert "999999" in bad  # the undefined bus is surfaced in the bad file
    assert "C_BAD_BRANCH" in bad
    assert "151" in bad and "153" in bad  # the undefined branch is surfaced

    # The kv-removed contingency never reaches the area files.
    assert "C_KV_REMOVED" not in good
    assert "C_KV_REMOVED" not in bad


def test_raw_reference_csvs_written(output_dir: Path):
    """The pipeline tail (formerly broken by the `args` global) runs cleanly."""
    for name in (
        "raw_bus.csv",
        "raw_acline.csv",
        "raw_generator.csv",
        "raw_load.csv",
        "raw_transformer.csv",
    ):
        assert (output_dir / name).exists(), f"missing {name}"
