"""
test_compare.py — ModelComparison tests.

Ported from tests/legacy_tests/test_compare.py; updated for the current API
and project layout after refactoring.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psse_model_util.common.constants import RangeFilterType
from psse_model_util.common.dirs import clear_cache
from psse_model_util.compare import ModelComparison
from psse_model_util.model import Model

DATA_DIR = Path(__file__).resolve().parent / "data"

# Area numbers that exist in Model_1.raw / Model_2.raw
INCLUDE_AREAS = {1: "CENTRAL", 2: "EAST", 3: "CENTRAL_DC",
                 4: "EAST_COGEN1", 5: "WEST", 6: "EAST_COGEN2"}

# Wide voltage range so all buses in the fixture pass the filter
DEFAULT_KV_FILTER = RangeFilterType(1, 10_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buses_in_alt_paths(alt_paths) -> set[int]:
    """Collect every bus number appearing in a sectionalization/bypass row.

    ``alt_paths`` is a list of paths; each path is a list of node tuples such
    as ``('bus', 3013)``. Returns the set of bus numbers across all paths.
    """
    buses: set[int] = set()
    for path in alt_paths:
        for node in path:
            if isinstance(node, tuple) and len(node) >= 2 and node[0] == "bus":
                buses.add(node[1])
    return buses


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_models():
    """Load Model_1 and Model_2 from the test fixtures (cache cleared first)."""
    clear_cache()
    model1 = Model(DATA_DIR / "Model_1.raw")
    model2 = Model(DATA_DIR / "Model_2.raw")
    return model1, model2


@pytest.fixture(scope="module")
def model_comparison(raw_models):
    model1, model2 = raw_models
    return ModelComparison(model1, model2)


@pytest.fixture(scope="module")
def compared(model_comparison):
    """Run both comparisons once and return the results."""
    df_comp = model_comparison.compare_network_dfs()
    graph_comp = model_comparison.compare_graph()
    return df_comp, graph_comp


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_init(model_comparison):
    assert isinstance(model_comparison, ModelComparison)
    assert model_comparison.model1.name == "Model_1"
    assert model_comparison.model2.name == "Model_2"


# ---------------------------------------------------------------------------
# bus_num_changes
# ---------------------------------------------------------------------------

def test_bus_num_changes(model_comparison):
    changes = model_comparison.bus_num_changes()
    assert isinstance(changes, pd.DataFrame)
    assert not changes.empty
    assert {"ibus_model1", "ibus_model2"}.issubset(changes.columns)
    # Known renames documented in Model_1 and 2 differences.txt
    assert any((changes["ibus_model1"] == 101) & (changes["ibus_model2"] == 111))
    assert any((changes["ibus_model1"] == 213) & (changes["ibus_model2"] == 219))


# ---------------------------------------------------------------------------
# compare_network_dfs
# ---------------------------------------------------------------------------

class TestCompareNetworkDfs:

    def test_returns_dict(self, compared):
        df_comp, _ = compared
        assert isinstance(df_comp, dict)

    def test_required_sections_present(self, compared):
        df_comp, _ = compared
        for section in ("bus", "generator", "load", "acline", "transformer"):
            assert section in df_comp
            assert isinstance(df_comp[section], pd.DataFrame)

    def test_presence_column_exists(self, compared):
        df_comp, _ = compared
        for section in ("bus", "generator", "acline", "load", "transformer"):
            assert "presence" in df_comp[section].columns

    def test_bus_added_and_removed(self, compared):
        df_comp, _ = compared
        bus_df = df_comp["bus"]
        # Bus 156 added in Model_2; Bus 155 removed
        assert any(bus_df["presence"] == "model2_only")
        assert any(bus_df["presence"] == "model1_only")


# ---------------------------------------------------------------------------
# compare_graph
# ---------------------------------------------------------------------------

class TestCompareGraph:

    def test_returns_dict_with_required_keys(self, compared):
        _, graph_comp = compared
        assert isinstance(graph_comp, dict)
        for key in ("added_edges", "removed_edges",
                    "path_sectionalizations", "path_bypasses"):
            assert key in graph_comp

    def test_added_and_removed_nodes(self, compared):
        _, graph_comp = compared
        # Ground truth for the expanded Model_1/Model_2 fixtures. Model_2 adds
        # buses 111, 156, 160, 161, 210, 219, 3013, 3014, 3111 (+ their
        # gens/loads/shunts/transformers) and deletes buses 101, 152, 213,
        # 2000, 2001, 3001, 3010 (+ their attached equipment).
        assert len(graph_comp["added_nodes"]) == 21
        assert len(graph_comp["removed_nodes"]) == 25

    def test_one_sectionalization(self, compared):
        _, graph_comp = compared
        sec = graph_comp["path_sectionalizations"]
        assert len(sec) == 1, "Expected exactly one path sectionalization"

    def test_sectionalization_involves_bus_3008(self, compared):
        _, graph_comp = compared
        sec = graph_comp["path_sectionalizations"]
        paths_with_3008 = [
            row[0] for row in sec.values
            if ("bus", 3008) in row[0]
        ]
        assert len(paths_with_3008) > 0

    def test_sectionalization_alt_path_traverses_new_buses(self, compared):
        """The 3008-3009 line was split by inserting new buses 3013 and 3014.

        Model_1:  3008 ---------------------------- 3009
        Model_2:  3008 --- 3013 --- 3014 --- 3009   (3013, 3014 are new)

        The single sectionalization's original edge is 3008-3009 and its
        alternate path must route through both newly-added intermediate buses.
        """
        _, graph_comp = compared
        sec = graph_comp["path_sectionalizations"]
        assert len(sec) == 1
        row = sec.iloc[0]
        assert set(row["original_path"]) == {("bus", 3008), ("bus", 3009)}
        alt_buses = _buses_in_alt_paths(row["alternate_paths"])
        assert {3013, 3014}.issubset(alt_buses)
        # New intermediate buses must not have existed in Model_1.
        assert {("bus", 3013), ("bus", 3014)}.issubset(graph_comp["added_nodes"])

    def test_no_forward_bypasses(self, compared):
        """No graph-level bypass is detectable in the model1->model2 direction.

        The two documented AC-line merges do not surface as topological
        bypasses, by design:
          * 213-2000-214 -> 219-214 is masked by the concurrent 213->219 bus
            rename (bus 219 does not exist in Model_1, so the merged edge has no
            Model_1 endpoint to trace an alternate path from).
          * 3003-2001-3005 -> 3003-3005 names the merged line CIRCUIT 2, but
            Model_1 already carries a parallel 3003-3005 circuit 1, so the
            simple graph already had that edge (no *new* edge is created).
        A genuine bypass is exercised by the reversed comparison below.
        """
        _, graph_comp = compared
        assert len(graph_comp["path_bypasses"]) == 0


# ---------------------------------------------------------------------------
# compare_graph — reversed (bypass)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reversed_graph_comp(raw_models):
    """Graph comparison with the models swapped (Model_2 -> Model_1).

    Reversing the comparison turns the Model_1->Model_2 line *split*
    (sectionalization) into a line *merge* (bypass): the 3008-3013-3014-3009
    chain in Model_2 collapses back to the direct 3008-3009 edge in Model_1.
    """
    model1, model2 = raw_models
    return ModelComparison(model2, model1).compare_graph()


class TestCompareGraphBypass:
    """The reversed comparison exercises path_bypasses (line merge)."""

    def test_one_bypass(self, reversed_graph_comp):
        byp = reversed_graph_comp["path_bypasses"]
        assert len(byp) == 1, "Expected exactly one path bypass"

    def test_no_sectionalizations_in_reverse(self, reversed_graph_comp):
        # The split, viewed backwards, is a merge — not a split.
        assert len(reversed_graph_comp["path_sectionalizations"]) == 0

    def test_bypass_merges_3008_3009(self, reversed_graph_comp):
        """The merged edge is 3008-3009; its Model_2 path ran through the
        now-removed intermediate buses 3013 and 3014."""
        byp = reversed_graph_comp["path_bypasses"]
        row = byp.iloc[0]
        assert set(row["original_path"]) == {("bus", 3008), ("bus", 3009)}
        alt_buses = _buses_in_alt_paths(row["alternate_paths"])
        assert {3013, 3014}.issubset(alt_buses)
        # From the reversed viewpoint the intermediate buses are *removed*.
        assert {("bus", 3013), ("bus", 3014)}.issubset(
            reversed_graph_comp["removed_nodes"]
        )

    def test_bypass_named_columns_present(self, reversed_graph_comp):
        byp = reversed_graph_comp["path_bypasses"]
        for col in ("original_path_named", "alternate_paths_named"):
            assert col in byp.columns
        assert "3008" in byp.iloc[0]["original_path_named"]


# ---------------------------------------------------------------------------
# New structural scenarios enabled by the expanded fixtures
# ---------------------------------------------------------------------------

class TestNewScenarios:
    """DataFrame-level coverage for scenarios documented in
    ``Model_1 and 2 differences.txt`` that the expanded Model_2 now exercises."""

    @pytest.mark.parametrize("bus_num", [2000, 2001, 3010])
    def test_bus_deletions(self, compared, bus_num):
        """Buses 2000 (merge + gen removal), 2001 (merge), 3010 (3W->2W)."""
        df_comp, _ = compared
        bus_df = df_comp["bus"]
        assert bus_num in bus_df.index
        assert bus_df.loc[bus_num, "presence"] == "model1_only"

    @pytest.mark.parametrize("bus_num", [3013, 3014, 156, 161, 210])
    def test_bus_additions(self, compared, bus_num):
        """New buses: 3013/3014 (line split), 156 (new load/gen/branch),
        161 (new shunts), 210 (2W->3W transformer)."""
        df_comp, _ = compared
        bus_df = df_comp["bus"]
        assert bus_num in bus_df.index
        assert bus_df.loc[bus_num, "presence"] == "model2_only"

    def test_load_id_change(self, compared):
        """Load at bus 153 changed loadid from '1' to '10'."""
        df_comp, _ = compared
        load = df_comp["load"]
        assert load.loc[(153, "1"), "presence"] == "model1_only"
        assert load.loc[(153, "10"), "presence"] == "model2_only"

    def test_load_added_and_removed(self, compared):
        """Load (156,'AA') added; load (205,'C') removed."""
        df_comp, _ = compared
        load = df_comp["load"]
        assert load.loc[(156, "AA"), "presence"] == "model2_only"
        assert load.loc[(205, "C"), "presence"] == "model1_only"

    def test_load_id_rename_201(self, compared):
        """Load at bus 201 changed loadid from 'SC' to 'SA'."""
        df_comp, _ = compared
        load = df_comp["load"]
        assert load.loc[(201, "SC"), "presence"] == "model1_only"
        assert load.loc[(201, "SA"), "presence"] == "model2_only"

    def test_generator_deleted_with_bus(self, compared):
        """The generator at bus 2000 is removed along with its bus (merge)."""
        df_comp, _ = compared
        gen = df_comp["generator"]
        assert gen.loc[(2000, "1"), "presence"] == "model1_only"

    def test_generator_added(self, compared):
        """New generator 'GG' added at new bus 156."""
        df_comp, _ = compared
        gen = df_comp["generator"]
        assert gen.loc[(156, "GG"), "presence"] == "model2_only"

    def test_generator_bus_and_id_rename(self, compared):
        """Generator moved from (101,'1') to (111,'A') with the 101->111 bus
        rename."""
        df_comp, _ = compared
        gen = df_comp["generator"]
        assert gen.loc[(101, "1"), "presence"] == "model1_only"
        assert gen.loc[(111, "A"), "presence"] == "model2_only"

    @pytest.mark.parametrize("old,new", [(101, 111), (213, 219), (152, 160)])
    def test_bus_num_changes_detected(self, model_comparison, old, new):
        """Renumberings that preserve the bus name+area are detected."""
        changes = model_comparison.bus_num_changes()
        assert any((changes["ibus_model1"] == old)
                   & (changes["ibus_model2"] == new)), f"{old}->{new} not found"

    def test_bus_num_change_with_name_change_not_detected(self, model_comparison):
        """Bus 3001->3111 also changed name (MINE->YOURS). bus_num_changes()
        inner-joins on name+area, so a simultaneous number+name change is *not*
        matched as a renumbering — a documented limitation of the heuristic."""
        changes = model_comparison.bus_num_changes()
        assert not any((changes["ibus_model1"] == 3001)
                       & (changes["ibus_model2"] == 3111))


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------

def test_to_csv(model_comparison, compared, tmp_path):
    model_comparison.csv_folder = tmp_path
    model_comparison.to_csv(df_comparison_to_csv=True, graph_comparison_to_csv=True)
    assert (tmp_path / "network_bus.csv").exists()
    assert (tmp_path / "graph_added_edges.csv").exists()
    assert (tmp_path / "graph_removed_edges.csv").exists()


# ---------------------------------------------------------------------------
# filter_by_area
# ---------------------------------------------------------------------------

def test_filter_by_area_default(raw_models):
    model1, model2 = raw_models
    comp = ModelComparison(model1, model2)
    filtered = comp.model1.filter_by_area(areas=INCLUDE_AREAS)
    assert set(filtered.network.bus["area"]).issubset(set(INCLUDE_AREAS.keys()))


def test_filter_by_area_custom(raw_models):
    model1, model2 = raw_models
    comp = ModelComparison(model1, model2)
    custom_areas = {1: "Area1", 2: "Area2"}
    filtered = comp.model1.filter_by_area(areas=custom_areas)
    assert set(filtered.network.bus["area"]) == {1, 2}


def test_filter_by_area_empty_raises(raw_models):
    model1, model2 = raw_models
    comp = ModelComparison(model1, model2)
    with pytest.raises(ValueError):
        comp.model1.filter_by_area(areas={})


# ---------------------------------------------------------------------------
# bus_kv_filter
# ---------------------------------------------------------------------------

def test_bus_kv_filter(model_comparison, compared):
    filtered_buses = model_comparison.bus_kv_filter()
    assert isinstance(filtered_buses, list)
    assert all(isinstance(bus_id, int) for bus_id in filtered_buses)
    bus_index = model_comparison.model1.network.bus.index
    for bus_id in filtered_buses:
        if bus_id in bus_index:
            baskv = model_comparison.model1.network.bus.loc[bus_id, "baskv"]
            assert DEFAULT_KV_FILTER.min <= baskv <= DEFAULT_KV_FILTER.max


# ---------------------------------------------------------------------------
# query_network_df_comparison
# ---------------------------------------------------------------------------

def test_query_network_df_comparison_returns_dfs(model_comparison, compared):
    filtered = model_comparison.query_network_df_comparison()
    assert isinstance(filtered, dict)
    assert all(isinstance(df, pd.DataFrame) for df in filtered.values())


def test_query_network_df_comparison_missing_bus_raises(model_comparison, compared):
    # Re-run compare so the dict is fresh before we mutate it
    model_comparison.compare_network_dfs()
    del model_comparison.network_df_comparison["bus"]
    with pytest.raises(KeyError):
        model_comparison.query_network_df_comparison()


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def test_bus_num_changes_performance(raw_models):
    """bus_num_changes on a 100k-bus model should complete in under 1 second."""
    model1, _ = raw_models
    large_model = model1.copy()
    num_buses = 100_000
    large_model.network.bus = pd.DataFrame({
        "ibus": range(num_buses),
        "name": [f"Bus{i}" for i in range(num_buses)],
        "area": [i % 10 for i in range(num_buses)],
        "baskv": np.random.uniform(100, 500, num_buses),
    }).set_index("ibus")
    comp = ModelComparison(large_model, large_model)
    start = time.perf_counter()
    comp.bus_num_changes()
    assert time.perf_counter() - start < 1.0
