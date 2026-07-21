"""Unit tests for the SFT-only appraisal training wiring.

Guards the launch surface without spending: the rendered appraisal configs parse in
Flash 1.0 as budget-safe SFT runs (no structured_outputs — Flash forbids it on SFT),
the emitted schema artifact matches the appraisal grammar, the offline flash.cost
estimate is well under the $45 cap, and build_all + the bundle ship the appraisal
train split (with its held-out-field split excluded) WITHOUT disturbing the frozen
ops splits."""

from __future__ import annotations

import json

import pytest

from cortesol.core.assessment import assessment_json_schema
from cortesol.train.bundle import PUBLISHED_SPLITS, build_bundle
from cortesol.train.config_artifacts import APPRAISAL_TEMPLATES, render_configs
from cortesol.train.datasets import build_all

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("appraisal-train-data")
    manifest = build_all(path)
    return path, manifest


def test_rendered_appraisal_configs_are_budget_safe_sft(tmp_path):
    from flash.cost import estimate_cost, runconfig_from_spec
    from flash.schema import spec_from_file

    paths = render_configs(tmp_path, environment_id="owner/cortesol-appraisal", target="appraisal")
    assert set(paths) == set(APPRAISAL_TEMPLATES)
    # the appraisal grammar is emitted as the pinned schema artifact
    assert json.loads((tmp_path / "assessment.schema.json").read_text()) == assessment_json_schema()

    total = 0.0
    for path in paths.values():
        spec = spec_from_file(str(path))
        assert spec.algorithm == "sft"
        assert spec.model == "Qwen/Qwen3.5-4B"
        assert spec.thinking is False
        # Flash rejects structured_outputs on SFT; the grammar rides on the gold rows.
        assert not spec.train.structured_outputs
        total += estimate_cost(runconfig_from_spec(spec)).total_usd
    assert total < 45.0  # a few dollars for 213 examples — nowhere near the cap

    sft = spec_from_file(str(paths["appraisal_sft"]))
    assert sft.train.max_examples == 213
    assert sft.train.max_steps == 200
    assert sft.train.save_at_steps == (200,)
    assert estimate_cost(runconfig_from_spec(sft)).total_usd < 10.0


def test_appraisal_split_is_built_and_registered_separately(built):
    path, manifest = built
    # the ops files map is untouched (add, don't replace) — appraisal lives elsewhere
    assert "appraisal_sft_train" not in manifest["files"]
    appraisal = manifest["appraisal"]
    assert appraisal["files"]["appraisal_sft_train"]["rows"] == 213
    assert appraisal["files"]["appraisal_sft_heldout"]["rows"] == 31
    assert set(appraisal["heldout_fields"]) == {"math", "q-bio"}
    assert set(appraisal["train_fields"]).isdisjoint(appraisal["heldout_fields"])
    assert (path / "appraisal_sft_train.jsonl").is_file()
    # every shipped row is Flash transport shape and leaks no held-out field tag
    for line in (path / "appraisal_sft_train.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert set(row) == {"input", "output", "metadata"}
        assert row["metadata"]["field"] not in appraisal["heldout_fields"]


def test_bundle_ships_appraisal_train_but_not_heldout(built):
    path, _ = built
    assert "appraisal_sft_train" in PUBLISHED_SPLITS
    bundle_dir = path.parent / "appraisal-bundle"
    manifest = build_bundle(bundle_dir, data_dir=path)
    shipped = {item.name for item in bundle_dir.rglob("*") if item.is_file()}
    assert "appraisal_sft_train.jsonl" in shipped
    assert "appraisal_sft_heldout.jsonl" not in shipped
    assert manifest["files"]["dataset/appraisal_sft_train.jsonl"]
    public = json.loads((bundle_dir / "dataset-manifest.json").read_text())
    assert public["files"]["appraisal_sft_train"]["rows"] == 213
