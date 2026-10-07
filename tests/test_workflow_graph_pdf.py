"""
The workflow graph must mirror the SOP PDFs in data/ (Goal A, Tugas 1):
step counts, actor = grammatical subject, verbatim evidence per step,
and one linear NEXT_STEP path per SOP.
"""
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph_sop.workflow_graph import SOPWorkflowGraph

DATA_DIR = ROOT / "data"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=None)
def pdf_text(pdf_name: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(DATA_DIR / pdf_name))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def numbered_steps_in_pdf(pdf_name: str) -> int:
    """Count the consecutive '1.', '2.', ... items that start a line in the PDF."""
    numbers = [int(n) for n in re.findall(r"(?m)^\s*(\d{1,2})\.\s", pdf_text(pdf_name))]
    count = 0
    for n in numbers:
        if n == count + 1:
            count = n
    return count


@pytest.fixture(scope="module")
def wg():
    return SOPWorkflowGraph()


def _sops(wg):
    return list(wg.sop_metadata.values())


def test_every_sop_points_to_an_existing_pdf(wg):
    pdfs = {p.name for p in DATA_DIR.glob("*.pdf")}
    assert {s["source_pdf"] for s in _sops(wg)} == pdfs


def test_step_count_matches_numbered_steps_in_pdf(wg):
    for sop in _sops(wg):
        assert len(sop["steps"]) == numbered_steps_in_pdf(sop["source_pdf"]), sop["id"]


def test_total_step_count_is_44(wg):
    assert sum(len(s["steps"]) for s in _sops(wg)) == 44
    assert len([n for n, d in wg.graph.nodes(data=True) if d["type"] == "SOP_STEP"]) == 44


def test_step_numbers_are_consecutive(wg):
    for sop in _sops(wg):
        assert [s["step_num"] for s in sop["steps"]] == list(range(1, len(sop["steps"]) + 1)), sop["id"]


def test_every_step_evidence_appears_in_its_pdf(wg):
    for sop in _sops(wg):
        text = normalize_ws(pdf_text(sop["source_pdf"]))
        for step in sop["steps"]:
            ev = normalize_ws(step["evidence"])
            assert ev, (sop["id"], step["step_num"])
            assert ev in text, (sop["id"], step["step_num"], ev)


def test_max_duration_evidence_appears_in_pdf(wg):
    for sop in _sops(wg):
        ev = sop["max_duration_evidence"]
        if sop["max_duration"] is None:
            assert ev is None, sop["id"]
            continue
        assert normalize_ws(ev) in normalize_ws(pdf_text(sop["source_pdf"])), sop["id"]


def test_durations_only_where_pdf_gives_waktu(wg):
    for sop in _sops(wg):
        has_waktu = "Waktu:" in pdf_text(sop["source_pdf"])
        for step in sop["steps"]:
            if "duration" in step:
                assert has_waktu, (sop["id"], step["step_num"])
                assert normalize_ws(f"Waktu: {step['duration']}") in normalize_ws(pdf_text(sop["source_pdf"]))


def test_outputs_come_from_pdf_output_lines(wg):
    for sop in _sops(wg):
        text = normalize_ws(pdf_text(sop["source_pdf"]))
        for step in sop["steps"]:
            if step["output"]:
                assert normalize_ws(f"Output: {step['output']}") in text, (sop["id"], step["step_num"])


def test_cuti_steps_2_and_3_are_performed_by_mahasiswa(wg):
    ctx = wg.get_workflow_context("SOP_CUTI_AKADEMIK")
    assert "2. Pelaksana / Aktor: Mahasiswa" in ctx
    assert "3. Pelaksana / Aktor: Mahasiswa" in ctx
    # The addressed officials stay in the action text.
    steps = wg.sop_metadata["SOP_CUTI_AKADEMIK"]["steps"]
    assert "Ketua Program Studi" in steps[1]["action"]
    assert "Dekan" in steps[2]["action"]


def test_each_sop_is_one_linear_next_step_path(wg):
    for sop in _sops(wg):
        step_ids = [f"{sop['id']}_STEP_{s['step_num']}" for s in sop["steps"]]
        next_edges = [
            (u, v) for u, v, d in wg.graph.edges(data=True)
            if d.get("relation") == "NEXT_STEP" and u in step_ids
        ]
        path = nx.DiGraph(next_edges)
        path.add_nodes_from(step_ids)
        assert nx.is_directed_acyclic_graph(path), sop["id"]
        assert len(next_edges) == len(step_ids) - 1, sop["id"]
        assert list(nx.topological_sort(path)) == step_ids, sop["id"]
        assert all(path.in_degree(n) <= 1 and path.out_degree(n) <= 1 for n in step_ids), sop["id"]
        assert all(v in step_ids for _, v in next_edges), sop["id"]


def test_graph_sha256_is_deterministic(wg):
    assert wg.graph_sha256() == SOPWorkflowGraph().graph_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", wg.graph_sha256())


def test_v1_snapshot_is_preserved():
    import json
    snap = json.loads((ROOT / "docs" / "workflow_graph_v1.json").read_text(encoding="utf-8"))
    assert snap["n_steps_total"] == 42
    assert snap["graph_sha256"] != SOPWorkflowGraph().graph_sha256()
