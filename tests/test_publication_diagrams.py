"""The recorded-state renderer must retain its source-integrity checks."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('crosses_renderer', ROOT / 'figsrc/figure1/render_crosses.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
BUNDLE = ROOT / 'results/figure1_crosses'


def test_archived_cross_states_accept_only_audited_sources():
    evidence = json.loads((BUNDLE / 'figure_evidence.json').read_text())
    renderer.verify_model_sources(evidence['source_sha256'], BUNDLE)


def test_archived_cross_states_reject_unknown_current_code(monkeypatch):
    evidence = json.loads((BUNDLE / 'figure_evidence.json').read_text())
    current = renderer.source_hashes()
    current['figure1_crosses.py'] = '0' * 64
    monkeypatch.setattr(renderer, 'source_hashes', lambda: current)
    with pytest.raises(ValueError, match='source differs'):
        renderer.verify_model_sources(evidence['source_sha256'], BUNDLE)


def test_archived_cross_states_reject_unknown_training_code():
    archived = json.loads((BUNDLE / 'figure_evidence.json').read_text())['source_sha256']
    archived['figure1_crosses.py'] = '0' * 64
    with pytest.raises(ValueError, match='source differs'):
        renderer.verify_model_sources(archived, BUNDLE)
