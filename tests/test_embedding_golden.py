"""
Golden embedding fixture test.

Reference vectors generated with sentence-transformers/all-MiniLM-L6-v2
via optimum ONNX on Python 3.11.15. Saved to embedding_golden_reference.npz.

Asserts:
1. Ordering around 0.95 threshold (the property that makes the cache safe)
2. Strong paraphrases above 0.95 (cache hits)
3. Entity substitutions below 0.95 (must NOT be cache hits)

Does NOT assert 1e-7 vector equality — too tight, fires on legitimate
onnxruntime build differences. Asserts ordering and threshold crossing,
which is what actually matters for cache correctness.

Skipped if embeddings.py not present (pre-merge). On first run after
merge, catches silent embedder behavior changes.
"""

import json
import numpy as np
import pytest
from pathlib import Path

FIXTURE = Path("tests/fixtures/embedding_golden.json")
REFERENCE = Path("tests/fixtures/embedding_golden_reference.npz")
EMBEDDING_THRESHOLD = 0.95
# Tolerance for cosine similarity comparison against reference
# Tight enough to catch model changes, loose enough for onnxruntime build variance
SIM_TOLERANCE = 0.01


try:
    from throttle.embeddings import get_embedding, EMBEDDINGS_AVAILABLE
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False
    EMBEDDINGS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _IMPORT_OK or not EMBEDDINGS_AVAILABLE,
    reason="embeddings.py not present — skipping golden fixture test"
)


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE.exists():
        pytest.skip("Reference .npz not yet generated")
    data = np.load(REFERENCE, allow_pickle=False)
    return {
        "vectors": data["vectors"],
        "cosines": data["cosines"],
    }


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE.exists():
        pytest.skip("Fixture JSON not yet generated — regenerate with optimum")
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def live_cosines(fixture):
    """Compute pairwise cosines for fixture pairs using current embedder."""
    prompts = fixture["prompts"]
    vectors = [get_embedding(p) for p in prompts]
    sims = {}
    for pair in fixture["pairs"]:
        a, b = pair["a"], pair["b"]
        va, vb = vectors[a], vectors[b]
        if va is None or vb is None:
            sims[(a, b)] = None
        else:
            sims[(a, b)] = float(np.dot(va, vb))
    return sims


def test_fixture_structure(fixture):
    """Fixture file is well-formed."""
    assert "prompts" in fixture
    assert "pairs" in fixture
    assert "ordering_assertions" in fixture
    assert "threshold_assertions" in fixture
    assert len(fixture["prompts"]) >= 10


def test_reference_file_exists():
    """Reference .npz must exist and have correct shape."""
    if not REFERENCE.exists():
        pytest.skip("Reference .npz not yet generated — regenerate with optimum")
    data = np.load(REFERENCE, allow_pickle=False)
    assert data["vectors"].shape == (20, 384)
    assert data["cosines"].shape == (20, 20)


def test_strong_paraphrases_above_threshold(fixture, live_cosines):
    """Strong paraphrases must be above 0.95 — they should be cache hits."""
    for pair in fixture["threshold_assertions"]:
        if "must_be_above" not in pair:
            continue
        a, b = pair["pair"]
        sim = live_cosines.get((a, b))
        assert sim is not None, f"Embedding failed for ({a},{b})"
        assert sim >= pair["must_be_above"], (
            f"Pair ({a},{b}) scored {sim:.4f}, expected >= {pair['must_be_above']}. "
            f"Reference: {pair['measured']:.4f}. Note: {pair['note']}"
        )


def test_entity_substitutions_below_threshold(fixture, live_cosines):
    """Entity substitutions must be below 0.95 — must NOT be cache hits."""
    for pair in fixture["threshold_assertions"]:
        if "must_be_below" not in pair:
            continue
        a, b = pair["pair"]
        sim = live_cosines.get((a, b))
        assert sim is not None, f"Embedding failed for ({a},{b})"
        assert sim < pair["must_be_below"], (
            f"Entity substitution ({a},{b}) scored {sim:.4f}, expected < {pair['must_be_below']}. "
            f"Reference: {pair['measured']:.4f}. "
            f"This pair would be INCORRECTLY served as a cache hit. "
            f"Note: {pair['note']}"
        )


def test_ordering_invariant(fixture, live_cosines):
    """
    Core ordering assertion: genuine paraphrases must score higher than
    entity substitutions. If this fails after an implementation swap,
    the new embedder has different semantic geometry.
    """
    for assertion in fixture["ordering_assertions"]:
        higher = tuple(assertion["higher"])
        lower = tuple(assertion["lower"])
        sim_h = live_cosines.get(higher)
        sim_l = live_cosines.get(lower)
        assert sim_h is not None, f"Embedding failed for {higher}"
        assert sim_l is not None, f"Embedding failed for {lower}"
        assert sim_h > sim_l, (
            f"Ordering violated: {higher}={sim_h:.4f} not > {lower}={sim_l:.4f}. "
            f"Note: {assertion['note']}"
        )


def test_similarity_within_tolerance_of_reference(fixture, reference, live_cosines):
    """
    Live similarities must be within SIM_TOLERANCE of reference values.
    Catches silent embedder changes. Tolerance is loose enough for
    legitimate onnxruntime build differences, tight enough to catch
    model swaps.
    """
    cosines_ref = reference["cosines"]
    for pair in fixture["pairs"]:
        a, b = pair["a"], pair["b"]
        sim_live = live_cosines.get((a, b))
        sim_ref = float(cosines_ref[a, b])
        if sim_live is None:
            pytest.skip(f"Embedding failed for ({a},{b})")
        assert abs(sim_live - sim_ref) <= SIM_TOLERANCE, (
            f"Pair ({a},{b}) similarity changed: live={sim_live:.4f}, "
            f"reference={sim_ref:.4f}, delta={abs(sim_live-sim_ref):.4f} > {SIM_TOLERANCE}. "
            f"Label: {pair['label']}. "
            f"If this is a legitimate embedder change, regenerate the reference."
        )
