import numpy as np
import pandas as pd
import pytest

from donor_targeting.features import (
    compute_features,
    eligible_sends,
    engagements,
    write_features,
)
from donor_targeting.ml_harness import run
from donor_targeting.mocks.crm import MockCRM
from donor_targeting.warehouse_io import load_content, load_features, load_timeline


def test_centroid_feature_matches_brute_force(small_warehouse):
    timeline, content = load_timeline(small_warehouse), load_content(small_warehouse)
    sends = eligible_sends(timeline).sample(200, random_state=0)
    eng = engagements(timeline)
    feats = compute_features(sends, eng, content).set_index("timeline_id")
    emb = dict(zip(content["id"], content["dense_embedding"]))
    for s in sends.itertuples():
        e = eng[(eng.person_id == s.person_id) & (eng.engaged_at < s.sent_at)]
        e = e[e.engaged_at >= s.sent_at - pd.Timedelta(days=30)]
        got = feats.loc[s.timeline_id, "centroid_cosine_similarity"]
        if e.empty:
            assert np.isnan(got)
            continue
        c = np.mean([emb[i] for i in e.content_id], axis=0)
        want = c @ emb[s.content_id] / np.linalg.norm(c)
        assert got == pytest.approx(want, abs=1e-5)


def test_features_parquet_reads_back_with_centroids(small_warehouse, tmp_path):
    timeline, content = load_timeline(small_warehouse), load_content(small_warehouse)
    sends = eligible_sends(timeline).sample(300, random_state=0)
    feats = compute_features(sends, engagements(timeline), content)
    write_features(feats, tmp_path / "features.parquet")
    back = load_features(tmp_path, with_centroid=True)
    has = feats["centroid_cosine_similarity"].notna().to_numpy()
    assert 0 < has.sum() < len(feats)
    assert back["centroid_1m"].isna().to_numpy().tolist() == (~has).tolist()
    np.testing.assert_allclose(
        np.stack(back["centroid_1m"][has].to_numpy()),
        np.stack(feats["centroid_1m"][has].to_numpy()),
        rtol=1e-6,
    )


def test_harness_returns_model_metrics_and_predictions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4_000)
    X = pd.DataFrame({"x": np.where(rng.random(4_000) < 0.2, np.nan, x)})
    y = pd.DataFrame({"y": (rng.random(4_000) < 1 / (1 + np.exp(3 - x))).astype(int)})
    result = run(X, y)
    assert result.metrics["roc_auc"] > 0.6
    assert len(result.predictions) == 1_000
    result = run(X, y, test_mask=np.arange(4_000) >= 3_000)
    assert result.predictions.index.min() == 3_000


def test_crm_takes_the_whole_list_in_one_call(tmp_path):
    crm = MockCRM(tmp_path, latency_s=0)
    list_id = crm.create_list("test", [3, 1, 2, 3], content_id=2001)
    assert crm.get_list(list_id)["member_count"] == 3
    assert pd.read_csv(tmp_path / f"{list_id}.csv")["person_id"].tolist() == [3, 1, 2]
