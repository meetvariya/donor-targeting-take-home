import numpy as np
import pandas as pd
import pytest

from donor_targeting.features import (
    centroid_weights,
    compute_features,
    eligible_sends,
    engagements,
    write_features,
)
from donor_targeting.ml_harness import run
from donor_targeting.mocks.crm import MockCRM
from donor_targeting.policy import replay_mask, selected_positions, summary
from donor_targeting.serving_features import compact_cosine, prepare_snapshot
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


@pytest.mark.parametrize("batch_size", [1, 3, 4096])
def test_compact_cosine_matches_existing_features(batch_size):
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(4, 1024)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    content = pd.DataFrame({"id": [10, 20, 30, 40], "dense_embedding": list(embeddings)})
    sent_at = pd.Timestamp("2026-09-01T05:00:00Z")
    sends = pd.DataFrame(
        {"timeline_id": [1, 2, 3, 4], "person_id": [1, 2, 3, 4],
         "content_id": [40] * 4, "sent_at": [sent_at] * 4}
    )
    engaged = pd.DataFrame(
        {"person_id": [1, 1, 2, 3], "content_id": [10, 20, 30, 10],
         "engaged_at": [sent_at - pd.Timedelta(days=1), sent_at - pd.Timedelta(days=30),
                        sent_at, sent_at - pd.Timedelta(days=31)]}
    )
    weights = centroid_weights(sends, engaged, content["id"].to_numpy())
    expected = compute_features(sends, engaged, content)["centroid_cosine_similarity"]
    actual = compact_cosine(weights, embeddings, embeddings[3], batch_size=batch_size)
    np.testing.assert_allclose(actual, expected, atol=1e-5, equal_nan=True)
    scaled = compact_cosine(weights, embeddings, embeddings[3] * 7, batch_size=batch_size)
    np.testing.assert_allclose(scaled, actual, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("hours_after_snapshot", [0, 1, 11])
def test_prepared_snapshot_matches_offline_features(serving_warehouse, tmp_path, hours_after_snapshot):
    from donor_targeting.config import SNAPSHOT_AT

    snapshot = prepare_snapshot(serving_warehouse, tmp_path / "serving")
    assert snapshot.person_ids.tolist() == [1, 2, 5, 6]
    as_of = pd.Timestamp(SNAPSHOT_AT) + pd.Timedelta(hours=hours_after_snapshot)
    content = load_content(serving_warehouse)
    sends = pd.DataFrame(
        {"timeline_id": np.arange(4), "person_id": snapshot.person_ids,
         "content_id": [1004] * 4, "sent_at": [as_of] * 4}
    )
    expected = compute_features(sends, engagements(load_timeline(serving_warehouse)), content)
    actual = snapshot.score(content.iloc[3]["dense_embedding"], as_of.to_pydatetime())
    np.testing.assert_allclose(actual, expected["centroid_cosine_similarity"], atol=1e-5, equal_nan=True)
    assert np.isnan(actual[2:]).all()
    assert prepare_snapshot(serving_warehouse, tmp_path / "serving").generation == snapshot.generation
    with pytest.raises(ValueError, match="stale"):
        snapshot.score(content.iloc[3]["dense_embedding"], SNAPSHOT_AT + pd.Timedelta(days=1))


def test_audience_ranking_is_deterministic_and_per_email():
    person_ids = np.array([9, 3, 8, 1])
    scores = np.array([0.5, 0.5, np.nan, 0.1])
    assert selected_positions(person_ids, scores, 0.5).tolist() == [1, 0]
    frame = pd.DataFrame({"person_id": person_ids, "content_id": [10, 10, 20, 20]})
    assert replay_mask(frame, scores, 0.5).tolist() == [0, 1, 0, 1]


def test_policy_metrics_use_bau_denominators():
    frame = pd.DataFrame({"donated": [1, 0, 1, 0], "amount": [10, 0, 30, 0],
                          "unsubscribed": [0, 1, 0, 1]})
    result = summary(frame, np.full(4, 0.75))
    assert result["donors_kept_vs_bau"] == 1
    assert result["dollars_kept_vs_bau"] == 1
    assert result["unsubscribes_avoided_vs_bau"] == 0
