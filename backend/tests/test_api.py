"""Independent integration & unit tests for FastAPI Backend endpoints."""

import math
from pathlib import Path
from typing import Dict, List

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

import backend.app as backend_app
from backend.app import (
    _get_candidate_sources,
    _get_data_scale_mode,
    _load_model_once,
    _normalize_popularity,
    _row_to_track_item,
    app,
)
from src.recommender import EmbeatRecommender


def _build_synthetic_tracks_df(
    *,
    include_artist_id: bool = True,
    unit_popularity: bool = False,
) -> pd.DataFrame:
    """Create a self-contained synthetic catalog for fast, disk-independent tests."""
    rows: List[Dict[str, object]] = []
    artists = [
        ("art_1", "Alpha Echo", "pop, synthpop"),
        ("art_2", "Bravo Pulse", "hip hop, rap"),
        ("art_3", "Charlie Acoustic", "acoustic, indie folk"),
        ("art_4", "Delta Ambient", "ambient, classical"),
        ("art_5", "Echo Drive", "rock, electronic"),
        ("art_6", "Foxtrot Chill", "lo-fi, chillhop"),
    ]

    for i in range(24):
        artist_id, artist_name, genres = artists[i % len(artists)]
        pop_100 = 95 - i * 3
        pop_val = round(pop_100 / 100.0, 2) if unit_popularity else pop_100
        row: Dict[str, object] = {
            "track_id": f"trk_{i:03d}",
            "track_name": f"Aurora Track {i:02d}",
            "artist_name": artist_name,
            "artist_genres": genres,
            "popularity": pop_val,
            "danceability": round(0.25 + (i % 7) * 0.1, 4),
            "energy": round(0.20 + (i % 8) * 0.09, 4),
            "valence": round(0.15 + (i % 6) * 0.14, 4),
            "tempo": float(75 + (i % 8) * 10),
            "acousticness": round(0.05 + ((7 - (i % 8)) * 0.1), 4),
            "instrumentalness": round(0.02 + (i % 5) * 0.18, 4),
            "liveness": round(0.08 + (i % 4) * 0.05, 4),
            "speechiness": round(0.03 + (i % 5) * 0.04, 4),
            "loudness": float(-18 + (i % 7) * 2),
        }
        if include_artist_id:
            row["artist_id"] = artist_id
        rows.append(row)

    return pd.DataFrame(rows)


def _reset_backend_state(monkeypatch: pytest.MonkeyPatch, tmp_dir: Path) -> None:
    """Reset module-level model state and isolate path constants away from workspace data/artifacts."""
    monkeypatch.setattr(backend_app, "model", None)
    monkeypatch.setattr(backend_app, "model_mode", "unloaded")
    monkeypatch.setattr(backend_app, "loaded_artifact_path", None)
    monkeypatch.setattr(backend_app, "PROJECT_ROOT", tmp_dir)
    monkeypatch.setattr(backend_app, "ENV_FILE_PATH", tmp_dir / ".env.missing")
    monkeypatch.setattr(backend_app, "ARTIFACT_PATH", tmp_dir / "missing_default.joblib")
    monkeypatch.setattr(backend_app, "COMPACT_ARTIFACT_PATH", tmp_dir / "missing_compact.joblib")
    monkeypatch.setattr(backend_app, "SAMPLE_ARTIFACT_PATH", tmp_dir / "missing_sample.joblib")
    monkeypatch.setattr(backend_app, "FULL_ARTIFACT_PATH", tmp_dir / "missing_full.joblib")
    monkeypatch.setattr(backend_app, "FALLBACK_CSV_PATHS", [])
    monkeypatch.delenv("DATA_SCALE_MODE", raising=False)
    monkeypatch.delenv("MODEL_ARTIFACT_PATH", raising=False)


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    """Initialize TestClient using an isolated temporary artifact (never reading 3.8GB disk files)."""
    tmp_dir = tmp_path_factory.mktemp("api_suite")
    df = _build_synthetic_tracks_df(include_artist_id=True, unit_popularity=False)
    rec = EmbeatRecommender().fit(df)
    artifact_file = tmp_dir / "embeat_test_compact.joblib"
    rec.save_artifact(artifact_file)

    mp = pytest.MonkeyPatch()
    _reset_backend_state(mp, tmp_dir)
    mp.setattr(backend_app, "COMPACT_ARTIFACT_PATH", artifact_file)
    mp.setenv("DATA_SCALE_MODE", "compact")

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        mp.undo()


def test_health_check(client: TestClient):
    """Verify /api/v1/health and /health return 200 with loaded model metadata and relative artifact_path."""
    for path in ("/api/v1/health", "/health"):
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["mode"] == "artifact_joblib"
        assert data["total_tracks"] == 24
        assert data["artifact_path"] is not None
        assert not Path(data["artifact_path"]).is_absolute()


def test_search_tracks_and_empty_or_no_match(client: TestClient):
    """Verify /api/v1/search for matching, empty/whitespace, and non-matching queries."""
    response = client.get("/api/v1/search", params={"q": "Aurora", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == len(data["results"]) == 5

    for idx, item in enumerate(data["results"], start=1):
        assert item["rank"] == idx
        assert item["track_id"]
        assert item["spotify_embed_url"] == f"https://open.spotify.com/embed/track/{item['track_id']}"
        assert item["audio_features"] is not None
        assert len(item["audio_features"]) == 9
        for val in item["audio_features"].values():
            assert isinstance(val, float)
            assert round(val, 4) == val

    # Empty and whitespace-only queries should return 200 with 0 results
    for empty_q in ("", "   "):
        empty_res = client.get("/api/v1/search", params={"q": empty_q, "limit": 5})
        assert empty_res.status_code == 200
        empty_data = empty_res.json()
        assert empty_data["status"] == "success"
        assert empty_data["total"] == 0
        assert empty_data["results"] == []

    # Non-matching query
    nomatch_res = client.get("/api/v1/search", params={"q": "zzzz_no_such_track_xyz"})
    assert nomatch_res.status_code == 200
    nomatch_data = nomatch_res.json()
    assert nomatch_data["total"] == 0
    assert nomatch_data["results"] == []


def test_get_track_details_valid_and_404(client: TestClient):
    """Verify /api/v1/tracks/{track_id} returns 200 for a valid ID and 404 with detail for unknown ID."""
    valid_res = client.get("/api/v1/tracks/trk_000")
    assert valid_res.status_code == 200
    track_data = valid_res.json()
    assert track_data["track_id"] == "trk_000"
    assert track_data["popularity"] == 95
    assert track_data["audio_features"] is not None

    not_found_res = client.get("/api/v1/tracks/non_existent_id_999")
    assert not_found_res.status_code == 404
    err_data = not_found_res.json()
    assert "Track ID not found" in err_data["detail"]
    assert "non_existent_id_999" in err_data["detail"]


def test_recommend_by_track_strategies(client: TestClient):
    """Verify /api/v1/recommend/track/{track_id} across audio, hybrid, and mmr strategies with [0, 1] scores."""
    seed_id = "trk_000"

    for strategy in ("audio", "hybrid", "mmr"):
        response = client.get(
            f"/api/v1/recommend/track/{seed_id}",
            params={"k": 5, "strategy": strategy},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["mode"] == strategy
        assert data["seed_track"]["track_id"] == seed_id

        recs = data["recommendations"]
        assert len(recs) == 5
        assert seed_id not in {item["track_id"] for item in recs}

        for idx, item in enumerate(recs, start=1):
            assert item["rank"] == idx
            assert item["similarity_score"] is not None
            assert 0.0 <= item["similarity_score"] <= 1.0
            assert item["audio_features"] is not None
            if strategy == "mmr":
                assert item["mmr_score"] is not None
            else:
                assert item["mmr_score"] is None


def test_recommend_mmr_similarity_score_bounds(client: TestClient):
    """Verify strategy=mmr keeps similarity_score in [0.0, 1.0] even with strong diversity weight."""
    response = client.get(
        "/api/v1/recommend/track/trk_000",
        params={"k": 10, "strategy": "mmr", "mmr_lambda": 0.3},
    )
    assert response.status_code == 200
    recs = response.json()["recommendations"]
    assert len(recs) == 10
    for item in recs:
        assert item["similarity_score"] is not None
        assert 0.0 <= item["similarity_score"] <= 1.0
        assert item["mmr_score"] is not None


def test_exclude_same_artist_true_and_false(client: TestClient):
    """Verify exclude_same_artist=True excludes Alpha Echo tracks while False allows them."""
    # trk_000 belongs to Alpha Echo (trk_000, trk_006, trk_012, trk_018)
    res_excluded = client.get(
        "/api/v1/recommend/track/trk_000",
        params={"k": 10, "strategy": "hybrid", "exclude_same_artist": True},
    )
    assert res_excluded.status_code == 200
    for item in res_excluded.json()["recommendations"]:
        assert item["artist_name"] != "Alpha Echo"
        assert 0.0 <= item["similarity_score"] <= 1.0

    res_included = client.get(
        "/api/v1/recommend/track/trk_000",
        params={"k": 15, "strategy": "audio", "exclude_same_artist": False},
    )
    assert res_included.status_code == 200
    recs_included = res_included.json()["recommendations"]
    assert len(recs_included) == 15
    assert any(item["artist_name"] == "Alpha Echo" for item in recs_included)
    for item in recs_included:
        assert 0.0 <= item["similarity_score"] <= 1.0


def test_exclude_same_artist_fallback_to_artist_name_without_artist_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Verify exclude_same_artist=True still filters by artist_name when artist_id is absent."""
    df_no_artist_id = _build_synthetic_tracks_df(include_artist_id=False)
    csv_file = tmp_path / "no_artist_id.csv"
    df_no_artist_id.to_csv(csv_file, index=False)

    _reset_backend_state(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_app, "FALLBACK_CSV_PATHS", [csv_file])
    monkeypatch.setenv("DATA_SCALE_MODE", "sample")

    with TestClient(app) as local_client:
        health = local_client.get("/api/v1/health").json()
        assert health["mode"] == "fallback_csv"
        assert health["artifact_path"] == "no_artist_id.csv"

        rec_res = local_client.get(
            "/api/v1/recommend/track/trk_000",
            params={"k": 8, "strategy": "hybrid", "exclude_same_artist": True},
        )
        assert rec_res.status_code == 200
        recs = rec_res.json()["recommendations"]
        assert len(recs) == 8
        assert all(item["artist_name"] != "Alpha Echo" for item in recs)
        assert all(0.0 <= item["similarity_score"] <= 1.0 for item in recs)

        profile_res = local_client.post(
            "/api/v1/recommend/profile",
            json={"track_ids": ["trk_000", "trk_001"], "k": 5, "exclude_same_artist": True},
        )
        assert profile_res.status_code == 200
        p_recs = profile_res.json()["recommendations"]
        assert all(item["artist_name"] not in {"Alpha Echo", "Bravo Pulse"} for item in p_recs)
        assert all(0.0 <= item["similarity_score"] <= 1.0 for item in p_recs)


def test_recommend_by_track_404(client: TestClient):
    """Verify /api/v1/recommend/track/invalid_id returns 404 and checks detail content."""
    response = client.get("/api/v1/recommend/track/invalid_id")
    assert response.status_code == 404
    err_data = response.json()
    assert "Track ID not found" in err_data["detail"]
    assert "invalid_id" in err_data["detail"]


def test_recommend_taste_profile(client: TestClient):
    """Verify POST /api/v1/recommend/profile with deduplication and [0, 1] similarity scores."""
    response = client.post(
        "/api/v1/recommend/profile",
        json={"track_ids": ["trk_000", "trk_001", "trk_000"], "k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["seed_tracks"]) == 2
    assert len(data["recommendations"]) == 5
    rec_ids = {item["track_id"] for item in data["recommendations"]}
    assert "trk_000" not in rec_ids and "trk_001" not in rec_ids
    for item in data["recommendations"]:
        assert 0.0 <= item["similarity_score"] <= 1.0


def test_recommend_taste_profile_404(client: TestClient):
    """Verify POST /api/v1/recommend/profile with unknown track_id returns 404."""
    response = client.post(
        "/api/v1/recommend/profile",
        json={"track_ids": ["non_existent_track_999"], "k": 5},
    )
    assert response.status_code == 404
    err_data = response.json()
    assert "Track ID not found" in err_data["detail"]


def test_recommend_popular(client: TestClient):
    """Verify /api/v1/recommend/popular?k=10 returns top 10 tracks ordered by popularity."""
    response = client.get("/api/v1/recommend/popular", params={"k": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    recs = data["recommendations"]
    assert len(recs) == 10
    pops = [item["popularity"] for item in recs]
    assert pops == sorted(pops, reverse=True)
    assert all(item["similarity_score"] is None for item in recs)


def test_recommend_all_six_moods_and_400(client: TestClient):
    """Verify all 6 supported moods (happy, sad, calm, chill, energetic, focus) and invalid_mood -> 400."""
    for mood in ("happy", "sad", "calm", "chill", "energetic", "focus"):
        res = client.get(f"/api/v1/recommend/mood/{mood}", params={"k": 5})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["mode"] == f"mood_{mood}"
        assert len(data["recommendations"]) == 5
        for item in data["recommendations"]:
            assert item["similarity_score"] is not None
            assert 0.0 <= item["similarity_score"] <= 1.0

    invalid_res = client.get("/api/v1/recommend/mood/invalid_mood", params={"k": 5})
    assert invalid_res.status_code == 400
    err_data = invalid_res.json()
    assert "Unsupported mood" in err_data["detail"]
    assert "invalid_mood" in err_data["detail"]


def test_validation_422_errors(client: TestClient):
    """Verify 422 Unprocessable Entity for invalid k, strategy, and track_ids."""
    for invalid_k in (0, 51):
        res_k = client.get("/api/v1/recommend/track/trk_000", params={"k": invalid_k})
        assert res_k.status_code == 422
        assert any("k" in err.get("loc", []) for err in res_k.json()["detail"])

    res_strategy = client.get(
        "/api/v1/recommend/track/trk_000",
        params={"k": 5, "strategy": "invalid_strategy"},
    )
    assert res_strategy.status_code == 422
    assert any("strategy" in err.get("loc", []) for err in res_strategy.json()["detail"])

    res_empty = client.post("/api/v1/recommend/profile", json={"track_ids": [], "k": 5})
    assert res_empty.status_code == 422
    assert any("track_ids" in err.get("loc", []) for err in res_empty.json()["detail"])

    res_too_many = client.post(
        "/api/v1/recommend/profile",
        json={"track_ids": ["trk_000"] * 51, "k": 5},
    )
    assert res_too_many.status_code == 422
    assert any("track_ids" in err.get("loc", []) for err in res_too_many.json()["detail"])


def test_normalize_popularity_boundaries_both_scales():
    """Verify _normalize_popularity handles integers, floats, NaN/None, and clamping on both scales."""
    # Dataset on 0..1 unit scale
    assert _normalize_popularity(0, is_unit_scale=True) == 0
    assert _normalize_popularity(1, is_unit_scale=True) == 100
    assert _normalize_popularity(0.0, is_unit_scale=True) == 0
    assert _normalize_popularity(1.0, is_unit_scale=True) == 100
    assert _normalize_popularity(0.456, is_unit_scale=True) == 46
    assert _normalize_popularity(-0.2, is_unit_scale=True) == 0
    assert _normalize_popularity(1.5, is_unit_scale=True) == 100
    assert _normalize_popularity(None, is_unit_scale=True) is None
    assert _normalize_popularity(float("nan"), is_unit_scale=True) is None
    assert _normalize_popularity(float("inf"), is_unit_scale=True) is None

    # Dataset on 0..100 scale (0.5 should NOT be multiplied by 100)
    assert _normalize_popularity(0, is_unit_scale=False) == 0
    assert _normalize_popularity(1, is_unit_scale=False) == 1
    assert _normalize_popularity(0.0, is_unit_scale=False) == 0
    assert _normalize_popularity(1.0, is_unit_scale=False) == 1
    assert _normalize_popularity(0.5, is_unit_scale=False) == 0
    assert _normalize_popularity(0.8, is_unit_scale=False) == 1
    assert _normalize_popularity(82, is_unit_scale=False) == 82
    assert _normalize_popularity(-15, is_unit_scale=False) == 0
    assert _normalize_popularity(150, is_unit_scale=False) == 100
    assert _normalize_popularity(None, is_unit_scale=False) is None
    assert _normalize_popularity(float("nan"), is_unit_scale=False) is None
    assert _normalize_popularity(float("-inf"), is_unit_scale=False) is None


def test_popularity_scale_follows_swapped_model_without_stale_global(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Verify swapping model between 0..1 and 0..100 datasets updates popularity scale cleanly."""
    _reset_backend_state(monkeypatch, tmp_path)

    unit_model = EmbeatRecommender().fit(
        _build_synthetic_tracks_df(unit_popularity=True)
    )
    hundred_model = EmbeatRecommender().fit(
        _build_synthetic_tracks_df(unit_popularity=False)
    )

    monkeypatch.setattr(backend_app, "model", unit_model)
    with TestClient(app) as local_client:
        unit_res = local_client.get("/api/v1/tracks/trk_000")
        assert unit_res.status_code == 200
        assert unit_res.json()["popularity"] == 95

        monkeypatch.setattr(backend_app, "model", hundred_model)
        hundred_res = local_client.get("/api/v1/tracks/trk_000")
        assert hundred_res.status_code == 200
        assert hundred_res.json()["popularity"] == 95


def test_nan_and_inf_in_audio_features_and_similarity_clamping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Verify +/-Inf and NaN in audio features and out-of-range similarity scores serialize safely."""
    row_inf = {
        "track_id": "trk_inf",
        "track_name": "Inf Track",
        "artist_name": "Test Artist",
        "popularity": 50,
        "danceability": float("inf"),
        "energy": float("-inf"),
        "valence": math.nan,
        "tempo": 120.0,
        "acousticness": 0.2,
        "instrumentalness": 0.1,
        "liveness": 0.1,
        "speechiness": 0.05,
        "loudness": -6.0,
        "hybrid_similarity": 1.25,
    }
    item = _row_to_track_item(row_inf, rank=1, is_unit_scale=False)
    assert item.audio_features is None
    assert item.similarity_score == 1.0

    row_neg_sim = dict(row_inf, danceability=0.5, energy=0.5, valence=0.5, hybrid_similarity=-0.4)
    item_neg = _row_to_track_item(row_neg_sim, rank=2, is_unit_scale=False)
    assert item_neg.audio_features is not None
    assert item_neg.similarity_score == 0.0

    # End-to-end JSON serialization check through FastAPI endpoint
    class _EdgeCaseRecommender:
        df = pd.DataFrame([{"track_id": "trk_inf", "popularity": 50}])

        def search_tracks(self, query, limit=15, as_dict=True):
            return [row_inf]

    _reset_backend_state(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_app, "model", _EdgeCaseRecommender())
    with TestClient(app) as local_client:
        res = local_client.get("/api/v1/search", params={"q": "Inf"})
        assert res.status_code == 200
        payload = res.json()
        assert payload["results"][0]["audio_features"] is None
        assert payload["results"][0]["similarity_score"] == 1.0


def test_data_scale_mode_configuration_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Verify DATA_SCALE_MODE (sample, compact, full) from .env/env and CSV/MemoryError fallback."""
    _reset_backend_state(monkeypatch, tmp_path)

    # 1. Reading DATA_SCALE_MODE from .env file
    env_file = tmp_path / ".env"
    env_file.write_text('DATA_SCALE_MODE="sample"\n', encoding="utf-8")
    monkeypatch.setattr(backend_app, "ENV_FILE_PATH", env_file)
    assert _get_data_scale_mode() == "sample"

    # Invalid mode falls back to compact
    monkeypatch.setenv("DATA_SCALE_MODE", "invalid_scale")
    assert _get_data_scale_mode() == "compact"
    monkeypatch.delenv("DATA_SCALE_MODE")

    # 2. Candidate ordering for compact vs full vs sample
    compact_arts, _ = _get_candidate_sources("compact")
    full_arts, _ = _get_candidate_sources("full")
    assert compact_arts[0] == backend_app.COMPACT_ARTIFACT_PATH
    assert full_arts[0] == backend_app.FULL_ARTIFACT_PATH

    # 3. sample mode prefers CSV over large default ARTIFACT_PATH when SAMPLE_ARTIFACT_PATH is missing
    df = _build_synthetic_tracks_df()
    sample_csv = tmp_path / "embeat_sample_tracks.csv"
    df.to_csv(sample_csv, index=False)

    default_art = tmp_path / "embeat_recommender.joblib"
    EmbeatRecommender().fit(df).save_artifact(default_art)

    monkeypatch.setattr(backend_app, "ARTIFACT_PATH", default_art)
    monkeypatch.setattr(backend_app, "FALLBACK_CSV_PATHS", [sample_csv])
    env_file.write_text("DATA_SCALE_MODE=sample\n", encoding="utf-8")

    _load_model_once()
    assert backend_app.model_mode == "fallback_csv"
    assert backend_app.loaded_artifact_path == "embeat_sample_tracks.csv"

    # 4. compact mode loads compact artifact over default artifact, and survives MemoryError fallback
    _reset_backend_state(monkeypatch, tmp_path)
    compact_art = tmp_path / "embeat_recommender_compact.joblib"
    EmbeatRecommender().fit(df).save_artifact(compact_art)
    monkeypatch.setattr(backend_app, "COMPACT_ARTIFACT_PATH", compact_art)
    monkeypatch.setattr(backend_app, "ARTIFACT_PATH", default_art)
    monkeypatch.setattr(backend_app, "FALLBACK_CSV_PATHS", [sample_csv])
    monkeypatch.setenv("DATA_SCALE_MODE", "compact")

    _load_model_once()
    assert backend_app.model_mode == "artifact_joblib"
    assert backend_app.loaded_artifact_path == compact_art.name

    # 5. full mode prioritizes FULL_ARTIFACT_PATH over default ARTIFACT_PATH
    _reset_backend_state(monkeypatch, tmp_path)
    full_art = tmp_path / "embeat_recommender_full.joblib"
    EmbeatRecommender().fit(df).save_artifact(full_art)
    monkeypatch.setattr(backend_app, "FULL_ARTIFACT_PATH", full_art)
    monkeypatch.setattr(backend_app, "ARTIFACT_PATH", default_art)
    monkeypatch.setenv("DATA_SCALE_MODE", "full")

    _load_model_once()
    assert backend_app.model_mode == "artifact_joblib"
    assert backend_app.loaded_artifact_path == full_art.name

    # Simulate MemoryError on artifact load -> falls back to CSV cleanly
    _reset_backend_state(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_app, "COMPACT_ARTIFACT_PATH", compact_art)
    monkeypatch.setattr(backend_app, "FALLBACK_CSV_PATHS", [sample_csv])
    monkeypatch.setenv("DATA_SCALE_MODE", "compact")

    def _raise_oom(_path):
        raise MemoryError("Simulated OOM")

    monkeypatch.setattr(backend_app, "_try_load_artifact", _raise_oom)
    _load_model_once()
    assert backend_app.model_mode == "fallback_csv"
    assert backend_app.loaded_artifact_path == "embeat_sample_tracks.csv"
