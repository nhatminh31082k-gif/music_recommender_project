"""FastAPI Backend Server for Embeat Music Recommender System."""

import asyncio
from contextlib import asynccontextmanager
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from backend.schemas import (
    AudioFeatures,
    HealthResponse,
    RecommendationResponse,
    SearchResponse,
    TasteProfileRequest,
    TrackItem,
)
from src.preprocess import AUDIO_FEATURES
from src.recommender import EmbeatRecommender

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "embeat_recommender.joblib"
COMPACT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "embeat_recommender_compact.joblib"
SAMPLE_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "embeat_recommender_sample.joblib"
FULL_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "embeat_recommender_full.joblib"
VALID_SCALE_MODES = ("sample", "compact", "full")
DEFAULT_SCALE_MODE = "compact"
FALLBACK_CSV_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "balanced_eda.csv",
    PROJECT_ROOT / "data" / "processed" / "embeat_sample_tracks.csv",
    PROJECT_ROOT / "data" / "processed" / "sample_tracks.csv",
]

AUDIO_FEATURE_FIELDS = (
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
)

SCORE_COLUMNS = (
    "hybrid_similarity",
    "similarity",
    "mood_similarity",
)

# Global model state initialized once during startup lifespan
model: Optional[EmbeatRecommender] = None
model_mode: str = "unloaded"
loaded_artifact_path: Optional[str] = None
_model_load_lock = threading.Lock()


def _read_env_file(env_path: Path) -> Dict[str, str]:
    """Parse key=value pairs from a .env file without external dependencies."""
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            if key:
                values[key] = val
    except OSError as exc:
        logger.warning("Could not read .env file at %s: %s", env_path, exc)
    return values


def _get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read config from process environment first, falling back to .env."""
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip() != "":
        return env_val.strip()
    file_val = _read_env_file(ENV_FILE_PATH).get(key)
    if file_val is not None and file_val.strip() != "":
        return file_val.strip()
    return default


def _get_data_scale_mode() -> str:
    """Resolve DATA_SCALE_MODE (sample | compact | full), defaulting to compact."""
    raw_mode = _get_config_value("DATA_SCALE_MODE", DEFAULT_SCALE_MODE)
    mode = (raw_mode or DEFAULT_SCALE_MODE).strip().lower()
    if mode not in VALID_SCALE_MODES:
        logger.warning(
            "Invalid DATA_SCALE_MODE='%s'; falling back to '%s'.",
            raw_mode,
            DEFAULT_SCALE_MODE,
        )
        return DEFAULT_SCALE_MODE
    return mode


def _get_candidate_sources(scale_mode: str) -> tuple[List[Path], List[Path]]:
    """Return ordered (artifact_candidates, csv_candidates) for the given scale mode."""
    custom_artifact = _get_config_value("MODEL_ARTIFACT_PATH")
    artifact_candidates: List[Path] = []
    if custom_artifact:
        custom_path = Path(custom_artifact)
        if not custom_path.is_absolute():
            custom_path = PROJECT_ROOT / custom_path
        artifact_candidates.append(custom_path)

    if scale_mode == "full":
        artifact_candidates.extend([FULL_ARTIFACT_PATH, ARTIFACT_PATH, COMPACT_ARTIFACT_PATH])
        csv_candidates = list(FALLBACK_CSV_PATHS)
    elif scale_mode == "sample":
        artifact_candidates.extend([SAMPLE_ARTIFACT_PATH, COMPACT_ARTIFACT_PATH, ARTIFACT_PATH])
        csv_candidates = [
            PROJECT_ROOT / "data" / "processed" / "embeat_sample_tracks.csv",
            PROJECT_ROOT / "data" / "processed" / "sample_tracks.csv",
            PROJECT_ROOT / "data" / "processed" / "balanced_eda.csv",
        ]
    else:  # compact
        artifact_candidates.extend([COMPACT_ARTIFACT_PATH, ARTIFACT_PATH])
        csv_candidates = list(FALLBACK_CSV_PATHS)

    # Deduplicate while preserving priority order
    unique_artifacts = list(dict.fromkeys(artifact_candidates))
    unique_csvs = list(dict.fromkeys(csv_candidates + FALLBACK_CSV_PATHS))
    return unique_artifacts, unique_csvs


def _detect_unit_popularity_scale(df: Optional[pd.DataFrame]) -> bool:
    """Return True if dataset popularity column uses 0.0..1.0 unit scale instead of 0..100."""
    if df is None or "popularity" not in df.columns:
        return False
    pop_series = pd.to_numeric(df["popularity"], errors="coerce").dropna()
    if pop_series.empty:
        return False
    finite_pop = pop_series[np.isfinite(pop_series)]
    if finite_pop.empty:
        return False
    return float(finite_pop.min()) >= 0.0 and float(finite_pop.max()) <= 1.0


def _to_relative_path(path: Path) -> str:
    """Return path relative to PROJECT_ROOT using POSIX slashes, or basename if outside."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _try_load_artifact(artifact_path: Path) -> Optional[EmbeatRecommender]:
    """Load a joblib artifact with explicit error logging; returns None on failure."""
    logger.info("Loading model artifact from %s", artifact_path)
    return EmbeatRecommender.load_artifact(artifact_path)


def _try_load_csv(csv_path: Path) -> EmbeatRecommender:
    """Fit an EmbeatRecommender from a fallback CSV file."""
    logger.info("Fitting fallback model from CSV %s", csv_path)
    df = pd.read_csv(csv_path)
    recommender = EmbeatRecommender()
    recommender.fit(df)
    return recommender


def _load_model_once() -> EmbeatRecommender:
    """Load the EmbeatRecommender artifact or fallback CSV according to DATA_SCALE_MODE."""
    global model, model_mode, loaded_artifact_path

    with _model_load_lock:
        if model is not None and model.df is not None:
            return model

        scale_mode = _get_data_scale_mode()
        artifact_candidates, csv_candidates = _get_candidate_sources(scale_mode)
        logger.info("Initializing recommender model with DATA_SCALE_MODE=%s", scale_mode)

        last_error: Optional[BaseException] = None

        # In sample mode, prefer an explicit MODEL_ARTIFACT_PATH or dedicated sample artifact,
        # then sample CSVs before falling back to potentially large default joblib artifacts.
        if scale_mode == "sample":
            custom_artifact = _get_config_value("MODEL_ARTIFACT_PATH")
            custom_path: Optional[Path] = None
            if custom_artifact:
                custom_path = Path(custom_artifact)
                if not custom_path.is_absolute():
                    custom_path = PROJECT_ROOT / custom_path

            load_plan: List[tuple[str, Path]] = []
            if custom_path is not None:
                load_plan.append(("artifact_joblib", custom_path))
            if SAMPLE_ARTIFACT_PATH in artifact_candidates and SAMPLE_ARTIFACT_PATH != custom_path:
                load_plan.append(("artifact_joblib", SAMPLE_ARTIFACT_PATH))
            for csv_path in csv_candidates:
                load_plan.append(("fallback_csv", csv_path))
            for art_path in artifact_candidates:
                if art_path not in {custom_path, SAMPLE_ARTIFACT_PATH}:
                    load_plan.append(("artifact_joblib", art_path))
        else:
            load_plan = [("artifact_joblib", p) for p in artifact_candidates] + [
                ("fallback_csv", p) for p in csv_candidates
            ]

        for source_mode, candidate_path in load_plan:
            if not candidate_path.exists():
                continue
            try:
                loaded_rec = (
                    _try_load_artifact(candidate_path)
                    if source_mode == "artifact_joblib"
                    else _try_load_csv(candidate_path)
                )
                model = loaded_rec
                model_mode = source_mode
                loaded_artifact_path = _to_relative_path(candidate_path)
                logger.info(
                    "Model loaded successfully (mode=%s, scale=%s, source=%s, tracks=%d)",
                    model_mode,
                    scale_mode,
                    loaded_artifact_path,
                    len(model.df) if model.df is not None else 0,
                )
                return model
            except MemoryError as exc:
                last_error = exc
                logger.exception(
                    "MemoryError while loading %s (%s). Consider setting DATA_SCALE_MODE=sample or compact.",
                    candidate_path,
                    source_mode,
                )
            except Exception as exc:
                last_error = exc
                logger.exception(
                    "Failed to load %s (%s): %s",
                    candidate_path,
                    source_mode,
                    exc,
                )

        if last_error is not None:
            raise RuntimeError(
                f"Failed to load model in DATA_SCALE_MODE='{scale_mode}' from all available sources: {last_error}"
            ) from last_error

        raise RuntimeError(
            f"Could not load model (DATA_SCALE_MODE='{scale_mode}'): neither artifact files in artifacts/ "
            "nor fallback CSV files in data/processed/ were found."
        )


def _get_model() -> EmbeatRecommender:
    """Retrieve the active recommender model, ensuring it is loaded."""
    if model is None or model.df is None:
        return _load_model_once()
    return model


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """FastAPI lifespan handler to load the recommendation model once off the event loop."""
    loaded = await asyncio.to_thread(_load_model_once)
    app_instance.state.model = loaded
    app_instance.state.mode = model_mode
    app_instance.state.artifact_path = loaded_artifact_path
    yield


app = FastAPI(
    title="Embeat Music Recommender API",
    description="Content-Based & Hybrid Music Recommendation REST API powered by Embeat 45M Spotify Tracks.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clean_scalar(value: Any) -> Any:
    """Convert NaN, np.nan, or +/-inf values to None and normalize numpy scalars."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _normalize_popularity(
    raw_pop: Any,
    is_unit_scale: bool = False,
) -> Optional[int]:
    """Convert popularity to integer in [0, 100] based on dataset scale."""
    if isinstance(raw_pop, (bool, np.bool_)):
        return None
    cleaned_pop = _clean_scalar(raw_pop)
    if cleaned_pop is None or isinstance(cleaned_pop, bool):
        return None
    try:
        pop_float = float(cleaned_pop)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pop_float):
        return None

    scaled = round(pop_float * 100.0) if is_unit_scale else round(pop_float)
    return max(0, min(100, int(scaled)))


def _clamp_similarity_score(raw_score: Any) -> Optional[float]:
    """Clamp finite similarity scores into [0.0, 1.0] and round to 4 decimal places."""
    cleaned_score = _clean_scalar(raw_score)
    if cleaned_score is None or isinstance(cleaned_score, bool):
        return None
    try:
        val = float(cleaned_score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return round(max(0.0, min(1.0, val)), 4)


def _row_to_track_item(
    row: dict,
    rank: int | None = None,
    *,
    is_unit_scale: bool = False,
) -> TrackItem:
    """Convert a track dictionary row into a validated TrackItem schema."""
    cleaned: Dict[str, Any] = {k: _clean_scalar(v) for k, v in row.items()}

    track_id = str(cleaned.get("track_id") or "")
    track_name = str(cleaned.get("track_name") or "Unknown Track")

    artist_name = cleaned.get("artist_name")
    if artist_name is not None:
        artist_name = str(artist_name)

    artist_genres = cleaned.get("artist_genres")
    if artist_genres is not None:
        artist_genres = str(artist_genres)

    popularity = _normalize_popularity(row.get("popularity"), is_unit_scale=is_unit_scale)

    # Extract similarity score from priority score columns, clamped to [0.0, 1.0]
    similarity_score: Optional[float] = None
    for col in SCORE_COLUMNS:
        if col in cleaned and cleaned[col] is not None:
            similarity_score = _clamp_similarity_score(cleaned[col])
            break
    if similarity_score is None and cleaned.get("similarity_score") is not None:
        similarity_score = _clamp_similarity_score(cleaned["similarity_score"])

    mmr_score: Optional[float] = None
    if cleaned.get("mmr_score") is not None:
        mmr_val = float(cleaned["mmr_score"])
        if math.isfinite(mmr_val):
            mmr_score = round(mmr_val, 4)

    # Group the 9 acoustic features if present in row (rounded to 4 decimal places as standard float)
    audio_features: Optional[AudioFeatures] = None
    if all(feat in cleaned and cleaned[feat] is not None for feat in AUDIO_FEATURE_FIELDS):
        audio_features = AudioFeatures(
            **{feat: round(float(cleaned[feat]), 4) for feat in AUDIO_FEATURE_FIELDS}
        )
    elif isinstance(cleaned.get("audio_features"), AudioFeatures):
        af_dict = cleaned["audio_features"].model_dump()
        audio_features = AudioFeatures(
            **{feat: round(float(af_dict[feat]), 4) for feat in AUDIO_FEATURE_FIELDS}
        )
    elif isinstance(cleaned.get("audio_features"), dict):
        af_clean = {k: _clean_scalar(v) for k, v in cleaned["audio_features"].items()}
        if all(feat in af_clean and af_clean[feat] is not None for feat in AUDIO_FEATURE_FIELDS):
            audio_features = AudioFeatures(
                **{feat: round(float(af_clean[feat]), 4) for feat in AUDIO_FEATURE_FIELDS}
            )

    resolved_rank = rank if rank is not None else cleaned.get("rank")
    if resolved_rank is not None:
        resolved_rank = int(resolved_rank)

    return TrackItem(
        rank=resolved_rank,
        track_id=track_id,
        track_name=track_name,
        artist_name=artist_name,
        artist_genres=artist_genres,
        popularity=popularity,
        similarity_score=similarity_score,
        mmr_score=mmr_score,
        spotify_embed_url=f"https://open.spotify.com/embed/track/{track_id}",
        audio_features=audio_features,
    )


@app.get("/api/v1/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return server health and loaded model status."""
    recommender = _get_model()
    is_loaded = recommender is not None and recommender.df is not None
    total_tracks = int(len(recommender.df)) if is_loaded else 0
    return HealthResponse(
        status="ok" if is_loaded else "error",
        model_loaded=is_loaded,
        mode=model_mode,
        total_tracks=total_tracks,
        artifact_path=loaded_artifact_path,
    )


@app.get("/api/v1/search", response_model=SearchResponse)
def search_tracks(
    q: str = Query(..., description="Search query for track name or artist name"),
    limit: int = Query(15, ge=1, le=100, description="Maximum number of search results"),
) -> SearchResponse:
    """Search tracks by keyword and attach their 9 audio features."""
    start = time.perf_counter()
    recommender = _get_model()
    is_unit_scale = _detect_unit_popularity_scale(getattr(recommender, "df", None))

    raw_results = recommender.search_tracks(q, limit=limit, as_dict=True)
    items: List[TrackItem] = []

    for idx, row in enumerate(raw_results, start=1):
        tid = row.get("track_id")
        merged_row = dict(row)
        has_all_audio = all(
            feat in merged_row and merged_row[feat] is not None
            for feat in AUDIO_FEATURE_FIELDS
        )
        if tid and not has_all_audio:
            try:
                details = recommender.get_track_details(tid)
                if details:
                    merged_row = {**details, **row}
            except ValueError:
                pass
        items.append(_row_to_track_item(merged_row, rank=idx, is_unit_scale=is_unit_scale))

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return SearchResponse(
        status="success",
        query=q,
        total=len(items),
        results=items,
        latency_ms=latency_ms,
    )


@app.get("/api/v1/tracks/{track_id}", response_model=TrackItem)
def get_track(track_id: str) -> TrackItem:
    """Retrieve metadata and 9 audio features for a single track by track_id."""
    recommender = _get_model()
    is_unit_scale = _detect_unit_popularity_scale(getattr(recommender, "df", None))
    try:
        details = recommender.get_track_details(track_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not details:
        raise HTTPException(
            status_code=404, detail=f"Track ID not found: {track_id}"
        )

    return _row_to_track_item(details, is_unit_scale=is_unit_scale)


@app.get("/api/v1/recommend/track/{track_id}", response_model=RecommendationResponse)
def recommend_by_track(
    track_id: str,
    k: int = Query(10, ge=1, le=50, description="Number of tracks to recommend"),
    strategy: Literal["audio", "hybrid", "mmr"] = Query(
        "hybrid",
        description="Recommendation strategy: audio, hybrid, or mmr",
    ),
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="Audio vs genre weight"),
    mmr_lambda: float = Query(
        0.7, ge=0.0, le=1.0, description="Relevance vs diversity weight for MMR"
    ),
    exclude_same_artist: bool = Query(
        True, description="Exclude tracks from the same artist as the seed track"
    ),
) -> RecommendationResponse:
    """Recommend top-K tracks similar to a single seed track."""
    start = time.perf_counter()
    recommender = _get_model()
    is_unit_scale = _detect_unit_popularity_scale(getattr(recommender, "df", None))

    strategy_norm = strategy.strip().lower()

    try:
        seed_details = recommender.get_track_details(track_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    seed_item = _row_to_track_item(seed_details, is_unit_scale=is_unit_scale)

    try:
        if strategy_norm == "audio":
            raw_recs = recommender.recommend_by_track(
                track_id,
                exclude_same_artist=exclude_same_artist,
                top_k=k,
                as_dict=True,
            )
        elif strategy_norm == "hybrid":
            raw_recs = recommender.recommend_hybrid(
                track_id,
                alpha=alpha,
                exclude_same_artist=exclude_same_artist,
                top_k=k,
                as_dict=True,
            )
        else:
            raw_recs = recommender.recommend_hybrid_mmr(
                track_id,
                alpha=alpha,
                mmr_lambda=mmr_lambda,
                candidate_pool=max(k * 3, 150),
                exclude_same_artist=exclude_same_artist,
                top_k=k,
                as_dict=True,
            )
    except ValueError as exc:
        err_msg = str(exc)
        if "Track ID not found" in err_msg:
            raise HTTPException(status_code=404, detail=err_msg) from exc
        raise HTTPException(status_code=400, detail=err_msg) from exc

    recommendations = [
        _row_to_track_item(row, rank=idx, is_unit_scale=is_unit_scale)
        for idx, row in enumerate(raw_recs, start=1)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return RecommendationResponse(
        status="success",
        mode=strategy_norm,
        seed_track=seed_item,
        recommendations=recommendations,
        latency_ms=latency_ms,
    )


@app.post("/api/v1/recommend/profile", response_model=RecommendationResponse)
def recommend_by_profile(payload: TasteProfileRequest) -> RecommendationResponse:
    """Recommend top-K tracks based on a multi-track user taste profile."""
    start = time.perf_counter()
    recommender = _get_model()
    is_unit_scale = _detect_unit_popularity_scale(getattr(recommender, "df", None))

    unique_track_ids = list(dict.fromkeys(payload.track_ids))

    seed_items: List[TrackItem] = []
    for tid in unique_track_ids:
        try:
            details = recommender.get_track_details(tid)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        seed_items.append(_row_to_track_item(details, is_unit_scale=is_unit_scale))

    try:
        raw_recs = recommender.recommend_by_taste_profile(
            unique_track_ids,
            alpha=payload.alpha,
            exclude_same_artist=payload.exclude_same_artist,
            top_k=payload.k,
            as_dict=True,
        )
    except ValueError as exc:
        err_msg = str(exc)
        if "Track ID not found" in err_msg:
            raise HTTPException(status_code=404, detail=err_msg) from exc
        raise HTTPException(status_code=400, detail=err_msg) from exc

    recommendations = [
        _row_to_track_item(row, rank=idx, is_unit_scale=is_unit_scale)
        for idx, row in enumerate(raw_recs, start=1)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return RecommendationResponse(
        status="success",
        mode="taste_profile",
        seed_tracks=seed_items,
        recommendations=recommendations,
        latency_ms=latency_ms,
    )


@app.get("/api/v1/recommend/popular", response_model=RecommendationResponse)
def recommend_popular(
    k: int = Query(10, ge=1, le=50, description="Number of popular tracks to return"),
) -> RecommendationResponse:
    """Return top-K most popular tracks as cold-start baseline."""
    start = time.perf_counter()
    recommender = _get_model()
    is_unit_scale = _detect_unit_popularity_scale(getattr(recommender, "df", None))

    raw_recs = recommender.recommend_by_popularity(top_k=k, as_dict=True)
    recommendations = [
        _row_to_track_item(row, rank=idx, is_unit_scale=is_unit_scale)
        for idx, row in enumerate(raw_recs, start=1)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return RecommendationResponse(
        status="success",
        mode="popular",
        recommendations=recommendations,
        latency_ms=latency_ms,
    )


@app.get("/api/v1/recommend/mood/{mood_name}", response_model=RecommendationResponse)
def recommend_by_mood(
    mood_name: str,
    k: int = Query(10, ge=1, le=50, description="Number of mood tracks to return"),
) -> RecommendationResponse:
    """Recommend top-K tracks matching a target acoustic mood profile."""
    start = time.perf_counter()
    recommender = _get_model()
    is_unit_scale = _detect_unit_popularity_scale(getattr(recommender, "df", None))

    try:
        raw_recs = recommender.recommend_by_mood(mood_name, top_k=k, as_dict=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recommendations = [
        _row_to_track_item(row, rank=idx, is_unit_scale=is_unit_scale)
        for idx, row in enumerate(raw_recs, start=1)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return RecommendationResponse(
        status="success",
        mode=f"mood_{mood_name.strip().lower()}",
        recommendations=recommendations,
        latency_ms=latency_ms,
    )
