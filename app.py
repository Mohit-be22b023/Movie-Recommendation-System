"""
CineLatent -- Streamlit front-end for the MovieLens deep-autoencoder recommender.

SINGLE-FILE BY DESIGN. This app imports nothing from the repository: no `src`
package, no relative imports, no sys.path manipulation. Deploying it needs
exactly three files pushed to the repo:

    app.py
    models/movie_embeddings.npy
    models/movie_id_mapping.pkl

That is deliberate. `from src.recommender import ...` fails the moment a
subfolder does not reach the host -- a common outcome when files are uploaded
through the GitHub web UI, or when a folder is named `Src` on a case-insensitive
filesystem and `src` on Linux. Removing the import removes the failure.

The recommendation logic here is a trimmed copy of `src/recommender.py`, which
remains the canonical version used by the notebook. `tests/test_app_parity.py`
asserts that both produce identical recommendations, so the copy cannot drift
silently.

The app never trains: it loads saved arrays and does a dot product.

    streamlit run app.py
"""

from __future__ import annotations

import difflib
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
POSTER_FALLBACK = "https://placehold.co/500x750/1b2130/64748b?text=No+poster"

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p"

st.set_page_config(
    page_title="CineLatent — movie recommendations",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# Recommender  (mirror of src/recommender.py — kept in sync by tests)
# ═══════════════════════════════════════════════════════════════════════════
class RecommenderError(Exception):
    """Base class for recommender failures."""


class MovieNotFoundError(RecommenderError):
    def __init__(self, message: str, suggestions: Optional[Sequence[str]] = None):
        super().__init__(message)
        self.suggestions = list(suggestions or [])


class AmbiguousTitleError(RecommenderError):
    def __init__(self, message: str, candidates: Optional[Sequence[str]] = None):
        super().__init__(message)
        self.candidates = list(candidates or [])


class MovieRecommender:
    """Cosine-similarity Top-N over the autoencoder's latent movie vectors."""

    def __init__(self, embeddings: np.ndarray, catalog: pd.DataFrame,
                 default_min_support: int = 10) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        if len(catalog) != embeddings.shape[0]:
            raise ValueError(
                f"catalog rows ({len(catalog)}) != embeddings rows "
                f"({embeddings.shape[0]}) -- artifacts are out of sync."
            )

        self.embeddings = embeddings
        self.catalog = catalog.reset_index(drop=True)
        self.default_min_support = int(default_min_support)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self._zero_norm = norms.squeeze(-1) == 0
        self._unit = self.embeddings / np.maximum(norms, 1e-10)

        self._title_to_pos = {str(t).lower(): i for i, t in enumerate(self.catalog["Title"])}
        if "TrainRatingCount" in self.catalog:
            self._support = self.catalog["TrainRatingCount"].to_numpy()
        elif "RatingCount" in self.catalog:
            self._support = self.catalog["RatingCount"].to_numpy()
        else:
            self._support = np.full(len(self.catalog), np.inf)
        self._threshold_cache: Dict[int, np.ndarray] = {}

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def load(cls, models_dir: "str | Path", **kwargs) -> "MovieRecommender":
        models_dir = Path(models_dir)
        emb_path = models_dir / "movie_embeddings.npy"
        map_path = models_dir / "movie_id_mapping.pkl"
        for path in (emb_path, map_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing artifact: {path}")
        embeddings = np.load(emb_path)
        with open(map_path, "rb") as fh:
            mapping = pickle.load(fh)
        catalog = mapping["catalog"] if isinstance(mapping, dict) else mapping
        if not isinstance(catalog, pd.DataFrame):
            catalog = pd.DataFrame(catalog)
        return cls(embeddings, catalog, **kwargs)

    # -- neighbourhood truncation ----------------------------------------- #
    def neighbourhood_threshold(self, k: int, chunk: int = 512) -> np.ndarray:
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        if k in self._threshold_cache:
            return self._threshold_cache[k]
        n = self._unit.shape[0]
        k_eff = min(k, n - 1)
        thresholds = np.empty(n, dtype=np.float32)
        for start in range(0, n, chunk):
            block = self._unit[start : start + chunk] @ self._unit.T
            np.fill_diagonal(block[:, start : start + block.shape[0]], -np.inf)
            thresholds[start : start + block.shape[0]] = np.partition(block, -k_eff, axis=1)[:, -k_eff]
        self._threshold_cache[k] = thresholds
        return thresholds

    # -- title resolution -------------------------------------------------- #
    @property
    def titles(self) -> List[str]:
        return self.catalog["Title"].tolist()

    def search_titles(self, query: str, limit: int = 20) -> List[str]:
        if not query or not str(query).strip():
            return []
        mask = self.catalog["Title"].str.lower().str.contains(str(query).strip().lower(), regex=False)
        return (self.catalog[mask].sort_values("RatingCount", ascending=False)
                ["Title"].head(limit).tolist())

    def resolve_title(self, movie_title: str, on_ambiguous: str = "best") -> int:
        if movie_title is None or not str(movie_title).strip():
            raise MovieNotFoundError("Please provide a movie title.")
        query = str(movie_title).strip()
        lowered = query.lower()
        if lowered in self._title_to_pos:
            return self._title_to_pos[lowered]

        titles_lower = self.catalog["Title"].str.lower()
        prefix = np.flatnonzero(titles_lower.str.startswith(lowered).to_numpy())
        matches = prefix if len(prefix) else np.flatnonzero(
            titles_lower.str.contains(lowered, regex=False).to_numpy())

        if len(matches) == 0:
            suggestions = difflib.get_close_matches(query, self.titles, n=5, cutoff=0.6)
            raise MovieNotFoundError(
                f"'{query}' is not in the MovieLens catalogue."
                + (f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""),
                suggestions=suggestions)
        if len(matches) == 1:
            return int(matches[0])

        ranked = matches[np.argsort(-self._support[matches])]
        candidates = [self.catalog["Title"].iloc[i] for i in ranked[:10]]
        if on_ambiguous == "raise":
            raise AmbiguousTitleError(
                f"'{query}' matches {len(matches)} titles: {', '.join(candidates[:5])}...",
                candidates=candidates)
        return int(ranked[0])

    # -- recommendations ---------------------------------------------------- #
    def _assemble(self, sims: np.ndarray, eligible: np.ndarray, top_n: int) -> pd.DataFrame:
        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            raise RecommenderError(
                "No candidate movies left after filtering. Lower the minimum-ratings "
                "setting and try again.")
        k = min(top_n, n_eligible)
        masked = np.where(eligible, sims, -np.inf)
        top_idx = np.argpartition(-masked, k - 1)[:k]
        top_idx = top_idx[np.argsort(-masked[top_idx])]

        out = self.catalog.iloc[top_idx][
            ["MovieID", "Title", "Genres", "RatingCount", "MeanRating"]].copy()
        out.insert(0, "Rank", np.arange(1, len(out) + 1))
        out.insert(3, "Similarity", np.round(sims[top_idx], 4))
        out = out.reset_index(drop=True)
        out.attrs["n_candidates"] = n_eligible
        if k < top_n:
            out.attrs["warning"] = f"Only {k} candidates available (requested {top_n})."
        return out

    def recommend_movies(self, movie_title: str, top_n: int = 10,
                         min_support: Optional[int] = None,
                         on_ambiguous: str = "best") -> pd.DataFrame:
        if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool):
            raise ValueError(f"top_n must be an integer, got {type(top_n).__name__}.")
        top_n = int(top_n)
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}.")
        min_support = self.default_min_support if min_support is None else int(min_support)

        pos = self.resolve_title(movie_title, on_ambiguous=on_ambiguous)
        query_row = self.catalog.iloc[pos]
        if self._zero_norm[pos]:
            raise MovieNotFoundError(
                f"'{query_row['Title']}' has an all-zero embedding "
                "(no usable training ratings), so similarity is undefined.")

        sims = self._unit @ self._unit[pos]
        eligible = (self._support >= min_support) & (~self._zero_norm)
        eligible[pos] = False

        out = self._assemble(sims, eligible, top_n)
        out.attrs["query_title"] = query_row["Title"]
        out.attrs["query_movie_id"] = int(query_row["MovieID"])
        out.attrs["min_support"] = min_support
        if query_row.get("RatingCount", np.inf) < 20:
            out.attrs["low_support_query"] = (
                f"'{query_row['Title']}' has only {int(query_row['RatingCount'])} "
                "ratings; its embedding is noisy.")
        return out

    def recommend_from_profile(self, liked_titles, top_n: int = 10,
                               min_support: Optional[int] = None,
                               neighbourhood_k: Optional[int] = 50) -> pd.DataFrame:
        liked_titles = [t for t in liked_titles if str(t).strip()]
        if not liked_titles:
            raise MovieNotFoundError("Provide at least one movie you liked.")
        top_n = int(top_n)
        if top_n < 1:
            raise ValueError("top_n must be >= 1.")
        min_support = self.default_min_support if min_support is None else int(min_support)

        positions = [self.resolve_title(t) for t in liked_titles]
        sim_block = self._unit @ self._unit[positions].T
        if neighbourhood_k:
            thresholds = self.neighbourhood_threshold(neighbourhood_k)
            sims = np.where(sim_block >= thresholds[:, None], sim_block, 0.0).sum(axis=1)
        else:
            sims = sim_block.mean(axis=1)

        eligible = (self._support >= min_support) & (~self._zero_norm)
        eligible[positions] = False
        out = self._assemble(sims, eligible, top_n)
        out.attrs["profile"] = [self.catalog["Title"].iloc[p] for p in positions]
        return out


# ═══════════════════════════════════════════════════════════════════════════
# TMDB client  (mirror of src/tmdb_api.py — posters and metadata only)
# ═══════════════════════════════════════════════════════════════════════════
_ARTICLES = {"the", "a", "an", "la", "le", "les", "l'", "un", "une", "il", "lo",
             "gli", "i", "el", "los", "las", "der", "die", "das", "ein", "eine",
             "de", "het", "een", "os", "as", "o"}


class TMDBError(RuntimeError):
    """TMDB returned an unexpected, non-recoverable error."""


class TMDBAuthError(TMDBError):
    """The credential is missing, malformed or rejected."""


def _uninvert_article(title: str) -> str:
    if "," not in title:
        return title
    head, _, tail = title.rpartition(",")
    candidate = tail.strip()
    if candidate.lower() in _ARTICLES:
        joiner = "" if candidate.endswith("'") else " "
        return f"{candidate}{joiner}{head.strip()}"
    return title


def parse_movielens_title(raw_title: str) -> Tuple[str, Optional[int], List[str]]:
    """'Postino, Il (The Postman) (1994)' -> ('Il Postino', 1994, ['The Postman'])."""
    if raw_title is None:
        return "", None, []
    text = str(raw_title).strip()

    year: Optional[int] = None
    year_match = re.search(r"\((\d{4})\)\s*$", text)
    if year_match:
        year = int(year_match.group(1))
        text = text[: year_match.start()].strip()

    alternates: List[str] = []
    for group in re.findall(r"\(([^()]*)\)", text):
        alt = re.sub(r"^\s*a\.?k\.?a\.?\s*", "", group, flags=re.IGNORECASE).strip()
        if alt:
            alternates.append(_uninvert_article(alt))
    text = re.sub(r"\s*\([^()]*\)", "", text).strip()

    clean = _uninvert_article(text)
    seen = {clean.lower()}
    unique_alts = []
    for alt in alternates:
        if alt.lower() not in seen:
            seen.add(alt.lower())
            unique_alts.append(alt)
    return clean, year, unique_alts


def get_poster_url(poster_path: Optional[str], size: str = "w500") -> Optional[str]:
    if not poster_path:
        return None
    if str(poster_path).startswith("http"):
        return str(poster_path)
    return f"{TMDB_IMAGE_URL}/{size}{poster_path}"


class TMDBClient:
    """Minimal, defensive TMDB client. Never raises for a merely missing film."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0,
                 language: str = "en-US", min_interval: float = 0.05) -> None:
        import requests  # imported lazily so the app runs even without it
        from requests.adapters import HTTPAdapter

        try:
            from urllib3.util.retry import Retry
        except ImportError:  # pragma: no cover
            from requests.packages.urllib3.util.retry import Retry  # type: ignore

        credential = api_key or os.getenv("TMDB_API_KEY") or os.getenv("TMDB_API_TOKEN")
        if not credential or not str(credential).strip():
            raise TMDBAuthError("No TMDB credential found.")
        self._credential = str(credential).strip()
        self._is_bearer = self._credential.startswith("eyJ") or len(self._credential) > 60

        self.timeout = timeout
        self.language = language
        self.min_interval = min_interval
        self._last_request_ts = 0.0
        self._cache: Dict[str, Any] = {}
        self._requests = requests

        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET"]), raise_on_status=False)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        if self._is_bearer:
            self.session.headers.update({"Authorization": f"Bearer {self._credential}"})
        self.session.headers.update({"accept": "application/json"})

    def __repr__(self) -> str:  # never expose the credential
        return f"TMDBClient(auth={'bearer-token' if self._is_bearer else 'api-key'})"

    __str__ = __repr__

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[dict]:
        params = dict(params or {})
        params.setdefault("language", self.language)
        if not self._is_bearer:
            params["api_key"] = self._credential

        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_ts = time.time()

        exc_mod = self._requests.exceptions
        try:
            response = self.session.get(f"{TMDB_BASE_URL}{endpoint}", params=params,
                                        timeout=self.timeout)
        except exc_mod.Timeout as exc:
            raise TMDBError(f"TMDB request timed out after {self.timeout}s") from exc
        except exc_mod.ConnectionError as exc:
            raise TMDBError("Could not reach TMDB (network/DNS unavailable)") from exc
        except exc_mod.RequestException as exc:
            raise TMDBError(f"TMDB request failed: {type(exc).__name__}") from exc

        if response.status_code in (401, 403):
            raise TMDBAuthError(f"TMDB rejected the credential (HTTP {response.status_code}).")
        if response.status_code == 404:
            return None
        if response.status_code == 429:
            raise TMDBError("TMDB rate limit exceeded; retry shortly.")
        if not response.ok:
            raise TMDBError(f"TMDB returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise TMDBError("TMDB returned a malformed JSON payload") from exc

    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[dict]:
        if not title or not str(title).strip():
            return None
        key = f"search::{title.lower()}::{year or ''}"
        if key in self._cache:
            return self._cache[key]
        params: Dict[str, Any] = {"query": title, "include_adult": "false"}
        if year:
            params["year"] = year
        results = (self._get("/search/movie", params) or {}).get("results") or []
        best = None
        if results:
            if year:
                exact = [r for r in results if str(r.get("release_date", ""))[:4] == str(year)]
                best = exact[0] if exact else results[0]
            else:
                best = results[0]
        self._cache[key] = best
        return best

    def get_movie_details(self, tmdb_id: int) -> Optional[dict]:
        if tmdb_id is None:
            return None
        key = f"details::{int(tmdb_id)}"
        if key in self._cache:
            return self._cache[key]
        details = self._get(f"/movie/{int(tmdb_id)}")
        self._cache[key] = details
        return details

    def enrich_movie(self, movielens_title: str, with_details: bool = True) -> Dict[str, Any]:
        clean, year, alternates = parse_movielens_title(movielens_title)
        record: Dict[str, Any] = {
            "Title": movielens_title, "Year": year, "TMDB_ID": None, "Poster_URL": None,
            "Overview": None, "Genres": None, "Release_Date": None,
            "TMDB_Rating": None, "tmdb_found": False, "tmdb_error": None,
        }
        attempts = [(clean, year)] + [(a, year) for a in alternates] + [(clean, None)]
        hit = None
        try:
            for candidate_title, candidate_year in attempts:
                hit = self.search_movie(candidate_title, candidate_year)
                if hit:
                    break
        except TMDBAuthError:
            raise
        except TMDBError as exc:
            record["tmdb_error"] = str(exc)
            return record

        if not hit:
            return record

        record.update({
            "TMDB_ID": hit.get("id"),
            "Poster_URL": get_poster_url(hit.get("poster_path")),
            "Overview": hit.get("overview") or None,
            "Release_Date": hit.get("release_date") or None,
            "TMDB_Rating": hit.get("vote_average"),
            "tmdb_found": True,
        })
        if with_details and hit.get("id"):
            try:
                details = self.get_movie_details(hit["id"])
            except TMDBError as exc:
                details, record["tmdb_error"] = None, str(exc)
            if details:
                record["Genres"] = ", ".join(
                    g["name"] for g in details.get("genres", []) if g.get("name")) or None
                record["Overview"] = details.get("overview") or record["Overview"]
                record["Release_Date"] = details.get("release_date") or record["Release_Date"]
                record["TMDB_Rating"] = details.get("vote_average", record["TMDB_Rating"])
                record["Poster_URL"] = (get_poster_url(details.get("poster_path"))
                                        or record["Poster_URL"])
        return record


# ═══════════════════════════════════════════════════════════════════════════
# Styling: projection-booth palette, condensed display type, ticket-stub meter
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
      :root{
        --ink:#0f141f; --slate:#1b2130; --line:#2b3547;
        --amber:#f0b429; --mist:#c8d0dd; --muted:#7d8798;
      }
      .stApp { background: var(--ink); }
      h1,h2,h3, .cl-title { font-family:'Barlow Condensed',sans-serif; letter-spacing:.4px; }
      html, body, [class*="css"] { font-family:'Inter',sans-serif; }
      .cl-head{ border-bottom:1px solid var(--line); padding-bottom:.6rem; margin-bottom:1.2rem; }
      .cl-head h1{ color:#f5f7fa; font-size:2.35rem; margin:0; font-weight:600; }
      .cl-head p{ color:var(--muted); margin:.25rem 0 0; font-size:.92rem; }
      .cl-card{
        background:var(--slate); border:1px solid var(--line); border-radius:10px;
        padding:.65rem .7rem .8rem; height:100%;
      }
      .cl-card img{ border-radius:6px; width:100%; display:block; }
      .cl-name{ font-family:'Barlow Condensed',sans-serif; font-size:1.12rem; font-weight:600;
                color:#f5f7fa; margin:.55rem 0 .15rem; line-height:1.15; }
      .cl-meta{ color:var(--muted); font-size:.76rem; margin-bottom:.4rem; }
      .cl-chip{ display:inline-block; border:1px solid var(--line); color:var(--mist);
                border-radius:999px; padding:.05rem .45rem; font-size:.68rem; margin:0 .2rem .2rem 0; }
      .cl-stub{
        display:flex; align-items:center; gap:.5rem; margin-top:.5rem;
        border-top:1px dashed var(--line); padding-top:.45rem;
      }
      .cl-stub .num{ font-family:'IBM Plex Mono',monospace; font-size:.74rem; color:var(--amber); }
      .cl-bar{ flex:1; height:4px; background:#121826; border-radius:2px; overflow:hidden; }
      .cl-bar span{ display:block; height:100%; background:var(--amber); }
      .cl-rank{ font-family:'IBM Plex Mono',monospace; color:var(--muted); font-size:.7rem; }
      @media (prefers-reduced-motion: no-preference){
        .cl-card{ transition: border-color .18s ease; }
        .cl-card:hover{ border-color:var(--amber); }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# Cached resources
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading movie embeddings…")
def load_recommender(models_dir: str) -> MovieRecommender:
    return MovieRecommender.load(models_dir)


@st.cache_resource(show_spinner=False)
def load_tmdb_client() -> Optional[TMDBClient]:
    """Credential from Streamlit secrets, the environment, or a local .env."""
    credential = None
    try:
        credential = st.secrets.get("TMDB_API_KEY") or st.secrets.get("TMDB_API_TOKEN")
    except Exception:
        credential = None
    if not credential:
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        credential = os.getenv("TMDB_API_KEY") or os.getenv("TMDB_API_TOKEN")
    if not credential:
        return None
    try:
        return TMDBClient(credential)
    except (TMDBAuthError, ImportError):
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_metadata(title: str) -> dict:
    """One TMDB lookup per title, cached for a day. Never raises."""
    client = load_tmdb_client()
    if client is None:
        return {"tmdb_found": False, "tmdb_error": "no-credential"}
    try:
        return client.enrich_movie(title)
    except TMDBAuthError:
        return {"tmdb_found": False, "tmdb_error": "auth"}
    except TMDBError as exc:
        return {"tmdb_found": False, "tmdb_error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════
def render_card(row: pd.Series, meta: dict, use_tmdb: bool) -> None:
    poster = meta.get("Poster_URL") if use_tmdb else None
    genres = (meta.get("Genres") if (use_tmdb and meta.get("Genres"))
              else str(row["Genres"]).replace("|", ", "))
    year = str(meta.get("Release_Date") or "")[:4]
    rating = meta.get("TMDB_Rating")

    bits = []
    if year:
        bits.append(year)
    if rating:
        bits.append(f"TMDB {float(rating):.1f}/10")
    bits.append(f"{int(row['RatingCount'])} MovieLens ratings")

    chips = "".join(f"<span class='cl-chip'>{g.strip()}</span>"
                    for g in str(genres).split(",") if g.strip())
    pct = max(0.0, min(1.0, float(row["Similarity"]))) * 100

    st.markdown(
        f"""
        <div class="cl-card">
          <img src="{poster or POSTER_FALLBACK}" alt="Poster for {row['Title']}">
          <div class="cl-name">{row['Title']}</div>
          <div class="cl-meta">{' · '.join(bits)}</div>
          <div>{chips}</div>
          <div class="cl-stub">
            <span class="cl-rank">#{int(row['Rank']):02d}</span>
            <span class="cl-bar"><span style="width:{pct:.0f}%"></span></span>
            <span class="num">{float(row['Similarity']):.3f}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    overview = meta.get("Overview")
    if use_tmdb and overview:
        with st.expander("Overview"):
            st.write(overview)


def render_results(recs: pd.DataFrame, use_tmdb: bool, columns: int = 5) -> None:
    metas: Dict[str, dict] = {}
    if use_tmdb:
        progress = st.progress(0.0, text="Fetching posters from TMDB…")
        for i, title in enumerate(recs["Title"], start=1):
            metas[title] = fetch_metadata(title)
            progress.progress(i / len(recs), text=f"Fetching posters from TMDB… ({i}/{len(recs)})")
        progress.empty()
        failed = [t for t, m in metas.items() if not m.get("tmdb_found")]
        if failed and len(failed) == len(recs):
            st.warning("TMDB returned nothing for these titles — showing MovieLens metadata instead.")
        elif failed:
            st.caption(f"No TMDB match for: {', '.join(failed)} — MovieLens data shown for those.")

    for start in range(0, len(recs), columns):
        chunk = recs.iloc[start : start + columns]
        for col, (_, row) in zip(st.columns(columns), chunk.iterrows()):
            with col:
                render_card(row, metas.get(row["Title"], {}), use_tmdb)


def show_missing_artifacts(error: Exception) -> None:
    """Turn a redacted crash into something actionable."""
    st.error(f"Could not load the model artifacts: {error}")
    st.markdown(
        f"""
The app needs these two files, and they are not where it looked:

```
{MODELS_DIR / 'movie_embeddings.npy'}
{MODELS_DIR / 'movie_id_mapping.pkl'}
```

**Running locally?** Open `movie_recommender.ipynb` and run it top to bottom —
section 20 writes both files.

**Deployed on Streamlit Cloud?** They are probably not committed. `.gitignore`
un-ignores them for exactly this reason; confirm with
`git check-ignore -v models/movie_embeddings.npy` (it should print nothing),
then `git add models/movie_embeddings.npy models/movie_id_mapping.pkl && git push`.
"""
    )
    with st.expander("What the app can actually see on disk"):
        try:
            here = sorted(p.name + ("/" if p.is_dir() else "") for p in ROOT.iterdir())
            st.write(f"**{ROOT}**"); st.code("\n".join(here) or "(empty)")
            if MODELS_DIR.exists():
                st.write(f"**{MODELS_DIR}**")
                st.code("\n".join(sorted(p.name for p in MODELS_DIR.iterdir())) or "(empty)")
            else:
                st.write(f"**{MODELS_DIR}** does not exist.")
        except OSError as exc:
            st.write(f"Could not list files: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="cl-head">
      <h1>CineLatent</h1>
      <p>Neighbours in the latent space of a deep autoencoder trained on 1,000,209 MovieLens ratings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    rec = load_recommender(str(MODELS_DIR))
except FileNotFoundError as exc:
    show_missing_artifacts(exc)
    st.stop()
except (pickle.UnpicklingError, AttributeError, ModuleNotFoundError) as exc:
    st.error(f"The saved catalogue could not be unpickled: {exc}")
    st.info(
        "This usually means the artifacts were written by a much older or newer "
        "pandas than the one installed here. Re-run the notebook in the same "
        "environment, or align the pandas version in requirements.txt."
    )
    st.stop()
except Exception as exc:  # corrupt or mismatched artifacts
    st.error(f"Could not load the saved artifacts: {exc}")
    st.stop()

tmdb_client = load_tmdb_client()

with st.sidebar:
    st.subheader("Settings")
    top_n = st.slider("How many recommendations", 5, 20, 10, step=5)
    min_support = st.slider(
        "Minimum ratings per recommended movie", 0, 200, 10, step=10,
        help="Movies rated by only a handful of users have noisy embeddings. "
             "Raise this for safer, more mainstream suggestions.",
    )
    want_tmdb = st.toggle("Show posters and TMDB metadata", value=tmdb_client is not None)
    if want_tmdb and tmdb_client is None:
        st.warning(
            "No TMDB credential found. On Streamlit Cloud add it under "
            "Settings → Secrets as `TMDB_API_KEY`; locally put it in `.env`.",
            icon="🔑",
        )
        want_tmdb = False
    st.divider()
    st.caption(
        f"{len(rec.catalog):,} movies · {rec.embeddings.shape[1]}-dimensional embeddings\n\n"
        "Similarity is cosine distance between latent vectors learned from rating "
        "patterns alone — no plot, cast or genre information is used."
    )

tab_similar, tab_profile = st.tabs(["Similar to one movie", "Blend a few favourites"])

with tab_similar:
    default_idx = rec.titles.index("Toy Story (1995)") if "Toy Story (1995)" in rec.titles else 0
    choice = st.selectbox("Pick a movie you liked", rec.titles, index=default_idx,
                          help="Type to search the MovieLens catalogue.")
    if st.button("Recommend", type="primary", key="btn_similar"):
        try:
            recs = rec.recommend_movies(choice, top_n=top_n, min_support=min_support)
        except (MovieNotFoundError, AmbiguousTitleError, RecommenderError) as exc:
            st.error(str(exc))
        else:
            if recs.attrs.get("low_support_query"):
                st.info(recs.attrs["low_support_query"])
            st.caption(f"Closest neighbours to **{recs.attrs['query_title']}** "
                       f"among {recs.attrs['n_candidates']:,} eligible movies.")
            render_results(recs, want_tmdb)

with tab_profile:
    picks = st.multiselect(
        "Pick two or more movies you liked", rec.titles,
        default=[t for t in ["Alien (1979)", "Blade Runner (1982)"] if t in rec.titles],
    )
    if st.button("Blend these", type="primary", key="btn_profile"):
        if len(picks) < 1:
            st.error("Choose at least one movie first.")
        else:
            try:
                recs = rec.recommend_from_profile(picks, top_n=top_n, min_support=min_support)
            except (MovieNotFoundError, RecommenderError) as exc:
                st.error(str(exc))
            else:
                st.caption(f"Averaged latent neighbours of {len(picks)} selected films.")
                render_results(recs, want_tmdb)
