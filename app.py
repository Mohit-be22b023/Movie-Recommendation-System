"""
CineLatent -- Streamlit front-end for the MovieLens deep-autoencoder recommender.

This app is a *consumer* of artifacts. It loads:
    models/movie_embeddings.npy      (one latent vector per movie)
    models/movie_id_mapping.pkl      (MovieID <-> row index + catalogue)
and never trains anything. Run the notebook first to produce them.

    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.recommender import (  # noqa: E402
    AmbiguousTitleError,
    MovieNotFoundError,
    MovieRecommender,
    RecommenderError,
)
from src.tmdb_api import TMDBAuthError, TMDBClient, TMDBError  # noqa: E402

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
POSTER_FALLBACK = "https://placehold.co/500x750/1b2130/64748b?text=No+poster"

st.set_page_config(
    page_title="CineLatent — movie recommendations",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling: projection-booth palette, condensed display type, ticket-stub meter #
# --------------------------------------------------------------------------- #
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
      /* signature element: the similarity readout as a perforated ticket stub */
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


# --------------------------------------------------------------------------- #
# Cached resources                                                             #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading movie embeddings…")
def load_recommender(models_dir: str) -> MovieRecommender:
    return MovieRecommender.load(models_dir)


@st.cache_resource(show_spinner=False)
def load_tmdb_client() -> TMDBClient | None:
    """Read the credential from Streamlit secrets, the environment, or .env."""
    credential = None
    try:
        credential = st.secrets.get("TMDB_API_KEY")  # type: ignore[attr-defined]
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
        return TMDBClient(credential, cache_path=ROOT / "outputs" / "tmdb_cache.json")
    except TMDBAuthError:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_metadata(title: str) -> dict:
    """One TMDB lookup per title, cached for a day. Never raises."""
    client = load_tmdb_client()
    if client is None:
        return {"tmdb_found": False, "tmdb_error": "no-credential"}
    try:
        record = client.enrich_movie(title)
        client.save_cache()
        return record
    except TMDBAuthError:
        return {"tmdb_found": False, "tmdb_error": "auth"}
    except TMDBError as exc:
        return {"tmdb_found": False, "tmdb_error": str(exc)}


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def render_card(row: pd.Series, meta: dict, use_tmdb: bool) -> None:
    poster = meta.get("Poster_URL") if use_tmdb else None
    genres = meta.get("Genres") if (use_tmdb and meta.get("Genres")) else str(row["Genres"]).replace("|", ", ")
    year = str(meta.get("Release_Date") or "")[:4]
    rating = meta.get("TMDB_Rating")

    bits = []
    if year:
        bits.append(year)
    if rating:
        bits.append(f"TMDB {float(rating):.1f}/10")
    bits.append(f"{int(row['RatingCount'])} MovieLens ratings")

    chips = "".join(
        f"<span class='cl-chip'>{g.strip()}</span>" for g in str(genres).split(",") if g.strip()
    )
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
    metas = {}
    if use_tmdb:
        progress = st.progress(0.0, text="Fetching posters from TMDB…")
        for i, title in enumerate(recs["Title"], start=1):
            metas[title] = fetch_metadata(title)
            progress.progress(i / len(recs), text=f"Fetching posters from TMDB… ({i}/{len(recs)})")
        progress.empty()
        failed = [t for t, m in metas.items() if not m.get("tmdb_found")]
        if failed and len(failed) == len(recs):
            st.warning(
                "TMDB returned nothing for these titles — showing MovieLens metadata instead."
            )
        elif failed:
            st.caption(f"No TMDB match for: {', '.join(failed)} — MovieLens data shown for those.")

    for start in range(0, len(recs), columns):
        chunk = recs.iloc[start : start + columns]
        for col, (_, row) in zip(st.columns(columns), chunk.iterrows()):
            with col:
                render_card(row, metas.get(row["Title"], {}), use_tmdb)


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #
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
    st.error(str(exc))
    st.info(
        "Open `notebooks/movie_recommender.ipynb`, run it top to bottom, and the "
        "`models/` folder will be created. Then reload this page."
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
        "Minimum ratings per recommended movie",
        0, 200, 10, step=10,
        help="Movies rated by only a handful of users have noisy embeddings. "
             "Raise this for safer, more mainstream suggestions.",
    )
    want_tmdb = st.toggle("Show posters and TMDB metadata", value=tmdb_client is not None)
    if want_tmdb and tmdb_client is None:
        st.warning(
            "No TMDB credential found. Add `TMDB_API_KEY` to `.env` or to "
            "`.streamlit/secrets.toml`, then reload.",
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
    default_idx = (
        rec.titles.index("Toy Story (1995)") if "Toy Story (1995)" in rec.titles else 0
    )
    choice = st.selectbox(
        "Pick a movie you liked", rec.titles, index=default_idx,
        help="Type to search the MovieLens catalogue.",
    )
    if st.button("Recommend", type="primary", key="btn_similar"):
        try:
            recs = rec.recommend_movies(choice, top_n=top_n, min_support=min_support)
        except (MovieNotFoundError, AmbiguousTitleError, RecommenderError) as exc:
            st.error(str(exc))
        else:
            note = recs.attrs.get("low_support_query")
            if note:
                st.info(note)
            st.caption(
                f"Closest neighbours to **{recs.attrs['query_title']}** "
                f"among {recs.attrs['n_candidates']:,} eligible movies."
            )
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
