# Movie-Recommendation-System

🎞️ CineLatent — Deep Autoencoder Movie Recommender (MovieLens 1M)

A complete, end-to-end collaborative-filtering system: a deep autoencoder learns 128-dimensional latent representations of 3,706 films from 1,000,209 ratings, cosine similarity in that space produces recommendations, the TMDB API adds posters and metadata, and a Streamlit app serves the result.

Every number in this README was printed by the notebook. Nothing is estimated, rounded up, or carried over from a paper.

	
Test RMSE (held-out ratings)	0.8604 vs 0.9321 for the best baseline
Best Top-10 scorer	Precision@10 0.0927, Recall@10 0.1137, Hit-Rate@10 0.5179, NDCG@10 0.1271
Popularity baseline (same protocol)	Precision@10 0.0733, Recall@10 0.0861, Hit-Rate@10 0.4369, NDCG@10 0.0986
Training	30 epochs, best at epoch 20, ~3 minutes on a single CPU core
Table of contents

Project overview · Business problem · Dataset · Technologies · System architecture · Data preprocessing · Collaborative filtering · Deep autoencoder · Embedding generation · Recommendation algorithm · Evaluation · TMDB API integration · Streamlit application · Colab / Jupyter setup · Local setup · API configuration · How to run · Deploying to Streamlit Cloud · Results · Limitations · Future improvements

Project overview

CineLatent answers one question — "I liked this film, what else would I like?" — using nothing but rating behaviour. No plot summaries, no cast lists, no genre labels enter the model. Two films end up close together because the same people rated them the same way, which is what collaborative filtering means.

The project is split cleanly in two:

movie_recommender.ipynb — the ML workflow: load, clean, explore, split, train, evaluate, extract embeddings, save artifacts. This is where the work happens, and it runs top to bottom in Jupyter or Google Colab.
app.py — the interface. It loads saved artifacts and never trains anything, so it starts in about a second.

The notebook writes src/recommender.py, src/tmdb_api.py and app.py to disk with %%writefile, so the notebook and the app can never drift apart.

Business problem

A catalogue with thousands of titles and a user who will scroll through maybe twenty of them. Editorial curation does not scale, and the obvious fallback — show whatever is most popular — is a genuinely strong baseline that a recommender has to beat, not merely claim to replace. This project measures that comparison explicitly rather than assuming the win.

The commercial framing: recommendations drive a large share of streaming watch-time, and the marginal cost of a better ranking function is close to zero once the pipeline exists.

Dataset

MovieLens 1M (GroupLens Research), three ::-delimited, Latin-1 encoded files:

file	rows	schema
ratings.dat	1,000,209	UserID :: MovieID :: Rating :: Timestamp
movies.dat	3,883	MovieID :: Title :: Genres
users.dat	6,040	UserID :: Gender :: Age :: Occupation :: Zip-code

Measured during validation (section 04 of the notebook):

6,040 users, 3,706 films that received at least one rating
177 films in movies.dat that nobody rated — kept in the catalogue, but they cannot receive an embedding
zero missing values, zero duplicate rows, zero duplicate (user, movie) pairs
ratings are whole stars 1–5; every user has at least 20 ratings
matrix sparsity 95.53% (density 4.47%)
ratings span 2000-04-25 to 2003-02-28

Nothing was dropped. All 1,000,209 ratings reach the model.

Technologies

Python 3.10+ · pandas · NumPy · SciPy (sparse matrices) · scikit-learn (PCA, splitting utilities) · TensorFlow / Keras 3 · Matplotlib · seaborn · requests · python-dotenv · Streamlit · TMDB REST API

Dependencies are split deliberately: requirements.txt holds only what app.py needs at runtime (streamlit, numpy, pandas, requests, python-dotenv), and requirements-dev.txt adds the training stack. The app loads saved arrays and never imports TensorFlow, so a deployment does not need a 600 MB wheel.

System architecture
MovieLens 1M (.dat)
      │
      ▼  cleaning + validation report
   EDA (9 figures)
      │
      ▼  movies × users sparse matrix   (missing ≠ zero)
 per-user 80/10/10 split                (leakage asserted, not assumed)
      │
      ▼  deep autoencoder, masked MSE loss
 latent layer ──────────► movie embeddings (3706 × 128)
      │                          │
      ▼                          ▼
 reconstructed ratings     cosine similarity + top-50 neighbourhood
 (RMSE / MAE)              (Top-N recommendations)
                                 │
                                 ▼
                        TMDB API (posters, overviews, ratings)
                                 │
                                 ▼
                          Streamlit application
Data preprocessing

Two decisions carry the entire model, and both are the places where recommender projects usually go quietly wrong.

1. Missing ratings are not zeros. The matrix is stored sparse, where an absent cell means unobserved. Ratings are scaled as r / 5, so observed values live in {0.2, 0.4, 0.6, 0.8, 1.0} and 0.0 is reserved to mean "no rating". The common alternative, (r - 1) / 4, maps a one-star rating to 0.0 and makes it indistinguishable from an empty cell.

2. The loss is masked. Only observed cells contribute:

L = Σ mᵢᵤ (rᵢᵤ − r̂ᵢᵤ)² / Σ mᵢᵤ ,   mᵢᵤ = 1 when the rating exists

Without the mask, 95.53% of the targets would be zeros and the network would learn to predict that everybody dislikes everything.

Split. Each user's ratings are shuffled with a fixed seed and cut 80/10/10, so every user appears in all three sets: 802,553 train / 100,273 validation / 97,383 test. The notebook asserts that no (user, movie) pair appears in two splits and that every validation and test cell is empty in the matrix the network reads. Validation targets are held-out cells scored against the training input, so val_loss measures generalisation with no leakage.

Collaborative filtering

The orientation of the matrix is a modelling decision, not a formatting one:

orientation	one sample is	the bottleneck yields
user-based (U-AutoRec)	a user's ratings over all films	one vector per user
item-based (I-AutoRec) ← used here	a film's ratings from all users	one vector per film

Because the product recommends films similar to a film, the item-based orientation gives exactly the representation needed straight from the bottleneck, with no reinterpretation step. It is also the better-conditioned problem here: 3,706 films averaging 270 ratings each is a denser signal per sample than 6,040 users averaging 166.

Deep autoencoder
input   6,040 user ratings for one film   (0 = unobserved)
  │  Dense 512  ReLU     + L2(1e-5), Dropout 0.2
  │  Dense 256  ReLU     + L2(1e-5), Dropout 0.2
  │  Dense 128  linear   ◄── the movie embedding
  │  Dense 256  ReLU     + L2(1e-5), Dropout 0.2
  │  Dense 512  ReLU     + L2(1e-5)
output  6,040 reconstructed ratings       (sigmoid, ×5 → stars)

6,520,344 parameters. Adam (lr 1e-3, halved on plateau), batch size 64, early stopping on validation masked RMSE with patience 10 and best-weight restore. Trained 30 epochs, best at epoch 20, validation masked RMSE 0.8695 stars.

Design rationale: the bottleneck is linear because ReLU would clamp every negative coordinate to zero and squash the range of cosine similarity; dropout sits on the hidden layers rather than the input because dropping ratings from an already 95%-empty vector adds noise without regularising; the output is a sigmoid because ratings are bounded.

Embedding generation

The encoder is the input-to-bottleneck half of the trained network. Running the rating matrix through it yields (3706, 128) float32 — one row per film, in the same order as the catalogue.

The notebook asserts, rather than assumes: one embedding per film, the MovieID ↔ index mapping is invertible, encoding a single film in isolation reproduces its row of the batch output, no non-finite values, and zero films with a degenerate all-zero vector.

What is not claimed: there is no per-user latent vector in this model. Obtaining one would require training the transposed network — the case where "movie embeddings" would have to be derived from decoder weights, and where the term is most often used loosely. Reconstructed ratings are used only for what they are: predicted ratings, evaluated with RMSE/MAE.

Recommendation algorithm
python
from src.recommender import MovieRecommender

rec = MovieRecommender.load("models")
rec.recommend_movies("Godfather, The (1972)", top_n=10)
resolve the title (exact → prefix → substring, ties broken by popularity)
look up its embedding
cosine similarity against all 3,706 embeddings
exclude the query film and films below min_support training ratings
return the Top-N with title, genres, similarity and rating counts

Handled explicitly, and demonstrated in section 15 of the notebook: unknown titles (with close-match suggestions), misspellings, ambiguous partial titles ("Toy Story" → two candidates), top_n that is zero, negative or non-integer, top_n larger than the candidate pool, films with almost no ratings, and a min_support so high that nothing qualifies.

recommend_from_profile(["Alien (1979)", "Blade Runner (1982)"]) blends several films using the top-50 neighbourhood scorer — the same algorithm the evaluation section measures.

A sample of actual output:

Because you watched  Godfather, The (1972)
   1. Godfather: Part II, The (1974)              cos 0.991
   2. Close Encounters of the Third Kind (1977)   cos 0.941
   3. Apocalypse Now (1979)                       cos 0.932
   4. Unforgiven (1992)                           cos 0.931
   5. Dog Day Afternoon (1975)                    cos 0.914

Because you watched  Toy Story (1995)
   1. Toy Story 2 (1999)                          cos 0.981
   2. Aladdin (1992)                              cos 0.958
   3. Bug's Life, A (1998)                        cos 0.944
   4. Lion King, The (1994)                       cos 0.907
Evaluation

Two different questions, two different measurements. Conflating them is how recommender results get oversold.

(a) Rating prediction — 97,383 held-out test ratings
model	RMSE	MAE
Deep autoencoder	0.8604	0.6720
Movie + user bias	0.9321	0.7314
Movie mean	0.9800	0.7830
User mean	1.0334	0.8268
Global mean	1.1168	0.9332

7.7% lower RMSE than the strongest baseline. A model that cannot beat "predict this film's average" has learned nothing, so the baselines are the point.

(b) Top-10 ranking — 5,862 users with at least one relevant test film

Relevance = a test rating of 4 or 5. Films already in the user's training set are excluded from the candidate pool.

scorer	Precision@10	Recall@10	Hit-Rate@10	NDCG@10
Embedding item-item, top-50 neighbours	0.0927	0.1137	0.5179	0.1271
Popularity baseline	0.0733	0.0861	0.4369	0.0986
Autoencoder reconstruction	0.0596	0.0669	0.3623	0.0813
Embedding item-item, all neighbours	0.0528	0.0728	0.3748	0.0735

Two findings worth stating plainly:

The autoencoder's raw reconstruction ranks worse than popularity despite winning clearly on RMSE. Predicting the rating of films a user already chose to watch is a different task from choosing which film to surface. This result is reported rather than hidden.
Neighbourhood truncation is what makes the embedding scorer win. Letting every film vote for every other drowns the signal in thousands of weak similarities (Precision@10 0.0528); restricting each film to its 50 closest peers lifts it to 0.0927, ahead of popularity on all four metrics.

Precision ceilings are low by construction: a user has only a handful of test ratings, so most of the ten slots cannot be hits even for a perfect model. Hit-Rate@10 is the most intuitive figure — 51.8% of users got at least one film they later rated 4+, against 43.7% for popularity.

(c) Do the embeddings mean anything?

Genre labels were never shown to the model, so they make an independent audit. Mean Jaccard genre overlap with the ten nearest films: 0.4179, against 0.1749 for ten random films — a 2.4× separation. The PCA projection in section 13 shows the same thing visually.

No plain "accuracy" is reported anywhere: for a 1–5 star prediction there is no meaningful correct/incorrect binary outcome.

TMDB API integration

TMDB is an external REST API — nothing to upload, nothing to scrape, and it plays no part in training. src/tmdb_api.py provides search_movie(), get_movie_details(), get_poster_url() and enrich_movie(), and handles:

both credential styles — a v3 API key goes in the query string, a v4 read token in an Authorization: Bearer header; the style is auto-detected
MovieLens title quirks — "American President, The (1995)" → The American President (1995); "Postino, Il (The Postman) (1994)" → Il Postino with The Postman as a fallback query; "Professional, The (a.k.a. Leon: The Professional) (1994)" handled too. Searched as written, these fail
failure modes — timeouts, retries with backoff, 401/403 (auth), 404, 429 (rate limit), 5xx, malformed JSON
caching — in-memory and on-disk, so a film is never requested twice
graceful degradation — a film TMDB cannot find returns a record with the MovieLens title intact and tmdb_found = False, never an exception mid-page

The credential is never printed, never logged, and is masked in __repr__.

Verification note. The TMDB client is covered by 12 offline test groups using a stubbed HTTP transport (see Tests below). Live calls against api.themoviedb.org were not exercised in the environment where this was built — outbound access to that host was unavailable. Add your key and run section 18 of the notebook to confirm against the real API.

Streamlit application
bash
streamlit run app.py

Two tabs — Similar to one movie and Blend a few favourites — over a searchable catalogue of all 3,706 films. Each result is a card with poster, title, genres, release year, TMDB rating, an expandable overview, and the cosine similarity rendered as a ticket-stub meter. Sidebar controls set the number of recommendations and the minimum-ratings threshold.

app.py is a single self-contained file: it imports only third-party packages, never src/, so it deploys anywhere the two artifact files reach. It loads movie_embeddings.npy and movie_id_mapping.pkl and never trains. Without a TMDB key it still works: posters fall back to a placeholder and MovieLens genres are shown instead. Missing artifacts produce a clear instruction to run the notebook, not a stack trace.

Google Colab / Jupyter setup

The same notebook runs in both — it detects Colab and adapts.

Open movie_recommender.ipynb in Colab (File → Upload notebook).
Run section 01. It creates data/, models/, outputs/, src/.
Get the dataset. Section 03 prints the exact commands if the files are absent:
python
   !wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip
   !unzip -q -o ml-1m.zip && mkdir -p data && mv ml-1m/*.dat data/

or upload ratings.dat, movies.dat, users.dat with files.upload(). 4. Run every cell top to bottom. On a free CPU runtime the whole notebook takes roughly 5 minutes, training included; a GPU runtime is faster but unnecessary. 5. Optional: set USE_GOOGLE_DRIVE = True in section 01 to keep artifacts on Drive across runtime restarts. 6. Optional: section 20 zips models/ and downloads it to your machine.

Only genuinely missing packages are installed — on Colab the install cell usually prints "nothing to install". The notebook uses relative paths throughout and contains no machine-specific paths, no Docker, and no database.

Local setup
bash
git clone <your-repo-url> && cd cinelatent
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                   # notebook + app
# pip install -r requirements.txt                     # app only (no TensorFlow)

# dataset
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip && mkdir -p data && mv ml-1m/*.dat data/

jupyter lab movie_recommender.ipynb

The notebook must live at the project root (it steps up one directory automatically if you move it into a subfolder).

API configuration

Get a free key at https://www.themoviedb.org/settings/api. Either credential style works.

bash
cp .env.example .env
# then edit .env:
TMDB_API_KEY=your_tmdb_api_token

Resolution order is environment variable → .env → interactive getpass prompt (which does not echo). For Streamlit Cloud, copy .streamlit/secrets.toml.example to .streamlit/secrets.toml instead.

Keeping the key out of a public repository

The credential is never written in code — it is read at runtime from Streamlit secrets, the environment, or a git-ignored .env. On Streamlit Cloud the secret lives in the platform, not the repository, so a public repo is fine.

Four defences, each verified by tests/test_no_secrets.py:

.env and .streamlit/secrets.toml are git-ignored (only the .example files, holding placeholders, are committed).
The notebook reads the key with getpass and never echoes it; the test scans every cell source and text output.
TMDBClient.__repr__ masks the credential.
Error messages are redacted. A v3 key travels in the URL query string, so a proxy or gateway error page can echo the whole URL back — _redact() strips the key before the message reaches the UI or the logs. The test simulates exactly that echo and asserts the key does not survive.

The scanner is checked against a planted dummy key, so it fails loudly rather than passing vacuously.

If you have ever committed a real key, .gitignore does not untrack it and it stays in the history. Revoke it at themoviedb.org and issue a new one — that is faster and safer than rewriting history.

The recommender itself does not need TMDB. Only posters and overviews do.

How to run
bash
# 1. train and evaluate (writes models/, outputs/, src/, app.py)
jupyter lab movie_recommender.ipynb        # or open it in Colab

# 2. serve
streamlit run app.py

# 3. tests
python tests/test_modules.py               # 18 groups, no network needed
python tests/test_app_smoke.py             # headless Streamlit render check
python tests/test_app_parity.py            # app.py vs src/ must agree exactly
python tests/test_posters.py               # poster rendering + TMDB status reporting
python tests/test_no_secrets.py            # run before every push to a public repo
Project structure
movie_recommender.ipynb     the complete ML workflow (Jupyter + Colab)
app.py                      Streamlit interface — no local imports (45 KB)
app_embedded.py             same app with artifacts baked in — deploy alone
src/
    recommender.py          MovieRecommender: title resolution, cosine Top-N
    tmdb_api.py             TMDBClient: search, details, posters, caching
tests/
    test_modules.py         offline tests for both modules
    test_app_smoke.py       headless app render test
    test_app_parity.py      app.py's embedded copy == src/recommender.py
    test_posters.py         poster rendering and TMDB credential reporting
    test_no_secrets.py      scans for committed credentials; checks redaction
models/
    autoencoder.keras       full trained model  (regenerated by the notebook)
    encoder.keras           input → latent half; regenerates the embeddings
    movie_embeddings.npy    (3706, 128) float32
    movie_id_mapping.pkl    catalogue + MovieID↔index dicts + run metadata
    training_history.json   per-epoch loss and RMSE
outputs/
    eda/                    9 figures (11 panels)
    evaluation/             metrics CSVs, summary.json, comparison charts
    recommendations/        example Top-10 tables, incl. one with TMDB columns
requirements.txt            app runtime only — the file Streamlit Cloud installs
requirements-dev.txt        the above plus TensorFlow, sklearn, matplotlib, Jupyter
.env.example  ·  .gitignore  ·  .streamlit/config.toml

Re-running the notebook after training reloads the saved model instead of retraining (FORCE_RETRAIN = False), so a full re-run takes seconds.

Deploying to Streamlit Cloud

The app runs on share.streamlit.io with no code changes, but three things have to be true in the repository.

1. app.py needs no other source files. It carries its own copy of the recommender and TMDB client, so ModuleNotFoundError: No module named 'src' — the classic Streamlit Cloud failure when a subfolder does not reach the host, or when a folder is Src on Windows and src on Linux — cannot happen. The trade-off is a duplicated ~250 lines, which tests/test_app_parity.py guards by asserting that app.py and src/recommender.py return identical recommendations, similarity scores and exceptions.

2. The artifacts travel inside app.py, so there is nothing else to commit. If you would rather ship them as separate files, models/movie_embeddings.npy (1.9 MB) and models/movie_id_mapping.pkl (332 KB) are un-ignored in .gitignore and take priority over the embedded copy:

bash
git check-ignore -v models/movie_embeddings.npy   # should print nothing

The 75 MB autoencoder.keras stays out either way — the app never opens it.

3. Keep requirements.txt at the repository root and slim. Streamlit Cloud installs that file and nothing else. If it is missing, only Streamlit's own dependencies get installed and every import in app.py fails; if it contains TensorFlow, the build is slow and can exhaust the free tier's memory.

Then, in the Streamlit Cloud UI:

Main file path → app.py
Settings → Secrets → paste TMDB_API_KEY = "your_token" (this is where the key belongs on Cloud; never commit secrets.toml)

Without the secret the app still deploys and works — posters fall back to a placeholder and MovieLens genres are shown.

Deployment checklist
repo-root/
├── app.py                        ← "Main file path" in the Streamlit UI  (45 KB)
├── requirements.txt              ← slim; Cloud installs this
└── models/
    ├── movie_embeddings.npy      (1.9 MB)
    └── movie_id_mapping.pkl      (332 KB)

app.py imports nothing from src/, so a missing subfolder cannot break it.

If you would rather deploy a single file, app_embedded.py is the same code with the embeddings and catalogue baked in as compressed base64 (2.4 MB): rename it to app.py and neither models/ nor anything else is needed. Regenerate it after retraining with python app.py --embed. Files in models/ still take priority when they exist, so retraining keeps working; run python app.py --embed afterwards to refresh the copy inside the file. The payload is float32 by measurement, not habit: float16 halves it but reorders 17% of top-10 lists, and int8 reorders 80%.

If models/ is absent the app does not crash: it shows the paths it looked in, lists what it can actually see on disk, and gives the git commands to fix it.

Results
Rating prediction: test RMSE 0.8604 / MAE 0.6720, 7.7% below the best of four baselines (0.9321). For context, published I-AutoRec results on MovieLens 1M sit near 0.83 RMSE under a different split protocol, so this is in the expected range for the architecture.
Top-10 ranking: Precision@10 0.0927, Recall@10 0.1137, Hit-Rate@10 0.5179, NDCG@10 0.1271 — ahead of the popularity baseline on all four.
Embedding quality: genre overlap with learned neighbours 0.4179 vs 0.1749 random, with genres never shown to the model.
Cost: 30 epochs in about 3 minutes on one CPU core; the app loads in about a second.

All figures come from a single run with SEED = 42; the notebook reproduces them.

Limitations
Cold start. A film with no ratings gets no embedding and a new user has no profile. This architecture cannot fix either.
Temporal cohorts. Nearly half of MovieLens 1M was rated within a few months of 2000, so the model partly learns "watched in the same period" rather than "liked by the same taste". It is visible in the output — Close Encounters of the Third Kind ranks second for The Godfather, which is a co-rating artefact rather than a content judgement.
Popularity bias. Ratings concentrate on blockbusters and the recommender inherits that. min_support guards against the opposite failure (noise from barely-rated films) but does not remove the bias.
Offline metrics only. Precision@10 measures agreement with what users happened to rate later, not what they would have enjoyed had it been shown. Only an online A/B test measures that.
One seed. Differences between scorers here are large enough to be meaningful; small differences would need repeated runs to be trusted.
Dataset age. MovieLens 1M stops in 2003. Nothing after that exists to the model.
TMDB live calls unverified in this build — see the note in the TMDB section.
Future improvements
Hybrid content signal. Embed TMDB overviews with a sentence transformer and concatenate with the latent vectors — the direct fix for cold start.
Ranking-native loss. Train with BPR or a sampled softmax instead of masked MSE, so the model optimises the objective it is actually judged on.
Denoising / variational variants. Mask-and-reconstruct training (CDAE) or Mult-VAE, both of which are the natural next architectures.
Time-aware split and features. Evaluate against a temporal cut and add rating recency, which would address the cohort artefact directly.
Approximate nearest neighbours (FAISS, Annoy) if the catalogue grows past a few hundred thousand titles; exact cosine over 3,706 films is instant.
Repeated seeds and confidence intervals on every reported metric.
Online evaluation — the only measurement that settles whether any of this works in production.
Acknowledgements

MovieLens 1M is provided by GroupLens Research at the University of Minnesota. This product uses the TMDB API but is not endorsed or certified by TMDB.

The architecture follows Sedhain et al., AutoRec: Autoencoders Meet Collaborative Filtering (WWW 2015), extended to a deeper encoder/decoder.
