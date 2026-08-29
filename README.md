# Movie-Recommendation-System

Absolutely — what you have is already very detailed, but it reads more like a technical project report than a GitHub README. For GitHub, I’d make it more reader-friendly and explanatory: first explain what CineLatent does and why, then progressively show the architecture, methodology, results, and how to run it.

Here’s a polished README structure that keeps your technical depth but makes the project easier to understand:

🎞️ CineLatent — Deep Autoencoder Movie Recommender

A deep-learning collaborative filtering system that learns what movies are similar from user rating behaviour.

CineLatent uses the MovieLens 1M dataset to learn 128-dimensional latent representations for 3,706 movies. Movies that receive similar ratings from similar users are placed close together in the learned latent space. Recommendations are then generated using cosine similarity between these movie embeddings.

The project also integrates the TMDB API to enrich recommendations with posters and movie metadata, and provides a Streamlit web application for interacting with the recommender.

The key idea: If people who liked The Godfather also tended to like another movie, the model learns that relationship — without ever being told about the movie's plot, actors, or genres.

📊 Results at a Glance
Metric	CineLatent
Test RMSE	0.8604
Test MAE	0.6720
Precision@10	0.0927
Recall@10	0.1137
Hit-Rate@10	0.5179
NDCG@10	0.1271
Movies embedded	3,706
Embedding dimension	128
Training	30 epochs
Best epoch	20
Training time	~3 min, single CPU core

The model achieves a 7.7% lower RMSE than the strongest rating-prediction baseline (0.9321).

For Top-10 recommendations, the embedding-based recommender also beats the popularity baseline on all four reported ranking metrics.

🧠 What Is CineLatent?

Imagine thousands of users rating thousands of movies.

Most users have only rated a small fraction of the catalogue, so the resulting movie × user matrix is extremely sparse.

CineLatent asks:

Can we compress each movie's entire rating pattern into a small vector that captures its relationships with other movies?

The answer is yes.

Each movie starts as a vector containing the ratings it received from 6,040 users:

Movie A
[5, 4, 0, 0, 3, 0, 5, 4, ...]


That vector is passed through a deep autoencoder:

6,040 ratings
      ↓
   Dense 512
      ↓
   Dense 256
      ↓
   Dense 128   ← movie embedding
      ↓
   Dense 256
      ↓
   Dense 512
      ↓
6,040 reconstructed ratings


The 128-dimensional bottleneck becomes the movie's learned representation.

Movies with similar rating behaviour tend to end up close together in this space.

CineLatent then uses cosine similarity to find the nearest movies.

🎯 The Problem

A catalogue can contain thousands of movies, while a user may only consider a few dozen.

A simple recommendation system can always show the most popular movies, but popularity is a surprisingly strong baseline.

Therefore, CineLatent evaluates two separate questions:

1. Can the model predict ratings?

Given a movie/user pair that was hidden during training, how close is the predicted rating to the actual rating?

This is measured using:

RMSE
MAE
2. Can the model recommend the right movies?

Given a user's previous ratings, can the system place movies they later rated highly inside the Top-10?

This is measured using:

Precision@10
Recall@10
Hit-Rate@10
NDCG@10

These are deliberately evaluated separately because good rating prediction does not automatically mean good ranking.

🗂️ Dataset

CineLatent uses MovieLens 1M, provided by GroupLens Research.

The original dataset contains:

File	Rows	Contents
ratings.dat	1,000,209	User ratings
movies.dat	3,883	Movie titles and genres
users.dat	6,040	User demographics

After validation:

6,040 users
3,706 rated movies
177 unrated catalogue movies
1,000,209 ratings
95.53% matrix sparsity
4.47% matrix density
Ratings from 1–5 stars
Every user has at least 20 ratings

The 177 movies with no ratings remain in the catalogue, but cannot receive learned embeddings because the model has no collaborative signal for them.

🔬 How the Model Works
1. Build the Movie × User Matrix

Instead of representing each user by their movie ratings, CineLatent represents each movie by the ratings it received from every user.

That gives:

                    Users
              U1  U2  U3  U4  ... U6040
             ───────────────────────────
Movie 1       5   0   4   0   ...   3
Movie 2       0   5   4   0   ...   4
Movie 3       2   0   0   5   ...   0
...
Movie 3706    4   0   5   0   ...   2


The resulting matrix has:

3,706 movies × 6,040 users


This orientation is intentional.

Because the final product is a movie-to-movie recommender, each training sample is already a movie and the autoencoder's bottleneck directly becomes that movie's embedding.

2. Missing Ratings Are Not Zero

This is one of the most important details in the implementation.

A missing rating does not mean the user disliked the movie.

It means:

The user has never rated it.

Therefore:

0 = unobserved
0.2 = 1 star
0.4 = 2 stars
0.6 = 3 stars
0.8 = 4 stars
1.0 = 5 stars


Ratings are scaled using:

rating / 5


rather than:

(rating - 1) / 4


because the latter would turn a genuine 1-star rating into 0.0, making it indistinguishable from a missing rating.

🎭 Masked Loss

Because 95.53% of the matrix is empty, a normal MSE loss would be problematic.

If missing values were treated as ordinary zero targets, the model could achieve a deceptively low loss simply by learning:

"Most users haven't rated most movies, so predict zero everywhere."

CineLatent therefore uses a masked MSE:

                    Σ mᵢᵤ (rᵢᵤ − r̂ᵢᵤ)²
Loss =              ───────────────────────
                         Σ mᵢᵤ


where:

mᵢᵤ = 1  if a rating exists
mᵢᵤ = 0  otherwise


Only observed ratings contribute to the loss.

This allows the model to learn actual rating behaviour rather than learning the sparsity pattern.

✂️ Train / Validation / Test Split

Ratings are split per user using a fixed seed:

80% training
10% validation
10% test


Resulting in:

Train:      802,553 ratings
Validation: 100,273 ratings
Test:        97,383 ratings


Every user appears in all three sets.

The notebook explicitly verifies that:

no (user, movie) pair appears in multiple splits
validation and test ratings are absent from the model's input matrix
held-out ratings are evaluated only after training
the split is reproducible with SEED = 42

This prevents train/test leakage.

🏗️ Deep Autoencoder Architecture

The model contains 6,520,344 parameters.

Input
3,706 × 6,040 movie-user ratings
              │
              ▼
       Dense(512, ReLU)
          L2 + Dropout
              │
              ▼
       Dense(256, ReLU)
          L2 + Dropout
              │
              ▼
       Dense(128, Linear)
          ★ LATENT SPACE ★
              │
              ▼
       Dense(256, ReLU)
          L2 + Dropout
              │
              ▼
       Dense(512, ReLU)
            L2
              │
              ▼
       Dense(6,040, Sigmoid)
              │
              ▼
      Reconstructed ratings

Why a linear bottleneck?

The 128-dimensional latent layer uses a linear activation rather than ReLU.

This avoids forcing every embedding coordinate to be non-negative and preserves a more flexible representation for cosine similarity.

Why dropout?

Dropout is applied to hidden layers rather than directly to the input.

The input is already approximately 95% empty, so randomly removing additional ratings would introduce unnecessary noise.

Why sigmoid at the output?

The ratings are scaled into [0, 1], so sigmoid naturally constrains reconstructed values to the same range.

⚙️ Training

The model uses:

Optimizer: Adam
Initial learning rate: 1e-3
Learning-rate reduction: halved on plateau
Batch size: 64
L2 regularisation: 1e-5
Dropout: 0.2
Early stopping: patience 10
Best-weight restoration: enabled
Random seed: 42

Training ran for 30 epochs, with the best validation result at epoch 20.

Best validation masked RMSE:

0.8695 stars


Training took approximately 3 minutes on a single CPU core.

A GPU is therefore useful for convenience, but not required.

🧬 Movie Embeddings

Once training is complete, only the encoder portion is needed:

6,040 ratings
      ↓
Dense 512
      ↓
Dense 256
      ↓
Dense 128
      ↓
movie embedding


The encoder produces:

(3706, 128)


Each row corresponds to one movie.

The notebook verifies that:

every rated movie has exactly one embedding
MovieID ↔ embedding index mapping is invertible
embeddings contain no NaN or infinite values
no embedding is an all-zero vector
encoding a movie individually reproduces its batch embedding

The saved embedding matrix is:

models/movie_embeddings.npy

🎬 Recommendation Algorithm

For a query such as:

rec.recommend_movies(
    "Godfather, The (1972)",
    top_n=10
)


CineLatent performs the following steps:

Movie title
    │
    ▼
Resolve title
    │
    ▼
Find movie embedding
    │
    ▼
Calculate cosine similarity
against all movie embeddings
    │
    ▼
Remove the query movie
    │
    ▼
Apply minimum rating support
    │
    ▼
Return Top-N


Title resolution supports:

exact matches
prefix matches
substring matches
close-match suggestions
ambiguous titles
invalid top_n
candidate pools smaller than requested top_n

The recommender can also blend several favourite movies:

rec.recommend_from_profile([
    "Alien (1979)",
    "Blade Runner (1982)"
])


The multi-movie profile uses the Top-50 neighbourhood scoring strategy used during ranking evaluation.

🍿 Example Recommendations
The Godfather
Because you watched The Godfather (1972)

1. The Godfather: Part II (1974)       0.991
2. Close Encounters of the Third Kind  0.941
3. Apocalypse Now (1979)               0.932
4. Unforgiven (1992)                    0.931
5. Dog Day Afternoon (1975)             0.914

Toy Story
Because you watched Toy Story (1995)

1. Toy Story 2 (1999)                   0.981
2. Aladdin (1992)                       0.958
3. A Bug's Life (1998)                  0.944
4. The Lion King (1994)                 0.907


The similarity score is cosine similarity in the learned latent space, not a rating prediction.

📈 Evaluation

CineLatent evaluates rating prediction and recommendation ranking independently.

Rating Prediction

Test set:

97,383 held-out ratings

Model	RMSE	MAE
Deep Autoencoder	0.8604	0.6720
Movie + User Bias	0.9321	0.7314
Movie Mean	0.9800	0.7830
User Mean	1.0334	0.8268
Global Mean	1.1168	0.9332

The autoencoder reduces RMSE by 7.7% compared with the strongest baseline.

Top-10 Recommendation Ranking

Evaluation uses 5,862 users who had at least one relevant test movie.

A relevant movie is defined as a held-out rating of:

4 or 5 stars


Movies already present in the user's training set are excluded.

Scorer	Precision@10	Recall@10	Hit-Rate@10	NDCG@10
Embedding + Top-50 neighbours	0.0927	0.1137	0.5179	0.1271
Popularity	0.0733	0.0861	0.4369	0.0986
Autoencoder reconstruction	0.0596	0.0669	0.3623	0.0813
Embedding + all neighbours	0.0528	0.0728	0.3748	0.0735

The strongest result is:

Top-50 neighbourhood embedding recommendations outperform popularity on all four ranking metrics.

🔍 An Important Finding

One of the most interesting results is that the autoencoder's rating reconstruction performs worse as a recommendation ranker than popularity:

Autoencoder reconstruction
Precision@10 = 0.0596

Popularity
Precision@10 = 0.0733


This is not contradictory.

The two tasks are different.

Rating prediction asks:

"What rating would this user give this movie?"

Recommendation ranking asks:

"Which movies should I put in the user's Top-10?"

A model can be good at reconstructing ratings while still being mediocre at selecting the ten most useful candidates.

CineLatent therefore does not hide this result.

🧩 Why Top-50 Neighbours Matter

Another important finding is the effect of neighbourhood size.

Using every movie as a possible neighbour produces:

Precision@10 = 0.0528


Restricting each movie to its 50 closest neighbours produces:

Precision@10 = 0.0927


The likely explanation is that thousands of weak similarities introduce noise.

The Top-50 neighbourhood preserves the strongest collaborative relationships while filtering out weak connections.

🎨 Do the Embeddings Actually Capture Movie Similarity?

Genres are never provided to the model.

That makes them useful as an independent sanity check.

For each movie, the genres of its ten nearest neighbours are compared with the query movie's genres using Jaccard similarity.

Results:

Learned neighbours: 0.4179
Random movies:      0.1749


That is approximately a 2.4× separation.

This does not mean the model is learning genres directly.

Instead, it suggests that rating behaviour contains enough structure for genre-related movies to emerge naturally in the latent space.

The project also includes a PCA visualization of the learned embeddings.

🌐 TMDB Integration

The machine-learning model does not depend on TMDB.

TMDB is used only after recommendation generation to enrich results with:

posters
movie overviews
release information
TMDB ratings
additional metadata

The integration is implemented in:

src/tmdb_api.py


The client supports:

TMDB v3 API keys
TMDB v4 bearer tokens
request timeouts
retries with backoff
rate-limit handling
authentication errors
404 responses
5xx errors
malformed JSON
in-memory caching
disk caching
graceful failure when a movie cannot be found

MovieLens title quirks are also handled.

For example:

American President, The (1995)


can be resolved to:

The American President


without modifying the original MovieLens catalogue title.

Important verification note

The TMDB client has offline tests using a stubbed HTTP transport.

Live requests to api.themoviedb.org were not executed in the original build environment because outbound access to that host was unavailable.

🖥️ Streamlit Application

The trained model is exposed through a Streamlit interface.

Run:

streamlit run app.py


The application contains two modes:

Similar to one movie

Choose a movie and receive similar films.

Blend a few favourites

Choose multiple movies and generate recommendations based on their combined latent neighbourhood.

Each recommendation can display:

poster
title
genres
release year
TMDB rating
overview
cosine similarity
rating count

The app does not train the model.

It simply loads the saved embeddings and catalogue.

This keeps deployment lightweight and avoids requiring TensorFlow at runtime.

📁 Project Structure
cinelatent/
│
├── movie_recommender.ipynb
├── app.py
├── app_embedded.py
│
├── src/
│   ├── recommender.py
│   └── tmdb_api.py
│
├── tests/
│   ├── test_modules.py
│   ├── test_app_smoke.py
│   ├── test_app_parity.py
│   ├── test_posters.py
│   └── test_no_secrets.py
│
├── models/
│   ├── autoencoder.keras
│   ├── encoder.keras
│   ├── movie_embeddings.npy
│   ├── movie_id_mapping.pkl
│   └── training_history.json
│
├── outputs/
│   ├── eda/
│   ├── evaluation/
│   └── recommendations/
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── .streamlit/
    └── config.toml

🧪 Testing

The project includes offline tests covering both the recommender and application.

python tests/test_modules.py
python tests/test_app_smoke.py
python tests/test_app_parity.py
python tests/test_posters.py
python tests/test_no_secrets.py


The tests verify:

recommendation behaviour
title resolution
edge cases
TMDB client behaviour
poster rendering
Streamlit rendering
parity between app.py and src/
API credential redaction
absence of accidentally committed secrets

The notebook also performs assertions throughout the ML pipeline rather than relying only on visual inspection.

🚀 Running Locally
1. Clone the repository
git clone <your-repo-url>
cd cinelatent

2. Create an environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

3. Install dependencies

For the complete ML workflow:

pip install -r requirements-dev.txt


For the application only:

pip install -r requirements.txt


The runtime requirements intentionally exclude TensorFlow.

📥 Download MovieLens 1M
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip

mkdir -p data
mv ml-1m/*.dat data/


Then open:

jupyter lab movie_recommender.ipynb


Run the notebook from beginning to end.

The notebook creates:

data/
models/
outputs/
src/
app.py

🔑 TMDB API Configuration

Create a .env file:

cp .env.example .env


Then add your TMDB credential:

TMDB_API_KEY=your_tmdb_api_token


The application resolves credentials in this order:

Environment variable
        ↓
.env
        ↓
Interactive getpass prompt


The credential is never hard-coded into the application.

For Streamlit Cloud, use Streamlit's Secrets configuration rather than committing a secret file.

Never commit a real API key to Git. If one has already been committed, revoke it and create a new one.

☁️ Deploying to Streamlit Cloud

The application can be deployed using:

Main file:
app.py


The runtime requirements are intentionally small:

streamlit
numpy
pandas
requests
python-dotenv


TensorFlow is not required because the deployed application uses the precomputed embeddings.

For a lightweight deployment, the important files are:

app.py
requirements.txt

models/
├── movie_embeddings.npy
└── movie_id_mapping.pkl


The full 75 MB autoencoder does not need to be deployed.

There is also an optional:

app_embedded.py


which contains the catalogue and embeddings inside the application itself, allowing the app to run as a single file.

⚠️ Limitations
Cold Start

A movie with no ratings cannot receive a collaborative embedding.

Likewise, a completely new user has no rating profile.

A content-based or hybrid model would be required to solve this.

Dataset Age

MovieLens 1M ends in 2003.

The learned relationships therefore reflect an old movie catalogue and historical viewing behaviour.

Temporal Effects

The data is not evaluated using a strict temporal split.

As a result, the model can learn correlations caused by movies being popular during the same period rather than purely by user taste.

Popularity Bias

Popular movies receive much more collaborative signal and can dominate the learned space.

min_support reduces noise from extremely rare movies but does not eliminate popularity bias.

Offline Evaluation

A high Precision@10 does not prove that users would actually enjoy the recommendations in production.

The only definitive production measurement would be an online experiment such as an A/B test.

One Random Seed

The reported results come from:

SEED = 42


Repeated runs would be required to calculate confidence intervals and establish how stable smaller metric differences are.

TMDB Live Verification

The TMDB integration was tested offline, but live API calls were unavailable in the original build environment.

🔮 Future Improvements

Several extensions naturally follow from the current system.

Hybrid Recommendations

Combine collaborative embeddings with semantic embeddings generated from TMDB overviews.

This would provide a solution for cold-start movies.

Ranking-Native Training

Instead of optimising masked MSE, train directly for ranking using approaches such as:

BPR
sampled softmax
pairwise ranking losses

This would align the training objective more closely with Top-N evaluation.

Denoising Autoencoder

A denoising architecture such as CDAE could intentionally corrupt observed ratings and learn to reconstruct them.

Variational Models

Mult-VAE is another natural next step for implicit or sparse recommendation settings.

Temporal Modelling

Introduce:

rating timestamps
recency
temporal splits
time-dependent user/movie preferences

This could reduce historical cohort effects.

Approximate Nearest Neighbours

The current catalogue contains only 3,706 rated movies, so exact cosine similarity is effectively instantaneous.

For a catalogue containing hundreds of thousands or millions of movies, an ANN library such as FAISS or Annoy would become more appropriate.

Repeated Experiments

Run multiple random seeds and report:

mean metrics
standard deviation
confidence intervals
Online Evaluation

Ultimately, deploy multiple ranking strategies and measure actual user behaviour.

Offline metrics can estimate usefulness, but an online experiment is the final test.

📌 Reproducibility

All headline numbers in this README were produced by the notebook.

The experiment uses:

SEED = 42


The notebook records:

training history
evaluation metrics
recommendation examples
generated embeddings
dataset validation statistics
model artifacts

Re-running the notebook with:

FORCE_RETRAIN = False


loads the existing trained model rather than retraining it.

📚 References
MovieLens 1M

GroupLens Research, University of Minnesota.

AutoRec

Sedhain et al., "AutoRec: Autoencoders Meet Collaborative Filtering", WWW 2015.

CineLatent follows the AutoRec idea and extends the architecture with a deeper encoder/decoder and a 128-dimensional latent representation.

🙏 Acknowledgements

MovieLens 1M is provided by GroupLens Research at the University of Minnesota.

This project uses the TMDB API for optional movie metadata and posters. It is not endorsed or certified by TMDB.

⭐ Summary

CineLatent demonstrates an end-to-end recommendation pipeline:

MovieLens 1M
     │
     ▼
Data validation
     │
     ▼
Movie × User sparse matrix
     │
     ▼
80 / 10 / 10 per-user split
     │
     ▼
Masked-loss deep autoencoder
     │
     ▼
128-dimensional movie embeddings
     │
     ▼
Cosine similarity
     │
     ▼
Top-50 neighbourhood filtering
     │
     ▼
Top-N recommendations
     │
     ├──────────────► TMDB metadata
     │
     ▼
Streamlit application


The central result is simple:

A deep autoencoder can learn useful movie-to-movie relationships directly from rating behaviour, and those learned relationships can outperform popularity-based recommendations when the latent neighbourhood is used as a ranking signal.

The project deliberately reports both its successes and its failures: strong rating prediction does not guarantee strong ranking, popularity remains competitive, and offline evaluation cannot substitute for a real production experiment.

This version keeps the technical credibility of your original README, but makes the narrative much easier to follow: problem → intuition → data → model → embeddings → recommendations → evaluation → deployment → limitations. It should also read much more naturally to someone reviewing the project on GitHub rather than someone auditing the notebook.
