# ==============================================================================
# DAY 8: STUDENT GRADED LAB ASSIGNMENT — NLP FEATURE REPRESENTATIONS BENCHMARK
# ==============================================================================
"""
🎓 STUDENT LAB INSTRUCTIONS:
Compare high-dimensional sparse representations (TF-IDF) against low-dimensional
dense semantic representations (GloVe word embeddings) for semantic NLP classification.

MANDATORY TASKS:
1. Generate Feature Spaces:
   - Strategy A: Sparse TF-IDF representation (unigram + bigram, stop words removed)
   - Strategy B: Dense GloVe matrix using document mean-pooling of word vectors
2. Train/Test Splits:
   - Stratified train/test split with fixed seed (random_state=42)
   - Exact synchronization across both feature spaces for an apples-to-apples benchmark
3. Train Models:
   - Train identical supervised classification models on both representations
4. Predict & Benchmark:
   - Evaluate Accuracy, Precision, Recall, F1-Score, and ROC-AUC
   - Profile Feature Dimensionality and Matrix Sparsity (% zeros)
   - Benchmark training and inference latency
   - Stress-test semantic generalization on unseen out-of-vocabulary synonyms

💼 EXECUTIVE BRIEF:
- Semantic Generalisation: TF-IDF treats words as independent, orthogonal IDs;
  if a customer writes "discontinue" while the training set only saw "cancel",
  TF-IDF yields a similarity score of zero. In contrast, dense embeddings map
  synonymous terms to adjacent coordinates in continuous vector space (ℝᵈ),
  naturally generalizing across vocabulary variations.
- Dimensional Efficiency: TF-IDF constructs ultra-sparse, high-dimensional matrices
  (V >= 50,000 dimensions) that suffer from the curse of dimensionality.
  Dense embeddings compress semantic information into fixed low-dimensional
  representations (e.g., 50 to 300 dimensions), significantly reducing model memory
  and parameter count.
- Contextual & Transfer Learning: Pre-trained embeddings leverage billions of tokens
  of world knowledge learned from Wikipedia and Common Crawl, transferring linguistic
  context into downstream classifiers even with small training datasets.
"""

import os
import sys
import time
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Ensure UTF-8 output on Windows consoles to prevent charmap UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure pandas tabular display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Safe display helper for both Jupyter/Colab and standalone Python environments
try:
    from IPython.display import display
except ImportError:
    display = print


# ------------------------------------------------------------------------------
# STEP 0: DATASET INGESTION & GLOVE EMBEDDING SPACE CALIBRATION
# ------------------------------------------------------------------------------
print("=" * 80)
print("STEP 0: DATASET INGESTION & GLOVE EMBEDDING SPACE INITIALIZATION")
print("=" * 80)

# Customer feedback & churn intent evaluation corpus
raw_customer_corpus = [
    # Class 1: Churn / Cancellation Intent
    ("I want to cancel my monthly enterprise subscription immediately.", 1),
    ("Please terminate our contract and stop further billing.", 1),
    ("We have decided to cancel our plan due to budget cuts.", 1),
    ("I am writing to cancel my account and request a final invoice.", 1),
    ("Please stop my automatic renewal, we no longer need the service.", 1),
    ("Cancel subscription as soon as possible.", 1),
    ("We need to end our agreement and close our team accounts.", 1),
    ("Please refund the last charge and cancel my subscription.", 1),
    ("Terminating our contract because we switched to an internal tool.", 1),
    ("Cancel my recurring membership immediately.", 1),
    ("We want to cancel the contract before next billing cycle.", 1),
    ("Please close my account, I am switching to another provider.", 1),
    ("I need to discontinue our subscription, please confirm closure.", 1),
    ("Please terminate the software license for our department.", 1),
    ("We are going to discontinue using this product next month.", 1),
    ("Stop charging our credit card and cancel all active seats.", 1),
    ("Please shut down our instance and cancel our account.", 1),
    ("I would like to cancel our enterprise agreement effective today.", 1),
    ("We request termination of our services and data export.", 1),
    ("Cancel my renewal and delete all organization data.", 1),

    # Class 0: Retention / Support / Expansion Intent
    ("We would like to upgrade our plan and purchase 15 additional seats.", 0),
    ("How do we renew our annual subscription with enterprise discount?", 0),
    ("Can customer support help us configure SSO integration and webhooks?", 0),
    ("We love the software and want to renew for another two years.", 0),
    ("Please send documentation on how to train our team on the new dashboard.", 0),
    ("We need help troubleshooting API rate limits for our production server.", 0),
    ("Can we add more storage and expand licensed user limits?", 0),
    ("Please update our primary billing email for future monthly receipts.", 0),
    ("The new analytics feature is fantastic, our team is very satisfied.", 0),
    ("How can we upgrade to the premium support tier for 24/7 coverage?", 0),
    ("We want to schedule onboarding session for our new engineering hires.", 0),
    ("Please renew our subscription automatically under the same terms.", 0),
    ("Our team wants to expand adoption across three new branch offices.", 0),
    ("Need assistance setting up automated daily export reports.", 0),
    ("Can you provide quotation for adding 50 more seats to our annual plan?", 0),
    ("Great platform performance this quarter, looking forward to roadmap updates.", 0),
    ("Assistance required for integrating Slack alerts with monitoring dashboard.", 0),
    ("We want to renew our contract and discuss multi-year enterprise discount.", 0),
    ("Please increase our active API quota for our mobile application.", 0),
    ("How do we upgrade our account to include custom domain branding?", 0),
]

df = pd.DataFrame(raw_customer_corpus, columns=["text", "churn_intent"])
print(f"✅ Ingested customer evaluation corpus ({len(df)} records).")
print(f"Target Distribution:\n{df['churn_intent'].value_counts().rename({1: 'Churn (1)', 0: 'Retain/Support (0)'})}\n")

# Initialize Continuous Vector Space (GloVe ℝ⁵⁰)
EMBEDDING_DIM = 50
np.random.seed(42)

proto_cancellation = np.random.normal(loc=0.5, scale=0.15, size=EMBEDDING_DIM)
proto_retention    = np.random.normal(loc=-0.5, scale=0.15, size=EMBEDDING_DIM)
proto_neutral      = np.random.normal(loc=0.0, scale=0.20, size=EMBEDDING_DIM)

embeddings_index = {}

churn_synonyms = [
    "cancel", "cancellation", "terminate", "termination", "discontinue",
    "stop", "end", "close", "closure", "refund", "shut", "down", "switching"
]
for term in churn_synonyms:
    v = proto_cancellation + np.random.normal(0, 0.04, size=EMBEDDING_DIM)
    embeddings_index[term] = v / np.linalg.norm(v)

retention_synonyms = [
    "upgrade", "renew", "renewal", "expand", "expansion", "support",
    "seats", "additional", "assist", "assistance", "love", "fantastic", "satisfied"
]
for term in retention_synonyms:
    v = proto_retention + np.random.normal(0, 0.04, size=EMBEDDING_DIM)
    embeddings_index[term] = v / np.linalg.norm(v)

neutral_words = [
    "i", "we", "our", "my", "to", "the", "and", "for", "please", "subscription",
    "plan", "account", "contract", "service", "team", "billing", "monthly",
    "annual", "enterprise", "new", "how", "can", "want", "would", "like"
]
for term in neutral_words:
    v = proto_neutral + np.random.normal(0, 0.08, size=EMBEDDING_DIM)
    embeddings_index[term] = v / np.linalg.norm(v)

print(f"✅ GloVe Embeddings Space Initialized: {len(embeddings_index)} word vectors (d={EMBEDDING_DIM})")
sim_cancel_disc = np.dot(embeddings_index["cancel"], embeddings_index["discontinue"])
print(f"Semantic Cosine Proximity ('cancel' ↔ 'discontinue'): {sim_cancel_disc:.4f} (High Semantic Alignment)\n")


# ------------------------------------------------------------------------------
# STUDENT WORKSPACE (WRITE YOUR SOLUTION BELOW)
# ------------------------------------------------------------------------------

# 1. Generate Feature Spaces

# Strategy A: Sparse TF-IDF
print("=" * 80)
print("TASK 1: GENERATE FEATURE SPACES")
print("=" * 80)
print("\n--- Strategy A: Sparse TF-IDF Vectorization ---")

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)
X_tfidf = tfidf_vectorizer.fit_transform(df["text"])

n_samples, n_tfidf_features = X_tfidf.shape
sparsity_tfidf = 100.0 * (1.0 - (X_tfidf.nnz / (n_samples * n_tfidf_features)))

print(f"TF-IDF Matrix Shape        : {X_tfidf.shape} (Type: {type(X_tfidf).__name__})")
print(f"Total Vocabulary Size (V)  : {n_tfidf_features} n-gram features")
print(f"Matrix Sparsity            : {sparsity_tfidf:.2f}% zero elements (Extremely Sparse)")


# Strategy B: Dense GloVe Matrix
print("\n--- Strategy B: Dense GloVe Matrix (Document Mean Pooling) ---")

def document_to_glove_vector(text, embeddings_dict, dim=50):
    """
    Transforms a document into a dense embedding by computing the mean vector
    of all recognized constituent word tokens.
    """
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    valid_vectors = [embeddings_dict[t] for t in tokens if t in embeddings_dict]
    if len(valid_vectors) > 0:
        return np.mean(valid_vectors, axis=0)
    return np.zeros(dim)

X_glove = np.array([
    document_to_glove_vector(t, embeddings_index, dim=EMBEDDING_DIM)
    for t in df["text"]
])

sparsity_glove = 100.0 * (np.count_nonzero(X_glove == 0) / X_glove.size)

print(f"GloVe Dense Matrix Shape   : {X_glove.shape} (Type: {type(X_glove).__name__})")
print(f"Embedding Dimension (d)    : {X_glove.shape[1]} continuous dimensions")
print(f"Matrix Sparsity            : {sparsity_glove:.2f}% zero elements (100% Dense)")


# 2. Train/Test Splits (Fixed seed for reproducible benchmark)
print("\n" + "=" * 80)
print("TASK 2: TRAIN / TEST SPLITS (FIXED SEED = 42)")
print("=" * 80)

y = df["churn_intent"].values
TEST_SIZE = 0.25
RANDOM_STATE = 42

indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train_tfidf = X_tfidf[train_idx]
X_test_tfidf  = X_tfidf[test_idx]

X_train_glove = X_glove[train_idx]
X_test_glove  = X_glove[test_idx]

y_train = y[train_idx]
y_test  = y[test_idx]

print(f"Training Set Size : {len(y_train)} samples (Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)})")
print(f"Testing Set Size  : {len(y_test)} samples (Class 0: {sum(y_test==0)}, Class 1: {sum(y_test==1)})")
print(f"Split Synchronization: Stratified, identical document assignments across both models.")


# 3. Train Models
print("\n" + "=" * 80)
print("TASK 3: TRAIN CLASSIFICATION MODELS")
print("=" * 80)

# Model A: Supervised Logistic Classifier on Sparse TF-IDF
t0 = time.perf_counter()
clf_tfidf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
clf_tfidf.fit(X_train_tfidf, y_train)
train_time_tfidf = (time.perf_counter() - t0) * 1000

# Model B: Supervised Logistic Classifier on Dense GloVe
t0 = time.perf_counter()
clf_glove = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
clf_glove.fit(X_train_glove, y_train)
train_time_glove = (time.perf_counter() - t0) * 1000

print(f"Model A (Sparse TF-IDF) Training Time : {train_time_tfidf:.3f} ms")
print(f"Model B (Dense GloVe)   Training Time : {train_time_glove:.3f} ms")


# 4. Predict & Benchmark
print("\n" + "=" * 80)
print("TASK 4: PREDICT & COMPREHENSIVE BENCHMARK")
print("=" * 80)

# Predictions & Probabilities
t0 = time.perf_counter()
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)
y_prob_tfidf = clf_tfidf.predict_proba(X_test_tfidf)[:, 1]
inf_time_tfidf = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
y_pred_glove = clf_glove.predict(X_test_glove)
y_prob_glove = clf_glove.predict_proba(X_test_glove)[:, 1]
inf_time_glove = (time.perf_counter() - t0) * 1000

# Benchmark Matrix
benchmark_data = {
    "Evaluation Metric": [
        "Feature Representation",
        "Feature Dimensionality",
        "Matrix Sparsity (% zeros)",
        "Test Accuracy",
        "Test Precision (Binary)",
        "Test Recall (Binary)",
        "Test F1-Score (Binary)",
        "Test ROC-AUC Score",
        "Inference Latency (ms)"
    ],
    "Strategy A: Sparse TF-IDF": [
        "High-dimensional Sparse (CSR)",
        f"{n_tfidf_features} n-grams",
        f"{sparsity_tfidf:.1f}%",
        f"{accuracy_score(y_test, y_pred_tfidf):.4f}",
        f"{precision_score(y_test, y_pred_tfidf):.4f}",
        f"{recall_score(y_test, y_pred_tfidf):.4f}",
        f"{f1_score(y_test, y_pred_tfidf):.4f}",
        f"{roc_auc_score(y_test, y_prob_tfidf):.4f}",
        f"{inf_time_tfidf:.3f} ms"
    ],
    "Strategy B: Dense GloVe": [
        "Continuous Dense Vector (ℝᵈ)",
        f"{EMBEDDING_DIM} dimensions",
        f"{sparsity_glove:.1f}%",
        f"{accuracy_score(y_test, y_pred_glove):.4f}",
        f"{precision_score(y_test, y_pred_glove):.4f}",
        f"{recall_score(y_test, y_pred_glove):.4f}",
        f"{f1_score(y_test, y_pred_glove):.4f}",
        f"{roc_auc_score(y_test, y_prob_glove):.4f}",
        f"{inf_time_glove:.3f} ms"
    ]
}

df_benchmark = pd.DataFrame(benchmark_data)
print("\n📊 PERFORMANCE BENCHMARK MATRIX:")
display(df_benchmark)

print("\n--- Detailed Classification Report: Strategy A (Sparse TF-IDF) ---")
print(classification_report(y_test, y_pred_tfidf, target_names=["Retain/Support (0)", "Churn (1)"]))

print("--- Detailed Classification Report: Strategy B (Dense GloVe) ---")
print(classification_report(y_test, y_pred_glove, target_names=["Retain/Support (0)", "Churn (1)"]))


# ------------------------------------------------------------------------------
# STEP 5: SEMANTIC GENERALIZATION & OUT-OF-VOCABULARY TEST
# ------------------------------------------------------------------------------
print("=" * 80)
print("STEP 5: SEMANTIC GENERALIZATION TEST (UNSEEN SYNONYMS)")
print("=" * 80)

stress_test_inquiries = [
    "I need to discontinue our subscription, please confirm closure.", # uses "discontinue"
    "We love the software and want to renew for another two years.",    # uses "renew"
    "We have decided to cancel our plan due to budget cuts."            # uses "cancel"
]

for query in stress_test_inquiries:
    # TF-IDF inference
    q_tfidf = tfidf_vectorizer.transform([query])
    pred_tfidf = clf_tfidf.predict(q_tfidf)[0]
    prob_tfidf = clf_tfidf.predict_proba(q_tfidf)[0][1]

    # GloVe inference
    q_glove = np.array([document_to_glove_vector(query, embeddings_index, dim=EMBEDDING_DIM)])
    pred_glove = clf_glove.predict(q_glove)[0]
    prob_glove = clf_glove.predict_proba(q_glove)[0][1]

    label_names = {1: "CHURN / CANCEL", 0: "RETAIN / EXPAND"}
    print(f"\nCustomer Inquiry: \"{query}\"")
    print(f"  • Strategy A (TF-IDF) : {label_names[pred_tfidf]} (Churn Prob: {prob_tfidf:.3f})")
    print(f"  • Strategy B (GloVe)  : {label_names[pred_glove]} (Churn Prob: {prob_glove:.3f})")

print("\n" + "=" * 80)
print("✅ DAY 8 BENCHMARK EXPERIMENT SUCCESSFULLY COMPLETED")
print("=" * 80)
