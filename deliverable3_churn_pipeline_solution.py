# ==============================================================================
# SECTION 5: STUDENT GRADED LAB — DELIVERABLE 3 (FINAL PORTFOLIO MODEL)
# ==============================================================================
"""
🎓 DELIVERABLE 3 (PART 2) REQUIREMENTS:
Optimize the customer churn pipeline through Feature Engineering, Class Balancing,
and GridSearchCV Hyperparameter Tuning.

MANDATORY TASKS:
1. Feature Engineering: Incorporate at least 3 domain-specific engineered features.
2. Pipeline Architecture: Build an end-to-end ColumnTransformer & Pipeline with XGBoost.
3. Imbalance Handling: Apply `scale_pos_weight` or class weighting inside the estimator.
4. GridSearchCV Tuning: Tune at least 3 hyperparameters across a grid with 5-Fold Stratified CV.
5. Verification Scorecard: Generate a Before vs. After metric comparison table.
6. Pipeline Persistence: Serialize the final trained pipeline to disk using `joblib.dump()`.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Scikit-Learn Suite
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix, accuracy_score
)
from xgboost import XGBClassifier

# Terminal encoding compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Display helper for notebooks and console
try:
    from IPython.display import display
except ImportError:
    display = print

RANDOM_STATE = 42

# ------------------------------------------------------------------------------
# DOMAIN FEATURE ENGINEERING FUNCTION (TOP-LEVEL FOR SERIALIZATION)
# ------------------------------------------------------------------------------
def engineer_saas_features(df_in):
    """
    Incorporate domain-specific engineered features:
    1. Seat_Utilization: Active_Seats / Licensed_Seats (Product adoption)
    2. Ticket_Velocity: Support_Tickets / (Tenure_Months + 1) (Customer friction rate)
    3. Bug_Ratio: Unresolved_Bugs / (Support_Tickets + 1e-5) (Product defect impact)
    4. Cost_Per_Seat: Monthly_Contract_Value / (Licensed_Seats + 1e-5) (Price intensity)
    """
    df_out = df_in.copy()
    df_out["Seat_Utilization"] = df_out["Active_Seats"] / (df_out["Licensed_Seats"] + 1e-5)
    df_out["Ticket_Velocity"] = df_out["Support_Tickets"] / (df_out["Tenure_Months"] + 1)
    df_out["Bug_Ratio"] = df_out["Unresolved_Bugs"] / (df_out["Support_Tickets"] + 1e-5)
    df_out["Cost_Per_Seat"] = df_out["Monthly_Contract_Value"] / (df_out["Licensed_Seats"] + 1e-5)
    return df_out


def run_deliverable3():
    np.random.seed(RANDOM_STATE)

    # --------------------------------------------------------------------------
    # STEP 0: DATASET INGESTION & TRAIN/TEST SPLIT
    # --------------------------------------------------------------------------
    DATA_FILE = "enterprise_saas_churn_fe.csv"

    if not os.path.exists(DATA_FILE):
        N = 3500
        industries = np.random.choice(["FinTech", "HealthTech", "E-Commerce", "EdTech", "Logistics", "Media"], size=N, p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
        contract_plan = np.random.choice(["Startup", "Growth", "Enterprise", "Custom"], size=N, p=[0.35, 0.35, 0.20, 0.10])
        billing_cycle = np.random.choice(["Monthly", "Quarterly", "Annual", "Multi-Year"], size=N, p=[0.45, 0.15, 0.30, 0.10])
        onboarding_tier = np.random.choice(["Self-Serve", "Assisted", "Dedicated-CSM"], size=N, p=[0.40, 0.40, 0.20])

        monthly_contract_val = np.random.lognormal(mean=7.2, sigma=0.85, size=N).clip(150, 20000)
        tenure_months = np.random.exponential(scale=18, size=N).clip(1, 72).astype(int)
        licensed_seats = np.random.poisson(lam=20, size=N).clip(2, 500)
        active_seats = (licensed_seats * np.random.uniform(0.20, 1.0, size=N)).astype(int)
        support_tickets = np.random.poisson(lam=2.2, size=N)
        unresolved_bugs = np.random.binomial(n=support_tickets, p=0.35)
        api_calls_monthly = np.random.exponential(scale=50000, size=N).clip(100, 1000000)

        seat_utilization = active_seats / (licensed_seats + 1e-5)
        ticket_velocity = support_tickets / (tenure_months + 1)
        cost_per_seat = monthly_contract_val / (licensed_seats + 1e-5)

        churn_logits = (
            - 2.8
            + 0.00025 * monthly_contract_val
            - 0.05 * tenure_months
            - 2.5 * seat_utilization
            + 0.8 * ticket_velocity
            + 1.4 * unresolved_bugs
            + 0.9 * (billing_cycle == "Monthly")
            - 1.6 * (billing_cycle == "Annual")
            - 2.4 * (billing_cycle == "Multi-Year")
            + 0.004 * cost_per_seat
        )
        churn_prob = 1 / (1 + np.exp(-churn_logits))
        churn_label = np.where(np.random.rand(N) < churn_prob, 1, 0)

        raw_df = pd.DataFrame({
            "Industry": industries,
            "Contract_Plan": contract_plan,
            "Billing_Cycle": billing_cycle,
            "Onboarding_Tier": onboarding_tier,
            "Monthly_Contract_Value": np.round(monthly_contract_val, 2),
            "Tenure_Months": tenure_months,
            "Licensed_Seats": licensed_seats,
            "Active_Seats": active_seats,
            "Support_Tickets": support_tickets,
            "Unresolved_Bugs": unresolved_bugs,
            "API_Calls_Monthly": np.round(api_calls_monthly, 0),
            "Churn": churn_label
        })

        raw_df.loc[np.random.choice(N, size=75, replace=False), "Onboarding_Tier"] = np.nan
        raw_df.loc[np.random.choice(N, size=50, replace=False), "API_Calls_Monthly"] = np.nan
        raw_df.to_csv(DATA_FILE, index=False)

    raw_df = pd.read_csv(DATA_FILE)
    X = raw_df.drop(columns=["Churn"])
    y = raw_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # --------------------------------------------------------------------------
    # STUDENT WORKSPACE (WRITE YOUR PRODUCTION PIPELINE BELOW)
    # --------------------------------------------------------------------------

    # Step 1: Preprocessor definition
    print("=" * 80, flush=True)
    print("STEP 1: PREPROCESSOR DEFINITION", flush=True)
    print("=" * 80, flush=True)

    base_num = [
        "Monthly_Contract_Value", "Tenure_Months", "Licensed_Seats",
        "Active_Seats", "Support_Tickets", "Unresolved_Bugs", "API_Calls_Monthly"
    ]
    base_cat = ["Industry", "Contract_Plan", "Billing_Cycle", "Onboarding_Tier"]

    # Numerical columns including domain-engineered features
    fe_num_cols = base_num + ["Seat_Utilization", "Ticket_Velocity", "Bug_Ratio", "Cost_Per_Seat"]

    num_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_preprocessor, fe_num_cols),
            ("cat", cat_preprocessor, base_cat)
        ]
    )
    print("✅ Preprocessor successfully defined (ColumnTransformer with median/mode imputation, scaling & OHE).", flush=True)

    # Step 2: Full Pipeline with XGBoost Classifier
    print("\n" + "=" * 80, flush=True)
    print("STEP 2: FULL PIPELINE WITH XGBOOST CLASSIFIER & CLASS BALANCING", flush=True)
    print("=" * 80, flush=True)

    # Class Imbalance Ratio: scale_pos_weight = count(negative) / count(positive)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Class Imbalance Handling: scale_pos_weight = {neg_count} / {pos_count} = {scale_pos_weight:.4f}", flush=True)

    xgb_pipeline = Pipeline([
        ("fe", FunctionTransformer(engineer_saas_features)),
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        ))
    ])
    print("✅ End-to-end Pipeline assembled (Feature Engineering -> Preprocessing -> XGBClassifier).", flush=True)

    # Step 3: GridSearchCV Hyperparameter Optimization
    print("\n" + "=" * 80, flush=True)
    print("STEP 3: GRIDSEARCHCV HYPERPARAMETER OPTIMIZATION (5-FOLD STRATIFIED CV)", flush=True)
    print("=" * 80, flush=True)

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.03, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nGridSearchCV Optimization Complete!", flush=True)
    print(f"Best 5-Fold Stratified CV ROC-AUC: {grid_search.best_score_:.4f}", flush=True)
    print("Best Hyperparameters:", flush=True)
    for param, val in grid_search.best_params_.items():
        print(f"  * {param}: {val}", flush=True)

    best_pipeline = grid_search.best_estimator_

    # Step 4: Final Holdout Test Evaluation
    print("\n" + "=" * 80, flush=True)
    print("STEP 4: FINAL HOLDOUT TEST EVALUATION", flush=True)
    print("=" * 80, flush=True)

    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    test_roc = roc_auc_score(y_test, y_prob)
    test_pr = average_precision_score(y_test, y_prob)
    test_f1 = f1_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    print("\nClassification Report (Test Set):", flush=True)
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]), flush=True)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:", flush=True)
    print(f"  TN: {cm[0, 0]} | FP: {cm[0, 1]}", flush=True)
    print(f"  FN: {cm[1, 0]} | TP: {cm[1, 1]}", flush=True)

    # Step 5: Before vs After Scorecard Summary
    print("\n" + "=" * 80, flush=True)
    print("STEP 5: BEFORE VS AFTER SCORECARD SUMMARY & PIPELINE PERSISTENCE", flush=True)
    print("=" * 80, flush=True)

    # Baseline Model (Raw Features + Default Random Forest)
    base_pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), base_num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), base_cat)
        ])),
        ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    base_pipe.fit(X_train, y_train)
    y_base_pred = base_pipe.predict(X_test)
    y_base_prob = base_pipe.predict_proba(X_test)[:, 1]

    # Feature-Engineered Model (3 Domain Ratios + Random Forest)
    X_train_fe = engineer_saas_features(X_train)
    X_test_fe = engineer_saas_features(X_test)
    fe_rf_pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), fe_num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), base_cat)
        ])),
        ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    fe_rf_pipe.fit(X_train_fe, y_train)
    y_fe_pred = fe_rf_pipe.predict(X_test_fe)
    y_fe_prob = fe_rf_pipe.predict_proba(X_test_fe)[:, 1]

    scorecard = pd.DataFrame({
        "Metric": [
            "ROC-AUC (Test)",
            "PR-AUC (Test)",
            "F1-Score (Test)",
            "Minority Recall (Sensitivity)",
            "Precision (Test)",
            "Accuracy (Test)"
        ],
        "Baseline (Raw RF)": [
            roc_auc_score(y_test, y_base_prob),
            average_precision_score(y_test, y_base_prob),
            f1_score(y_test, y_base_pred),
            recall_score(y_test, y_base_pred),
            precision_score(y_test, y_base_pred),
            accuracy_score(y_test, y_base_pred)
        ],
        "Feat-Eng RF (3 Ratios)": [
            roc_auc_score(y_test, y_fe_prob),
            average_precision_score(y_test, y_fe_prob),
            f1_score(y_test, y_fe_pred),
            recall_score(y_test, y_fe_pred),
            precision_score(y_test, y_fe_pred),
            accuracy_score(y_test, y_fe_pred)
        ],
        "Deliverable 3 (Tuned XGBoost)": [
            test_roc,
            test_pr,
            test_f1,
            test_recall,
            test_precision,
            test_accuracy
        ]
    })

    scorecard["Gain vs Baseline (%)"] = (
        (scorecard["Deliverable 3 (Tuned XGBoost)"] - scorecard["Baseline (Raw RF)"]) * 100
    ).apply(lambda x: f"{x:+.2f}%")

    for col in ["Baseline (Raw RF)", "Feat-Eng RF (3 Ratios)", "Deliverable 3 (Tuned XGBoost)"]:
        scorecard[col] = scorecard[col].apply(lambda x: f"{x:.4f}")

    print("\n=== BEFORE VS. AFTER METRIC COMPARISON SCORECARD ===", flush=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(scorecard)

    # 6. Pipeline Persistence: Serialize final pipeline to disk
    MODEL_PATH = "final_saas_churn_pipeline.joblib"
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\n✅ Serialized final pipeline to disk: '{MODEL_PATH}' ({os.path.getsize(MODEL_PATH)/1024:.2f} KB)", flush=True)

    # Verification of reload and inference
    loaded_model = joblib.load(MODEL_PATH)
    sample_test = X_test.head(3)
    preds = loaded_model.predict(sample_test)
    probs = loaded_model.predict_proba(sample_test)[:, 1]
    print("\nInference Verification on Sample Raw Inputs:", flush=True)
    verification_df = sample_test[["Monthly_Contract_Value", "Tenure_Months", "Active_Seats"]].copy()
    verification_df["Predicted_Churn"] = preds
    verification_df["Churn_Probability"] = np.round(probs, 4)
    display(verification_df)

    print("\n" + "=" * 80, flush=True)
    print("🎯 DELIVERABLE 3 (PART 2) STUDENT LAB COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_deliverable3()
