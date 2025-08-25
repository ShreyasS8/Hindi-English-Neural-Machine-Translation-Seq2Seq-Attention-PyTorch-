import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Try to silence InconsistentVersionWarning if present
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

import joblib
import json
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

# ML libs
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, balanced_accuracy_score
)
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

# try xgboost
try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("Please install xgboost: pip install xgboost") from e

# Try imbalanced-learn, but do not raise if it's unavailable. Proceed without SMOTE.
IMBLEARN_AVAILABLE = True
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except Exception as e:
    IMBLEARN_AVAILABLE = False
    ImbPipeline = None
    SMOTE = None
    print("WARNING: imbalanced-learn (SMOTE) not available or incompatible with installed scikit-learn.")
    print("SMOTE-based pipeline and SMOTE hyperparameter tuning will be skipped.")
    print("To enable SMOTE: try installing a compatible version, for example:")
    print("  pip install -U imbalanced-learn scikit-learn")
    print("Then restart the kernel and re-run this script.\n")

# optional shap
SHAP_AVAILABLE = True
try:
    import shap
except Exception:
    SHAP_AVAILABLE = False

# -------------------------
# CONFIG (edit values here)
# -------------------------
SEED = 42
np.random.seed(SEED)

# Paths
DATA_FILE = "/kaggle/input/creditcardfraud/creditcard.csv"          # path to dataset CSV
OUTPUT_DIR = "models_outputs"         # where to save models/artefacts

# Toggles
DO_TUNING = True                      # run RandomizedSearchCV (set False to skip)
SHOW_PLOTS = True                     # show PR and importance plots (set False for headless)

# Train/test split
TEST_SIZE = 0.2
STRATIFY = True                       # use stratified split

# Bagging params
BAGGING_N_ESTIMATORS = 100

# XGBoost default params (used when not tuning)
XGB_DEFAULT = {
    "n_estimators": 200,
    "learning_rate": 0.01,
    "max_depth": 6,
    "n_jobs": -1,
    "random_state": SEED,
    # Note: use_label_encoder removed in newer xgboost; keep eval_metric
    "use_label_encoder": False,
    "eval_metric": "aucpr"
}

# RandomizedSearchCV params (small budget)
RS_CV_SPLITS = 3
RS_N_ITER = 6
RS_RANDOM_STATE = SEED
RS_N_JOBS = -1
RS_VERBOSE = 1

XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1]
}

SMOTE_PIPE_PARAM_DIST = {
    "xgb__n_estimators": [100, 200],
    "xgb__max_depth": [4, 6],
    "xgb__learning_rate": [0.01, 0.05]
}

# Precision@K setting
PRECISION_AT_K = 100

# Permutation importance
PERM_N_REPEATS = 20

# SHAP sampling
SHAP_SAMPLE_N = 2000

# -------------------------
# End CONFIG
# -------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Utility functions
# -------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Put the dataset at: {path}. Download from Kaggle: mlg-ulb/creditcardfraud")
    df = pd.read_csv(path)
    return df


def safe_sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -709, 709)
    return 1.0 / (1.0 + np.exp(-x))


def get_model_proba(model, X):
    # robust proba extraction
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            elif proba.ndim == 1:
                return proba
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
            return safe_sigmoid(scores)
        except Exception:
            pass
    if hasattr(model, "predict"):
        try:
            preds = model.predict(X)
            return np.asarray(preds, dtype=float)
        except Exception:
            pass
    # check for pipeline steps
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            try:
                if hasattr(step, "predict_proba"):
                    p = step.predict_proba(X)
                    p = np.asarray(p)
                    if p.ndim == 2 and p.shape[1] >= 2:
                        return p[:, 1]
            except Exception:
                continue
    raise RuntimeError("Model doesn't support predict_proba / decision_function / predict for probability extraction.")


def compute_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.asarray(thresholds)
    if thresholds.size == 0:
        th = 0.5
        y_pred = (np.asarray(y_proba) >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return th, f1
    p = precision[1:]
    r = recall[1:]
    f1_scores = 2 * (p * r) / (p + r + 1e-12)
    idx = int(np.nanargmax(f1_scores))
    return float(thresholds[idx]), float(f1_scores[idx])


def eval_at_threshold(y_true, y_proba, threshold):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm, "y_pred": y_pred}


def precision_at_k(y_true, y_proba, k=PRECISION_AT_K):
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    if 0 < k < 1:
        k = int(len(y_proba_arr) * k)
    k = min(len(y_proba_arr), int(k))
    if k <= 0:
        return 0.0
    order = np.argsort(y_proba_arr)[::-1][:k]
    return float(y_true_arr[order].sum() / k)


def lift_at_k(y_true, y_proba, k=PRECISION_AT_K):
    base_rate = np.asarray(y_true).mean()
    if base_rate == 0:
        return np.nan
    p_at_k = precision_at_k(y_true, y_proba, k)
    return p_at_k / base_rate


def average_precision_scorer_for_perm(estimator, X, y):
    try:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_score = proba[:, 1]
            elif proba.ndim == 1:
                y_score = proba
            else:
                y_score = proba
        elif hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X)
            y_score = safe_sigmoid(scores)
        else:
            y_score = get_model_proba(estimator, X)
    except Exception:
        try:
            preds = estimator.predict(X)
            y_score = np.asarray(preds, dtype=float)
        except Exception:
            raise
    return float(average_precision_score(y, y_score))


def format_and_print_confusion_matrix(cm, model_name, output_dir, show_plot=SHOW_PLOTS):
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    total = tn + fp + fn + tp
    cm_df = pd.DataFrame(cm, index=["Actual_Negative(0)","Actual_Positive(1)"], columns=["Pred_Negative(0)","Pred_Positive(1)"])
    print(f"\nConfusion matrix for {model_name} (counts):")
    print(cm_df.to_string())
    pct_df = cm_df / total
    print(f"\nConfusion matrix for {model_name} (percent of total):")
    print(pct_df.to_string(float_format=lambda x: f"{x*100:.2f}%"))
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    print(f"\nCounts: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Total={total}")
    print(f"Per-class rates: Recall_Positive (TPR)={recall_pos:.4f}, Recall_Negative (TNR)={recall_neg:.4f}")
    print(f"Precision_Positive={precision_pos:.4f}, Precision_Negative={precision_neg:.4f}")
    out_df = cm_df.copy().astype(int)
    out_df.columns = [f"{c}_count" for c in out_df.columns]
    pct_cols = [f"{c}_pct" for c in cm_df.columns]
    pct_df_renamed = (cm_df / total).copy()
    pct_df_renamed.columns = pct_cols
    combined = pd.concat([out_df, pct_df_renamed], axis=1)
    combined.to_csv(os.path.join(output_dir, f"{model_name}_confusion_matrix.csv"))
    print(f"Saved confusion matrix CSV to: {os.path.join(output_dir, f'{model_name}_confusion_matrix.csv')}")
    if show_plot:
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=["Pred_Neg(0)","Pred_Pos(1)"],
                        yticklabels=["Actual_Neg(0)","Actual_Pos(1)"])
            plt.title(f"{model_name} — Confusion Matrix (counts)")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

# -------------------------
# Main pipeline
# -------------------------

def main():
    df = load_data(DATA_FILE)
    print("Dataset shape:", df.shape)
    if 'Class' not in df.columns:
        raise ValueError("Expected 'Class' column as target.")
    print(df['Class'].value_counts())
    X = df.drop(columns=['Class'])
    y = df['Class']
    if STRATIFY:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    print("Train/test shapes:", X_train.shape, X_test.shape)

    # Scale Amount and Time
    X_train = X_train.copy()
    X_test = X_test.copy()
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    if 'Amount' in X_train.columns and 'Time' in X_train.columns:
        X_train['scaled_amount'] = scaler_amount.fit_transform(X_train['Amount'].values.reshape(-1,1))
        X_train['scaled_time']   = scaler_time.fit_transform(X_train['Time'].values.reshape(-1,1))
        X_train.drop(['Amount','Time'], axis=1, inplace=True)
        X_test['scaled_amount'] = scaler_amount.transform(X_test['Amount'].values.reshape(-1,1))
        X_test['scaled_time']   = scaler_time.transform(X_test['Time'].values.reshape(-1,1))
        X_test.drop(['Amount','Time'], axis=1, inplace=True)
        joblib.dump(scaler_amount, os.path.join(OUTPUT_DIR, "scaler_amount.joblib"))
        joblib.dump(scaler_time, os.path.join(OUTPUT_DIR, "scaler_time.joblib"))
        print("Saved scalers to", OUTPUT_DIR)

    cols = X_train.columns.tolist()
    for c in ['scaled_amount','scaled_time']:
        if c in cols:
            cols.insert(0, cols.pop(cols.index(c)))
    X_train = X_train[cols]
    X_test = X_test[cols]

    models = {}
    skipped_steps = []

    # Logistic
    print("\nTraining LogisticRegression (baseline)...")
    lr = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=SEED, max_iter=1000)
    lr.fit(X_train, y_train)
    models['LogisticRegression'] = lr

    # Bagging (with robust fallback)
    print("\nTraining BaggingClassifier (DecisionTree base) [serial to avoid joblib issues]...")
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=SEED),
        n_estimators=BAGGING_N_ESTIMATORS,
        n_jobs=1,                # <-- serial execution avoids joblib / signature mismatch errors
        random_state=SEED
    )
    try:
        bag.fit(X_train, y_train)
        models['Bagging'] = bag
    except TypeError as e:
        print("BaggingClassifier.fit raised TypeError (parallel/signature mismatch).")
        print("Falling back to a single DecisionTreeClassifier so pipeline can continue.")
        print("Error:", e)
        dt = DecisionTreeClassifier(random_state=SEED)
        dt.fit(X_train, y_train)
        models['Bagging_fallback_DecisionTree'] = dt
        skipped_steps.append("Bagging ensemble (failed; used single DecisionTree fallback)")
    except Exception as e:
        print("BaggingClassifier.fit failed with Exception; falling back to single DecisionTree.")
        print("Error:", e)
        dt = DecisionTreeClassifier(random_state=SEED)
        dt.fit(X_train, y_train)
        models['Bagging_fallback_DecisionTree'] = dt
        skipped_steps.append("Bagging ensemble (failed; used single DecisionTree fallback)")

    # -------------------------
    # --- XGBoost / TUNING (replacement integrated)
    # -------------------------

    print("\nPreparing XGBoost (no SMOTE)...")
    neg_cnt = int(y_train.value_counts().get(0, 0))
    pos_cnt = int(y_train.value_counts().get(1, 0))
    spw = 1
    if pos_cnt > 0:
        spw = max(1, int(neg_cnt / pos_cnt))

    xgb_clf = XGBClassifier(**{**XGB_DEFAULT, "scale_pos_weight": spw})
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED
    )
    print("Training XGBoost (no SMOTE) with early stopping...")
    try:
        xgb_clf.fit(
            X_tr_sub, y_tr_sub,
            eval_set=[(X_val_sub, y_val_sub)],
            early_stopping_rounds=20,
            verbose=False
        )
        models['XGBoost_noSMOTE'] = xgb_clf
    except Exception as e:
        print("Failed to fit XGBoost_noSMOTE with early stopping:", e)
        print(traceback.format_exc())
        try:
            xgb_clf.fit(X_train, y_train)
            models['XGBoost_noSMOTE'] = xgb_clf
        except Exception as e2:
            print("Fallback XGBoost fit also failed:", e2)
            print(traceback.format_exc())
            skipped_steps.append("XGBoost_noSMOTE (failed)")

    # SMOTE pipeline (if imblearn available): use XGBClassifier directly inside pipeline
    pipe_smote = None
    if IMBLEARN_AVAILABLE:
        print("Preparing SMOTE + XGBoost pipeline (imblearn pipeline)...")
        smote = SMOTE(random_state=SEED)
        xgb_sm = XGBClassifier(**{**XGB_DEFAULT, "scale_pos_weight": 1})
        pipe_smote = ImbPipeline(steps=[('smote', smote), ('xgb', xgb_sm)])
    else:
        skipped_steps.append("SMOTE pipeline (imbalanced-learn missing)")

    # RandomizedSearchCV tuning (use XGBClassifier directly; safer to set n_jobs=1 while debugging)
    rs_no_smote = None
    rs_smote = None
    if DO_TUNING:
        print("\nRunning RandomizedSearchCV (budget-limited)...")
        skf = StratifiedKFold(n_splits=RS_CV_SPLITS, shuffle=True, random_state=SEED)

        # Debug tip: set RS_N_JOBS = 1 if you're seeing pickling errors; change back to -1 later.
        safe_n_jobs = RS_N_JOBS if RS_N_JOBS != -1 else 1

        rs_no_smote = RandomizedSearchCV(
            estimator=XGBClassifier(**{**XGB_DEFAULT, "scale_pos_weight": spw}),
            param_distributions=XGB_PARAM_DIST,
            scoring='average_precision',
            n_iter=RS_N_ITER,
            cv=skf,
            random_state=RS_RANDOM_STATE,
            n_jobs=safe_n_jobs,
            verbose=RS_VERBOSE,
            refit=True
        )
        try:
            rs_no_smote.fit(X_train, y_train)
            print("Best (no SMOTE) CV AUPRC:", rs_no_smote.best_score_)
            print("Best params (no SMOTE):", rs_no_smote.best_params_)
            models['XGBoost_noSMOTE_tuned'] = rs_no_smote.best_estimator_
            joblib.dump(rs_no_smote, os.path.join(OUTPUT_DIR, "rs_no_smote.joblib"))
            with open(os.path.join(OUTPUT_DIR, "rs_no_smote_best_params.json"), "w") as fh:
                json.dump(rs_no_smote.best_params_, fh, indent=2)
        except Exception as e:
            print("RandomizedSearchCV (no SMOTE) failed:", e)
            print(traceback.format_exc())
            rs_no_smote = None

        # SMOTE pipeline tuning (if available)
        if IMBLEARN_AVAILABLE and pipe_smote is not None:
            rs_smote = RandomizedSearchCV(
                estimator=pipe_smote,
                param_distributions=SMOTE_PIPE_PARAM_DIST,
                scoring='average_precision',
                n_iter=RS_N_ITER,
                cv=skf,
                random_state=RS_RANDOM_STATE,
                n_jobs=safe_n_jobs,
                verbose=RS_VERBOSE,
                refit=True
            )
            try:
                rs_smote.fit(X_train, y_train)
                print("Best (SMOTE pipeline) CV AUPRC:", rs_smote.best_score_)
                print("Best params (SMOTE pipeline):", rs_smote.best_params_)
                models['SMOTE_XGBoost_tuned'] = rs_smote.best_estimator_
                joblib.dump(rs_smote, os.path.join(OUTPUT_DIR, "rs_smote.joblib"))
                with open(os.path.join(OUTPUT_DIR, "rs_smote_best_params.json"), "w") as fh:
                    json.dump(rs_smote.best_params_, fh, indent=2)
            except Exception as e:
                print("RandomizedSearchCV (SMOTE pipeline) failed:", e)
                print(traceback.format_exc())
                rs_smote = None
        else:
            print("Skipping SMOTE RandomizedSearchCV because imbalanced-learn is not available.")
            skipped_steps.append("RandomizedSearchCV for SMOTE pipeline (skipped)")
    else:
        if IMBLEARN_AVAILABLE and pipe_smote is not None:
            models['SMOTE_XGBoost'] = pipe_smote
        else:
            skipped_steps.append("SMOTE_XGBoost model (imbalanced-learn missing)")

    # Evaluate on test set
    print("\nEvaluating models on TEST set...")
    if SHOW_PLOTS:
        plt.figure(figsize=(10,6))
    eval_rows = []
    y_test_array = y_test.reset_index(drop=True).astype(int).values
    for name, model in models.items():
        print(f"\nModel: {name}")
        try:
            y_proba = get_model_proba(model, X_test)
        except Exception as e:
            print(f"Skipping {name}: can't extract probabilities: {e}")
            continue
        y_proba = np.asarray(y_proba).ravel()
        auprc = average_precision_score(y_test_array, y_proba)
        roc = roc_auc_score(y_test_array, y_proba)
        best_th, best_f1 = compute_best_threshold(y_test_array, y_proba)
        met = eval_at_threshold(y_test_array, y_proba, best_th)
        y_pred = met['y_pred']
        p_at_k = precision_at_k(pd.Series(y_test_array), y_proba, k=PRECISION_AT_K)
        lift_k = lift_at_k(pd.Series(y_test_array), y_proba, k=PRECISION_AT_K)
        cm = met['confusion_matrix']
        format_and_print_confusion_matrix(cm, name, OUTPUT_DIR, show_plot=SHOW_PLOTS)
        print(f"\nClassification report for {name} (threshold={best_th:.4f}):")
        print(classification_report(y_test_array, y_pred, digits=4, zero_division=0))
        balanced_acc = balanced_accuracy_score(y_test_array, y_pred)
        eval_rows.append({
            'Model': name,
            'AUPRC': auprc,
            'ROC_AUC': roc,
            'BestThresh': best_th,
            'BestF1_onPR': best_f1,
            'Precision_atBest': met['precision'],
            'Recall_atBest': met['recall'],
            'F1_atBest': met['f1'],
            'Precision@{}'.format(PRECISION_AT_K): p_at_k,
            'Lift@{}'.format(PRECISION_AT_K): lift_k,
            'TP': int(cm[1,1]),
            'FP': int(cm[0,1]),
            'FN': int(cm[1,0]),
            'TN': int(cm[0,0]),
            'BalancedAcc': balanced_acc
        })
        precision_vals, recall_vals, _ = precision_recall_curve(y_test_array, y_proba)
        if SHOW_PLOTS:
            plt.plot(recall_vals, precision_vals, lw=2, label=f"{name} (AUPRC={auprc:.3f})")
    if SHOW_PLOTS:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves (Test)")
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    summary_df = pd.DataFrame(eval_rows).sort_values('AUPRC', ascending=False) if eval_rows else pd.DataFrame()
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print("\n=== Test set performance summary ===")
    if summary_df.empty:
        print("No models evaluated successfully.")
    else:
        print(summary_df.to_string(index=False))
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary.csv"), index=False)
    for name, model in models.items():
        outp = os.path.join(OUTPUT_DIR, f"{name}.joblib")
        try:
            joblib.dump(model, outp)
        except Exception as e:
            print(f"Failed to save model {name}: {e}")
    print("\nSaved models and summary to:", OUTPUT_DIR)
    if skipped_steps:
        with open(os.path.join(OUTPUT_DIR, "skipped_steps.txt"), "w") as fh:
            fh.write("\n".join(skipped_steps))
        print("\nThe following steps were skipped or replaced (see skipped_steps.txt):")
        for s in skipped_steps:
            print(" -", s)

    # Permutation importance for best model
    if not summary_df.empty:
        print("\nComputing permutation importance on best model (by AUPRC)...")
        best_model_name = summary_df.iloc[0]['Model']
        best_model = models[best_model_name]
        try:
            r = permutation_importance(
                best_model,
                X_test,
                y_test_array,
                scoring=average_precision_scorer_for_perm,
                n_repeats=PERM_N_REPEATS,
                random_state=SEED,
                n_jobs=-1
            )
            imp_df = pd.DataFrame({'feature': X_test.columns, 'importance_mean': r.importances_mean, 'importance_std': r.importances_std})
            imp_df.sort_values('importance_mean', ascending=False, inplace=True)
            print("\nTop 15 features by permutation importance (drop in AUPRC):")
            print(imp_df.head(15).to_string(index=False))
            if SHOW_PLOTS:
                import seaborn as sns
                plt.figure(figsize=(8,6))
                sns.barplot(x='importance_mean', y='feature', data=imp_df.head(15))
                plt.title(f"Permutation importance (drop in average_precision) — {best_model_name}")
                plt.xlabel("Mean drop in average_precision")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print("Permutation importance failed:", e)

    # SHAP (optional)
    if SHAP_AVAILABLE and not summary_df.empty:
        try:
            print("\nRunning SHAP for the best model (sampled)...")
            best_model_name = summary_df.iloc[0]['Model']
            best_model = models[best_model_name]
            model_for_shap = best_model
            if hasattr(best_model, 'named_steps') and 'xgb' in best_model.named_steps:
                step = best_model.named_steps['xgb']
                model_for_shap = getattr(step, "model", step)
            elif hasattr(best_model, "get_booster"):
                model_for_shap = best_model
            explainer = shap.TreeExplainer(model_for_shap)
            sample = X_test.sample(n=min(SHAP_SAMPLE_N, len(X_test)), random_state=SEED)
            try:
                shap_values = explainer.shap_values(sample) if hasattr(explainer, "shap_values") else explainer(sample)
            except Exception:
                shap_values = explainer(sample)
            print("SHAP summary (mean abs):")
            shap.summary_plot(shap_values, sample, plot_type="bar", show=True)
            shap.summary_plot(shap_values, sample, show=True)
        except Exception as e:
            print("SHAP analysis failed:", e)
            print("Try: pip install shap (compatible version) or set SHAP_AVAILABLE=False in config.")
    else:
        if not SHAP_AVAILABLE:
            print("\nSHAP not installed. To enable SHAP, run: pip install shap")

    print("\nNext recommended steps:")
    print("- If you want SMOTE, install a compatible imbalanced-learn and re-run.")
    print("- Run larger hyperparameter search (Optuna) optimizing average_precision.")
    print("- Try stacking/ensembling and probability calibration.")
    print("- Add cost-sensitive evaluation (expected loss).")
    print("- Build a small demo showing top-k flagged items + SHAP explanations.")

if __name__ == "__main__":
    main()
