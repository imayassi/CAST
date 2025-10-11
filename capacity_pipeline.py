# capacity_pipeline.py (Final Consolidated Version)

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
import plotly.graph_objects as go

from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import norm

# =========================
# Configs
# =========================
@dataclass
class ColumnMap:
    date: str = "_date_"
    alias: str = "alias"
    group: str = "StaffGroup"
    num_cases: str = "numCases"
    open_cases: str = "numOpenCases"
    backlog: str = "backlog"
    time_spent: str = "TimeSpent"
    avg_dtc: str = "avgDaysToClose"
    efficiency_raw: str = "som"
    weekend_flag: str = "IsClosedOnWeekend"
    x24_flag: str = "Is24X7OptedIn"
    linearity: str = "LinearityScore"
    aics: str = "AICS"
    tenure: str = "tenure"
    active_time_ratio: str = "activeTimeRatio"
    hours_to_sla_exp: str = "hoursToSLAExpires"
    is_ir_met: str = "IsIRMet"
    sev_cols: Tuple[str, ...] = ("currentSev1","currentSevA","currentSevB","currentSevC")

@dataclass
class ScoreConfig:
    winsor_limits: Tuple[float,float] = (0.01,0.01)
    bench_percentile: float = 0.70
    small_group_min_n: int = 15
    percentile_clip: Tuple[int,int] = (1,99)
    scale_scope: str = "global"
    method: str = "DSLI"
    dsli_anchor_pct: float = 0.70
    dsli_min_features: int = 5
    direction_map: Optional[Dict[str,bool]] = None
    random_state: int = 42
    distribution: str = "linear"
    score_range: Tuple[int, int] = (0, 100)
    percentile_weights: Optional[Dict[str, float]] = None

@dataclass
class WSIConfig:
    baseline_window_months: int = 12
    persist_window_months: int = 8
    high_cap_thresh: float = 75.0
    weights: Dict[str,float] = field(default_factory=lambda: dict(
        w_workload=0.30, cap_dev=0.30, persist=0.15, complexity=0.15, time=0.10
    ))
    dev_tau: float = 0.25
    q_low: float = 0.05
    q_high: float = 0.95
    team_aggregate: str = "median"
    enforce_month_start: bool = True
    distribution: str = "linear"
    feature_range: Tuple[int, int] = (0, 100)

@dataclass
class ScenarioConfig:
    realloc_strategy: str = "proportional"
    aggregate: str = "median"
    random_state: int = 42
    capacity_col_out: str = "Capacity_Score_0_100"

@dataclass
class OptimizeConfig:
    budget_moves: int = 20
    objective_weights: Dict[str, float] = field(default_factory=lambda: dict(
        WSI_0_100=-1.0, efficiency=+0.6, days_to_close=-0.6, backlog=-0.4, numCases=+0.25
    ))
    random_state: int = 42

# =========================
# Constants
# =========================
FEATURE_DIRECTION: Dict[str, bool] = {
    "numCases": True, "numOpenCases": True, "backlog": True, "TimeSpent": True,
    "currentSevA": True, "Is24X7OptedIn": True, "IsClosedOnWeekend": True,
    "tenure": False, "som": False, "avgDaysToClose": True, "efficiency": False,
    "CustomerSatisfaction": False, "ActiveCases": True, "TenureInMonths": False,
    "EfficiencyRating": False, "AvgTimeToResolve": True, "TasksCompleted": True,
    "BacklogSize": True
}


# =========================
# Utils
# =========================
def _mstart(x):
    """Converts a date or Series of dates to the start of the month."""
    dates = pd.to_datetime(x, errors="coerce")
    if isinstance(dates, pd.Series):
        return dates.dt.to_period("M").dt.to_timestamp()
    else:
        return dates.to_period("M").to_timestamp()

def robust_minmax(series: pd.Series, feature_range=(0, 100), qlow=1, qhigh=99, distribution="linear"):
    arr = np.asarray(series.dropna())
    lo_val, hi_val = feature_range
    if len(arr) < 2: return pd.Series(np.full_like(series, (lo_val + hi_val) / 2.0, dtype=float), index=series.index)

    if distribution == "gaussian":
        ranks = series.rank(method='average', pct=True)
        z_scores = norm.ppf(ranks.clip(0.001, 0.999))
        scaled = (z_scores - (-3)) / (3 - (-3)) 
        final_scores = scaled * (hi_val - lo_val) + lo_val
        return pd.Series(np.clip(final_scores, lo_val, hi_val), index=series.index)
    else: # linear
        low_p_val = np.percentile(arr, qlow)
        high_p_val = np.percentile(arr, qhigh)
        if np.isclose(high_p_val, low_p_val): return pd.Series(np.full_like(series, (lo_val + hi_val) / 2.0, dtype=float), index=series.index)
        scaled = (np.clip(series, low_p_val, high_p_val) - low_p_val) / (high_p_val - low_p_val)
        return pd.Series(scaled * (hi_val - lo_val) + lo_val, index=series.index)


# =========================
# DataPrep & Scoring Classes
# =========================
class DataPrep:
    def __init__(self, columns: ColumnMap, features: List[str], config: ScoreConfig):
        self.columns, self.features, self.config = columns, features, config
        self.kept_features_, self.dev_cols_, self.direction_map_ = [], [], {}
        self.scaler_: Optional[StandardScaler] = None
        self.benchmarks_: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "DataPrep":
        d = df.copy()
        self.kept_features_ = [c for c in self.features if c in d.columns and pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 1]
        
        gsize = d.groupby(self.columns.group).size()
        small_groups = gsize[gsize < self.config.small_group_min_n].index
        
        bench_global = d[self.kept_features_].quantile(q=self.config.bench_percentile).add_suffix("_Benchmark")
        bench_by_group = (d[~d[self.columns.group].isin(small_groups)].groupby(self.columns.group)[self.kept_features_]
                          .quantile(q=self.config.bench_percentile).add_suffix("_Benchmark"))
        
        d = d.merge(bench_by_group, on=self.columns.group, how="left")
        for c in self.kept_features_: d[f"{c}_Benchmark"].fillna(bench_global[f"{c}_Benchmark"], inplace=True)
        
        self.benchmarks_ = d[[self.columns.group] + [f"{c}_Benchmark" for c in self.kept_features_]].drop_duplicates(self.columns.group)
        self.direction_map_ = self.config.direction_map or FEATURE_DIRECTION
        
        for f in self.kept_features_:
            direction = 1 if self.direction_map_.get(f, True) else -1
            d[f"{f}_Dev"] = direction * (d[f] - d[f"{f}_Benchmark"]) / d[f"{f}_Benchmark"].replace(0, 1e-6)
        
        self.dev_cols_ = [f"{c}_Dev" for c in self.kept_features_]
        self.scaler_ = StandardScaler().fit(d[self.dev_cols_].fillna(0.0))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.merge(self.benchmarks_, on=self.columns.group, how="left")
        for f in self.kept_features_:
            direction = 1 if self.direction_map_.get(f, True) else -1
            d[f"{f}_Dev"] = direction * (d[f] - d[f"{f}_Benchmark"]) / d[f"{f}_Benchmark"].replace(0, 1e-6)
        
        Z = self.scaler_.transform(d[self.dev_cols_].fillna(0.0))
        for i, c in enumerate(self.dev_cols_): d[f"Z_{c}"] = Z[:, i]
        return d

# In capacity_pipeline.py, replace the entire CapacityScorer class

class CapacityScorer:
    def __init__(self, config: ScoreConfig):
        self.cfg = config
        self.method = config.method.lower()
        self.dsli_model_: Optional[LogisticRegression] = None
        self.dev_cols_: List[str] = []
        self.clip_: Tuple[int,int] = config.percentile_clip

    def fit(self, df_prepped: pd.DataFrame, group_col: Optional[str] = None, id_cols: Optional[Tuple[str, ...]] = None) -> "CapacityScorer":
        self.dev_cols_ = [c for c in df_prepped.columns if c.startswith("Z_") and c.endswith("_Dev")]
        if not self.dev_cols_:
            raise ValueError("No standardized deviation columns found. Run DataPrep.transform first.")

        rng = self.cfg.random_state
        X = df_prepped[self.dev_cols_].values

        # Hybrid approach: Always try to fit the DSLI model for insights
        try:
            raw_features = [c.replace("Z_","").replace("_Dev","") for c in self.dev_cols_]
            preferred = ["numCases","backlog","numOpenCases","TimeSpent"]
            anchors = [c for c in preferred if c in raw_features] or raw_features[:max(1, min(self.cfg.dsli_min_features, len(raw_features)))]
            z_anchor_cols = [f"Z_{c}_Dev" for c in anchors]
            A = df_prepped[z_anchor_cols].mean(axis=1)
            
            q = self.cfg.dsli_anchor_pct
            y = (A >= A.quantile(q)).astype(int).values

            lr = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=rng)
            lr.fit(X, y)
            self.dsli_model_ = lr
        except Exception as e:
            warnings.warn(f"Could not fit DSLI model for insights. Contribution analysis will not be available. Reason: {e}")
            self.dsli_model_ = None
        
        return self

    def transform(self, df_prepped: pd.DataFrame, group_col: Optional[str] = None, id_cols: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
        id_cols_to_use = list(id_cols) if id_cols else [c for c in ["alias", "_date_"] if c in df_prepped.columns]
        group_col_to_use = group_col if group_col and group_col in df_prepped.columns else "StaffGroup"
        out_cols = id_cols_to_use + [group_col_to_use]
        
        out = df_prepped[out_cols].copy()
        Z = df_prepped[self.dev_cols_].values
        
        raw_score = pd.Series(np.nan, index=df_prepped.index)

        if self.method == "percentile":
            ranks = df_prepped[self.dev_cols_].rank(pct=True)
            if self.cfg.percentile_weights:
                weights_keys = [f"Z_{k}_Dev" for k in self.cfg.percentile_weights.keys()]
                weights = pd.Series(self.cfg.percentile_weights.values(), index=weights_keys)
                valid_weights = weights[weights.index.isin(ranks.columns)]
                if not valid_weights.empty:
                    valid_weights /= valid_weights.sum()
                    raw_score = ranks[valid_weights.index].mul(valid_weights).sum(axis=1)
                else:
                    raw_score = ranks.mean(axis=1)
            else:
                raw_score = ranks.mean(axis=1)
        
        elif self.method == "dsli":
            if self.dsli_model_:
                raw_score = self.dsli_model_.predict_proba(Z)[:, 1]
            else:
                 raise RuntimeError("DSLI model is not fitted, cannot use 'dsli' method.")
        
        out["Raw_Score"] = raw_score
        out["Capacity_Score_0_100"] = robust_minmax(
            out["Raw_Score"],
            feature_range=self.cfg.score_range,
            qlow=self.cfg.percentile_clip[0],
            qhigh=self.cfg.percentile_clip[1],
            distribution=self.cfg.distribution
        ).round(2)
        return out

    # --- ADDED BACK: feature_importances method ---
    def feature_importances(self) -> pd.DataFrame:
        if self.dsli_model_ is None:
            raise RuntimeError("DSLI model not available for feature importances.")
        
        features = [c.replace("Z_","").replace("_Dev","") for c in self.dev_cols_]
        coefs = self.dsli_model_.coef_.ravel()
        return pd.DataFrame({"feature": features, "importance": np.abs(coefs)}).sort_values("importance", ascending=False)

    # --- ADDED BACK: contributions method ---
    def contributions(self, df_prepped: pd.DataFrame, alias: str, date) -> pd.DataFrame:
        if self.dsli_model_ is None:
            raise RuntimeError("DSLI model not available for contributions analysis.")
            
        mask = (df_prepped["alias"]==alias) & (_mstart(df_prepped["_date_"])==_mstart(date))
        if not mask.any():
            raise KeyError("Alias/date not found.")
        
        x = df_prepped.loc[mask, self.dev_cols_].values[0]
        coefs = self.dsli_model_.coef_.ravel()
        parts = coefs * x
        df = pd.DataFrame({
            "feature":[c.replace("Z_","").replace("_Dev","") for c in self.dev_cols_],
            "contribution": parts
        }).sort_values(by="contribution", key=abs, ascending=False).reset_index(drop=True)
        return df

class WSIComputer:
    def __init__(self, columns: ColumnMap, config: WSIConfig):
        self.columns, self.cfg = columns, config

    def compute(self, df_scored: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        c, cfg = self.columns, self.cfg
        d = df_scored.copy(); d[c.date] = _mstart(d[c.date]); d.sort_values([c.alias, c.date], inplace=True)
        
        cap = pd.to_numeric(d["Capacity_Score_0_100"], errors="coerce").clip(0, 100)
        num_cases = pd.to_numeric(d[c.num_cases], errors="coerce")
        
        workload_norm = (num_cases - num_cases.min()) / (num_cases.max() - num_cases.min() + 1e-6)
        base = d.groupby(c.alias)["Capacity_Score_0_100"].transform(lambda s: s.rolling(cfg.baseline_window_months, min_periods=3).median()).fillna(cap.median())
        cap_dev_norm = np.clip(np.maximum(0, (cap/base.replace(0,np.nan)).fillna(1)-1)/cfg.dev_tau, 0, 1)
        persist = d.groupby(c.alias)["Capacity_Score_0_100"].transform(lambda s: s.rolling(cfg.persist_window_months).apply(lambda x: (x>=cfg.high_cap_thresh).all(), raw=True)).fillna(0)
        
        sev_terms = [pd.to_numeric(d[scn],errors='coerce').fillna(0)/(num_cases.replace(0,np.nan))*w for scn,w in zip(c.sev_cols, [1,.5,.2]) if scn in d.columns]
        complexity = (sum(sev_terms) if sev_terms else 0).clip(0, 1)
        timep = (0.6*d.get(c.weekend_flag, 0) + 0.4*d.get(c.x24_flag, 0)).fillna(0).clip(0,1)
        
        W_raw = (cfg.weights["w_workload"] * workload_norm + cfg.weights["cap_dev"] * cap_dev_norm + 
                 cfg.weights["persist"] * persist + cfg.weights["complexity"] * complexity + 
                 cfg.weights["time"] * timep)
        
        WSI = robust_minmax(W_raw, cfg.feature_range, cfg.q_low*100, cfg.q_high*100, cfg.distribution).round(2)
        
        alias_df = d[[c.alias, c.group, c.date, "Capacity_Score_0_100"]].copy()
        alias_df["WSI_0_100"] = WSI
        alias_df["cap_dev"], alias_df["complexity"] = cap_dev_norm, complexity
        
        team_df = alias_df.groupby([c.group, c.date]).agg({"WSI_0_100": cfg.team_aggregate, "cap_dev": cfg.team_aggregate, "complexity": cfg.team_aggregate}).reset_index()
        return {"alias_metrics": alias_df, "team_metrics": team_df}

# =========================
# Simulation & Prediction Pipeline
# =========================
def _coalesce_names_for_training(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "efficiency" not in d.columns and "som" in d.columns: d["efficiency"] = pd.to_numeric(d["som"], errors="coerce")
    if "efficiency" not in d.columns and "EfficiencyRating" in d.columns: d["efficiency"] = pd.to_numeric(d["EfficiencyRating"], errors="coerce")
    if "days_to_close" not in d.columns and "avgDaysToClose" in d.columns: d["days_to_close"] = pd.to_numeric(d["avgDaysToClose"], errors="coerce")
    if "days_to_close" not in d.columns and "AvgTimeToResolve" in d.columns: d["days_to_close"] = pd.to_numeric(d["AvgTimeToResolve"], errors="coerce")
    return d

def prepare_training_df(df_in: pd.DataFrame, scores_baseline: pd.DataFrame, *, compute_wsi_fn, wsi_kwargs: Optional[dict] = None) -> pd.DataFrame:
    df = df_in.copy(); df["_date_"] = _mstart(df["_date_"])
    sc = scores_baseline.rename(columns={"Capacity_Score_0_100": "Capacity"})
    train = df.merge(sc[["_date_", "alias", "Capacity"]], on=["_date_", "alias"], how="left")
    if "Capacity_Score_0_100" not in train.columns: train["Capacity_Score_0_100"] = train["Capacity"]
    train = _coalesce_names_for_training(train)
    wsi = compute_wsi_fn(train.copy(), **(wsi_kwargs or {}))["alias_metrics"]
    train = train.merge(wsi[["_date_", "alias", "WSI_0_100"]], on=["_date_", "alias"], how="left")
    keep = ["StaffGroup", "alias", "_date_", "Capacity", "WSI_0_100", "efficiency", "days_to_close", "backlog", "numCases"]
    return train[[c for c in keep if c in train.columns]].dropna()

def fit_tree_models(train_df: pd.DataFrame, *, targets):
    models, mse_report = {}, {}
    for t in targets:
        X = train_df[["Capacity", "StaffGroup"]]; y = train_df[t]
        preproc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["StaffGroup"])])
        pipe = Pipeline([("prep", preproc), ("reg", xgb.XGBRegressor(random_state=42))])
        pipe.fit(X, y); models[t] = pipe; mse_report[t] = mean_squared_error(y, pipe.predict(X))
    return models, mse_report

# def simulate_post_capacity(df_in: pd.DataFrame, scores_baseline: pd.DataFrame, month: str, moves: List[Dict], realloc_strategy="proportional", random_state=42):
#     d = df_in.copy(); d["_date_"] = _mstart(d["_date_"]); m0 = _mstart(month)
#     sc = scores_baseline.rename(columns={"Capacity_Score_0_100": "Capacity"})
#     pre = d[d["_date_"] == m0].merge(sc, on=["_date_", "alias"], how="left")
    
#     knn_by_sg = {}
#     for sg, g in pre.groupby("StaffGroup"):
#         valid = g["Capacity"].notna() & g["numCases"].notna()
#         if valid.sum() >= 5:
#             knn_by_sg[sg] = KNeighborsRegressor(n_neighbors=min(5, valid.sum())).fit(g.loc[valid, ["numCases"]], g.loc[valid, "Capacity"])
            
#     cur = pre.copy(); rng = np.random.default_rng(random_state)
#     for mv in moves:
#         sg_from, sg_to = mv.get("from"), mv.get("to")
#         if not sg_from or not sg_to: continue
#         pool = cur[cur["StaffGroup"] == sg_from]["alias"].unique()
#         n = int(mv.get("n", 0) or len(pool) * mv.get("pct", 0))
#         moved = rng.choice(pool, size=min(n, len(pool)), replace=False)
#         if len(moved) == 0: continue
        
#         removed_workload = cur.loc[cur["alias"].isin(moved), "numCases"].sum()
#         cur.loc[cur["alias"].isin(moved), "numCases"] = 0
#         remain = cur[(cur["StaffGroup"] == sg_from) & (~cur["alias"].isin(moved))]
#         if not remain.empty: cur.loc[remain.index, "numCases"] += removed_workload / len(remain)
#         cur.loc[cur["alias"].isin(moved), "StaffGroup"] = sg_to

#     def predict_group_capacity(g):
#         sg_name = g.name
#         if sg_name in knn_by_sg: g["Capacity"] = knn_by_sg[sg_name].predict(g[["numCases"]])
#         else: g["Capacity"] = pre[pre["StaffGroup"] == sg_name]["Capacity"].median()
#         return g
#     cur = cur.groupby("StaffGroup").apply(predict_group_capacity).reset_index(drop=True)
#     return cur.groupby("StaffGroup")["Capacity"].median().reset_index()

def simulate_post_capacity(df_in: pd.DataFrame, scores_baseline: pd.DataFrame, month: str, moves: List[Dict], realloc_strategy="proportional", random_state=42):
    d = df_in.copy()
    d["_date_"] = _mstart(d["_date_"])
    m0 = _mstart(month)
    
    # RENAME the capacity score column for the merge
    sc = scores_baseline.rename(columns={"Capacity_Score_0_100": "Capacity"})
    
    # --- FIX ---
    # Drop "StaffGroup" from the right-side DataFrame (sc) before merging
    # to prevent pandas from creating 'StaffGroup_x' and 'StaffGroup_y'
    if "StaffGroup" in sc.columns:
        sc = sc.drop(columns=["StaffGroup"])
        
    pre = d[d["_date_"] == m0].merge(sc, on=["_date_", "alias"], how="left")
    
    knn_by_sg = {}
    # The groupby will now work because 'pre' has a single 'StaffGroup' column
    for sg, g in pre.groupby("StaffGroup"):
        valid = g["Capacity"].notna() & g["numCases"].notna()
        if valid.sum() >= 5:
            knn_by_sg[sg] = KNeighborsRegressor(n_neighbors=min(5, valid.sum())).fit(g.loc[valid, ["numCases"]], g.loc[valid, "Capacity"])
            
    cur = pre.copy()
    rng = np.random.default_rng(random_state)
    for mv in moves:
        sg_from, sg_to = mv.get("from"), mv.get("to")
        if not sg_from or not sg_to: continue
        pool = cur[cur["StaffGroup"] == sg_from]["alias"].unique()
        n = int(mv.get("n", 0) or len(pool) * mv.get("pct", 0))
        moved = rng.choice(pool, size=min(n, len(pool)), replace=False)
        if len(moved) == 0: continue
        
        removed_workload = cur.loc[cur["alias"].isin(moved), "numCases"].sum()
        cur.loc[cur["alias"].isin(moved), "numCases"] = 0
        remain = cur[(cur["StaffGroup"] == sg_from) & (~cur["alias"].isin(moved))]
        if not remain.empty: cur.loc[remain.index, "numCases"] += removed_workload / len(remain)
        cur.loc[cur["alias"].isin(moved), "StaffGroup"] = sg_to

    def predict_group_capacity(g):
        sg_name = g.name
        if sg_name in knn_by_sg:
            # Ensure numCases is 2D for prediction
            g["Capacity"] = knn_by_sg[sg_name].predict(g[["numCases"]])
        else:
            # Fallback to the median of the original group
            median_cap = pre[pre["StaffGroup"] == sg_name]["Capacity"].median()
            g["Capacity"] = median_cap if pd.notna(median_cap) else pre["Capacity"].median()
        return g

    # The rest of the function remains the same
    cur = cur.groupby("StaffGroup").apply(predict_group_capacity).reset_index(drop=True)
    return cur.groupby("StaffGroup")["Capacity"].median().reset_index()

def predict_post_from_capacity(post_cap_df, models, targets):
    preds = post_cap_df.copy()
    for t in targets:
        if t in models: preds[t] = models[t].predict(preds[["Capacity", "StaffGroup"]])
    return preds

def plot_color_grid(grid_df, color_by="WSI_0_100", title="StaffGroup Grid"):
    cols = ["Capacity","WSI_0_100","efficiency","days_to_close","backlog"]
    z = np.tile(grid_df[color_by].values.reshape(-1,1), (1, len(cols))) / 100
    text = np.column_stack([grid_df[c].round(2).astype(str) for c in cols])
    fig = go.Figure(data=go.Heatmap(z=z, x=cols, y=grid_df["StaffGroup"], text=text, texttemplate="%{text}", colorscale="RdYlGn_r", reversescale=True))
    fig.update_layout(title=title, margin=dict(l=140, t=80))
    return fig

def pre_post_tree_pipeline(
    df_in: pd.DataFrame,
    scores_baseline: pd.DataFrame,
    month: str,
    moves: List[Dict], *,
    compute_wsi_fn,
    wsi_kwargs: Optional[dict] = None,
    targets=("WSI_0_100", "efficiency", "days_to_close", "backlog"),
    only_impacted: bool = True,
    team_level_post: bool = False,      # NEW: predict once per team from aggregated capacity
    show_predict_samples: bool = True   # NEW: print the exact X passed into .predict()
):
    """
    Runs the end-to-end simulation pipeline and returns results including a sanity check DataFrame.
    PRE grid uses observed actuals. POST grid is simulated using the trained outcome models.
    """
    # -----------------------------
    # 0) Prep + fit
    # -----------------------------
    train_df = prepare_training_df(df_in, scores_baseline,
                                   compute_wsi_fn=compute_wsi_fn, wsi_kwargs=wsi_kwargs)
    models, mse_report = fit_tree_models(train_df, targets=targets)

    m0 = _mstart(month)
    pre_month = train_df[train_df["_date_"] == m0].copy()

    # -----------------------------
    # 1) PRE grid (actuals, team median)
    # -----------------------------
    # Capacity + targets at team level (median by default to match prior behavior)
    pre_cols = ["Capacity"] + list(targets)
    pre_cols = [c for c in pre_cols if c in pre_month.columns]
    grid_pre = (pre_month.groupby("StaffGroup")[pre_cols]
                .median().reset_index())

    # Sanity ranges (IQR) for PRE (alias-level)
    sanity_ranges = {}
    for metric in ["Capacity"] + list(targets):
        if metric in pre_month.columns:
            q1 = pre_month[metric].quantile(0.25)
            q3 = pre_month[metric].quantile(0.75)
            sanity_ranges[metric] = f"{q1:.2f} to {q3:.2f}"
    sanity_df = (pd.DataFrame.from_dict(sanity_ranges, orient='index',
                                        columns=['Typical Range (25th-75th percentile)'])
                 .rename_axis("Metric"))

    # -----------------------------
    # 2) POST capacity simulation
    # -----------------------------
    post_cap = simulate_post_capacity(df_in, scores_baseline, month, moves).copy()
    # Ensure expected schema
    if "_date_" in post_cap.columns:
        post_cap["_date_"] = pd.to_datetime(post_cap["_date_"]).dt.to_period("M").dt.to_timestamp()
        post_cap = post_cap[post_cap["_date_"] == m0].copy()
    if "Capacity" not in post_cap.columns and "Capacity_Score_0_100" in post_cap.columns:
        post_cap = post_cap.rename(columns={"Capacity_Score_0_100": "Capacity"})

    # -----------------------------
    # 3) POST grid (two modes)
    # -----------------------------
    if not team_level_post:
        # Alias-level path (original behavior)
        # Show exactly what goes into the model: Capacity + StaffGroup (raw; OHE is inside pipeline)
        if show_predict_samples:
            X_alias_preview = post_cap[["Capacity", "StaffGroup"]].copy()
            X_alias_preview["StaffGroup"] = X_alias_preview["StaffGroup"].astype("string").fillna("__MISSING__")
            print("\n[DEBUG] Alias-level predict() sample (first 8 rows):")
            print(X_alias_preview.head(8).to_string(index=False))
            print("Rows to predict (alias-level):", len(X_alias_preview))

        grid_post = predict_post_from_capacity(post_cap, models, targets=targets)
        # Optional: bring team capacity to grid_post (median) so combined tables still show Capacity
        team_cap_post = (post_cap.groupby("StaffGroup", as_index=False)["Capacity"].median())
        if "Capacity" not in grid_post.columns:
            grid_post = grid_post.merge(team_cap_post, on="StaffGroup", how="left")

    else:
        # Team-level path (predict once per team from aggregated capacity)
        team_cap = (post_cap.groupby("StaffGroup", as_index=False)["Capacity"].median())
        # Build raw X exactly like training (let pipeline do OneHotEncoder internally)
        X_team = team_cap[["Capacity", "StaffGroup"]].copy()
        X_team["StaffGroup"] = X_team["StaffGroup"].astype("string").fillna("__MISSING__")

        if show_predict_samples:
            print("\n[DEBUG] Team-level predict() sample (first 20 rows):")
            print(X_team.head(20).to_string(index=False))
            print("Rows to predict (team-level):", len(X_team))

        preds = {"StaffGroup": team_cap["StaffGroup"].values}
        for t in targets:
            preds[t] = models[t].predict(X_team)
        grid_post = pd.DataFrame(preds)
        # Add capacity column for completeness in combined view
        grid_post = grid_post.merge(team_cap, on="StaffGroup", how="left")

    # -----------------------------
    # 4) Restrict to impacted teams (safely)
    # -----------------------------
    if only_impacted:
        impacted = get_impacted_sgs(moves)
        # Safety: only filter if we actually have overlap; otherwise leave as-is and warn
        pre_mask = grid_pre["StaffGroup"].isin(impacted)
        post_mask = grid_post["StaffGroup"].isin(impacted)
        if pre_mask.any():
            grid_pre = grid_pre[pre_mask]
        else:
            print("[WARN] No PRE rows match impacted teams; leaving PRE unfiltered.")
        if post_mask.any():
            grid_post = grid_post[post_mask]
        else:
            print("[WARN] No POST rows match impacted teams; leaving POST unfiltered.")

    # -----------------------------
    # 5) Combine in the original table-like format + plot
    # -----------------------------
    gp = grid_pre.copy(); gp["Scenario"] = "Pre"
    gq = grid_post.copy(); gq["Scenario"] = "Post (pred)"
    combined = (pd.concat([gp, gq], ignore_index=True)
                  .sort_values(["StaffGroup", "Scenario"])
                  .reset_index(drop=True))

    fig = plot_color_grid_with_sanity(
        combined,                  # uses StaffGroup + Scenario shape directly
        sanity_df,                 # IQR ranges
        title="Pre vs. Post Simulation (Sanity-flagged)",
        metrics=[m for m in ["Capacity","WSI_0_100","efficiency","days_to_close","backlog","numCases"] if m in combined.columns],
        soft_color="rgba(255, 99, 71, 0.18)"  # adjust softness as you like
    )

    return grid_pre, grid_post, combined, fig, mse_report, sanity_df

# def pre_post_tree_pipeline(
#     df_in: pd.DataFrame, scores_baseline: pd.DataFrame, month: str, moves: List[Dict], *,
#     compute_wsi_fn, wsi_kwargs: Optional[dict] = None,
#     targets=("WSI_0_100", "efficiency", "days_to_close", "backlog"),
#     only_impacted=True
# ):
#     """
#     Runs the end-to-end simulation pipeline and returns results including a sanity check DataFrame.
#     """
#     train_df = prepare_training_df(df_in, scores_baseline, compute_wsi_fn=compute_wsi_fn, wsi_kwargs=wsi_kwargs)
#     models, mse_report = fit_tree_models(train_df, targets=targets)
    
#     m0 = _mstart(month)
#     pre_month = train_df[train_df["_date_"] == m0]
#     grid_pre = pre_month.groupby("StaffGroup")[["Capacity"] + list(targets)].median().reset_index()
    
#     # --- NEW: Calculate the sanity check ranges (IQR) ---
#     sanity_ranges = {}
#     for metric in ["Capacity"] + list(targets):
#         if metric in pre_month.columns:
#             q1 = pre_month[metric].quantile(0.25)
#             q3 = pre_month[metric].quantile(0.75)
#             sanity_ranges[metric] = f"{q1:.2f} to {q3:.2f}"
            
#     sanity_df = pd.DataFrame.from_dict(sanity_ranges, orient='index', columns=['Typical Range (25th-75th percentile)'])
#     sanity_df.index.name = "Metric"
    
#     post_cap = simulate_post_capacity(df_in, scores_baseline, month, moves)
#     grid_post = predict_post_from_capacity(post_cap, models, targets=targets)
    
#     if only_impacted:
#         impacted = get_impacted_sgs(moves)
#         grid_pre = grid_pre[grid_pre["StaffGroup"].isin(impacted)]
#         grid_post = grid_post[grid_post["StaffGroup"].isin(impacted)]
        
#     gp = grid_pre.copy(); gp["Scenario"] = "Pre"
#     gq = grid_post.copy(); gq["Scenario"] = "Post (pred)"
#     combined = pd.concat([gp, gq], ignore_index=True).sort_values(["StaffGroup", "Scenario"]).reset_index(drop=True)
    
#     grid_for_plot = combined.copy()
#     grid_for_plot["StaffGroup"] = grid_for_plot["StaffGroup"] + " — " + grid_for_plot["Scenario"].astype(str)
#     fig = plot_color_grid(grid_for_plot, title="Pre vs. Post Simulation")
    
#     # --- UPDATED: Return the new sanity_df as well ---
#     return grid_pre, grid_post, combined, fig, mse_report, sanity_df

def compute_capacity_centric_wsi(df, **kwargs):
    cols = ColumnMap(**{k:v for k,v in kwargs.items() if hasattr(ColumnMap,k)})
    wsi_cfg = WSIConfig(**{k:v for k,v in kwargs.items() if hasattr(WSIConfig,k)})
    return WSIComputer(columns=cols, config=wsi_cfg).compute(df)


def get_impacted_sgs(moves: List[Dict]) -> List[str]:
    """Extracts a unique, sorted list of all StaffGroups mentioned in a move plan."""
    s = set()
    for mv in moves:
        if mv.get("from"): s.add(str(mv["from"]))
        if mv.get("to"):   s.add(str(mv["to"]))
    return sorted(s)
# =========================
# Plotting Helpers
# =========================
def make_sample_data(n_groups=4, aliases_per_group=25, months=("2025-06","2025-07","2025-08"), seed=42) -> pd.DataFrame:
    # This function is unchanged from your original script
    rng = np.random.default_rng(seed)
    sgs = [f"SG-{chr(65+i)}" for i in range(n_groups)]
    rows = []

    personas = {
        "High Performer": {"eff_mean": 0.90, "dtc_mean": 4, "workload_factor": 1.1, "backlog_factor": -0.5, "noise": 0.5},
        "Overworked Star": {"eff_mean": 0.85, "dtc_mean": 6, "workload_factor": 1.5, "backlog_factor": 1.0, "noise": 0.8},
        "Steady Performer": {"eff_mean": 0.78, "dtc_mean": 8, "workload_factor": 1.0, "backlog_factor": 0.0, "noise": 1.0},
        "Struggling": {"eff_mean": 0.65, "dtc_mean": 12, "workload_factor": 0.9, "backlog_factor": 1.5, "noise": 1.2},
    }
    persona_names = list(personas.keys())

    for sg in sgs:
        for j in range(aliases_per_group):
            alias = f"{sg}-E{j+1:02d}"
            
            persona_key = rng.choice(persona_names, p=[0.20, 0.20, 0.40, 0.20])
            p = personas[persona_key]
            
            base_workload = rng.normal(40, 5) * p['workload_factor']
            prev_bkl = max(0, rng.normal(30 + p['backlog_factor'] * 10, 5))

            for m in months:
                dt = _mstart(m)
                noise = p['noise']

                numCases = int(max(10, base_workload + rng.normal(0, 5 * noise)))
                efficiency = float(np.clip(rng.normal(p['eff_mean'], 0.03 * noise), 0.5, 0.98))
                dtc = float(np.clip(rng.normal(p['dtc_mean'], 1 * noise), 1, 30))
                backlog = int(max(0, prev_bkl * 0.7 + numCases * 0.1 + rng.normal(0, 5 * noise)))
                prev_bkl = backlog

                tenure = float(np.clip(rng.normal(24, 12), 3, 72))
                numOpen = int(max(0, numCases * rng.uniform(0.1, 0.3)))
                time_spent = float(numCases * (2.0 - (efficiency - 0.75)) + rng.normal(0, 5 * noise))
                sevA = int(rng.binomial(numCases, 0.15 + (p.get('complexity_factor', 0) * 0.1)))

                rows.append(dict(
                    _date_=dt, alias=alias, StaffGroup=sg,
                    numCases=numCases, numOpenCases=numOpen, backlog=backlog, TimeSpent=time_spent,
                    som=efficiency, avgDaysToClose=dtc, tenure=tenure,
                    currentSevA=sevA, IsClosedOnWeekend=int(rng.random() < 0.2), Is24X7OptedIn=int(rng.random() < 0.1),
                    TasksCompleted=numCases, ActiveCases=numOpen, BacklogSize=backlog, AvgTimeToResolve=dtc,
                    TenureInMonths=tenure, EfficiencyRating=efficiency, CustomerSatisfaction=rng.uniform(0.7, 0.98)
                ))
    return pd.DataFrame(rows)

# =========================
# --- NEW: Optimizer Pipeline (Search Mechanism) ---
# =========================

def _objective(grid: pd.DataFrame, weights: Dict[str, float]) -> float:
    """Calculates a single score for a simulation outcome grid."""
    score = 0.0
    for col, weight in weights.items():
        if col in grid.columns:
            val = grid[col].sum() if col in ['backlog', 'numCases'] else grid[col].mean()
            score += val * weight
    return score

def run_optimizer_pipeline(df_full, scores_baseline, opt_cfg, cols: ColumnMap, **kwargs):
    """
    Finds an optimal move plan by iteratively searching for the best single move
    that improves the objective score, using pre_post_tree_pipeline as the evaluation engine.
    """
    # 1. Get the initial state before any moves
    month_str = df_full['_date_'].max().strftime('%Y-%m')
    initial_results = pre_post_tree_pipeline(
        df_in=df_full,
        scores_baseline=scores_baseline,
        month=month_str,
        moves=[], # No moves for initial state
        **kwargs
    )
    initial_metrics = initial_results[0]
    best_obj = _objective(initial_metrics, opt_cfg.objective_weights)
    
    plan = []
    
    # 2. Iteratively find the best move for the given budget
    for i in range(opt_cfg.budget_moves):
        # Identify high-stress (from) and low-stress (to) teams based on current WSI
        current_state = pre_post_tree_pipeline(df_in=df_full, scores_baseline=scores_baseline, month=month_str, moves=plan, **kwargs)[0]
        from_groups = current_state.sort_values("WSI_0_100", ascending=False)['StaffGroup'].tolist()
        to_groups = current_state.sort_values("WSI_0_100", ascending=True)['StaffGroup'].tolist()
        
        best_move, best_local_obj = None, -np.inf

        # Search for the best single move (e.g., moving 1 person)
        for sg_from in from_groups[:3]: # Limit search space for speed
            for sg_to in to_groups[:3]:
                if sg_from == sg_to: continue
                
                # Try moving one person and evaluate the outcome
                test_moves = plan + [{"from": sg_from, "to": sg_to, "n": 1}]
                
                post_grid = pre_post_tree_pipeline(
                    df_in=df_full, scores_baseline=scores_baseline, month=month_str,
                    moves=test_moves, **kwargs
                )[1]
                
                obj = _objective(post_grid, opt_cfg.objective_weights)
                
                if obj > best_local_obj:
                    best_local_obj = obj
                    best_move = {"from": sg_from, "to": sg_to, "n": 1}
        
        # If a good move is found, add it to the plan
        if best_move and best_local_obj > best_obj:
            plan.append(best_move)
            best_obj = best_local_obj
        else:
            break # Stop if no improvement is found

    # 3. Get the final expected metrics after all moves
    final_results = pre_post_tree_pipeline(df_in=df_full, scores_baseline=scores_baseline, month=month_str, moves=plan, **kwargs)
    expected_metrics = final_results[1]

    return {"plan": plan, "initial_metrics": initial_metrics, "expected_metrics": expected_metrics}


def plot_score_distributions(alias_df: pd.DataFrame, score_cols: List[str], save_path: Optional[str] = None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, len(score_cols), figsize=(7 * len(score_cols), 5), sharey=True)
    if len(score_cols) == 1: axes = [axes]
    for ax, col in zip(axes, score_cols):
        sns.histplot(alias_df[col], kde=True, ax=ax, bins=20, color='skyblue')
        ax.set_title(f'Distribution of {col}', fontsize=14, weight='bold')
        ax.set_xlabel('Score (0-100)'); ax.set_ylabel('Number of Employees')
    plt.suptitle('Employee Score Distributions', fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=150); print(f"✅ Plot saved to: {save_path}")
    plt.show()

def plot_team_metrics(team_df: pd.DataFrame, metric_cols: List[str], save_path: Optional[str] = None):
    plt.style.use('seaborn-v0_8-whitegrid')
    df_melt = team_df.melt(id_vars=['StaffGroup', '_date_'], value_vars=metric_cols, var_name='Metric', value_name='Score')
    g = sns.catplot(data=df_melt, x='_date_', y='Score', hue='StaffGroup', col='Metric',
                    kind='bar', col_wrap=2, height=4, aspect=1.5, sharey=False, palette='viridis')
    g.fig.suptitle('Team Metrics Over Time', fontsize=18, weight='bold')
    g.set_titles("{col_name}", size=14)
    g.set_xticklabels(rotation=45, ha='right')
    g.set_axis_labels("Month", "Average Score")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path: g.savefig(save_path, bbox_inches='tight', dpi=150); print(f"✅ Plot saved to: {save_path}")
    plt.show()

def plot_feature_importances(scorer: CapacityScorer, save_path: Optional[str] = None):
    imp_df = scorer.feature_importances().nlargest(10, 'importance')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x='importance', y='feature', palette='mako')
    plt.title('Top 10 Drivers of the Capacity Score', fontsize=16, weight='bold')
    plt.xlabel('Importance (Coefficient Magnitude)'); plt.ylabel('Feature')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=150); print(f"✅ Plot saved to: {save_path}")
    plt.show()

def plot_contributions_waterfall(contrib_df: pd.DataFrame, alias: str, save_path: Optional[str] = None):
    contrib_df = contrib_df.head(10).sort_values('contribution')
    plt.figure(figsize=(10, 6))
    colors = ['crimson' if c < 0 else 'mediumseagreen' for c in contrib_df['contribution']]
    plt.barh(contrib_df['feature'], contrib_df['contribution'], color=colors)
    plt.title(f'Top Score Contributions for {alias}', fontsize=16, weight='bold')
    plt.xlabel('Contribution to Score (Log-Odds)'); plt.ylabel('Feature')
    plt.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=150); print(f"✅ Plot saved to: {save_path}")
    plt.show()

# in capacity_pipeline.py
def plot_simulation_heatmap(
    combined: pd.DataFrame,
    metric_for_color: str = "WSI_0_100_Optimized",
    zmin: float | None = None,
    zmax: float | None = None,
):
    """
    Heatmap where color is driven by `metric_for_color`.
    Defaults to optimized WSI so higher stress -> red, lower -> green.
    Pass "WSI_0_100_Initial" if you want the pre state, or "WSI_0_100_Delta" if you compute deltas.
    """
    import plotly.express as px

    # Ensure the column exists; fall back gracefully
    m = metric_for_color
    if m not in combined.columns:
        # try common fallbacks in order
        for cand in ["WSI_0_100_Optimized", "WSI_0_100_Initial", "WSI_0_100"]:
            if cand in combined.columns:
                m = cand
                break

    z = (combined
         .pivot(index="StaffGroup", columns="_metric", values=m)
         .sort_index())

    fig = px.imshow(
        z.values,
        x=z.columns,
        y=z.index,
        color_continuous_scale="RdYlGn_r",  # high WSI -> red, low -> green
        zmin=zmin,
        zmax=zmax,
        aspect="auto"
    )
    fig.update_layout(
        title=f"Grid Heatmap (color: {m})",
        xaxis_title="Metric",
        yaxis_title="StaffGroup",
        coloraxis_colorbar_title=m
    )
    return fig
    
def plot_optimization_plan(plan: List[Dict], save_path: Optional[str] = None):
    if not plan: return
    # This line expects 'from_group' and 'to_group'
    moves_summary = pd.DataFrame(plan).groupby(['from_group', 'to_group']).size().reset_index(name='count')
    net_moves = pd.concat([
        moves_summary.groupby('from_group')['count'].sum() * -1,
        moves_summary.groupby('to_group')['count'].sum()
    ]).groupby(level=0).sum().reset_index(name='Net Change').rename(columns={'index': 'StaffGroup'})

    plt.figure(figsize=(10, 6))
    sns.barplot(data=net_moves, x='Net Change', y='StaffGroup', palette='PiYG')
    plt.title('Optimal Reassignment Plan: Net Headcount Moves', fontsize=16, weight='bold')
    plt.xlabel('Number of Employees to Move'); plt.ylabel('Staffing Group')
    plt.axvline(0, color='grey', linestyle='--')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=150); print(f"✅ Plot saved to: {save_path}")
    plt.show()  

def plot_simulation_distributions(pre_df_employee: pd.DataFrame, post_df_employee: pd.DataFrame, metrics: List[str], save_path: Optional[str] = None):
    pre_df_employee['State'] = 'Pre-Simulation'
    post_df_employee['State'] = 'Post-Simulation'
    combined_df = pd.concat([pre_df_employee, post_df_employee])
    melted_df = combined_df.melt(id_vars=['State'], value_vars=metrics, var_name='Metric', value_name='Value')
    g = sns.displot(data=melted_df, x='Value', hue='State', col='Metric', kind='kde', fill=True, common_norm=False, facet_kws={'sharey': False, 'sharex': False}, palette={'Pre-Simulation': 'skyblue', 'Post-Simulation': 'coral'})
    for ax, metric in zip(g.axes.flat, metrics):
        mean_pre, mean_post = pre_df_employee[metric].mean(), post_df_employee[metric].mean()
        diff_text = f'Mean Diff (Post - Pre): {mean_post - mean_pre:+.2f}'
        ax.text(0.95, 0.95, diff_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    g.fig.suptitle('Pre vs. Post Simulation: Employee-Level Distributions', fontsize=16, weight='bold')
    g.set_titles("{col_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path: g.savefig(save_path, bbox_inches='tight', dpi=150); print(f"✅ Distribution plot saved to: {save_path}")
    plt.show()

def plot_volume_uplift_distributions(pre_df: pd.DataFrame, post_df: pd.DataFrame, metric: str, save_path: Optional[str] = None):
    pre_df['State'] = 'Before Uplift'
    post_df['State'] = 'After Uplift (Simulated)'
    combined_df = pd.concat([pre_df, post_df])
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=combined_df, x=metric, hue='State', fill=True, common_norm=False, palette={'Before Uplift': 'skyblue', 'After Uplift (Simulated)': 'coral'})
    mean_pre, mean_post = pre_df[metric].mean(), post_df[metric].mean()
    diff_text = f'Mean Diff (After - Before): {mean_post - mean_pre:+.2f}'
    plt.text(0.95, 0.95, diff_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.title(f'Overall Distribution of {metric} After Uplift', fontsize=16, weight='bold')
    plt.xlabel(f'Value of {metric}'); plt.ylabel('Density')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=150); print(f"✅ Volume uplift plot saved to: {save_path}")
    plt.show()

def plot_bucket_uplift_distributions(alias_df: pd.DataFrame, metric: str, from_bucket: Tuple, to_bucket: Tuple, save_path: Optional[str] = None):
    from_df = alias_df[(alias_df['Capacity_Score_0_100'] >= from_bucket[0]) & (alias_df['Capacity_Score_0_100'] < from_bucket[1])]
    to_df = alias_df[(alias_df['Capacity_Score_0_100'] >= to_bucket[0]) & (alias_df['Capacity_Score_0_100'] < to_bucket[1])]
    if from_df.empty or to_df.empty: return
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=from_df, x=metric, fill=True, color='skyblue', label=f'"{from_bucket[0]}-{from_bucket[1]}" Bucket (Before)')
    sns.kdeplot(data=to_df, x=metric, fill=True, color='coral', label=f'"{to_bucket[0]}-{to_bucket[1]}" Bucket (Target)')
    mean_from, mean_to = from_df[metric].mean(), to_df[metric].mean()
    diff_text = f'Mean Diff (Target - Before): {mean_to - mean_from:+.2f}'
    plt.text(0.95, 0.95, diff_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.title(f'Distribution of {metric} for Bucket Uplift Scenario', fontsize=16, weight='bold')
    plt.xlabel(f'Value of {metric}'); plt.ylabel('Density')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=150); print(f"✅ Bucket uplift plot saved to: {save_path}")
    plt.show()

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def _extract_q1_q3_from_sanity(sanity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      - a sanity_df like you currently build:
        index: Metric, column: 'Typical Range (25th-75th percentile)' with 'Q1 to Q3' string
      - OR a richer frame with numeric 'q1' and 'q3' columns.

    Returns a DataFrame with index=metric and numeric columns ['q1','q3'].
    """
    sd = sanity_df.copy()
    if {"q1", "q3"}.issubset(sd.columns):
        out = sd[["q1", "q3"]].copy()
        out.index = out.index.astype(str)
        return out

    # Parse "a.b to c.d"
    pat = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)\s*$")
    rows = []
    for metric, row in sd.iterrows():
        s = str(row.get("Typical Range (25th-75th percentile)", "")).strip()
        m = pat.match(s)
        if m:
            q1, q3 = float(m.group(1)), float(m.group(2))
        else:
            q1 = q3 = np.nan
        rows.append((str(metric), q1, q3))
    out = pd.DataFrame(rows, columns=["metric", "q1", "q3"]).set_index("metric")
    return out

def plot_color_grid_with_sanity(
    combined_table: pd.DataFrame,
    sanity_df: pd.DataFrame,
    *,
    title: str = "Pre vs. Post Simulation (Sanity-flagged)",
    metrics: list[str] = None,
    soft_color: str = "rgba(255, 99, 71, 0.18)",   # soft tomato red
    base_alt_color: str = "rgba(0,0,0,0)",         # transparent
) -> go.Figure:
    """
    Render the same table-like grid (StaffGroup — Scenario rows)
    and softly highlight any metric cell that falls outside its sanity IQR [q1, q3].

    Parameters
    ----------
    combined_table : DataFrame
        Output of your pre_post_tree_pipeline 'combined' (already stacked Pre/Post).
        Must include columns: ['StaffGroup','Scenario', metric...]
    sanity_df : DataFrame
        Your sanity table. Either has numeric ['q1','q3'] columns indexed by metric,
        or a single string column 'Typical Range (25th-75th percentile)' like '12.3 to 18.9'.
    metrics : list[str]
        Which metric columns to show/flag. Defaults to common ones found in combined_table.
    """

    df = combined_table.copy()
    # Preserve the original label style
    df["Row"] = df["StaffGroup"].astype(str) + " — " + df["Scenario"].astype(str)

    # Decide which metrics to show
    default_metrics = ["Capacity", "WSI_0_100", "efficiency", "days_to_close", "backlog", "numCases"]
    if metrics is None:
        metrics = [m for m in default_metrics if m in df.columns]
        # if Capacity missing (e.g., depending on your grid_post), ignore silently

    # Build the display frame (Row + metrics)
    display_cols = ["Row"] + metrics
    show_df = df[display_cols].copy()

    # Extract numeric q1/q3 for each metric
    qtbl = _extract_q1_q3_from_sanity(sanity_df)

    # Prepare cell fill colors: Plotly go.Table wants column-wise color lists
    fillcolors = []
    # First column (Row label) uses no highlighting; use transparent for all rows
    fillcolors.append([base_alt_color] * len(show_df))

    # For each metric column, build a color vector marking out-of-IQR cells
    for metric in metrics:
        col_vals = pd.to_numeric(show_df[metric], errors="coerce")
        q1 = qtbl.loc[metric, "q1"] if metric in qtbl.index else np.nan
        q3 = qtbl.loc[metric, "q3"] if metric in qtbl.index else np.nan

        if np.isnan(q1) or np.isnan(q3):
            # If no sanity range, keep base color
            fillcolors.append([base_alt_color] * len(col_vals))
        else:
            mask = (col_vals < q1) | (col_vals > q3)
            fillcolors.append([soft_color if bool(m) else base_alt_color for m in mask])

    # Build the table
    header_values = ["StaffGroup — Scenario"] + metrics
    cell_values = [show_df[c].tolist() for c in ["Row"] + metrics]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color="rgba(240,240,240,1)",
                    align="left",
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=fillcolors,
                    align="left",
                    format=[None] + [".2f"] * len(metrics),
                ),
            )
        ]
    )
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ---- NEW: generic fitter that can include numCases
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

def fit_tree_models_from_features(
    train_df: pd.DataFrame,
    targets: Tuple[str, ...] = ("WSI_0_100", "efficiency", "days_to_close", "backlog"),
    num_features: List[str] = ("Capacity", "numCases"),
    cat_features: List[str] = ("StaffGroup",),
    random_state: int = 42,
):
    """
    Train one sklearn pipeline per target using the specified numeric + categorical features.
    Uses OneHotEncoder(handle_unknown="ignore") for categorical features inside the pipeline.

    Returns
    -------
    models: Dict[str, sklearn.Pipeline]
        One fitted pipeline per target.
    mse_report: pd.DataFrame
        In-sample MSE per target (for transparency).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error

    # Prefer XGBRegressor if available, else fall back to RandomForestRegressor
    try:
        from xgboost import XGBRegressor
        Regr = lambda: XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        Regr = lambda: RandomForestRegressor(
            n_estimators=500, random_state=random_state, n_jobs=-1
        )

    # Build feature matrix X and ensure types
    X = train_df[list(num_features) + list(cat_features)].copy()
    for c in cat_features:
        X[c] = X[c].astype("string").fillna("__MISSING__")

    models: Dict[str, Pipeline] = {}
    rows = []

    # Preprocessor
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(num_features)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_features)),
        ],
        remainder="drop",
        n_jobs=None,
    )

    for t in targets:
        if t not in train_df.columns:
            continue
        y = pd.to_numeric(train_df[t], errors="coerce")
        mask = y.notna().values
        Xt = X.loc[mask]
        yt = y.loc[mask]

        pipe = Pipeline(steps=[("prep", pre), ("regr", Regr())])
        pipe.fit(Xt, yt)

        yhat = pipe.predict(Xt)
        mse  = float(mean_squared_error(yt, yhat))
        models[t] = pipe
        rows.append({"target": t, "mse": mse, "n": int(mask.sum())})

    mse_report = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    return models, mse_report


# ---- NEW: predictor that consumes post capacity & numCases and aggregates to team
def predict_post_from_cap_and_cases(
    post_df: pd.DataFrame,
    models: Dict[str, "Pipeline"],
    targets: Tuple[str, ...] = ("WSI_0_100", "efficiency", "days_to_close", "backlog"),
    team_aggregate: str = "median",
    *,
    debug_sample: bool = True,
):
    """
    Predict POST outcomes from post_df using models trained with features
    ["Capacity", "numCases", "StaffGroup"]. Aggregates predictions to team level.

    Parameters
    ----------
    post_df : DataFrame
        Must have columns: ["StaffGroup","Capacity","numCases"] at alias-level or team-level.
        If alias-level, predictions are made per alias and then aggregated.
        If team-level (one row per StaffGroup), predictions are made once per team.
    """
    agg = team_aggregate
    need_cols = ["Capacity", "numCases", "StaffGroup"]
    for c in need_cols:
        if c not in post_df.columns:
            raise ValueError(f"predict_post_from_cap_and_cases: missing required column '{c}' in post_df")

    X = post_df[need_cols].copy()
    X["StaffGroup"] = X["StaffGroup"].astype("string").fillna("__MISSING__")

    if debug_sample:
        print("\n[DEBUG] Predict() input sample (Capacity + numCases + StaffGroup):")
        print(X.head(12).to_string(index=False))
        print("Rows to predict:", len(X))

    preds = {}
    for t, mdl in models.items():
        if t not in targets:
            continue
        preds[t] = mdl.predict(X)

    pred_df = pd.DataFrame({"StaffGroup": X["StaffGroup"], **preds})

    # If alias-level, aggregate to team
    grid_post = (
        pred_df.groupby("StaffGroup", as_index=False)
        .agg({k: agg for k in preds.keys()})
    )

    # Carry Capacity & numCases at team level for display
    add_cols = {}
    for c in ["Capacity", "numCases"]:
        if c in post_df.columns:
            add_cols[c] = agg
    if add_cols:
        team_extras = post_df.groupby("StaffGroup", as_index=False).agg(add_cols)
        grid_post = grid_post.merge(team_extras, on="StaffGroup", how="left")

    return grid_post


# ---- NEW: a parallel pipeline that uses Capacity + numCases + StaffGroup
from typing import Optional, List, Dict, Tuple

def pre_post_tree_pipeline_cap_cases(
    df_in: pd.DataFrame, scores_baseline: pd.DataFrame, month: str, moves: List[Dict], *,
    compute_wsi_fn, wsi_kwargs: Optional[dict] = None,
    targets: Tuple[str, ...] = ("WSI_0_100", "efficiency", "days_to_close", "backlog"),
    only_impacted: bool = True,
    team_level_post: bool = False,       # keep both modes
    show_predict_samples: bool = True,
):
    """
    Same outputs as your original function, but the outcome models are trained with
    features: Capacity + numCases + StaffGroup, and predictions consume numCases_post.
    """
    # 0) Prepare training data (same helper you already use)
    train_df = prepare_training_df(df_in, scores_baseline,
                                   compute_wsi_fn=compute_wsi_fn, wsi_kwargs=wsi_kwargs)

    # Safety: ensure required columns exist
    need_train = {"StaffGroup", "_date_", "Capacity", "numCases", *targets}
    missing = [c for c in need_train if c not in train_df.columns]
    if missing:
        raise ValueError(f"pre_post_tree_pipeline_cap_cases: training data missing columns: {missing}")

    # Fit models using Capacity + numCases + StaffGroup
    models, mse_report = fit_tree_models_from_features(
        train_df=train_df,
        targets=targets,
        num_features=["Capacity", "numCases"],
        cat_features=["StaffGroup"],
    )

    # PRE month & grid (actuals, team median)
    m0 = _mstart(month)
    pre_month = train_df[train_df["_date_"] == m0].copy()
    grid_pre = (pre_month.groupby("StaffGroup")[["Capacity", "numCases", *targets]]
                .median().reset_index())

    # Sanity IQR based on alias-level PRE
    sanity_ranges = {}
    for metric in ["Capacity", "numCases", *targets]:
        if metric in pre_month.columns:
            q1 = pre_month[metric].quantile(0.25)
            q3 = pre_month[metric].quantile(0.75)
            sanity_ranges[metric] = f"{q1:.2f} to {q3:.2f}"
    sanity_df = (pd.DataFrame.from_dict(sanity_ranges, orient='index',
                                        columns=['Typical Range (25th-75th percentile)'])
                 .rename_axis("Metric"))

    # POST capacity simulation
    post_cap = simulate_post_capacity(df_in, scores_baseline, month, moves).copy()

    # Normalize month and column names
    if "_date_" in post_cap.columns:
        post_cap["_date_"] = pd.to_datetime(post_cap["_date_"]).dt.to_period("M").dt.to_timestamp()
        post_cap = post_cap[post_cap["_date_"] == m0].copy()
    if "Capacity" not in post_cap.columns and "Capacity_Score_0_100" in post_cap.columns:
        post_cap = post_cap.rename(columns={"Capacity_Score_0_100": "Capacity"})

    # Ensure numCases_post exists; if not present, try a reasonable fallback
    if "numCases" not in post_cap.columns:
        # Fallback: use PRE team median numCases
        team_cases_pre = pre_month.groupby("StaffGroup", as_index=False)["numCases"].median()
        post_cap = post_cap.merge(team_cases_pre, on="StaffGroup", how="left")
        print("[WARN] simulate_post_capacity did not provide 'numCases'; using PRE team medians as a fallback.")

    # POST prediction:
    # - alias-level (default): predict per alias then aggregate
    # - team-level: aggregate (Capacity,numCases) to team, predict once per team
    if not team_level_post:
        # Alias-level: print sample X then predict+aggregate
        if show_predict_samples:
            X_alias = post_cap[["Capacity", "numCases", "StaffGroup"]].copy()
            X_alias["StaffGroup"] = X_alias["StaffGroup"].astype("string").fillna("__MISSING__")
            print("\n[DEBUG] Alias-level POST predict() input (first 12 rows):")
            print(X_alias.head(12).to_string(index=False))
            print("Rows to predict (alias-level):", len(X_alias))

        grid_post = predict_post_from_cap_and_cases(
            post_df=post_cap,
            models=models,
            targets=targets,
            team_aggregate="median",
            debug_sample=False,
        )

    else:
        # Team-level: aggregate inputs first, then predict once per team
        team_post = (post_cap.groupby("StaffGroup", as_index=False)[["Capacity","numCases"]]
                     .median())
        X_team = team_post.copy()
        X_team["StaffGroup"] = X_team["StaffGroup"].astype("string").fillna("__MISSING__")

        if show_predict_samples:
            print("\n[DEBUG] Team-level POST predict() input (first 12 rows):")
            print(X_team.head(12).to_string(index=False))
            print("Rows to predict (team-level):", len(X_team))

        preds = {"StaffGroup": X_team["StaffGroup"].to_numpy()}
        for t in targets:
            preds[t] = models[t].predict(X_team[["Capacity","numCases","StaffGroup"]])
        grid_post = pd.DataFrame(preds).merge(team_post, on="StaffGroup", how="left")

    # Filter to impacted teams if requested (safe)
    if only_impacted:
        impacted = get_impacted_sgs(moves)
        pre_mask  = grid_pre["StaffGroup"].isin(impacted)
        post_mask = grid_post["StaffGroup"].isin(impacted)
        if pre_mask.any():
            grid_pre  = grid_pre[pre_mask]
        else:
            print("[WARN] No PRE rows match impacted teams; leaving PRE unfiltered.")
        if post_mask.any():
            grid_post = grid_post[post_mask]
        else:
            print("[WARN] No POST rows match impacted teams; leaving POST unfiltered.")

    # Combine to the same table-like shape you had before
    gp = grid_pre.copy(); gp["Scenario"] = "Pre"
    gq = grid_post.copy(); gq["Scenario"] = "Post (pred)"
    combined = (pd.concat([gp, gq], ignore_index=True)
                  .sort_values(["StaffGroup", "Scenario"])
                  .reset_index(drop=True))

    # Keep your original color table (or the sanity-highlighted one you just added)
    grid_for_plot = combined.copy()
    grid_for_plot["StaffGroup"] = grid_for_plot["StaffGroup"] + " — " + grid_for_plot["Scenario"].astype(str)
    fig = plot_color_grid(grid_for_plot, title="Pre vs. Post Simulation (Cap + numCases)")

    return grid_pre, grid_post, combined, fig, mse_report, sanity_df