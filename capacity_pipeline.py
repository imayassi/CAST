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
# FEATURE_DIRECTION: Dict[str, bool] = {
#     "numCases": True, "numOpenCases": True, "backlog": True, "TimeSpent": True,
#     "currentSevA": True, "Is24X7OptedIn": True, "IsClosedOnWeekend": True,
#     "tenure": False, "som": False, "avgDaysToClose": True, "efficiency": False,
#     "CustomerSatisfaction": False, "ActiveCases": True, "TenureInMonths": False,
#     "EfficiencyRating": False, "AvgTimeToResolve": True, "TasksCompleted": True,
#     "BacklogSize": True
# }


# =========================
# Utils
# =========================
def _mstart(s):
    """
    Normalize datetime-like Series/array/scalar to month-start Timestamp(s).
    Works for pandas Series/Index/list-likes and scalars.
    """
    x = pd.to_datetime(s, errors="coerce")
    # Series / Index / list-like → has .dt
    if hasattr(x, "dt"):
        return x.dt.to_period("M").dt.to_timestamp()
    # scalar datetime
    return pd.to_datetime(x).to_period("M").to_timestamp()

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
#     d = df_in.copy()
#     d["_date_"] = _mstart(d["_date_"])
#     m0 = _mstart(month)
    
#     # RENAME the capacity score column for the merge
#     sc = scores_baseline.rename(columns={"Capacity_Score_0_100": "Capacity"})
    
#     # --- FIX ---
#     # Drop "StaffGroup" from the right-side DataFrame (sc) before merging
#     # to prevent pandas from creating 'StaffGroup_x' and 'StaffGroup_y'
#     if "StaffGroup" in sc.columns:
#         sc = sc.drop(columns=["StaffGroup"])
        
#     pre = d[d["_date_"] == m0].merge(sc, on=["_date_", "alias"], how="left")
    
#     knn_by_sg = {}
#     # The groupby will now work because 'pre' has a single 'StaffGroup' column
#     for sg, g in pre.groupby("StaffGroup"):
#         valid = g["Capacity"].notna() & g["numCases"].notna()
#         if valid.sum() >= 5:
#             knn_by_sg[sg] = KNeighborsRegressor(n_neighbors=min(5, valid.sum())).fit(g.loc[valid, ["numCases"]], g.loc[valid, "Capacity"])
            
#     cur = pre.copy()
#     rng = np.random.default_rng(random_state)
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
#         if sg_name in knn_by_sg:
#             # Ensure numCases is 2D for prediction
#             g["Capacity"] = knn_by_sg[sg_name].predict(g[["numCases"]])
#         else:
#             # Fallback to the median of the original group
#             median_cap = pre[pre["StaffGroup"] == sg_name]["Capacity"].median()
#             g["Capacity"] = median_cap if pd.notna(median_cap) else pre["Capacity"].median()
#         return g

#     # The rest of the function remains the same
#     cur = cur.groupby("StaffGroup").apply(predict_group_capacity).reset_index(drop=True)
#     return cur.groupby("StaffGroup")["Capacity"].median().reset_index()



# =========================
# simulate_post_capacity()
# =========================
import numpy as np
import pandas as pd

def simulate_post_capacity(
    df_in: pd.DataFrame,
    scores_baseline: pd.DataFrame,
    month: str | pd.Timestamp,
    moves: list[dict],
    *,
    # column names
    date_col: str = "_date_",
    alias_col: str = "alias",
    sg_col: str = "StaffGroup",
    cap_col_baseline: str = "Capacity_Score_0_100",
    # optional deviation features for KNN
    features_dev_cols: list[str] | None = None,
    use_knn: bool = False,
    knn_k: int = 5,
    # workload rebalancing
    rebalance_workload: bool = True,
    workload_cols: list[str] = ["numCases"],
    rebalance_method: str = "proportional",  # or "even"
    # NEW: recompute post capacity sequence options
    recompute_capacity_post: bool = False,
    capacity_method: str = "scorer",         # "scorer" | "knn" | "numcases_model"
    prep=None,                                # fitted DataPrep (for "scorer"/"knn")
    scorer=None,                              # fitted CapacityScorer (for "scorer")
    capacity_features: list[str] | None = None,
    numcases_model=None,                      # sklearn Pipeline from fit_capacity_from_numcases_model
    numcases_col_for_model: str = "numCases",
    # misc
    random_state: int | None = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Simulate a post-move alias-level frame for a single month:
      1) Filter to month m0
      2) Merge baseline Capacity → 'Capacity'
      3) Apply move plan (update StaffGroup for moved aliases)
      4) Rebalance workload within donor/host teams (preserve PRE team totals)
      5) Optionally recompute alias-level Capacity (scorer/knn/numcases_model)
      6) Return alias-level post table
    """
    rng = np.random.default_rng(random_state)
    m0 = pd.to_datetime(month, errors="coerce").to_period("M").to_timestamp()

    base = df_in.copy()
    if date_col not in base.columns:
        raise ValueError(f"simulate_post_capacity: '{date_col}' not found in df_in.")
    base[date_col] = pd.to_datetime(base[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    base = base.loc[base[date_col] == m0].copy()
    if base.empty:
        raise ValueError("No rows in df_in for the requested month.")

    missing_min = {alias_col, sg_col} - set(base.columns)
    if missing_min:
        raise ValueError(f"simulate_post_capacity: missing columns in df_in month-slice: {missing_min}")

    # Merge baseline capacity
    if not {date_col, alias_col}.issubset(scores_baseline.columns):
        raise ValueError("simulate_post_capacity: scores_baseline must contain date_col and alias_col.")
    sb = scores_baseline[[date_col, alias_col, cap_col_baseline]].copy()
    sb[date_col] = pd.to_datetime(sb[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    sb = sb.loc[sb[date_col] == m0].copy()

    post = base.merge(sb, on=[date_col, alias_col], how="left")
    if "Capacity" not in post.columns and cap_col_baseline in post.columns:
        post = post.rename(columns={cap_col_baseline: "Capacity"})
    if "Capacity" not in post.columns:
        post["Capacity"] = np.nan

    if verbose:
        print(f"Simulating moves for month={m0.date()} ...")

    # Apply moves
    team_aliases = post.groupby(sg_col)[alias_col].apply(list).to_dict()
    affected_teams = set()
    for mv in moves or []:
        donor = mv.get("from"); host = mv.get("to"); k = int(mv.get("n", 0) or 0)
        if not donor or not host or k <= 0:
            continue
        donor_aliases = list(team_aliases.get(donor, []))
        if not donor_aliases:
            if verbose: print(f"  • [SKIP] Donor '{donor}' empty.")
            continue
        moved = donor_aliases[:] if len(donor_aliases) <= k else list(rng.choice(donor_aliases, size=k, replace=False))
        post.loc[post[alias_col].isin(moved), sg_col] = host
        team_aliases[donor] = [a for a in donor_aliases if a not in moved]
        team_aliases.setdefault(host, list(team_aliases.get(host, [])) + moved)
        affected_teams.update([donor, host])
        if verbose: print(f"  • Move {len(moved)} aliases: {donor} → {host}")

    # Rebalance workload (preserve PRE totals)
    if rebalance_workload and workload_cols:
        pre_totals = base.groupby(sg_col, as_index=False)[workload_cols].sum()

        def _distribute(total, size, weights=None):
            if size <= 0: return np.array([])
            if rebalance_method == "proportional" and weights is not None and weights.sum() > 0:
                w = weights / weights.sum()
                return total * w
            return np.full(size, total / size if size > 0 else 0.0)

        post_aliases = post.groupby(sg_col)[alias_col].apply(list).to_dict()
        teams_to_rebalance = set(post_aliases.keys()) if not affected_teams else affected_teams

        for team in teams_to_rebalance:
            aliases = post_aliases.get(team, [])
            if not aliases: continue

            row = pre_totals.loc[pre_totals[sg_col] == team]
            team_total = float(row[workload_cols[0]].values[0]) if not row.empty else float(post.loc[post[sg_col]==team, workload_cols[0]].sum())

            if rebalance_method == "proportional" and workload_cols[0] in base.columns:
                pre_alias = base.loc[base[sg_col]==team, [alias_col, workload_cols[0]]].groupby(alias_col, as_index=False)[workload_cols[0]].sum()
                pre_alias = pre_alias.set_index(alias_col).reindex(aliases).fillna(0.0)
                weights = pre_alias[workload_cols[0]].to_numpy()
            else:
                cur_alias = post.loc[post[sg_col]==team, [alias_col, workload_cols[0]]].groupby(alias_col, as_index=False)[workload_cols[0]].sum()
                cur_alias = cur_alias.set_index(alias_col).reindex(aliases).fillna(0.0)
                weights = cur_alias[workload_cols[0]].to_numpy()

            distr = _distribute(team_total, len(aliases), weights)
            if workload_cols[0] not in post.columns: post[workload_cols[0]] = np.nan
            post.loc[post[alias_col].isin(aliases), workload_cols[0]] = distr

        if verbose:
            post_totals = post.groupby(sg_col, as_index=False)[workload_cols].sum()
            chk = pre_totals.merge(post_totals, on=sg_col, suffixes=("_pre","_post"), how="outer").fillna(0.0)
            deltas = []
            for col in workload_cols:
                diff = (chk[f"{col}_pre"] - chk[f"{col}_post"]).abs().sum()
                deltas.append(f"{col}: Δ={diff:.3f}")
            print("  • Workload rebalanced (team totals preserved):", ", ".join(deltas))

    # ======= NEW: recompute post Capacity after redistribution (optional) =======
    if recompute_capacity_post:
        # Build a post-world raw frame for rescoring
        base_m = df_in.copy()
        base_m[date_col] = pd.to_datetime(base_m[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
        base_m = base_m.loc[base_m[date_col] == m0].copy()

        id_cols = [alias_col, date_col, sg_col]
        if capacity_features is None:
            keep_feats = base_m.select_dtypes(include=["number","bool"]).columns.tolist()
        else:
            keep_feats = [c for c in capacity_features if c in base_m.columns]
        post_raw = base_m[id_cols + keep_feats].copy()

        # override StaffGroup + numCases from simulated post
        override_cols = [c for c in [sg_col, "numCases"] if c in post.columns]
        post_raw = (post_raw
                    .drop(columns=[c for c in override_cols if c in post_raw.columns], errors="ignore")
                    .merge(post[[alias_col, date_col] + override_cols], on=[alias_col, date_col], how="left"))

        if capacity_method == "scorer":
            if prep is None or scorer is None:
                raise ValueError("recompute_capacity_post=True + capacity_method='scorer' requires fitted `prep` and `scorer`.")
            post_prepped = prep.transform(post_raw)
            post_scored  = scorer.transform(post_prepped, group_col=sg_col, id_cols=(alias_col, date_col))
            post_cap_new = post_scored[[alias_col, date_col, "Capacity_Score_0_100"]] \
                               .rename(columns={"Capacity_Score_0_100":"Capacity"})

        elif capacity_method == "knn":
            if prep is None:
                raise ValueError("capacity_method='knn' requires fitted `prep` to compute deviations.")
            from sklearn.neighbors import KNeighborsRegressor
            pre_prepped = prep.transform(base)
            z_cols = [c for c in pre_prepped.columns if c.startswith("Z_") or c.endswith("_Dev")]
            y_pre = scores_baseline[[date_col, alias_col, cap_col_baseline]].copy()
            y_pre[date_col] = pd.to_datetime(y_pre[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
            y_pre = y_pre.loc[y_pre[date_col] == m0]
            train = pre_prepped.merge(y_pre.rename(columns={cap_col_baseline:"Capacity_Score_0_100"}),
                                      on=[date_col, alias_col], how="left")
            post_prepped = prep.transform(post_raw)

            preds = {}
            for team, need_blk in post_prepped.groupby(sg_col):
                fit_blk = train[train[sg_col] == team]
                if fit_blk.empty or need_blk.empty: continue
                X_fit = fit_blk[z_cols].fillna(0.0).to_numpy()
                y_fit = pd.to_numeric(fit_blk["Capacity_Score_0_100"], errors="coerce").fillna(0.0).to_numpy()
                X_need = need_blk[z_cols].fillna(0.0).to_numpy()
                k_use = max(2, min(knn_k, len(fit_blk)))
                knn = KNeighborsRegressor(n_neighbors=k_use, weights="distance")
                knn.fit(X_fit, y_fit)
                y_pred = knn.predict(X_need)
                for a, d, yhat in zip(need_blk[alias_col].tolist(), need_blk[date_col].tolist(), y_pred.tolist()):
                    preds[(a, d)] = float(yhat)
            post_cap_new = post_raw[[alias_col, date_col]].copy()
            post_cap_new["Capacity"] = [preds.get((a, d), np.nan) for a, d in zip(post_cap_new[alias_col], post_cap_new[date_col])]

        elif capacity_method == "numcases_model":
            if numcases_model is None:
                raise ValueError("capacity_method='numcases_model' requires a trained `numcases_model`.")
            # predict Capacity from (numCases, StaffGroup) AFTER redistribution
            Xp = post_raw[[numcases_col_for_model, sg_col]].copy()
            Xp[sg_col] = Xp[sg_col].astype("string").fillna("__MISSING__")
            Xp[numcases_col_for_model] = pd.to_numeric(Xp[numcases_col_for_model], errors="coerce").fillna(0.0)
            yhat = numcases_model.predict(Xp)
            post_cap_new = post_raw[[alias_col, date_col]].copy()
            post_cap_new["Capacity"] = yhat

        else:
            raise ValueError("capacity_method must be 'scorer' | 'knn' | 'numcases_model'.")

        # merge back and guard
        post = (post.drop(columns=["Capacity"], errors="ignore")
                    .merge(post_cap_new, on=[alias_col, date_col], how="left"))
        post["Capacity"] = post["Capacity"].where(
            post["Capacity"].notna(),
            post.groupby(sg_col)["Capacity"].transform("median")
        )
        if post["Capacity"].isna().any():
            gmed = post["Capacity"].median()
            if np.isfinite(gmed):
                post["Capacity"] = post["Capacity"].fillna(gmed)
        post["Capacity"] = post["Capacity"].fillna(50.0)

    # Final ordering
    cols_front = [date_col, alias_col, sg_col, "Capacity"]
    cols_work  = [c for c in workload_cols if c in post.columns]
    other_cols = [c for c in post.columns if c not in set(cols_front + cols_work)]
    post = post[cols_front + cols_work + other_cols].copy()
    post = post.sort_values([alias_col, date_col]).drop_duplicates(subset=[alias_col, date_col], keep="last")
    return post

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


from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

# --- small internal helper to resolve/standardize numCases ---
def _resolve_and_standardize_numcases(
    train_df: pd.DataFrame,
    df_in: pd.DataFrame,
    post_cap: pd.DataFrame,
    m0: pd.Timestamp,
    verbose: bool = True
):
    """
    Ensure 'numCases' exists in train_df (for fitting), pre_month, and post_cap.
    1) Try to find an existing workload column from common candidates and map to 'numCases'.
    2) If still missing, pull alias-month values from df_in for m0.
    3) If still missing, fall back to PRE team median for post_cap; zeros as last resort.
    Returns: (train_df, post_cap, used_col_name_or_None)
    """
    # 1) Try to detect workload column in train_df first, else df_in
    candidates = [
        "numCases", "TasksCompleted", "tasksCompleted", "tasks_completed",
        "Tasks_Completed", "TotalTasks", "TotalEvent"  # add any you use
    ]
    used = None

    # a) standardize in train_df (for fitting)
    if "numCases" not in train_df.columns:
        for c in candidates:
            if c in train_df.columns:
                train_df["numCases"] = pd.to_numeric(train_df[c], errors="coerce")
                used = c
                break

    # b) if train_df still lacks numCases, try df_in by joining on alias/date for m0 to set it (for post_cap we’ll also use this)
    if "numCases" not in train_df.columns:
        join_cols = ["_date_", "alias"]
        if all(cc in train_df.columns for cc in join_cols) and all(cc in df_in.columns for cc in join_cols):
            df_in_m0 = df_in.copy()
            df_in_m0["_date_"] = pd.to_datetime(df_in_m0["_date_"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            df_in_m0 = df_in_m0.loc[df_in_m0["_date_"] == m0].copy()
            # look for a candidate in df_in
            for c in candidates:
                if c in df_in_m0.columns:
                    tmp = df_in_m0[join_cols + [c]].drop_duplicates()
                    train_df = train_df.merge(tmp.rename(columns={c: "numCases"}), on=join_cols, how="left")
                    used = c
                    break

    # c) ensure numeric, fill na (for fitting we can temporarily fill 0; predictions will use better fallbacks)
    if "numCases" not in train_df.columns:
        train_df["numCases"] = 0.0
        if verbose:
            print("⚠️  Could not find a workload column for training; defaulting train_df['numCases']=0.")
    else:
        train_df["numCases"] = pd.to_numeric(train_df["numCases"], errors="coerce").fillna(0.0)

    # 2) post_cap: ensure numCases exists
    if "numCases" not in post_cap.columns:
        # try alias join for m0 from df_in with the same detected column (or any candidate)
        if "alias" in post_cap.columns:
            df_in_m0 = df_in.copy()
            df_in_m0["_date_"] = pd.to_datetime(df_in_m0["_date_"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            df_in_m0 = df_in_m0.loc[df_in_m0["_date_"] == m0].copy()
            joined = False
            # prefer the column we used above if any
            cand_order = [used] + candidates if used else candidates
            for c in cand_order:
                if c and c in df_in_m0.columns:
                    tmp = df_in_m0[["alias", c]].drop_duplicates()
                    post_cap = post_cap.merge(tmp.rename(columns={c: "numCases"}), on="alias", how="left")
                    joined = True
                    break
            if verbose and joined and used:
                print(f"ℹ️  POST numCases recovered by alias-join from df_in using '{used}'.")
        # fallback: fill later from PRE team medians
    # numeric clean-up
    if "numCases" in post_cap.columns:
        post_cap["numCases"] = pd.to_numeric(post_cap["numCases"], errors="coerce")

    return train_df, post_cap, used

def _ensure_numcases_post(
    pre_month: pd.DataFrame,
    post_cap: pd.DataFrame,
    df_in: pd.DataFrame,
    m0: pd.Timestamp,
    feature_scope: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ensure post_cap has a numeric 'numCases' column.
    Strategy:
      1) If missing, create it (NaN).
      2) Try alias join from df_in at month m0 using feature_scope candidates.
      3) If still NaN, fall back to PRE team median (compute from pre_month or df_in).
      4) Final fill 0.0
    """
    # Build candidate list (prefer scope, then defaults)
    scope = list(feature_scope or [])
    scope_lower = {c.lower(): c for c in scope}
    likely = ["numcases","taskscompleted","tasks_completed","activecases","backlogsize","totaltasks","totalevent"]
    scope_cands = [scope_lower[k] for k in likely if k in scope_lower]
    defaults = ["numCases", "TasksCompleted", "ActiveCases", "BacklogSize", "TotalTasks", "TotalEvent"]
    candidates = scope_cands + [c for c in defaults if c not in scope_cands]

    # 0) Make sure the column exists
    if "numCases" not in post_cap.columns:
        post_cap["numCases"] = np.nan

    # 1) Try alias join from df_in@m0
    if "alias" in post_cap.columns:
        dfm = df_in.copy()
        dfm["_date_"] = pd.to_datetime(dfm["_date_"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        dfm = dfm.loc[dfm["_date_"] == m0]
        for c in candidates:
            if c in dfm.columns:
                tmp = dfm[["alias", c]].drop_duplicates()
                before_na = post_cap["numCases"].isna().sum()
                post_cap = post_cap.merge(tmp.rename(columns={c: "numCases_join"}), on="alias", how="left")
                post_cap["numCases"] = post_cap["numCases"].fillna(post_cap["numCases_join"])
                post_cap.drop(columns=["numCases_join"], inplace=True)
                after_na = post_cap["numCases"].isna().sum()
                if verbose and before_na != after_na:
                    print(f"ℹ️  Recovered {before_na - after_na} numCases via alias-join from '{c}'.")
                if post_cap["numCases"].notna().any():
                    break  # keep the first that helped

    # 2) If still all NaN, compute PRE team medians to fill
    if post_cap["numCases"].isna().all():
        # Make sure we can get a PRE numCases series
        pre_num = None
        if "numCases" in pre_month.columns:
            pre_num = pre_month[["StaffGroup","numCases"]].copy()
        else:
            # try to derive numCases in pre_month from candidates present
            for c in candidates:
                if c in pre_month.columns:
                    pre_num = pre_month[["StaffGroup", c]].rename(columns={c: "numCases"})
                    break
            if pre_num is None:
                # derive from df_in@m0 per StaffGroup
                dfm = df_in.copy()
                dfm["_date_"] = pd.to_datetime(dfm["_date_"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                dfm = dfm.loc[dfm["_date_"] == m0]
                for c in candidates:
                    if c in dfm.columns and "StaffGroup" in dfm.columns:
                        pre_num = dfm[["StaffGroup", c]].rename(columns={c: "numCases"})
                        break
        if pre_num is not None:
            pre_team_med = pre_num.groupby("StaffGroup", as_index=False)["numCases"].median() \
                                  .rename(columns={"numCases": "numCases_team_median"})
            post_cap = post_cap.merge(pre_team_med, on="StaffGroup", how="left")
            post_cap["numCases"] = post_cap["numCases"].fillna(post_cap["numCases_team_median"])
            post_cap.drop(columns=["numCases_team_median"], inplace=True, errors="ignore")

    # 3) Final guard
    post_cap["numCases"] = pd.to_numeric(post_cap["numCases"], errors="coerce").fillna(0.0)
    return post_cap


# ============================================================
# pre_post_tree_pipeline_cap_cases (FULL — original behavior + new options)
# ============================================================
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

def pre_post_tree_pipeline_cap_cases(
    df_in: pd.DataFrame,
    scores_baseline: pd.DataFrame,
    month: str,
    moves: List[Dict],
    *,
    # ----- EXISTING (unchanged) -----
    compute_wsi_fn,
    wsi_kwargs: Optional[dict] = None,
    targets: Tuple[str, ...] = ("WSI_0_100", "efficiency", "days_to_close", "backlog"),
    only_impacted: bool = True,
    team_level_post: bool = False,
    show_predict_samples: bool = True,
    use_causal_sequence: Optional[object] = False,  # False | True | "learn"
    feature_scope: Optional[List[str]] = None,
    verbose: bool = True,

    # ----- NEW (all optional; defaults keep original behavior) -----
    # Recompute post Capacity AFTER workload redistribution (your requested sequence)
    recompute_capacity_post: bool = False,
    # How to compute post alias Capacity when recompute_capacity_post=True
    #   "scorer"          → use fitted DataPrep/CapacityScorer to recompute post Capacity_Score_0_100
    #   "knn"             → per-team KNN on pre deviations predicting post Capacity
    #   "numcases_model"  → predict Capacity from (numCases + StaffGroup) using a model trained on ALL months
    capacity_method: str = "scorer",
    prep=None,                        # fitted DataPrep (required for "scorer" or "knn")
    scorer=None,                      # fitted CapacityScorer (required for "scorer")
    capacity_features: Optional[List[str]] = None,  # raw features the scorer expects (optional; scorer can infer)
    numcases_model=None,              # optional pre-trained model (if None and method="numcases_model", will auto-train)
    numcases_col_for_model: str = "numCases",
):
    """
    End-to-end simulator using Capacity + numCases + StaffGroup as base inputs.

    PRE  = observed medians for the selected month.
    POST = predictions (direct, fixed-causal, or DAG-learned) from models trained on PRE,
           supplied with simulated post alias/team inputs (Capacity, numCases, StaffGroup).

    NEW OPTIONS:
      - recompute_capacity_post=True recomputes post alias Capacity AFTER workload redistribution following your sequence.
      - capacity_method="scorer" | "knn" | "numcases_model" controls how post Capacity is built.
        If "numcases_model" and no model provided, this function auto-trains one on ALL months.
    """

    # -------------------- Local helpers (kept inside for easy paste) --------------------

    def _mstart(s):
        """
        Normalize datetime-like Series/array/scalar to month-start Timestamp(s).
        Works for pandas Series/Index/list-likes and scalars.
        """
        x = pd.to_datetime(s, errors="coerce")
        # Series / Index / list-like → has .dt
        if hasattr(x, "dt"):
            return x.dt.to_period("M").dt.to_timestamp()
        # scalar datetime
        return pd.to_datetime(x).to_period("M").to_timestamp()

    def _ensure_models_direct(train_df: pd.DataFrame):
        """Fit direct per-target models: y ~ Capacity + numCases + StaffGroup"""
        if "fit_tree_models_from_features" not in globals():
            raise RuntimeError("fit_tree_models_from_features is not available in globals().")
        _targets = tuple([t for t in targets if t in train_df.columns])
        return fit_tree_models_from_features(
            train_df=train_df,
            targets=_targets,
            num_features=["Capacity", "numCases"],
            cat_features=["StaffGroup"],
        )

    def _ensure_models_dag_fixed(train_df: pd.DataFrame, seq: List[str]):
        """Fit DAG/fixed causal models (stage-wise)."""
        if "fit_tree_models_dag" not in globals():
            raise RuntimeError("fit_tree_models_dag is not available in globals().")
        return fit_tree_models_dag(
            train_df=train_df,
            seq=seq,
            base_features=("Capacity","numCases","StaffGroup"),
            feature_scope=feature_scope
        )

    def _ensure_models_dag_learn(train_df: pd.DataFrame):
        """Learn DAG order and fit models."""
        if "learn_sequence_from_data" not in globals() or "fit_tree_models_dag" not in globals():
            raise RuntimeError("DAG helpers not available (learn_sequence_from_data / fit_tree_models_dag).")
        seq, _G = learn_sequence_from_data(
            train_df=train_df,
            candidate_targets=tuple([t for t in ["backlog","days_to_close","efficiency","WSI_0_100"] if t in train_df.columns]),
            base_features=("Capacity","numCases","StaffGroup"),
            feature_scope=feature_scope,
            k=6,
            verbose=verbose
        )
        models, mse_report = fit_tree_models_dag(
            train_df=train_df,
            seq=seq,
            base_features=("Capacity","numCases","StaffGroup"),
            feature_scope=feature_scope
        )
        return models, mse_report, seq

    def _plot_sanity_table(combined: pd.DataFrame, sanity_df: pd.DataFrame, title: str):
        """Prefer sanity-flagged table if available; else fallback to color grid."""
        metrics = [m for m in ["Capacity","numCases","WSI_0_100","efficiency","days_to_close","backlog"] if m in combined.columns]
        if "plot_color_grid_with_sanity_range" in globals():
            return plot_color_grid_with_sanity_range(
                combined_table=combined,
                sanity_df=sanity_df,
                metrics=metrics,
                low_col="p10", high_col="p95",
                title=title,
            )
        # Fallback to old plotter if sanity plot not present
        if "plot_color_grid" in globals():
            grid_for_plot = combined.copy()
            grid_for_plot["StaffGroup"] = grid_for_plot["StaffGroup"] + " — " + grid_for_plot["Scenario"].astype(str)
            return plot_color_grid(grid_for_plot, title=title)
        raise RuntimeError("No plotting helper found (plot_color_grid_with_sanity_range or plot_color_grid).")

    def _fit_capacity_from_numcases_model(df_full: pd.DataFrame,
                                        scores: pd.DataFrame,
                                        wsi_kwargs: dict,
                                        numcases_col: str = "numCases",
                                        random_state: int = 42):
        """
        Auto-train Capacity ~ f(numCases, StaffGroup) on ALL months.
        Column names are inferred from wsi_kwargs (alias_col, date_col, sg_col)
        with safe canonical defaults if missing.
        """
        # infer column names from wsi_kwargs (safe defaults)
        alias_col = wsi_kwargs.get("alias_col", "alias")
        date_col  = wsi_kwargs.get("date_col",  "_date_")
        group_col = wsi_kwargs.get("sg_col",    "StaffGroup")

        # If the global trainer exists and takes a ColumnMap, we can still call it,
        # but we won't rely on a global `cols`. Build features directly instead.
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline

        try:
            from xgboost import XGBRegressor
            Regr = XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.06,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective="reg:squarederror", n_jobs=-1, random_state=random_state
            )
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            Regr = RandomForestRegressor(n_estimators=600, random_state=random_state, n_jobs=-1)

        # Build the training set across ALL months
        df = df_full[[alias_col, date_col, group_col, numcases_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()

        y = scores[[alias_col, date_col, "Capacity_Score_0_100"]].copy()
        y[date_col] = pd.to_datetime(y[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()

        data = (df.merge(y, on=[alias_col, date_col], how="inner")
                .dropna(subset=[numcases_col, "Capacity_Score_0_100"]))
        if data.empty:
            raise ValueError("No rows to train capacity-from-numCases model; check inputs.")

        X = data[[numcases_col, group_col]].copy()
        X[group_col] = X[group_col].astype("string").fillna("__MISSING__")
        X[numcases_col] = pd.to_numeric(X[numcases_col], errors="coerce").fillna(0.0)
        y = pd.to_numeric(data["Capacity_Score_0_100"], errors="coerce")

        pre = ColumnTransformer(
            transformers=[
                ("num", "passthrough", [numcases_col]),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [group_col]),
            ],
            remainder="drop",
        )
        pipe = Pipeline([("prep", pre), ("regr", Regr)])
        pipe.fit(X, y)
        return pipe

    # -------------------- 0) Build TRAIN data --------------------
    # Note: prepare_training_df must exist in your module; we keep the original call
    train_df = prepare_training_df(
        df_in=df_in,
        scores_baseline=scores_baseline,
        compute_wsi_fn=compute_wsi_fn,
        wsi_kwargs=wsi_kwargs,
    )
    if "_date_" in train_df.columns:
        train_df["_date_"] = _mstart(train_df["_date_"])
    if "Capacity" not in train_df.columns and "Capacity_Score_0_100" in train_df.columns:
        train_df = train_df.rename(columns={"Capacity_Score_0_100": "Capacity"})

    m0 = _mstart(month)

    # -------------------- 1) Fit outcome models (original behavior preserved) --------------------
    learned_seq = None
    if use_causal_sequence == "learn":
        try:
            models, mse_report, learned_seq = _ensure_models_dag_learn(train_df)
        except Exception as e:
            if verbose:
                print(f"⚠️ DAG learn failed ({e}); falling back to fixed causal order.")
            use_causal_sequence = True

    if use_causal_sequence is True and learned_seq is None:
        # fixed order if available
        seq = [t for t in ["backlog","days_to_close","efficiency","WSI_0_100"] if t in train_df.columns]
        try:
            models, mse_report = _ensure_models_dag_fixed(train_df, seq=seq)
            learned_seq = seq
        except Exception as e:
            if verbose:
                print(f"⚠️ Fixed DAG fit failed ({e}); falling back to direct models.")
            use_causal_sequence = False

    if use_causal_sequence is False and learned_seq is None:
        models, mse_report = _ensure_models_direct(train_df)

    if verbose and isinstance(models, dict):
        rows = []
        for t, obj in models.items():
            if isinstance(obj, dict) and "features" in obj:
                rows.append({"target": t, "features": ", ".join(obj["features"])})
        if rows:
            print("\n[INFO] Final feature sets per target:")
            print(pd.DataFrame(rows).sort_values("target").to_string(index=False))

    # -------------------- 2) PRE grid (observed medians) --------------------
    pre_month = train_df[train_df["_date_"] == m0].copy()
    pre_cols = ["Capacity", "numCases"] + [t for t in targets if t in pre_month.columns]
    grid_pre = pre_month.groupby("StaffGroup")[pre_cols].median().reset_index()

    # -------------------- 3) Sanity ranges (p10–p95) --------------------
    sanity_rows = []
    for metric in [m for m in ["Capacity","numCases",*targets] if m in pre_month.columns]:
        s = pd.to_numeric(pre_month[metric], errors="coerce").dropna()
        if s.empty: continue
        sanity_rows.append({"metric": metric,
                            "p10": np.percentile(s,10),
                            "median": np.percentile(s,50),
                            "p95": np.percentile(s,95)})
    sanity_df = pd.DataFrame(sanity_rows).set_index("metric").sort_index()

    # -------------------- 4) POST alias simulation (simulate_post_capacity) --------------------
    # Auto-train numcases model if requested
    if recompute_capacity_post and capacity_method == "numcases_model" and numcases_model is None:
        numcases_model = _fit_capacity_from_numcases_model(
            df_full=df_in,
            scores=scores_baseline,
            wsi_kwargs=wsi_kwargs,               # <-- pass wsi_kwargs so we can infer col names
            numcases_col=numcases_col_for_model  # <-- typically "numCases"
        )

    # simulate_post_capacity must exist in your module. We call it with new kwargs; defaults keep old behavior
    post_cap = simulate_post_capacity(
        df_in=df_in,
        scores_baseline=scores_baseline,
        month=month,
        moves=moves,
        # NEW sequence control:
        recompute_capacity_post=recompute_capacity_post,
        capacity_method=capacity_method,          # "scorer" | "knn" | "numcases_model"
        prep=prep, scorer=scorer,                 # used if capacity_method="scorer"
        capacity_features=capacity_features,      # optional
        numcases_model=numcases_model,            # used if capacity_method="numcases_model"
        numcases_col_for_model=numcases_col_for_model,
        # Keep original redistributor behavior
        rebalance_workload=True, workload_cols=["numCases"],
        rebalance_method="proportional",
        # Keep original KNN off by default; you can toggle in simulate_post_capacity
        use_knn=False,
        verbose=verbose
    )
    if not isinstance(post_cap, pd.DataFrame):
        raise TypeError("simulate_post_capacity must return a pandas DataFrame")

    # -------------------- 5) POST predictions (original behavior preserved) --------------------
    alias_missing = "alias" not in post_cap.columns
    effective_team_level = team_level_post or alias_missing

    if not effective_team_level:
        if show_predict_samples:
            cols_preview = [c for c in ["Capacity","numCases","StaffGroup"] if c in post_cap.columns]
            print("\n[DEBUG] Alias-level POST input (first 12 rows):")
            print(post_cap[cols_preview].head(12).to_string(index=False))
            print("Rows to predict (alias-level):", len(post_cap))

        if use_causal_sequence in (True, "learn") and "predict_post_from_dag" in globals():
            grid_post = predict_post_from_dag(
                post_df=post_cap,
                models=models,
                seq=learned_seq,
                team_aggregate="median",
                debug_sample=show_predict_samples
            )
        else:
            # direct path
            grid_post = predict_post_from_cap_and_cases(
                post_df=post_cap,
                models=models,
                targets=targets,
                team_aggregate="median",
                debug_sample=False
            )

    else:
        # Team-level path: aggregate inputs first, then predict once per team
        agg_cols = [c for c in ["Capacity","numCases"] if c in post_cap.columns]
        team_post = post_cap.groupby("StaffGroup", as_index=False)[agg_cols].median()
        X_team = team_post.copy()
        X_team["StaffGroup"] = X_team["StaffGroup"].astype("string").fillna("__MISSING__")

        if show_predict_samples:
            print("\n[DEBUG] Team-level POST input (first 12 rows):")
            print(X_team.head(12).to_string(index=False))
            print("Rows to predict (team-level):", len(X_team))

        if use_causal_sequence in (True, "learn") and "predict_post_from_dag" in globals():
            grid_post = predict_post_from_dag(
                post_df=X_team,
                models=models,
                seq=learned_seq,
                team_aggregate="median",
                debug_sample=show_predict_samples
            )
            grid_post = grid_post.merge(team_post, on="StaffGroup", how="left")
        else:
            preds = {"StaffGroup": X_team["StaffGroup"].to_numpy()}
            for t in targets:
                mdl = models[t]["pipe"] if isinstance(models.get(t), dict) else models[t]
                cols_use = [c for c in ["Capacity","numCases","StaffGroup"] if c in X_team.columns]
                preds[t] = mdl.predict(X_team[cols_use])
            grid_post = pd.DataFrame(preds).merge(team_post, on="StaffGroup", how="left")

    # ---- Ensure Capacity & numCases exist in POST grid (for combined view) ----
    try:
        cap_team = (post_cap.groupby("StaffGroup", as_index=False)[["Capacity"]]
                    .median().rename(columns={"Capacity":"Capacity_team_median_post"}))
        grid_post = grid_post.merge(cap_team, on="StaffGroup", how="left")
        if "Capacity" in grid_post.columns:
            grid_post["Capacity"] = grid_post["Capacity"].combine_first(grid_post["Capacity_team_median_post"])
        else:
            grid_post["Capacity"] = grid_post["Capacity_team_median_post"]
        grid_post.drop(columns=["Capacity_team_median_post"], inplace=True)
    except Exception:
        if "Capacity" not in grid_post.columns and "team_post" in locals() and "Capacity" in team_post.columns:
            grid_post = grid_post.merge(team_post[["StaffGroup","Capacity"]], on="StaffGroup", how="left")

    if "numCases" not in grid_post.columns:
        try:
            nc_team = (post_cap.groupby("StaffGroup", as_index=False)[["numCases"]]
                       .median().rename(columns={"numCases":"numCases_team_median_post"}))
            grid_post = grid_post.merge(nc_team, on="StaffGroup", how="left")
            grid_post["numCases"] = grid_post["numCases_team_median_post"]
            grid_post.drop(columns=["numCases_team_median_post"], inplace=True)
        except Exception:
            if "team_post" in locals() and "numCases" in team_post.columns:
                grid_post = grid_post.merge(team_post[["StaffGroup","numCases"]], on="StaffGroup", how="left")

    # -------------------- 6) Impacted filter (unchanged) --------------------
    if only_impacted:
        if "get_impacted_sgs" in globals():
            impacted = get_impacted_sgs(moves)
        else:
            impacted = list({mv.get("from") for mv in moves} | {mv.get("to") for mv in moves})
        if not grid_pre.empty:
            grid_pre = grid_pre[grid_pre["StaffGroup"].isin(impacted)]
        if not grid_post.empty:
            grid_post = grid_post[grid_post["StaffGroup"].isin(impacted)]

    # -------------------- 7) Combine + sanity-plot (unchanged) --------------------
    if verbose:
        print("\n[DEBUG] grid_pre columns:", list(grid_pre.columns), "shape:", grid_pre.shape)
        print("[DEBUG] grid_post columns:", list(grid_post.columns), "shape:", grid_post.shape)

    gp = grid_pre.copy(); gp["Scenario"] = "Pre"
    gq = grid_post.copy(); gq["Scenario"] = "Post (pred)"
    combined = (pd.concat([gp, gq], ignore_index=True)
                .sort_values(["StaffGroup","Scenario"])
                .reset_index(drop=True))

    title = "Pre vs Post (sanity-flagged cells: outside p10–p95)"
    fig = _plot_sanity_table(combined, sanity_df, title=title)

    return grid_pre, grid_post, combined, fig, mse_report, sanity_df

# ---- Minimal regressor factory (place ABOVE fit_tree_models_causal) ----
def _make_regressor(random_state: int = 42):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
        )
    except Exception:
        # Fallback if xgboost isn't available
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1
        )
def fit_tree_models_causal(
    train_df: pd.DataFrame,
    random_state: int = 42,
    add_complexity: bool = True
):
    """
    Train one pipeline per stage with the correct causal parents.
    Returns dict of fitted models keyed by target name and a compact MSE report.
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error
    import numpy as np

    # Base features always present in POST
    base_num = ["Capacity", "numCases"]
    base_cat = ["StaffGroup"]

    # Optional complexity drivers if present in your training frame
    complexity_pool = [c for c in [
        "LinearityScore", "numCritsit", "currentSev1", "currentSevA", "currentSevB",
        "initialSev1", "initialSevA", "initialSevB",
    ] if c in train_df.columns]
    extra = complexity_pool if add_complexity else []

    # Stage feature sets
    FEAT = {
        "backlog":          base_num + base_cat + extra,
        "days_to_close":    base_num + base_cat + ["backlog"] + extra,
        "efficiency":       base_num + base_cat + ["backlog", "days_to_close"] + extra,
        "WSI_0_100":        base_num + base_cat + ["backlog", "days_to_close", "efficiency"] + extra,
    }

    models = {}
    mse_rows = []

    for tgt in ["backlog", "days_to_close", "efficiency", "WSI_0_100"]:
        if tgt not in train_df.columns:
            continue
        # Build training X,y – use actual parents available in train_df
        use_feats = [f for f in FEAT[tgt] if f in train_df.columns]
        X = train_df[use_feats].copy()
        X["StaffGroup"] = X["StaffGroup"].astype("string").fillna("__MISSING__")
        y = pd.to_numeric(train_df[tgt], errors="coerce")
        mask = y.notna()
        X = X.loc[mask]; y = y.loc[mask]

        pre = ColumnTransformer(
            transformers=[
                ("num", "passthrough", [c for c in use_feats if c != "StaffGroup"]),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["StaffGroup"]),
            ],
            remainder="drop"
        )
        pipe = Pipeline([("prep", pre), ("regr", _make_regressor(random_state))])
        pipe.fit(X, y)

        # MSE on fitted slice
        yhat = pipe.predict(X)
        mse = float(mean_squared_error(y, yhat)) if len(y) else np.nan
        models[tgt] = {"pipe": pipe, "features": use_feats}
        mse_rows.append({"target": tgt, "mse": mse, "n": int(mask.sum())})

    mse_report = pd.DataFrame(mse_rows).sort_values("target").reset_index(drop=True)
    return models, mse_report

def predict_post_from_cap_cases_causal(
    post_df: pd.DataFrame,
    models: dict,
    team_aggregate: str = "median",
    debug_sample: bool = True
):
    """
    Sequentially predict post metrics at ALIAS level using the learned causal order:
      backlog -> days_to_close -> efficiency -> WSI_0_100
    Aggregates to team at the end.
    """
    import pandas as pd
    import numpy as np

    df = post_df.copy()
    # Ensure required base columns
    need = ["Capacity", "numCases", "StaffGroup"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"predict_post_from_cap_cases_causal: missing '{c}' in post_df")
    df["StaffGroup"] = df["StaffGroup"].astype("string").fillna("__MISSING__")

    def _predict_stage(target, extra_cols_to_keep=None):
        if target not in models:
            return
        feats = models[target]["features"]
        X = df[feats].copy()
        X["StaffGroup"] = X["StaffGroup"].astype("string").fillna("__MISSING__")
        if debug_sample:
            print(f"\n[DEBUG] Predicting {target} with features: {feats}")
            print(X.head(8).to_string(index=False))
        df[target] = models[target]["pipe"].predict(X)

    # Stage 1: backlog
    _predict_stage("backlog")
    # Stage 2: days_to_close (uses backlog)
    _predict_stage("days_to_close")
    # Stage 3: efficiency (uses backlog, days_to_close)
    _predict_stage("efficiency")
    # Stage 4: WSI_0_100 (uses backlog, days_to_close, efficiency)
    _predict_stage("WSI_0_100")

    agg = team_aggregate
    keep_cols = ["StaffGroup", "Capacity", "numCases", "backlog", "days_to_close", "efficiency", "WSI_0_100"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    grid_post = (
        df[keep_cols].groupby("StaffGroup", as_index=False)
          .agg({c: agg for c in keep_cols if c != "StaffGroup"})
    )
    return grid_post


from typing import List, Dict, Tuple, Optional
import pandas as pd, numpy as np

def _resolve_and_standardize_numcases(
    train_df: pd.DataFrame,
    df_in: pd.DataFrame,
    post_cap: pd.DataFrame,
    m0: pd.Timestamp,
    verbose: bool = True,
    feature_scope: Optional[List[str]] = None
):
    """
    Ensure 'numCases' exists in train_df and post_cap, preferring feature_scope columns.
    """
    # Build candidate list: prefer feature_scope names that look like workload, then fallbacks
    scope = [c for c in (feature_scope or [])]
    scope_lower = {c.lower(): c for c in scope}
    likely_workload = [
        "numcases", "taskscompleted", "tasks_completed",
        "activecases", "backlogsize", "totaltasks", "totalevent"
    ]
    # map lower-case candidates in scope back to original-cased names
    scope_candidates = [scope_lower[k] for k in likely_workload if k in scope_lower]
    default_candidates = ["numCases", "TasksCompleted", "ActiveCases", "BacklogSize", "TotalTasks", "TotalEvent"]
    candidates_order = scope_candidates + [c for c in default_candidates if c not in scope_candidates]

    used = None

    # TRAIN side
    if "numCases" not in train_df.columns:
        for c in candidates_order:
            if c in train_df.columns:
                train_df["numCases"] = pd.to_numeric(train_df[c], errors="coerce")
                used = c
                break

    if "numCases" not in train_df.columns and {"_date_","alias"}.issubset(train_df.columns) and {"_date_","alias"}.issubset(df_in.columns):
        dfm = df_in.copy()
        dfm["_date_"] = pd.to_datetime(dfm["_date_"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        dfm = dfm.loc[dfm["_date_"] == m0]
        for c in candidates_order:
            if c in dfm.columns:
                tmp = dfm[["_date_","alias", c]].drop_duplicates()
                train_df = train_df.merge(tmp.rename(columns={c: "numCases"}), on=["_date_","alias"], how="left")
                used = c
                break

    if "numCases" not in train_df.columns:
        train_df["numCases"] = 0.0
        if verbose: print("⚠️  Could not resolve workload for training; defaulted train_df['numCases']=0.")

    train_df["numCases"] = pd.to_numeric(train_df["numCases"], errors="coerce").fillna(0.0)

    # POST side
    if "numCases" not in post_cap.columns and "alias" in post_cap.columns:
        dfm = df_in.copy()
        dfm["_date_"] = pd.to_datetime(dfm["_date_"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        dfm = dfm.loc[dfm["_date_"] == m0]
        for c in ([used] + candidates_order if used else candidates_order):
            if c and c in dfm.columns:
                tmp = dfm[["alias", c]].drop_duplicates()
                post_cap = post_cap.merge(tmp.rename(columns={c: "numCases"}), on="alias", how="left")
                break

    if "numCases" in post_cap.columns:
        post_cap["numCases"] = pd.to_numeric(post_cap["numCases"], errors="coerce")

    return train_df, post_cap, used


def _make_regressor(random_state: int = 42):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=350, max_depth=6, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=-1, random_state=random_state
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=random_state)

def learn_sequence_from_data(
    train_df: pd.DataFrame,
    candidate_targets: Tuple[str, ...] = ("backlog","days_to_close","efficiency","WSI_0_100"),
    base_features: Tuple[str, ...] = ("Capacity","numCases","StaffGroup"),
    feature_scope: Optional[List[str]] = None,
    k: int = 6,
    verbose: bool = True,
):
    """
    Build a weighted directed graph by feature-importances and return a DAG order.
    Cycle-breaking rule:
      1) if a cycle contains two 'preferred targets', drop the edge that violates the preferred order
      2) else drop the smallest-weight edge in the cycle
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import networkx as nx

    # --- soft preference among targets to bias cycle-breaking
    preferred_order = ["backlog", "days_to_close", "efficiency", "WSI_0_100"]
    pref_rank = {t: i for i, t in enumerate(preferred_order)}

    # restrict features to allowed set
    feats_all = [c for c in train_df.columns if c not in ["_date_","alias"]]
    allowed = set(list(base_features) + list(feature_scope or []) + list(candidate_targets))
    feats_all = [f for f in feats_all if f in allowed]

    G = nx.DiGraph()
    W = {}  # (u,v) -> weight (importance)

    # ensure roots exist
    for b in base_features:
        G.add_node(b)

    # helper
    def _make_regressor(random_state: int = 42):
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=350, max_depth=6, learning_rate=0.06,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective="reg:squarederror", n_jobs=-1, random_state=random_state
            )
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=random_state)

    # build weighted edges
    for tgt in candidate_targets:
        if tgt not in train_df.columns:
            continue
        pool = [f for f in feats_all if f != tgt]
        num = [f for f in pool if f != "StaffGroup"]
        cat = ["StaffGroup"] if "StaffGroup" in pool else []
        X = train_df[num + cat].copy()
        if "StaffGroup" in cat:
            X["StaffGroup"] = X["StaffGroup"].astype("string").fillna("__MISSING__")
        y = pd.to_numeric(train_df[tgt], errors="coerce")
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]
        if len(y) < 10:  # not enough data
            continue

        pre = ColumnTransformer([
            ("num", "passthrough", num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ])
        model = Pipeline([("prep", pre), ("regr", _make_regressor(42))])
        model.fit(X, y)

        # get importances; map back to numeric columns only (simple + robust)
        imp = getattr(model.named_steps["regr"], "feature_importances_", None)
        if imp is None:
            parents = list(base_features)
            for p in parents:
                if p != tgt:
                    G.add_edge(p, tgt); W[(p,tgt)] = W.get((p,tgt), 1e-6)
            continue

        # first len(num) are the numeric passthrough columns
        num_imp = np.asarray(imp[:len(num)]) if len(num) else np.array([])
        imp_series = pd.Series(num_imp, index=num).sort_values(ascending=False)

        parents = list(imp_series.head(k).index)
        # always ensure base drivers are allowed to connect with small weight if missing
        for b in base_features:
            if b in pool and b not in parents:
                parents.append(b)

        # add weighted edges
        for p in parents:
            if p == tgt:
                continue
            w = float(imp_series.get(p, 1e-6))  # tiny weight if not in top-k
            G.add_edge(p, tgt)
            W[(p, tgt)] = max(W.get((p, tgt), 0.0), w)

    # break cycles if any
    def _break_cycles(G, W):
        # try fast topo first
        try:
            order = list(nx.topological_sort(G))
            return order
        except nx.NetworkXUnfeasible:
            pass

        # iteratively remove edges until DAG
        max_iters = 1000
        it = 0
        while it < max_iters:
            it += 1
            try:
                order = list(nx.topological_sort(G))
                return order
            except nx.NetworkXUnfeasible:
                # find one cycle
                cycle = next(nx.simple_cycles(G))
                # pick an edge to remove
                drop_edge = None
                # 1) prefer removing backward edge vs. preferred_order if both ends are preferred targets
                best_score = None
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    w = W.get((u,v), 1e-6)
                    # if u and v are both in preferred targets, penalize backwards edge
                    if u in pref_rank and v in pref_rank and pref_rank[v] < pref_rank[u]:
                        # strong candidate to drop
                        score = (-1, w)  # prioritize these first; tie-break by lower weight
                    else:
                        score = (0, w)   # normal removal: prefer smaller weight
                    # choose minimal score
                    if (best_score is None) or (score < best_score):
                        best_score = score
                        drop_edge = (u, v)
                # remove chosen edge
                if drop_edge is not None and G.has_edge(*drop_edge):
                    G.remove_edge(*drop_edge)
                    W.pop(drop_edge, None)
                else:
                    # fallback: remove arbitrary edge
                    e = list(G.edges())[0]
                    G.remove_edge(*e)
                    W.pop(e, None)
        # if we somehow still failed, raise
        raise nx.NetworkXUnfeasible("Unable to break cycles within max iters")

    order = _break_cycles(G, W)

    # final target order = keep only your targets in topo order; append any missing (shouldn't happen)
    final_targets = [t for t in order if t in candidate_targets]
    final_targets += [t for t in candidate_targets if t not in final_targets]

    if verbose:
        print("🧭 Learned DAG order:", " → ".join(final_targets))
        for t in final_targets:
            parents = [u for u, v in G.in_edges(t)]
            if parents:
                print(f"  • {t} <= {parents}")

    return final_targets, G


def fit_tree_models_dag(
    train_df: pd.DataFrame,
    seq: List[str],
    base_features: Tuple[str, ...] = ("Capacity","numCases","StaffGroup"),
    feature_scope: Optional[List[str]] = None,
    random_state: int = 42
):
    """
    Fit one pipeline per target following a learned sequence.
    Each target uses: base features + already-defined earlier targets + (feature_scope ∩ train_df).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error

    models, rows = {}, []
    earlier = set()
    for tgt in seq:
        # feature set = base + earlier targets + any scoped features present (excluding tgt)
        extras = [f for f in (feature_scope or []) if f in train_df.columns and f != tgt]
        use_feats = list(dict.fromkeys(list(base_features) + list(earlier) + extras))  # dedup, preserve order
        # make X/y
        X = train_df[use_feats].copy()
        if "StaffGroup" in X.columns:
            X["StaffGroup"] = X["StaffGroup"].astype("string").fillna("__MISSING__")
        y = pd.to_numeric(train_df[tgt], errors="coerce")
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]
        if len(y) < 10:
            continue

        pre = ColumnTransformer([
            ("num", "passthrough", [c for c in use_feats if c != "StaffGroup"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["StaffGroup"] if "StaffGroup" in use_feats else [])
        ])
        pipe = Pipeline([("prep", pre), ("regr", _make_regressor(random_state))])
        pipe.fit(X, y)
        yhat = pipe.predict(X)
        mse = float(mean_squared_error(y, yhat))

        models[tgt] = {"pipe": pipe, "features": use_feats}
        rows.append({"target": tgt, "mse": mse, "n": int(mask.sum())})
        earlier.add(tgt)

    mse_report = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    return models, mse_report

def predict_post_from_dag(
    post_df: pd.DataFrame,
    models: Dict[str, Dict],
    seq: List[str],
    team_aggregate: str = "median",
    debug_sample: bool = True
):
    """
    Sequentially predict targets following the learned DAG order.
    """
    df = post_df.copy()
    if "StaffGroup" in df.columns:
        df["StaffGroup"] = df["StaffGroup"].astype("string").fillna("__MISSING__")

    for tgt in seq:
        if tgt not in models:  # skip if not fitted
            continue
        feats = models[tgt]["features"]
        X = df[[c for c in feats if c in df.columns]].copy()
        if "StaffGroup" in X.columns:
            X["StaffGroup"] = X["StaffGroup"].astype("string").fillna("__MISSING__")
        if debug_sample:
            print(f"\n[DEBUG] Predicting {tgt} with features: {feats}")
            print(X.head(8).to_string(index=False))
        df[tgt] = models[tgt]["pipe"].predict(X)

    agg = team_aggregate
    keep_cols = ["StaffGroup","Capacity","numCases"] + [t for t in seq if t in df.columns]
    keep_cols = [c for c in keep_cols if c in df.columns]
    grid_post = (
        df[keep_cols].groupby("StaffGroup", as_index=False)
          .agg({c: agg for c in keep_cols if c != "StaffGroup"})
    )

    return grid_post

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_color_grid_with_sanity_range(
    combined_table: pd.DataFrame,
    sanity_df: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    low_col: str = "p10",
    high_col: str = "p95",
    soft_color: str = "rgba(255, 99, 71, 0.18)",   # soft red
    base_color: str = "rgba(0,0,0,0)",             # transparent
    title: str = "Pre vs. Post (sanity-flagged cells: outside p10–p95)"
) -> go.Figure:
    """
    Render a table-like grid and softly highlight any metric cell whose value
    is outside the [p10, p95] sanity range for that metric.
    combined_table: rows with columns ['StaffGroup','Scenario', <metrics>...]
    sanity_df: index = metric name; columns = at least [low_col, high_col]
    """
    df = combined_table.copy()
    df["Row"] = df["StaffGroup"].astype(str) + " — " + df["Scenario"].astype(str)

    # Choose metrics to display in the table (exclude meta-columns)
    default_metrics = ["Capacity", "numCases", "WSI_0_100", "efficiency", "days_to_close", "backlog"]
    if metrics is None:
        metrics = [m for m in default_metrics if m in df.columns]

    # Build the table in the same row order
    show_df = df[["Row"] + metrics].copy()

    # Build color fills per column (first column is Row labels, no coloring)
    fillcolors = []
    fillcolors.append([base_color] * len(show_df))  # for the Row header

    # Ensure sanity_df has the needed columns
    # (We expect sanity_df to have rows per metric and columns including low_col and high_col)
    if low_col not in sanity_df.columns or high_col not in sanity_df.columns:
        raise ValueError(f"sanity_df must include columns '{low_col}' and '{high_col}'")

    # Make sure index is string metric names
    s_tbl = sanity_df.copy()
    s_tbl.index = s_tbl.index.astype(str)

    # Build colors for each metric column
    for metric in metrics:
        col = pd.to_numeric(show_df[metric], errors="coerce")
        if metric in s_tbl.index:
            lo = s_tbl.loc[metric, low_col]
            hi = s_tbl.loc[metric, high_col]
            # Array of flags
            mask = (col < lo) | (col > hi)
            fillcolors.append([soft_color if bool(m) else base_color for m in mask])
        else:
            # No range for this metric: keep base color
            fillcolors.append([base_color] * len(col))

    header_values = ["StaffGroup — Scenario"] + metrics
    cell_values = [show_df[c].tolist() for c in ["Row"] + metrics]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header_values, fill_color="rgba(240,240,240,1)", align="left"),
                cells=dict(values=cell_values, fill_color=fillcolors, align="left",
                           format=[None] + [".2f"] * len(metrics))
            )
        ]
    )
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig


import numpy as np
import pandas as pd

def rebalance_workload_after_moves(
    df_in_month: pd.DataFrame,
    post_cap: pd.DataFrame,
    moves: list[dict],
    *,
    group_col: str = "StaffGroup",
    alias_col: str = "alias",
    workload_cols: list[str] = ["numCases"],   # add others like "numOpenCases","backlog" if desired
    method: str = "proportional",              # "proportional" or "even"
    random_state: int | None = 42
) -> pd.DataFrame:
    """
    Ensure alias-level workload is rebalanced in donor and receiver teams BEFORE post predictions.
    df_in_month : the PRE month slice (observed), used as the authoritative 'total volume' per team
    post_cap    : the alias-level POST frame returned by simulate_post_capacity (after team assignment changes)
    moves       : list of {from,to,n}
    """
    if alias_col not in post_cap.columns or group_col not in post_cap.columns:
        # Nothing to do (no alias granularity); caller will do team-level predictions.
        return post_cap

    rng = np.random.default_rng(random_state)
    post = post_cap.copy()

    # authoritative PRE team totals for each workload col (we keep total team volume fixed)
    pre_totals = (
        df_in_month.groupby(group_col, as_index=False)[workload_cols].sum()
        if workload_cols else pd.DataFrame({group_col: df_in_month[group_col].unique()})
    )

    # helper to distribute total T across a set of aliases by weights or evenly
    def _distribute(total: float, size: int, weights: np.ndarray | None):
        if size <= 0:
            return np.array([])
        if method == "proportional" and weights is not None and weights.sum() > 0:
            w = weights / weights.sum()
            return total * w
        # even split
        return np.full(size, total / size if size > 0 else 0.0)

    # Build a view of current alias sets by team after simulate_post_capacity’s team assignment
    team_aliases = post.groupby(group_col)[alias_col].apply(list).to_dict()

    # For selection, we need donor team alias lists PRIOR to the move (already reflected in post)
    for mv in moves or []:
        donor, host, k = mv.get("from"), mv.get("to"), int(mv.get("n", 0))
        if not donor or not host or k <= 0: 
            continue

        current_donor_aliases = list(team_aliases.get(donor, []))
        if len(current_donor_aliases) <= k:
            # move whatever is available (or skip if none)
            moved = current_donor_aliases
        else:
            moved = list(rng.choice(current_donor_aliases, size=k, replace=False))

        # Reassign the moved aliases' team to the host in post_cap
        post.loc[post[alias_col].isin(moved), group_col] = host

        # Update our team_aliases map
        team_aliases[donor] = [a for a in current_donor_aliases if a not in moved]
        team_aliases.setdefault(host, list(team_aliases.get(host, [])) + moved)

    # After ALL moves, rebalance workload in each affected team
    affected_teams = set([mv.get("from") for mv in (moves or [])] + [mv.get("to") for mv in (moves or [])])
    affected_teams = {t for t in affected_teams if t is not None}

    for team in affected_teams:
        aliases = team_aliases.get(team, [])
        if not aliases:
            continue

        # total team workload from PRE (we preserve this team total for POST)
        if not pre_totals.empty and team in set(pre_totals[group_col]):
            row = pre_totals.loc[pre_totals[group_col] == team]
        else:
            # fallback: compute from df_in_month
            row = pd.DataFrame([{group_col: team, **{c: df_in_month.loc[df_in_month[group_col]==team, c].sum()
                                                   for c in workload_cols}}])

        for col in workload_cols:
            team_total = float(row[col].values[0]) if col in row.columns else np.nan
            if not np.isfinite(team_total):
                # fallback to current POST sum to avoid NaNs
                team_total = float(post.loc[post[group_col]==team, col].sum()) if col in post.columns else 0.0

            # current weights (use PRE alias splits if present, else POST, else even)
            if col in df_in_month.columns:
                pre_alias = df_in_month.loc[df_in_month[group_col]==team, [alias_col, col]]
                pre_alias = pre_alias.groupby(alias_col, as_index=False)[col].sum()
                # align to current alias set
                pre_alias = pre_alias.set_index(alias_col).reindex(aliases).fillna(0.0)
                weights = pre_alias[col].to_numpy()
            elif col in post.columns:
                cur_alias = post.loc[post[group_col]==team, [alias_col, col]]
                cur_alias = cur_alias.groupby(alias_col, as_index=False)[col].sum().set_index(alias_col).reindex(aliases).fillna(0.0)
                weights = cur_alias[col].to_numpy()
            else:
                weights = None

            distr = _distribute(team_total, len(aliases), weights)

            # write back
            if col not in post.columns:
                post[col] = np.nan
            post.loc[post[alias_col].isin(aliases), col] = distr

    return post

import numpy as np
import pandas as pd
from copy import deepcopy

def run_simulation_bootstrap(
    n_boot: int,
    *,
    df_full: pd.DataFrame,
    scores_baseline: pd.DataFrame,
    month: str,
    moves: list[dict],
    compute_wsi_fn,
    wsi_kwargs: dict,
    targets=("WSI_0_100","efficiency","days_to_close","backlog"),
    only_impacted: bool = True,
    team_level_post: bool = False,
    show_predict_samples: bool = False,     # turn off for speed during bootstraps
    use_causal_sequence: object = "learn",
    feature_scope: list[str] | None = None,
    metrics_for_ci: list[str] = None,       # restrict which metrics you want CI for
    base_seed: int = 12345,
    **pipeline_kwargs   # <---- NEW (forwards all new args)

):
    """
    Run your existing pre_post_tree_pipeline_cap_cases multiple times over bootstrap resamples
    of df_full and build CIs per StaffGroup × Scenario × metric.

    Returns
    -------
    point_estimates : (grid_pre, grid_post, combined, fig, mse_report, sanity_df) from the ORIGINAL (non-boot) run
    ci_df           : DataFrame with columns: StaffGroup, Scenario, metric, mean, p2_5, p50, p97_5
    delta_ci_df     : (optional) CIs for deltas: Post - Pre per StaffGroup × metric
    """
    if metrics_for_ci is None:
        metrics_for_ci = ["Capacity","numCases","WSI_0_100","efficiency","days_to_close","backlog"]

    # 1) Run ONCE on the full df for your point estimates (as you already do)
    grid_pre, grid_post, combined_pt, fig, mse_report, sanity_df = pre_post_tree_pipeline_cap_cases(
        df_in=df_full,
        scores_baseline=scores_baseline,
        month=month,
        moves=moves,
        compute_wsi_fn=compute_wsi_fn,
        wsi_kwargs=wsi_kwargs,
        targets=targets,
        only_impacted=only_impacted,
        team_level_post=team_level_post,
        show_predict_samples=show_predict_samples,
        use_causal_sequence=use_causal_sequence,
        feature_scope=feature_scope,
        **pipeline_kwargs        # <--- ADD THIS

    )

    # 2) Bootstrap loop
    rng = np.random.default_rng(base_seed)
    combined_runs = []   # list of DataFrames (combined) per bootstrap

    # NOTE: we resample df_full rows with replacement (you can switch to alias-level resampling below)
    aliases = df_full["alias"].unique() if "alias" in df_full.columns else None

    for b in range(n_boot):
        # Option A: simple row bootstrap (fast, generic)
        df_boot = df_full.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 2**31-1)))

        # Option B: alias-level bootstrap (keep alias months together) — uncomment to use:
        # if aliases is not None:
        #     boot_aliases = rng.choice(aliases, size=len(aliases), replace=True)
        #     df_boot = pd.concat([df_full.loc[df_full["alias"]==a] for a in boot_aliases], ignore_index=True)

        try:
            _, _, combined_b, _, _, _ = pre_post_tree_pipeline_cap_cases(
                df_in=df_boot,
                scores_baseline=scores_baseline,
                month=month,
                moves=moves,
                compute_wsi_fn=compute_wsi_fn,
                wsi_kwargs=wsi_kwargs,
                targets=targets,
                only_impacted=only_impacted,
                team_level_post=team_level_post,
                show_predict_samples=False,             # keep quiet in boot runs
                use_causal_sequence=use_causal_sequence,
                feature_scope=feature_scope,
                **pipeline_kwargs        # <--- ADD THIS

            )
            # Keep only needed columns to save memory
            keep_cols = ["StaffGroup","Scenario"] + [c for c in metrics_for_ci if c in combined_b.columns]
            combined_runs.append(combined_b[keep_cols].copy())
        except Exception as e:
            # Skip failed boot; log and continue
            print(f"[BOOT {b+1}/{n_boot}] skipped due to: {e}")

    # 3) Build CI tables
    if not combined_runs:
        raise RuntimeError("All bootstrap runs failed; no CI could be computed.")

    combo = pd.concat(combined_runs, keys=range(len(combined_runs)), names=["boot","row"]).reset_index("boot")

    # Keep only numeric columns among requested metrics (prevents weird dtype issues)
    metrics_for_ci = [m for m in metrics_for_ci if m in combo.columns]
    combo = combo[["boot","StaffGroup","Scenario", *metrics_for_ci]].copy()

    long = combo.melt(id_vars=["boot","StaffGroup","Scenario"], var_name="metric", value_name="value")

    def _agg_ci(s: pd.Series):
        return pd.Series({
            "mean": float(np.nanmean(s)),
            "p2_5": float(np.nanpercentile(s, 2.5)),
            "p50":  float(np.nanpercentile(s, 50)),
            "p97_5":float(np.nanpercentile(s, 97.5))
        })

    # Build CIs with named aggregations (robust; no apply-shape issues)
    ci_df = (
        long.groupby(["StaffGroup", "Scenario", "metric"])["value"]
            .agg(
                mean="mean",
                p2_5=lambda s: float(np.nanpercentile(s, 2.5)),
                p50=lambda s: float(np.nanpercentile(s, 50)),
                p97_5=lambda s: float(np.nanpercentile(s, 97.5)),
            )
            .reset_index()
    )

    # 4) Optional: delta CIs (Post − Pre) per StaffGroup × metric
    # Pivot each boot into Pre/Post rows and compute deltas; then aggregate across boots
    deltas = []
    for b, dfb in enumerate(combined_runs):
        pre_b  = dfb[dfb["Scenario"].eq("Pre")].set_index("StaffGroup")
        post_b = dfb[dfb["Scenario"].str.startswith("Post")].set_index("StaffGroup")
        common = pre_b.index.intersection(post_b.index)
        if common.empty:
            continue
        dif = (post_b.loc[common, metrics_for_ci] - pre_b.loc[common, metrics_for_ci]).copy()
        dif["StaffGroup"] = common
        dif["boot"] = b
        deltas.append(dif.reset_index(drop=True))
    delta_ci_df = None
    if deltas:
        deltab = pd.concat(deltas, ignore_index=True)
        longd  = deltab.melt(id_vars=["boot","StaffGroup"], var_name="metric", value_name="delta")

        delta_ci_df = (
            longd.groupby(["StaffGroup", "metric"])["delta"]
                .agg(
                    mean="mean",
                    p2_5=lambda s: float(np.nanpercentile(s, 2.5)),
                    p50=lambda s: float(np.nanpercentile(s, 50)),
                    p97_5=lambda s: float(np.nanpercentile(s, 97.5)),
                )
                .reset_index()
        )

    # 5) Return point estimates + CIs
    return (grid_pre, grid_post, combined_pt, fig, mse_report, sanity_df), ci_df, delta_ci_df


import pandas as pd
import numpy as np

def build_contributions_df(
    df_prepped: pd.DataFrame,
    df_full: pd.DataFrame,
    scores: pd.DataFrame,
    scorer,
    score_cfg,           # your ScoreConfig
    cols,                # your ColumnMap
    capacity_features: list[str],
    top_k: int | None = None,     # keep all by default; set e.g. 15 to keep top-k by magnitude
):
    """
    Returns a long dataframe with columns:
      ['alias','_date_','StaffGroup','feature','value','contribution','Capacity_Score_0_100']
    If the scorer exposes true contributions (DSLI), uses them.
    Else (percentile method), computes a transparent weighted proxy from
    deviation features and percentile_weights.
    """

    # Normalize month key
    def _mstart(s):
        """
        Normalize datetime-like Series/array/scalar to month-start Timestamp(s).
        Works for pandas Series/Index/list-likes and scalars.
        """
        x = pd.to_datetime(s, errors="coerce")
        # Series / Index / list-like → has .dt
        if hasattr(x, "dt"):
            return x.dt.to_period("M").dt.to_timestamp()
        # scalar datetime
        return pd.to_datetime(x).to_period("M").to_timestamp()

    # Base id table (one row per alias×month)
    base_ids = (df_full[[cols.alias, cols.date, cols.group]]
                .assign(**{cols.date: _mstart(df_full[cols.date])})
                .drop_duplicates())

    # Attach Capacity score to the id grid
    cap_scores = (scores[[cols.alias, cols.date, "Capacity_Score_0_100"]]
                  .assign(**{cols.date: _mstart(scores[cols.date])}))

    id_grid = (base_ids
               .merge(cap_scores, on=[cols.alias, cols.date], how="left"))

    # Path A: true contributions available (DSLI or similar)
    has_true_contribs = hasattr(scorer, "contributions")

    rows = []

    if has_true_contribs:
        # Loop over alias×month keys (bulk APIs vary; loop is safest)
        for (a, d, g) in id_grid[[cols.alias, cols.date, cols.group]].itertuples(index=False, name=None):
            try:
                # Many implementations accept (df_prepped, alias, date)
                contrib = scorer.contributions(df_prepped, alias=a, date=d)
                # Normalize to DataFrame
                contrib_df = pd.DataFrame(contrib)
                # Try to infer column names: feature, value, contribution
                # Common patterns:
                # - columns already named ['feature','value','contribution']
                # - or ['name','value','effect'] / ['term','x','impact']
                colmap = {}
                for c in contrib_df.columns:
                    lc = str(c).lower()
                    if lc in ("feature","name","term"): colmap[c] = "feature"
                    elif lc in ("value","x","val"):     colmap[c] = "value"
                    elif lc in ("contribution","impact","effect","weight"): colmap[c] = "contribution"
                if colmap:
                    contrib_df = contrib_df.rename(columns=colmap)
                # Keep only what we need
                keep = [c for c in ["feature","value","contribution"] if c in contrib_df.columns]
                contrib_df = contrib_df[keep].copy()

                contrib_df[cols.alias]  = a
                contrib_df[cols.date]   = d
                contrib_df[cols.group]  = g
                # attach overall capacity score for context
                cap_val = id_grid.loc[(id_grid[cols.alias]==a) & (id_grid[cols.date]==d), "Capacity_Score_0_100"].max()
                contrib_df["Capacity_Score_0_100"] = cap_val

                rows.append(contrib_df)
            except Exception:
                # if one key fails, just skip it
                continue

        contributions_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=[cols.alias, cols.date, cols.group, "feature","value","contribution","Capacity_Score_0_100"]
        )

    else:
        # Path B: percentile scorer → build a transparent proxy
        # Use deviation features (Z_*) if present; multiply by percentile_weights; this matches your scoring config philosophy.
        pw = getattr(score_cfg, "percentile_weights", None) or {}
        # Limit to real features you passed & that exist
        cap_feats = [f for f in capacity_features if f in df_prepped.columns]
        # Try to find their deviation columns (common patterns: 'Z_<feature>_Dev' or '<feature>_Dev')
        def _dev_col(f):
            candidates = [f"Z_{f}_Dev", f"{f}_Dev", f"Z_{f}"]
            for c in candidates:
                if c in df_prepped.columns:
                    return c
            return None

        dev_cols_map = {f: _dev_col(f) for f in cap_feats}
        any_dev = any(v is not None for v in dev_cols_map.values())

        # Build a long table per alias×month
        tmp = (df_prepped
               .assign(**{cols.date: _mstart(df_prepped[cols.date])})
               [[cols.alias, cols.date, cols.group] + cap_feats + [c for c in dev_cols_map.values() if c]]
               .drop_duplicates())

        long_rows = []
        for (a, d, g), row in tmp.groupby([cols.alias, cols.date, cols.group]):
            for f in cap_feats:
                val = row[f].values[0] if f in row else np.nan
                dev = row[dev_cols_map[f]].values[0] if dev_cols_map[f] and dev_cols_map[f] in row else np.nan
                w   = pw.get(f, 1.0)  # default weight 1 if not specified
                # proxy contribution: weighted deviation (directional)
                contrib = (dev if pd.notna(dev) else 0.0) * float(w)
                long_rows.append({
                    cols.alias: a,
                    cols.date:  d,
                    cols.group: g,
                    "feature":  f,
                    "value":    val,
                    "contribution": contrib,
                })

        contributions_df = pd.DataFrame(long_rows)
        # Attach capacity score for context
        contributions_df = contributions_df.merge(cap_scores, on=[cols.alias, cols.date], how="left")

    # Optional: keep only top_k features by |contribution|
    if top_k is not None and top_k > 0 and not contributions_df.empty:
        contributions_df["_abs"] = contributions_df["contribution"].abs()
        contributions_df = (contributions_df
                            .sort_values([cols.alias, cols.date, "_abs"], ascending=[True, True, False])
                            .groupby([cols.alias, cols.date])
                            .head(top_k)
                            .drop(columns="_abs"))

    # Nice ordering & return
    contributions_df = contributions_df[[c for c in [cols.alias, cols.date, cols.group,
                                                     "feature","value","contribution","Capacity_Score_0_100"]
                                         if c in contributions_df.columns]].copy()
    contributions_df = contributions_df.sort_values([cols.alias, cols.date, "contribution"], ascending=[True, True, False])
    return contributions_df

import pandas as pd
import numpy as np

def build_wsi_contributions_df(
    alias_metrics: pd.DataFrame,
    df_full: pd.DataFrame,
    wsi_cfg,              # your WSIConfig (has weights)
    wsi_kwargs: dict,     # the kwargs you used in WSI (lists the feature columns)
    cols,                 # your ColumnMap
    attach_inputs: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Returns long-form dataframe with per-alias×month WSI component 'contributions'.
    Tries hard to find component columns under multiple common names. If no components
    are found, falls back to a single 'WSI_total' component (=WSI_0_100, weight=1)
    so downstream code doesn't break, and prints a note on how to enable components.
    """

    def _mstart(s):
        """
        Normalize datetime-like Series/array/scalar to month-start Timestamp(s).
        Works for pandas Series/Index/list-likes and scalars.
        """
        x = pd.to_datetime(s, errors="coerce")
        # Series / Index / list-like → has .dt
        if hasattr(x, "dt"):
            return x.dt.to_period("M").dt.to_timestamp()
        # scalar datetime
        return pd.to_datetime(x).to_period("M").to_timestamp()

    # Normalize month
    am = alias_metrics.copy()
    am[cols.date] = _mstart(am[cols.date])

    # ---- TRY TO LOCATE COMPONENT COLUMNS (flexible naming) ----
    # Canonical names we want:
    canonical = {
        "workload_component":        ["workload_component","workload","workload_idx","w_workload_component"],
        "capacity_dev_component":    ["capacity_dev_component","cap_dev","capacity_deviation","cap_dev_idx","w_cap_dev_component"],
        "persistence_component":     ["persistence_component","persist_component","persistence_idx","w_persist_component"],
        "complexity_component":      ["complexity_component","complexity","complexity_idx","w_complex_component"],
        "time_component":            ["time_component","time_coverage_component","time_idx","w_time_component"],
    }

    rename_map = {}
    found_any = False
    for canon, cands in canonical.items():
        for c in cands:
            if c in am.columns:
                rename_map[c] = canon
                found_any = True
                break

    am_renamed = am.rename(columns=rename_map)
    comp_cols = [c for c in canonical.keys() if c in am_renamed.columns]

    # ---- Pull weights (robust; support both .weights dict and individual attrs) ----
    weights = {}
    raw_weights = getattr(wsi_cfg, "weights", None)
    if isinstance(raw_weights, dict) and raw_weights:
        weights = raw_weights.copy()
    else:
        weights = {
            "workload_component":      getattr(wsi_cfg, "w_workload", 0.30),
            "capacity_dev_component":  getattr(wsi_cfg, "w_cap_dev", 0.30),
            "persistence_component":   getattr(wsi_cfg, "w_persist", 0.15),
            "complexity_component":    getattr(wsi_cfg, "w_complex", 0.15),
            "time_component":          getattr(wsi_cfg, "w_time", 0.10),
        }

    # ---- If no component columns are present, fall back safely ----
    if not comp_cols:
        if verbose:
            print(
                "ℹ️  No WSI component columns found in alias_metrics. "
                "Building a fallback with a single 'WSI_total' component (=WSI_0_100). "
                "To get real components, re-run WSIComputer to include component columns."
            )
        # Fallback: single-row per (alias, month) with WSI_0_100 as 'component_value'
        # Keep the interface identical for downstream
        out = am_renamed[[cols.alias, cols.date, cols.group, "WSI_0_100"]].copy()
        out["component"] = "WSI_total"
        out["component_value"] = out["WSI_0_100"].astype(float)
        out["weight"] = 1.0
        out["weighted_contribution"] = out["component_value"] * out["weight"]

        # relative share = 1.0 since single component
        out["rel_share"] = 1.0

        # Attach raw WSI inputs if requested
        if attach_inputs:
            raw_cols = [cols.alias, cols.date, cols.group]
            wsi_input_cols = []
            for key in ["cases_col", "complexity_col", "linearity_col", "weekend_col", "x24_col"]:
                c = wsi_kwargs.get(key)
                if isinstance(c, str) and c in df_full.columns:
                    wsi_input_cols.append(c)
            sev_cols = list(wsi_kwargs.get("sev_cols", [])) or []
            wsi_input_cols += [c for c in sev_cols if c in df_full.columns]

            if wsi_input_cols:
                raw = df_full[raw_cols + wsi_input_cols].copy()
                raw[cols.date] = _mstart(raw[cols.date])
                raw = raw.groupby([cols.alias, cols.date, cols.group], as_index=False).last()
                out = out.merge(raw, on=[cols.alias, cols.date, cols.group], how="left")

        # Order columns
        return out[[cols.alias, cols.date, cols.group, "component", "component_value",
                    "weight", "weighted_contribution", "rel_share", "WSI_0_100"] +
                   [c for c in out.columns if c not in {cols.alias, cols.date, cols.group,
                                                        "component","component_value","weight",
                                                        "weighted_contribution","rel_share","WSI_0_100"}]] \
                 .sort_values([cols.alias, cols.date, "component"]).reset_index(drop=True)

    # ---- Real component path: melt to long ----
    base_cols = [cols.alias, cols.date, cols.group, "WSI_0_100"]
    long = am_renamed[base_cols + comp_cols] \
        .melt(id_vars=base_cols, value_vars=comp_cols,
              var_name="component", value_name="component_value")

    # Map weights; compute weighted contribution + relative share
    long["weight"] = long["component"].map(weights).fillna(0.0).astype(float)
    long["component_value"] = pd.to_numeric(long["component_value"], errors="coerce")
    long["weighted_contribution"] = long["component_value"] * long["weight"]

    def _rel_share(g: pd.DataFrame) -> pd.Series:
        denom = np.abs(g["weighted_contribution"]).sum()
        if denom == 0 or np.isnan(denom):
            return pd.Series([0.0] * len(g), index=g.index)
        return g["weighted_contribution"] / denom

    long["rel_share"] = long.groupby([cols.alias, cols.date]).apply(_rel_share) \
                            .reset_index(level=[cols.alias, cols.date], drop=True)

    # Attach raw inputs if requested
    if attach_inputs:
        raw_cols = [cols.alias, cols.date, cols.group]
        inputs = []
        for key in ["cases_col", "complexity_col", "linearity_col", "weekend_col", "x24_col"]:
            c = wsi_kwargs.get(key)
            if isinstance(c, str) and c in df_full.columns:
                inputs.append(c)
        sev_cols = list(wsi_kwargs.get("sev_cols", [])) or []
        inputs += [c for c in sev_cols if c in df_full.columns]

        if inputs:
            raw = df_full[raw_cols + inputs].copy()
            raw[cols.date] = _mstart(raw[cols.date])
            raw = raw.groupby([cols.alias, cols.date, cols.group], as_index=False).last()
            long = long.merge(raw, on=[cols.alias, cols.date, cols.group], how="left")

    # Order columns and return
    order = [cols.alias, cols.date, cols.group, "component", "component_value",
             "weight", "weighted_contribution", "rel_share", "WSI_0_100"]
    extra = [c for c in long.columns if c not in order]
    out = long[order + extra].sort_values([cols.alias, cols.date, "component"]).reset_index(drop=True)
    return out


# ========= helper: train capacity-from-numCases model =========
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

def fit_capacity_from_numcases_model(
    df_full: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    cols,                                  # ColumnMap (with .date, .alias, .group)
    numcases_col: str = "numCases",
    random_state: int = 42
):
    """
    Train a simple, robust model: Capacity_Score_0_100 ~ f(numCases, StaffGroup)
    using ALL months in df_full.

    Returns: sklearn Pipeline (ColumnTransformer[OneHotEncoder] + Regressor)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    # Regressor: prefer XGB, else RandomForest
    try:
        from xgboost import XGBRegressor
        Regr = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=-1, random_state=random_state
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        Regr = RandomForestRegressor(n_estimators=600, random_state=random_state, n_jobs=-1)

    # Join all months, normalize date to month-start
    df = df_full[[cols.alias, cols.date, cols.group, numcases_col]].copy()
    df[cols.date] = pd.to_datetime(df[cols.date], errors="coerce").dt.to_period("M").dt.to_timestamp()

    y = scores[[cols.alias, cols.date, "Capacity_Score_0_100"]].copy()
    y[cols.date] = pd.to_datetime(y[cols.date], errors="coerce").dt.to_period("M").dt.to_timestamp()

    data = df.merge(y, on=[cols.alias, cols.date], how="inner").dropna(subset=[numcases_col, "Capacity_Score_0_100"])
    if data.empty:
        raise ValueError("No rows to train capacity-from-numCases model; check inputs.")

    # Features: numCases + StaffGroup
    num_features = [numcases_col]
    cat_features = [cols.group]

    X_num = data[num_features].copy()
    X_cat = data[cat_features].copy().astype("string").fillna("__MISSING__")
    X = pd.concat([X_num, X_cat], axis=1)
    y = pd.to_numeric(data["Capacity_Score_0_100"], errors="coerce")

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
    )

    pipe = Pipeline([("prep", pre), ("regr", Regr)])
    pipe.fit(X, y)

    return pipe
