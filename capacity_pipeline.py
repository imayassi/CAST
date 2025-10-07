# capacity_pipeline.py


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Union
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# PASTE THE ENTIRETY OF THE capacity_pipeline.py CODE FROM THE PREVIOUS RESPONSE INTO THIS CELL
from dataclasses import dataclass, field

from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


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
    method: str = "DSLI"  # {'DSLI','kmeans','pca','percentile'}
    dsli_anchor_pct: float = 0.70
    dsli_min_features: int = 5
    direction_map: Optional[Dict[str,bool]] = None
    random_state: int = 42

@dataclass
class WSIConfig:
    baseline_window_months: int = 12
    persist_window_months: int = 8
    high_cap_thresh: float = 75.0
    weights: Dict[str,float] = field(default_factory=lambda: dict(
        cap_dev=0.45, persist=0.25, complexity=0.15, time=0.15
    ))
    dev_tau: float = 0.25
    q_low: float = 0.05
    q_high: float = 0.95
    team_aggregate: str = "median"  # {"median","mean"}
    enforce_month_start: bool = True

@dataclass
class ScenarioConfig:
    realloc_strategy: str = "proportional"  # {"proportional","equal"}
    aggregate: str = "median"
    random_state: int = 42
    capacity_col_out: str = "Capacity_Score_0_100"

@dataclass
class OptimizeConfig:
    budget_moves: int = 20
    allowed_from: Optional[Iterable[str]] = None
    allowed_to: Optional[Iterable[str]] = None
    limit_per_group_from: Optional[int] = None
    limit_per_group_to: Optional[int] = None
    objective_weights: Dict[str, float] = field(default_factory=lambda: dict(
        WSI_0_100=-1.0, efficiency=+0.6, days_to_close=-0.6, backlog=-0.4, numCases=+0.25
    ))
    max_iters: int = 150
    tabu: int = 8
    random_state: int = 42


# =========================
# Constants
# =========================
FEATURE_DIRECTION: Dict[str, bool] = {
    "numCases": True, "numOpenCases": True, "backlog": True, "TimeSpent": True,
    "currentSev1": True, "currentSevA": True, "currentSevB": True, "currentSevC": True,
    "Is24X7OptedIn": True, "IsClosedOnWeekend": True,
    "LinearityScore": False, "AICS": False, "tenure": False,
    "efficiency": False, "activeTimeRatio": True, "hoursToSLAExpires": False,
    "avgDaysToClose": True,
}


# =========================
# Utils
# =========================
def _mstart(x):
    return pd.to_datetime(x, errors="coerce").to_period("M").to_timestamp()

def validate_input_schema(df: pd.DataFrame, cols: ColumnMap) -> Dict:
    missing = [c for c in [cols.date, cols.alias, cols.group] if c not in df.columns]
    return {"ok": len(missing)==0, "missing": missing}

def coalesce_names(df: pd.DataFrame, cols: ColumnMap) -> pd.DataFrame:
    d = df.copy()
    if "efficiency" not in d.columns and cols.efficiency_raw in d.columns:
        d["efficiency"] = pd.to_numeric(d[cols.efficiency_raw], errors="coerce")
    if "days_to_close" not in d.columns and cols.avg_dtc in d.columns:
        d["days_to_close"] = pd.to_numeric(d[cols.avg_dtc], errors="coerce")
    return d

def winsorize_cols(df: pd.DataFrame, cols: List[str], limits=(0.01,0.01)) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        v = pd.to_numeric(out[c], errors="coerce")
        if v.notna().sum() >= 3:
            out[c] = winsorize(v.fillna(v.median()), limits=limits)
    return out

def robust_minmax(series: pd.Series, lo=0, hi=100, qlow=1, qhigh=99):
    arr = np.asarray(series.dropna())
    if len(arr) < 2:
        return pd.Series(np.full_like(series, (lo+hi)/2.0, dtype=float), index=series.index)
    low = np.percentile(arr, qlow)
    high = np.percentile(arr, qhigh)
    if np.isclose(high, low):
        return pd.Series(np.full_like(series, (lo+hi)/2.0, dtype=float), index=series.index)
    scaled = (np.clip(series, low, high) - low) / (high - low)
    return pd.Series(scaled*(hi-lo)+lo, index=series.index)


# =========================
# DataPrep
# =========================
class DataPrep:
    def __init__(self, columns: ColumnMap, features: List[str], config: ScoreConfig):
        self.columns = columns
        self.features = features
        self.config = config
        self.kept_features_: List[str] = []
        self.dev_cols_: List[str] = []
        self.direction_map_: Dict[str,bool] = {}
        self.scaler_: Optional[Union[StandardScaler, Dict[str, StandardScaler]]] = None
        self.benchmarks_: Optional[pd.DataFrame] = None
        self._fitted = False

    def _group_percentiles(self, df: pd.DataFrame, cols: List[str], group_col: str, q: float,
                           min_group_size: int) -> pd.DataFrame:
        out = df.copy()
        gsize = out.groupby(group_col).size()
        small = gsize[gsize < min_group_size].index
        bench_global = out[cols].quantile(q=q).rename(lambda c: f"{c}_Benchmark")
        bench_by_group = (out[~out[group_col].isin(small)]
                          .groupby(group_col)[cols].quantile(q=q)
                          .rename(columns=lambda c: f"{c}_Benchmark"))
        out = out.merge(bench_by_group, on=group_col, how="left")
        for c in cols:
            out[f"{c}_Benchmark"] = out[f"{c}_Benchmark"].fillna(bench_global[f"{c}_Benchmark"])
        return out

    def fit(self, df: pd.DataFrame) -> "DataPrep":
        ok = validate_input_schema(df, self.columns)
        if not ok["ok"]:
            raise ValueError(f"Missing required columns: {ok['missing']}")
        d = coalesce_names(df, self.columns)
        candidate = [c for c in self.features if c in d.columns]
        numeric = [c for c in candidate if pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 1]
        if not numeric:
            raise ValueError("No usable numeric features found.")
        self.kept_features_ = numeric
        
        d_bench = self._group_percentiles(
            d, numeric, self.columns.group, q=self.config.bench_percentile,
            min_group_size=self.config.small_group_min_n
        )
        self.benchmarks_ = (
            d_bench[[self.columns.group] + [f"{c}_Benchmark" for c in numeric]]
            .drop_duplicates(self.columns.group)
        )
        
        self.direction_map_ = self.config.direction_map or FEATURE_DIRECTION
        dev_cols = []
        for f in numeric:
            b = d_bench[f"{f}_Benchmark"]
            denom = np.where(np.abs(b) < 1e-6, 1e-6, b)
            direction = 1 if self.direction_map_.get(f, True) else -1
            d_bench[f"{f}_Dev"] = direction * (pd.to_numeric(d_bench[f], errors="coerce") - b) / denom
            dev_cols.append(f"{f}_Dev")
        
        d_dev = winsorize_cols(d_bench, dev_cols, limits=self.config.winsor_limits)
        self.dev_cols_ = dev_cols
        
        X_all_df = d_dev[self.dev_cols_].fillna(0.0)
        global_scaler = StandardScaler().fit(X_all_df)
        
        if getattr(self.config, "scale_scope", "global") == "by_group":
            self.scaler_ = {"__global__": global_scaler}
            for g, gdf in d_dev.groupby(self.columns.group, dropna=False):
                Xg_df = gdf[self.dev_cols_].fillna(0.0)
                self.scaler_[g] = global_scaler if len(Xg_df) < 3 else StandardScaler().fit(Xg_df)
        else:
            self.scaler_ = global_scaler
        
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("DataPrep must be fitted before transform.")
        d = coalesce_names(df, self.columns).copy()
        d = d.merge(self.benchmarks_, on=self.columns.group, how="left")
        
        for f in self.kept_features_:
            b = d[f"{f}_Benchmark"]
            denom = np.where(np.abs(b) < 1e-6, 1e-6, b)
            direction = 1 if self.direction_map_.get(f, True) else -1
            d[f"{f}_Dev"] = direction * (pd.to_numeric(d[f], errors="coerce") - b) / denom
        
        d = winsorize_cols(d, self.dev_cols_, limits=self.config.winsor_limits)
        
        Z = np.empty((len(d), len(self.dev_cols_)), dtype=float)
        
        if isinstance(self.scaler_, dict):
            global_scaler = self.scaler_.get("__global__")
            for g in d[self.columns.group].unique():
                mask = (d[self.columns.group] == g).values
                if not np.any(mask): continue
                Xg_df = d.loc[mask, self.dev_cols_].fillna(0.0)
                scaler_g = self.scaler_.get(g, global_scaler)
                Z[mask, :] = scaler_g.transform(Xg_df)
        else:
            Z = self.scaler_.transform(d[self.dev_cols_].fillna(0.0))
        
        for i, c in enumerate(self.dev_cols_):
            d[f"Z_{c}"] = Z[:, i]
        
        return d

    def explain(self) -> Dict:
        return dict(summary=f"Kept {len(self.kept_features_)} features.",
                    dev_cols=self.dev_cols_, bench_percentile=self.config.bench_percentile)


# =========================
# Capacity Scorer
# =========================
class CapacityScorer:
    def __init__(self, config: ScoreConfig):
        self.cfg = config
        self.method = config.method.lower()
        self.model_: Optional[LogisticRegression] = None
        self.kmeans_: Optional[KMeans] = None
        self.pca_: Optional[PCA] = None
        self.dev_cols_: List[str] = []
        self.anchor_cols_: List[str] = []
        self.clip_: Tuple[int,int] = config.percentile_clip

    def fit(self, df_prepped: pd.DataFrame, group_col: str, id_cols: Tuple[str, ...]) -> "CapacityScorer":
        self.dev_cols_ = [c for c in df_prepped.columns if c.startswith("Z_") and c.endswith("_Dev")]
        if not self.dev_cols_:
            raise ValueError("No standardized deviation columns found. Run DataPrep.transform first.")
        
        rng = self.cfg.random_state
        X = df_prepped[self.dev_cols_].values

        if self.method == "dsli":
            raw_features = [c.replace("Z_","").replace("_Dev","") for c in self.dev_cols_]
            preferred = ["numCases","backlog","numOpenCases","TimeSpent"]
            anchors = [c for c in preferred if c in raw_features] or raw_features[:max(1, min(5, len(raw_features)))]
            self.anchor_cols_ = anchors
            z_anchor_cols = [f"Z_{c}_Dev" for c in anchors]
            A = df_prepped[z_anchor_cols].mean(axis=1)
            
            q = self.cfg.dsli_anchor_pct
            y = (A >= A.quantile(q)).astype(int).values
            
            lr = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=rng)
            lr.fit(X, y)
            self.model_ = lr
        elif self.method == "kmeans":
            self.kmeans_ = KMeans(n_clusters=3, n_init=10, random_state=rng).fit(X)
        elif self.method == "pca":
            self.pca_ = PCA(n_components=1, random_state=rng).fit(X)
        elif self.method != "percentile":
            raise ValueError("Unknown scoring method.")
        
        self.clip_ = self.cfg.percentile_clip
        return self

    def transform(self, df_prepped: pd.DataFrame, group_col: str, id_cols: Tuple[str, ...]) -> pd.DataFrame:
        idx = list(id_cols)
        out = df_prepped[idx + [group_col]].copy()
        Z = df_prepped[self.dev_cols_].values
        
        if self.method == "dsli" and self.model_:
            raw = self.model_.predict_proba(Z)[:, 1]
            out["Raw_Score"] = raw
        elif self.method == "kmeans" and self.kmeans_:
            centers = self.kmeans_.cluster_centers_
            sums = centers.sum(axis=1)
            heavy_idx, light_idx = int(np.argmax(sums)), int(np.argmin(sums))
            d_heavy = np.linalg.norm(Z - centers[heavy_idx], axis=1)
            d_light = np.linalg.norm(Z - centers[light_idx], axis=1)
            raw = d_light / (d_heavy + d_light + 1e-9)
            out["Raw_Score"] = raw
        elif self.method == "pca" and self.pca_:
            raw = self.pca_.transform(Z)[:,0]
            # Ensure higher PC1 is "worse" for consistent scaling
            if np.corrcoef(raw, df_prepped[self.dev_cols_].mean(axis=1))[0, 1] < 0:
                raw = -raw
            out["Raw_Score"] = raw
        else: # percentile
            ranks = df_prepped[self.dev_cols_].rank(pct=True)
            raw = ranks.mean(axis=1).values
            out["Raw_Score"] = raw
            
        out["Capacity_Score_0_100"] = robust_minmax(out["Raw_Score"], 0, 100,
                                                    qlow=self.clip_[0], qhigh=self.clip_[1]).round(2)
        return out

    def feature_importances(self) -> pd.DataFrame:
        features = [c.replace("Z_","").replace("_Dev","") for c in self.dev_cols_]
        if self.method == "dsli" and self.model_:
            coefs = self.model_.coef_.ravel()
            return pd.DataFrame({"feature": features, "importance": np.abs(coefs)}).sort_values("importance", ascending=False)
        if self.method == "pca" and self.pca_:
            loadings = self.pca_.components_[0]
            return pd.DataFrame({"feature": features, "importance": np.abs(loadings)}).sort_values("importance", ascending=False)
        if self.method == "kmeans" and self.kmeans_:
            diffs = self.kmeans_.cluster_centers_.max(axis=0) - self.kmeans_.cluster_centers_.min(axis=0)
            return pd.DataFrame({"feature": features, "importance": np.abs(diffs)}).sort_values("importance", ascending=False)
        return pd.DataFrame({"feature": features, "importance": 1.0 / len(features)}, index=range(len(features)))

    def contributions(self, df_prepped: pd.DataFrame, alias: str, date) -> pd.DataFrame:
        if self.method != "dsli" or self.model_ is None:
            raise RuntimeError("Contributions only supported for DSLI method.")
        mask = (df_prepped["alias"]==alias) & (pd.to_datetime(df_prepped["_date_"])==_mstart(date))
        if not mask.any():
            raise KeyError("Alias/date not found.")
        x = df_prepped.loc[mask, self.dev_cols_].values[0]
        coefs = self.model_.coef_.ravel()
        parts = coefs * x
        df = pd.DataFrame({
            "feature":[c.replace("Z_","").replace("_Dev","") for c in self.dev_cols_],
            "contribution": parts
        }).sort_values(by="contribution", key=abs, ascending=False).reset_index(drop=True)
        return df

    def explain(self) -> Dict:
        return {"method": self.method.upper(), "clip_percentiles": self.clip_}


# =========================
# WSI
# =========================
class WSIComputer:
    def __init__(self, columns: ColumnMap, config: WSIConfig):
        self.columns = columns
        self.cfg = config

    def compute(self, df_scored: pd.DataFrame) -> Dict[str,pd.DataFrame]:
        c, cfg = self.columns, self.cfg
        d = df_scored.copy()
        d[c.date] = pd.to_datetime(d[c.date], errors="coerce")
        if cfg.enforce_month_start:
            d[c.date] = d[c.date].dt.to_period("M").dt.to_timestamp()
        d = d.sort_values([c.alias, c.date])
        cap = pd.to_numeric(d["Capacity_Score_0_100"], errors="coerce").clip(0,100)
        
        # Calculate historical baseline
        base = d.groupby(c.alias)["Capacity_Score_0_100"].transform(
            lambda s: s.rolling(cfg.baseline_window_months, min_periods=3).median()
        ).fillna(d.groupby(c.alias)["Capacity_Score_0_100"].transform("median")).fillna(cap.median())
        
        # --- START OF FIX ---
        # Calculate raw deviation ratio
        deviation_ratio = (cap / base.replace(0, np.nan)).fillna(1.0) - 1.0
        
        # Calculate a bipolar cap_dev from -1.0 to 1.0 for clearer analysis and plotting
        cap_dev = np.clip(deviation_ratio / max(cfg.dev_tau, 1e-6), -1.0, 1.0)
        # --- END OF FIX ---

        # Persistence of high scores
        def _runlen(s):
            r=0; out=[]
            for v in s:
                r = min(r+1, cfg.persist_window_months) if v >= cfg.high_cap_thresh else 0
                out.append(r)
            return pd.Series(out, index=s.index)
        persist = d.groupby(c.alias)["Capacity_Score_0_100"].transform(_runlen) / cfg.persist_window_months

        # Complexity and Time Pressure
        denom = pd.to_numeric(d.get(c.num_cases, 1), errors="coerce").replace(0,np.nan)
        sev_terms, w = [], [1.0,0.5,0.2,0.1]
        for i, scn in enumerate([x for x in c.sev_cols if x in d.columns]):
            sev_terms.append((pd.to_numeric(d[scn], errors="coerce").fillna(0) / denom) * w[i])
        complexity = (sum(sev_terms) if sev_terms else pd.Series(0.0, index=d.index)).clip(0.0, 1.0)
        timep = (0.6*d.get(c.weekend_flag, 0) + 0.4*d.get(c.x24_flag, 0)).fillna(0).clip(0,1)

        # WSI calculation now uses only the POSITIVE part of the new cap_dev, maintaining the original intent
        W = (cfg.weights["cap_dev"] * np.maximum(0, cap_dev) +
             cfg.weights["persist"] * persist +
             cfg.weights["complexity"] * complexity +
             cfg.weights["time"] * timep)
        WSI = robust_minmax(W, 0, 100, qlow=cfg.q_low*100, qhigh=cfg.q_high*100).round(2)

        alias_df = d[[c.alias, c.group, c.date, "Capacity_Score_0_100"]].copy()
        alias_df["WSI_0_100"] = WSI
        alias_df["_cap_dev"], alias_df["_persist"], alias_df["_complex"], alias_df["_time"] = cap_dev, persist, complexity, timep
        
        agg_func = "median" if self.cfg.team_aggregate=="median" else "mean"
        team_df = (alias_df.groupby([c.group, c.date])
                   .agg(WSI_0_100=("WSI_0_100", agg_func),
                        cap_dev=("_cap_dev", agg_func),
                        persist=("_persist", agg_func),
                        complexity=("_complex", agg_func),
                        time_load=("_time", agg_func))
                   .reset_index())
        return {"alias_metrics": alias_df, "team_metrics": team_df}

    def explain(self) -> Dict:
        return dict(definition="WSI blends capacity deviation, persistence, complexity, and time pressure into 0–100.",
                    weights=self.cfg.weights)


# =========================
# Outcome Modeler - UPDATED CLASS
# =========================
class OutcomeModeler:
    def __init__(self, targets: Tuple[str, ...]=("WSI_0_100","efficiency","days_to_close","backlog"),
                 random_state=42):
        self.targets = list(targets)
        self.random_state = random_state
        self.models_: Dict[str, Pipeline] = {}
        self.scores_: Dict[str,float] = {}

    def fit(self, train_df: pd.DataFrame) -> "OutcomeModeler":
        from sklearn.model_selection import cross_val_score
        # FIX: Added the missing import for OneHotEncoder
        from sklearn.preprocessing import OneHotEncoder
        
        tdf = coalesce_names(train_df, ColumnMap())
        features = ["Capacity_Score_0_100", "StaffGroup"]
        Xb = tdf[features].rename(columns={"Capacity_Score_0_100":"Capacity"})
        
        preproc = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["StaffGroup"]),
            ("num", "passthrough", ["Capacity"])
        ])
        
        for t in self.targets:
            if t not in tdf.columns: continue
            y = pd.to_numeric(tdf[t], errors="coerce")
            valid_idx = y.notna()
            if not valid_idx.any(): continue

            reg = HistGradientBoostingRegressor(random_state=self.random_state)
            pipe = Pipeline([("prep", preproc), ("reg", reg)])
            
            cv_scores = cross_val_score(pipe, Xb[valid_idx], y[valid_idx], cv=3, scoring='r2')
            self.scores_[t] = np.mean(cv_scores)
            
            pipe.fit(Xb[valid_idx], y[valid_idx])
            self.models_[t] = pipe
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        features = ["Capacity_Score_0_100", "StaffGroup"]
        Xb = X[features].rename(columns={"Capacity_Score_0_100":"Capacity"})
        out = X[["StaffGroup"]].copy()
        for t, m in self.models_.items():
            out[f"{t}_pred"] = m.predict(Xb)
        return out

    def r2_scores(self) -> Dict[str,float]:
        return dict(self.scores_)


# =========================
# Scenario Engine - UPDATED CLASS
# =========================

Move = Dict[str, Union[str, int, float, List[str]]]

class ScenarioEngine:
    def __init__(self, columns: ColumnMap, prep: DataPrep, scorer: CapacityScorer,
                 wsi: WSIComputer, modeler: OutcomeModeler, config: ScenarioConfig):
        self.c, self.prep, self.scorer, self.wsi, self.modeler, self.cfg = columns, prep, scorer, wsi, modeler, config


    # FIX: Added 'return_employee_level=False' to the method signature
    def _get_current_state(self, df_month: pd.DataFrame, return_employee_level=False) -> pd.DataFrame:
        c = self.c
        prep = self.prep.transform(df_month)
        scores = self.scorer.transform(prep, group_col=c.group, id_cols=(c.alias, c.date))
        df_scored = df_month.merge(scores[[c.alias, c.date, self.cfg.capacity_col_out]], on=[c.alias, c.date], how="left")
        wsi_res = self.wsi.compute(df_scored)
        df_wsi = df_scored.merge(wsi_res["alias_metrics"][[c.alias, c.date, "WSI_0_100"]], on=[c.alias, c.date], how="left")
        
        df_agg = coalesce_names(df_wsi, c).groupby(c.group).agg(
            Headcount=(c.alias, "nunique"),
            Capacity_Score_0_100=(self.cfg.capacity_col_out, self.cfg.aggregate),
            WSI_0_100=("WSI_0_100", self.cfg.aggregate),
            efficiency=("efficiency", self.cfg.aggregate),
            days_to_close=("days_to_close", self.cfg.aggregate),
            backlog=("backlog", "sum"),
            numCases=("numCases", "sum")
        ).reset_index()
        
        # FIX: Added logic to optionally return the detailed employee-level dataframe
        if return_employee_level:
            return df_agg, df_wsi
        
        return df_agg

    def _apply_moves(self, df_month: pd.DataFrame, moves: List[Move]) -> pd.DataFrame:
        c = self.c
        rng = np.random.default_rng(self.cfg.random_state)
        df_moved = df_month.copy()
        for mv in moves:
            sg_from, sg_to = mv.get("from"), mv.get("to")
            if not sg_from or not sg_to: continue
            
            pool = df_moved.loc[df_moved[c.group] == sg_from, c.alias].unique().tolist()
            n_move = int(mv.get("n", 0) or len(pool) * mv.get("pct", 0))
            n_move = max(0, min(n_move, len(pool)))
            if n_move == 0: continue
            
            moved_aliases = rng.choice(pool, size=n_move, replace=False)
            df_moved.loc[df_moved[c.alias].isin(moved_aliases), c.group] = sg_to
        return df_moved

    def simulate_moves(self, df_month: pd.DataFrame, moves: List[Move]) -> Dict[str, pd.DataFrame]:
        c = self.c
        
        # Get employee-level "pre" data
        pre_state_agg, pre_employee_df = self._get_current_state(df_month, return_employee_level=True)

        # Apply moves and get employee-level "post" data
        df_post_move_structure = self._apply_moves(df_month, moves)
        post_state_agg, post_employee_df = self._get_current_state(df_post_move_structure, return_employee_level=True)

        # Use outcome modeler for metrics that depend on group composition
        outcomes_pre = self.modeler.predict(pre_state_agg)
        outcomes_post = self.modeler.predict(post_state_agg)
        
        for t in self.modeler.targets:
            if f"{t}_pred" in outcomes_pre.columns: pre_state_agg[t] = outcomes_pre[f"{t}_pred"]
            if f"{t}_pred" in outcomes_post.columns: post_state_agg[t] = outcomes_post[f"{t}_pred"]

        combined = pre_state_agg.merge(post_state_agg, on=self.c.group, suffixes=("_Pre", "_Post"), how='outer').fillna(0)
        for m in ["Headcount", "Capacity_Score_0_100", "WSI_0_100", "efficiency", "days_to_close", "backlog", "numCases"]:
            if f"{m}_Pre" in combined.columns and f"{m}_Post" in combined.columns:
                combined[f"{m}_Δ"] = combined[f"{m}_Post"] - combined[f"{m}_Pre"]
        
        # Return employee-level dataframes as well
        return {
            "pre_agg": pre_state_agg, "post_agg": post_state_agg, "combined": combined,
            "pre_employee": pre_employee_df, "post_employee": post_employee_df
        }


# =========================
# Optimizer - UPDATED CLASS
# =========================
class Optimizer:
    def __init__(self, engine: ScenarioEngine, cfg: OptimizeConfig):
        self.engine = engine
        self.cfg = cfg

    def _objective(self, grid: pd.DataFrame) -> float:
        score = 0.0
        # FIX: The target columns for the objective function may have a "_pred" suffix
        # from the modeler, or they may be the original columns. This handles both.
        for col, weight in self.cfg.objective_weights.items():
            col_to_use = f"{col}_pred" if f"{col}_pred" in grid.columns else col
            if col_to_use in grid.columns:
                # Use mean for rates/averages, sum for totals like backlog and numCases
                val = grid[col_to_use].sum() if col in ['backlog', 'numCases'] else grid[col_to_use].mean()
                score += val * weight
        return score

    def optimize_reassignment(self, df_month: pd.DataFrame) -> Dict:
        # FIX: Access ColumnMap (c) through the engine object (self.engine.c)
        c = self.engine.c
        current_df = df_month.copy()
        initial_state_agg = self.engine._get_current_state(current_df)
        
        # FIX: Access the modeler through the engine object (self.engine.modeler)
        initial_outcomes = self.engine.modeler.predict(initial_state_agg)
        initial_state = initial_state_agg.merge(initial_outcomes, on=c.group)
        
        best_state = initial_state
        best_obj = self._objective(best_state)
        
        plan, audit, tabu_list = [], [], []

        for it in range(self.cfg.max_iters):
            current_state_agg = self.engine._get_current_state(current_df)
            current_outcomes = self.engine.modeler.predict(current_state_agg)
            current_state = current_state_agg.merge(current_outcomes, on=c.group)

            from_groups = current_state.sort_values("WSI_0_100", ascending=False)[c.group].tolist()
            to_groups = current_state.sort_values("WSI_0_100", ascending=True)[c.group].tolist()
            
            best_move, best_local_obj, best_local_state = None, -np.inf, None

            # Limit the search space to speed up optimization
            for sg_from in from_groups[:3]:
                for sg_to in to_groups[:3]:
                    if sg_from == sg_to: continue
                    
                    pool = current_df[current_df[c.group] == sg_from][c.alias].unique()
                    if len(pool) <= 1: continue

                    move_key = (sg_from, sg_to)
                    if move_key in tabu_list: continue

                    alias_to_move = np.random.choice(pool, 1)[0]
                    
                    sim_results = self.engine.simulate_moves(current_df, moves=[{"from": sg_from, "to": sg_to, "n": 1, "aliases": [alias_to_move]}])
                    sim_state = sim_results["post_agg"]
                    
                    obj = self._objective(sim_state)
                    
                    if obj > best_local_obj:
                        best_local_obj = obj
                        best_move = {"alias": alias_to_move, "from_group": sg_from, "to_group": sg_to}
                        best_local_state = sim_state
            
            if len(plan) >= self.cfg.budget_moves or best_move is None or best_local_obj <= best_obj:
                break
            
            plan.append(best_move)
            audit.append({"iter": it + 1, "objective": best_local_obj, "improvement": best_local_obj - best_obj})
            best_obj = best_local_obj
            best_state = best_local_state
            current_df.loc[current_df[c.alias] == best_move["alias"], c.group] = best_move["to_group"]

            tabu_list.append((best_move["from_group"], best_move["to_group"]))
            if len(tabu_list) > self.cfg.tabu:
                tabu_list.pop(0)

        return {"plan": plan, "initial_metrics": initial_state, "expected_metrics": best_state, "audit": pd.DataFrame(audit)}

# =========================
# Viz helpers (Complete End-to-End Code)
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from scipy.stats import gaussian_kde # Needed for MLE calculation


# --- Main Plotting Functions ---

def plot_score_distributions(alias_df: pd.DataFrame, score_cols: List[str], save_path: Optional[str] = None):
    """
    Plots a histogram and KDE for the distribution of specified scores.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, len(score_cols), figsize=(7 * len(score_cols), 5), sharey=True)
    if len(score_cols) == 1: axes = [axes]

    for ax, col in zip(axes, score_cols):
        sns.histplot(alias_df[col], kde=True, ax=ax, bins=20, color='skyblue')
        ax.set_title(f'Distribution of {col}', fontsize=14, weight='bold')
        ax.set_xlabel('Score (0-100)', fontsize=12)
        ax.set_ylabel('Number of Employees', fontsize=12)

    plt.suptitle('Employee Score Distributions', fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Plot saved to: {save_path}")

    plt.show()

def plot_team_metrics(team_df: pd.DataFrame, metric_cols: List[str], save_path: Optional[str] = None):
    """
    Plots a bar chart showing team-level metrics over time.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    df_melt = team_df.melt(id_vars=['StaffGroup', '_date_'], value_vars=metric_cols, var_name='Metric', value_name='Score')

    g = sns.catplot(data=df_melt, x='_date_', y='Score', hue='StaffGroup', col='Metric',
                    kind='bar', col_wrap=2, height=4, aspect=1.5, sharey=False, palette='viridis')

    g.fig.suptitle('Team Metrics Over Time', fontsize=18, weight='bold')
    g.set_titles("{col_name}", size=14)
    g.set_xticklabels(rotation=45, ha='right')
    g.set_axis_labels("Month", "Average Score")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        g.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Plot saved to: {save_path}")

    plt.show()

def plot_feature_importances(scorer: CapacityScorer, save_path: Optional[str] = None):
    """
    Plots the top 10 most important features for the Capacity Score.
    """
    imp_df = scorer.feature_importances().nlargest(10, 'importance')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x='importance', y='feature', palette='mako')
    plt.title('Top 10 Drivers of the Capacity Score', fontsize=16, weight='bold')
    plt.xlabel('Importance (Coefficient Magnitude)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Plot saved to: {save_path}")

    plt.show()

def plot_contributions_waterfall(contrib_df: pd.DataFrame, alias: str, save_path: Optional[str] = None):
    """
    Plots a waterfall chart showing score contributions for a single employee.
    """
    contrib_df = contrib_df.head(10).sort_values('contribution')

    plt.figure(figsize=(10, 6))
    colors = ['crimson' if c < 0 else 'mediumseagreen' for c in contrib_df['contribution']]
    plt.barh(contrib_df['feature'], contrib_df['contribution'], color=colors)

    plt.title(f'Top Score Contributions for {alias}', fontsize=16, weight='bold')
    plt.xlabel('Contribution to Score (Log-Odds)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Plot saved to: {save_path}")

    plt.show()

def plot_simulation_heatmap(pre_df: pd.DataFrame, post_df: pd.DataFrame, group_col="StaffGroup", save_path: Optional[str] = None):
    """
    Generates a heatmap comparing pre- and post-simulation metrics.
    """
    pre_df['State'] = 'Pre'
    post_df['State'] = 'Post (pred)'

    metric_cols = ['Capacity_Score_0_100', 'WSI_0_100', 'efficiency', 'days_to_close', 'backlog']
    plot_data = pd.concat([pre_df, post_df])
    plot_data['SortKey'] = plot_data[group_col] + '_' + plot_data['State']
    plot_data = plot_data.sort_values(by=['SortKey']).set_index([group_col, 'State'])

    heatmap_data = plot_data[metric_cols].rename(columns={'Capacity_Score_0_100': 'Capacity'})

    plt.figure(figsize=(10, len(plot_data) * 0.4))
    ax = sns.heatmap(
        heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn_r",
        cbar=True, cbar_kws={'label': 'WSI_0_100 (0-100)'}
    )

    norm = plt.Normalize(heatmap_data['WSI_0_100'].min(), heatmap_data['WSI_0_100'].max())
    cmap = plt.get_cmap('RdYlGn_r')

    for (y, row), (i, group_state) in zip(heatmap_data.iterrows(), enumerate(ax.get_yticklabels())):
        wsi_val = row['WSI_0_100']
        color = cmap(norm(wsi_val))
        for x in range(len(metric_cols)):
            ax.add_patch(plt.Rectangle((x, i), 1, 1, fill=True, color=color, ec='none'))
            val = heatmap_data.iloc[i, x]
            ax.text(x + 0.5, i + 0.5, f'{val:.2f}', ha='center', va='center', color='black')

    ax.set_title('Pre vs Post Simulation', fontsize=14, weight='bold')
    ax.set_ylabel('StaffGroup'); ax.set_xlabel('Metric')
    new_labels = [f"{idx[0]} -- {idx[1]}" for idx in heatmap_data.index]
    ax.set_yticklabels(new_labels, rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Plot saved to: {save_path}")
    plt.show()
    return plot_data

def plot_optimization_plan(plan: List[Dict], save_path: Optional[str] = None):
    """
    Plots a bar chart showing the net headcount moves from the optimizer.
    """
    if not plan: return
    moves_summary = pd.DataFrame(plan).groupby(['from_group', 'to_group']).size().reset_index(name='count')
    net_moves = pd.concat([
        moves_summary.groupby('from_group')['count'].sum() * -1,
        moves_summary.groupby('to_group')['count'].sum()
    ]).groupby(level=0).sum().reset_index(name='Net Change')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=net_moves, x='Net Change', y='index', palette='PiYG')
    plt.title('Optimal Reassignment Plan: Net Headcount Moves', fontsize=16, weight='bold')
    plt.xlabel('Number of Employees to Move', fontsize=12)
    plt.ylabel('Staffing Group', fontsize=12)
    plt.axvline(0, color='grey', linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Plot saved to: {save_path}")

    plt.show()

# --- Overlapping Distribution Plots with Statistical Annotations ---

def plot_simulation_distributions(pre_df_employee: pd.DataFrame, post_df_employee: pd.DataFrame, metrics: List[str], save_path: Optional[str] = None):
    """
    Plots overlapping KDEs for pre/post simulation at the employee level,
    with annotations for mean and MLE (peak of distribution).
    """
    pre_df_employee['State'] = 'Pre-Simulation'
    post_df_employee['State'] = 'Post-Simulation'

    combined_df = pd.concat([pre_df_employee, post_df_employee])
    melted_df = combined_df.melt(id_vars=['State'], value_vars=metrics, var_name='Metric', value_name='Value')

    g = sns.displot(
        data=melted_df, x='Value', hue='State', col='Metric',
        kind='kde', fill=True, common_norm=False,
        facet_kws={'sharey': False, 'sharex': False},
        palette={'Pre-Simulation': 'skyblue', 'Post-Simulation': 'coral'}
    )

    # Add annotations for mean and MLE
    for ax, metric in zip(g.axes.flat, metrics):
        pre_data = pre_df_employee[metric].dropna()
        post_data = post_df_employee[metric].dropna()

        if pre_data.empty or post_data.empty:
            continue

        mean_pre, mean_post = pre_data.mean(), post_data.mean()

        kde_pre, kde_post = gaussian_kde(pre_data), gaussian_kde(post_data)
        x_range = np.linspace(min(pre_data.min(), post_data.min()), max(pre_data.max(), post_data.max()), 500)
        mle_pre = x_range[np.argmax(kde_pre(x_range))]
        mle_post = x_range[np.argmax(kde_post(x_range))]

        ax.axvline(mean_pre, color='blue', linestyle='-', linewidth=2, label=f'Mean (Pre): {mean_pre:.2f}')
        ax.axvline(mean_post, color='red', linestyle='-', linewidth=2, label=f'Mean (Post): {mean_post:.2f}')
        ax.axvline(mle_pre, color='skyblue', linestyle='--', linewidth=2, label=f'MLE (Pre): {mle_pre:.2f}')
        ax.axvline(mle_post, color='coral', linestyle='--', linewidth=2, label=f'MLE (Post): {mle_post:.2f}')

        diff_text = f'Mean Diff (Post - Pre): {mean_post - mean_pre:+.2f}'
        ax.text(0.95, 0.95, diff_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        
        ax.legend()

    g.fig.suptitle('Pre vs. Post Simulation: Employee-Level Distributions', fontsize=16, weight='bold')
    g.set_titles("{col_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        g.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Distribution plot saved to: {save_path}")
    
    plt.show()

def plot_bucket_uplift_distributions(alias_df: pd.DataFrame, metric: str, from_bucket: Tuple, to_bucket: Tuple, save_path: Optional[str] = None):
    """
    Plots overlapping KDEs for a metric between two capacity score buckets,
    with annotations for mean and MLE.
    """
    from_df = alias_df[(alias_df['Capacity_Score_0_100'] >= from_bucket[0]) & (alias_df['Capacity_Score_0_100'] < from_bucket[1])]
    to_df = alias_df[(alias_df['Capacity_Score_0_100'] >= to_bucket[0]) & (alias_df['Capacity_Score_0_100'] < to_bucket[1])]

    if from_df.empty or to_df.empty:
        print(f"Warning: Not enough data in buckets to plot distribution for {metric}.")
        return

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=from_df, x=metric, fill=True, color='skyblue', label=f'"{from_bucket[0]}-{from_bucket[1]}" Bucket (Before)')
    sns.kdeplot(data=to_df, x=metric, fill=True, color='coral', label=f'"{to_bucket[0]}-{to_bucket[1]}" Bucket (Target)')

    mean_from, mean_to = from_df[metric].mean(), to_df[metric].mean()

    kde_from, kde_to = gaussian_kde(from_df[metric]), gaussian_kde(to_df[metric])
    x_range = np.linspace(alias_df[metric].min(), alias_df[metric].max(), 500)
    mle_from = x_range[np.argmax(kde_from(x_range))]
    mle_to = x_range[np.argmax(kde_to(x_range))]

    plt.axvline(mean_from, color='blue', linestyle='-', label=f'Mean (Before): {mean_from:.2f}')
    plt.axvline(mean_to, color='red', linestyle='-', label=f'Mean (Target): {mean_to:.2f}')
    plt.axvline(mle_from, color='deepskyblue', linestyle='--', label=f'MLE (Before): {mle_from:.2f}')
    plt.axvline(mle_to, color='tomato', linestyle='--', label=f'MLE (Target): {mle_to:.2f}')

    diff_text = f'Mean Diff (Target - Before): {mean_to - mean_from:+.2f}'
    plt.text(0.95, 0.95, diff_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.title(f'Distribution of {metric} for Bucket Uplift Scenario', fontsize=16, weight='bold')
    plt.xlabel(f'Value of {metric}'); plt.ylabel('Density')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Bucket uplift plot saved to: {save_path}")

    plt.show()

def plot_volume_uplift_distributions(pre_df: pd.DataFrame, post_df: pd.DataFrame, metric: str, save_path: Optional[str] = None):
    """
    Plots the overlapping KDE for a metric showing the entire population
    before and after a bucket uplift simulation, with annotations for mean and MLE.
    """
    pre_df['State'] = 'Before Uplift'
    post_df['State'] = 'After Uplift (Simulated)'

    combined_df = pd.concat([pre_df, post_df])

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=combined_df, x=metric, hue='State', fill=True, common_norm=False, palette={'Before Uplift': 'skyblue', 'After Uplift (Simulated)': 'coral'})

    mean_pre, mean_post = pre_df[metric].mean(), post_df[metric].mean()

    kde_pre, kde_post = gaussian_kde(pre_df[metric]), gaussian_kde(post_df[metric])
    x_range = np.linspace(combined_df[metric].min(), combined_df[metric].max(), 500)
    mle_pre = x_range[np.argmax(kde_pre(x_range))]
    mle_post = x_range[np.argmax(kde_post(x_range))]

    plt.axvline(mean_pre, color='blue', linestyle='-', label=f'Mean (Before): {mean_pre:.2f}')
    plt.axvline(mean_post, color='red', linestyle='-', label=f'Mean (After): {mean_post:.2f}')
    plt.axvline(mle_pre, color='deepskyblue', linestyle='--', label=f'MLE (Before): {mle_pre:.2f}')
    plt.axvline(mle_post, color='tomato', linestyle='--', label=f'MLE (After): {mle_post:.2f}')

    diff_text = f'Mean Diff (After - Before): {mean_post - mean_pre:+.2f}'
    plt.text(0.95, 0.95, diff_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.title(f'Overall Distribution of {metric} After Uplift', fontsize=16, weight='bold')
    plt.xlabel(f'Value of {metric}'); plt.ylabel('Density')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Volume uplift plot saved to: {save_path}")

    plt.show()


# =========================
# Sample data generator (Updated for realistic personas)
# =========================
def make_sample_data(n_groups=4, aliases_per_group=25, months=("2025-06","2025-07","2025-08"), seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sgs = [f"SG-{chr(65+i)}" for i in range(n_groups)]
    rows = []

    # Define employee personas with different performance characteristics
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
            
            # Assign a persona to each employee
            persona_key = rng.choice(persona_names, p=[0.20, 0.20, 0.40, 0.20]) # 40% are steady
            p = personas[persona_key]
            
            base_workload = rng.normal(40, 5) * p['workload_factor']
            prev_bkl = max(0, rng.normal(30 + p['backlog_factor'] * 10, 5))

            for m in months:
                dt = _mstart(m)
                noise = p['noise']

                # Generate metrics based on the persona's profile + some randomness
                numCases = int(max(10, base_workload + rng.normal(0, 5 * noise)))
                efficiency = float(np.clip(rng.normal(p['eff_mean'], 0.03 * noise), 0.5, 0.98))
                dtc = float(np.clip(rng.normal(p['dtc_mean'], 1 * noise), 1, 30))
                backlog = int(max(0, prev_bkl * 0.7 + numCases * 0.1 + rng.normal(0, 5 * noise)))
                prev_bkl = backlog

                # Other features
                tenure = float(np.clip(rng.normal(24, 12), 3, 72))
                numOpen = int(max(0, numCases * rng.uniform(0.1, 0.3)))
                time_spent = float(numCases * (2.0 - (efficiency - 0.75)) + rng.normal(0, 5 * noise))
                sevA = int(rng.binomial(numCases, 0.15 + (p.get('complexity_factor', 0) * 0.1)))

                rows.append(dict(
                    _date_=dt, alias=alias, StaffGroup=sg,
                    numCases=numCases, numOpenCases=numOpen, backlog=backlog, TimeSpent=time_spent,
                    som=efficiency, avgDaysToClose=dtc, tenure=tenure,
                    currentSevA=sevA, IsClosedOnWeekend=int(rng.random() < 0.2), Is24X7OptedIn=int(rng.random() < 0.1)
                ))
    return pd.DataFrame(rows)