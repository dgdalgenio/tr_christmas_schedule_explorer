"""Results Explorer for the Christmas 2026 grid search (christmas_search5/grid-search2).

Reads the grid-search output (results_*.csv + results_*_schedules/) and the demand forecast CSV,
lets you filter the evaluated configurations, compare the survivors, check how many agents break a
given set of break-policy rules (e.g. the defaults), and drill into one set-up (including a net
staffing curve in the style of the TarroTetris dashboard).

Run:  bash run_app.sh
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
GS2 = HERE.parent  # grid-search2/

# ─── Constants ───────────────────────────────────────────────────────────────────────────────

TIERS = ["XMAS_T2", "XMAS_T3", "XMAS_T3FC", "XMAS_Mentor"]
TIER_LABEL = {"XMAS_T2": "T2", "XMAS_T3": "T3", "XMAS_T3FC": "T3FC", "XMAS_Mentor": "Mentor"}
PATTERNS = ["4x11", "5x9", "6x7.5", "A1", "A2", "3x2", "4.5"]
NONFLEXI_PATTERNS = {"5x9"}  # every other pattern counts as flexi
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ABBR = {"Monday": "mon", "Tuesday": "tue", "Wednesday": "wed", "Thursday": "thu",
            "Friday": "fri", "Saturday": "sat", "Sunday": "sun"}
FORECAST_GROUPS = ["T1GD", "T2GD", "T3JPN", "T3CJN", "T3GD", "T3ENT"]  # the groups the optimiser covers

# Grid parameters, in display order, with the "default" each is compared against. Break-policy
# defaults are the backend DEFAULT_GENERATION_RULES; the others are the pre-grid-search settings.
# All editable in the sidebar.
PARAMS = [
    # ("xmasT3CostWeight", "1.03"),
    ("minPhoneBeforeFirstBreakMin", "60"),
    ("minPhoneAfterLastBreakMin", "60"),
    ("maxContiguousPhoneMin", "150"),
    ("minContiguousPhoneMin", "60"),
    ("maxCombinedAdjacentPhoneMin", "300"),
    ("longBreakPlacement", "middle"),
    # ("requireFridayOnShift", "false"),
    # ("peakIntervalCount", "1"),
    # ("saAlgorithm", "attempt1"),
    # ("maxHeadcountFlexi", "none"),
    ("workPatternConfig", "base"),
]
PARAM_SHORT = {
    # "xmasT3CostWeight": "T3 cost wt",
    "minPhoneBeforeFirstBreakMin": "Min phone time before 1st break",
    "minPhoneAfterLastBreakMin": "Min phone time after last break",
    "maxContiguousPhoneMin": "Max contiguous phone time",
    "minContiguousPhoneMin": "Min contiguous phone time",
    "maxCombinedAdjacentPhoneMin": "Max adjacent phone time",
    "longBreakPlacement": "Long break placement",
    # "requireFridayOnShift": "Fri on shift",
    # "peakIntervalCount": "Peak intervals with no break",
    "workPatternConfig": "Pattern set",
}

# The five break-policy rules whose violations can be counted from a saved schedule.
# key -> (grid parameter, plain-language rule, short label, backend default)
RULES = {
    "before": ("minPhoneBeforeFirstBreakMin", "phone time before the first break is LESS than", "Min before 1st break", 60),
    "after": ("minPhoneAfterLastBreakMin", "phone time after the last break is LESS than", "Min after last break", 60),
    "max": ("maxContiguousPhoneMin", "a contiguous phone stretch is MORE than", "Max contiguous phone", 150),
    "min": ("minContiguousPhoneMin", "a contiguous phone stretch is LESS than", "Min contiguous phone", 60),
    "adj": ("maxCombinedAdjacentPhoneMin", "two phone stretches split by a short (<45 min) break total MORE than",
            "Max adjacent phone", 300),
}
RULE_KEYS = list(RULES)
SLOT_MIN = 15

# Ranking options: label -> column. Violation columns are computed from the saved schedules.
RANK_OPTIONS = {
    "Final headcount": "hc_final",
    "Cost-weighted output (with flexi surcharge)": "objective_flexi_surcharge",
    "Cost-weighted output (base)": "objective_base",
    "Total deficit": "total_deficit_sum",
    "Staffing efficiency ratio": "efficiency_efficiencyRatio",
    "Surplus spread (stdev %)": "efficiency_surplusStdevPct",
    "Flexi headcount": "HC_flexi",
    "Compliance rate": "aqol_complianceRate",
    "Agents breaking any rule (from schedules)": "v_agents_any",
    **{f"Agents breaking: {RULES[k][2].lower()} (from schedules)": f"v_agents_{k}" for k in RULE_KEYS},
}
VIOLATION_RANKS = {c for c in RANK_OPTIONS.values() if c.startswith("v_agents_")}

# Tarro-style palette (matches the dashboard chart).
C_REQUIRED, C_PHONES, C_SHIFT = "#0f1218", "#009983", "#e6b75c"
C_BAD = "#ef4444"
TIER_COLORS = {"T2": "#009983", "T3": "#e6b75c", "T3FC": "#7c3aed", "Mentor": "#9aa1ad"}
PATTERN_COLORS = ["#009983", "#6c7382", "#0f1218", "#e6b75c", "#7c3aed", "#ef4444", "#00e590"]

import inspect

# st.plotly_chart only gained width="stretch" in newer Streamlit releases; fall back gracefully.
_PLOT_KW = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
            else {"use_container_width": True})


def plot(fig, **_ignored):
    st.plotly_chart(fig, **_PLOT_KW)


st.set_page_config(page_title="[Fifth Pass] Christmas 2026 Possible Schedules Explorer", layout="wide")


# ─── Loading ─────────────────────────────────────────────────────────────────────────────────

def norm(v) -> str:
    """Normalise a parameter value so CSV values and typed defaults compare equal."""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "none"
    s = str(v).strip().lower()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


@st.cache_data(show_spinner=False)
def load_results(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["error"] = df["error"].fillna("") if "error" in df else ""
    df["maxHeadcountFlexi"] = df["maxHeadcountFlexi"].fillna("none")
    df["label"] = "#" + df["taskIdx"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_forecast(path: str, mtime: float) -> Dict[str, Dict[str, Dict[int, float]]]:
    """{group: {Day: {minuteOfDay: required}}} from the multi-block forecast CSV."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    group_row, day_row = rows[0], rows[1]
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    starts = [i for i, g in enumerate(group_row) if g.strip()]
    for gi, c0 in enumerate(starts):
        group = group_row[c0].strip()
        width = (starts[gi + 1] if gi + 1 < len(starts) else len(group_row)) - c0
        day_cols = {}
        for j in range(c0, c0 + width):
            lab = day_row[j].strip().upper() if j < len(day_row) else ""
            for d in DAYS:
                if lab == d[:3].upper():
                    day_cols[d] = j
        if len(day_cols) < 7:  # per-weekday detail blocks etc. — not group blocks
            continue
        per_day: Dict[str, Dict[int, float]] = {d: {} for d in DAYS}
        prev = -1
        wrap = 0
        for r in rows[2:]:
            label = (r[0] if r else "").strip()
            if not label:
                continue
            m = parse_time(label)
            if m is None:
                continue
            m += wrap
            if m < prev:  # crossed midnight
                wrap += 1440
                m += 1440
            prev = m
            for d, j in day_cols.items():
                try:
                    per_day[d][m] = float(r[j]) if j < len(r) and r[j].strip() != "" else 0.0
                except ValueError:
                    per_day[d][m] = 0.0
        out[group] = per_day
    return out


def parse_time(label: str) -> Optional[int]:
    label = label.strip().upper()
    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            t = dt.datetime.strptime(label, fmt)
            return t.hour * 60 + t.minute
        except ValueError:
            continue
    return None


def fmt_time(m: float) -> str:
    m = int(m) % 1440
    h, mi = divmod(m, 60)
    return f"{h:02d}:{mi:02d}"


@st.cache_data(show_spinner=False)
def load_schedule(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@st.cache_data(show_spinner=False)
def schedule_index(schedules_dir: str, dir_mtime: float) -> Dict[int, str]:
    """taskIdx -> schedule file, from the '..._task<idx>_...' file names."""
    out: Dict[int, str] = {}
    if not schedules_dir or not os.path.isdir(schedules_dir):
        return out
    for name in sorted(os.listdir(schedules_dir)):
        m = re.search(r"_task(\d+)_", name)
        if m and name.endswith(".csv"):
            out.setdefault(int(m.group(1)), os.path.join(schedules_dir, name))
    return out


# ─── Rule-violation analysis (mirrors backend scheduler/rules.ts → phoneStretchViolations) ────

Thresholds = Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]  # before, after, max, min, adj


def _row_features(s: str):
    """Features of one agent-day string over '.', 'i' (phone), 'b' (break) slots."""
    core = s.strip(".")
    if not core:
        return None
    runs = [(ch, sum(1 for _ in g)) for ch, g in groupby(core)]
    stretches, cur = [], 0
    for ch, n in runs:
        if ch == "i":
            cur += n * SLOT_MIN
        else:
            stretches.append(cur)
            cur = 0
    stretches.append(cur)
    has_break = any(ch == "b" for ch, _ in runs)
    first_gap = (runs[0][1] * SLOT_MIN if runs[0][0] == "i" else 0) if has_break else None
    last_gap = (runs[-1][1] * SLOT_MIN if runs[-1][0] == "i" else 0) if has_break else None
    adj = []
    for t in range(1, len(runs) - 1):
        ch, n = runs[t]
        if ch == "b" and runs[t - 1][0] == "i" and runs[t + 1][0] == "i" and n * SLOT_MIN < 45:
            adj.append((runs[t - 1][1] + runs[t + 1][1]) * SLOT_MIN)
    return first_gap, last_gap, stretches, adj


@st.cache_data(show_spinner=False)
def violation_rows(path: str, mtime: float, thr: Thresholds) -> pd.DataFrame:
    """One row per working agent-day with the rule measurements and violation counts for `thr`."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    slot_cols = [c for c in df.columns if c not in ("Day", "AgentId", "AgentTier")]
    cells = np.where(df[slot_cols].to_numpy() == "", ".", df[slot_cols].to_numpy())
    strings = ["".join(r) for r in cells]
    t_before, t_after, t_max, t_min, t_adj = thr
    recs = []
    day, aid, tier = df["Day"].to_numpy(), df["AgentId"].to_numpy(), df["AgentTier"].to_numpy() if "AgentTier" in df else None
    for k, s in enumerate(strings):
        f = _row_features(s)
        if f is None:
            continue
        first_gap, last_gap, stretches, adj = f
        parts = str(aid[k]).split("|")
        recs.append({
            "Day": day[k], "AgentId": aid[k], "Tier": tier[k] if tier is not None else "",
            "Pattern": parts[1] if len(parts) > 1 else "",
            "first_gap": first_gap, "last_gap": last_gap,
            "max_stretch": max(stretches), "min_stretch": min(stretches),
            "max_adj": max(adj) if adj else 0,
            "n_before": int(first_gap is not None and t_before is not None and first_gap < t_before),
            "n_after": int(last_gap is not None and t_after is not None and last_gap < t_after),
            "n_max": sum(x > t_max for x in stretches) if t_max is not None else 0,
            "n_min": sum(x < t_min for x in stretches) if t_min is not None else 0,
            "n_adj": sum(x > t_adj for x in adj) if t_adj is not None else 0,
        })
    return pd.DataFrame(recs)


def summarise_violations(rows: pd.DataFrame) -> Dict[str, float]:
    """Agents (unique) and agent-days / events breaking each rule."""
    out: Dict[str, float] = {}
    if rows.empty:
        return out
    n_agents = rows["AgentId"].nunique()
    any_flag = np.zeros(len(rows), dtype=bool)
    per_agent_any = pd.Series(False, index=rows["AgentId"].unique())
    for k in RULE_KEYS:
        col = f"n_{k}"
        flag = rows[col] > 0
        any_flag |= flag.to_numpy()
        agents = rows.loc[flag, "AgentId"].nunique()
        per_agent_any.loc[rows.loc[flag, "AgentId"].unique()] = True
        out[f"v_agents_{k}"] = agents
        out[f"v_days_{k}"] = int(flag.sum())
        out[f"v_events_{k}"] = int(rows[col].sum())
    out["v_agents_any"] = int(per_agent_any.sum())
    out["v_days_any"] = int(any_flag.sum())
    out["v_headcount"] = n_agents
    out["v_pct_any"] = 100.0 * out["v_agents_any"] / max(1, n_agents)
    return out


@st.cache_data(show_spinner=False)
def violation_summary(path: str, mtime: float, thr: Thresholds) -> Dict[str, float]:
    return summarise_violations(violation_rows(path, mtime, thr))


def thresholds_for(row: pd.Series, mode: str, baseline: Dict[str, str]) -> Thresholds:
    """Thresholds to check against: the baseline ('default') values or the configuration's own."""
    vals: List[Optional[int]] = []
    for k in RULE_KEYS:
        param, _, _, backend_default = RULES[k]
        src = row[param] if mode == "own" and param in row.index else baseline.get(param, backend_default)
        try:
            vals.append(int(float(src)))
        except (TypeError, ValueError):
            vals.append(None)
    # The engine enforces ONE edge buffer, min(before, after), on both ends of the shift
    # (breakPlacement.ts: edgeBufferMin = Math.min(minPhoneBefore, minPhoneAfter)). Do the same so the
    # counts agree with the run's own compliance figures; identical when before == after (as in the defaults).
    if vals[0] is not None and vals[1] is not None:
        vals[0] = vals[1] = min(vals[0], vals[1])
    return tuple(vals)  # type: ignore[return-value]


# ─── Charts ──────────────────────────────────────────────────────────────────────────────────

def xlabels(df: pd.DataFrame) -> List[str]:
    return (df["label"] + "<br>" + df["workPatternConfig"]).tolist()


def stacked_bar(df: pd.DataFrame, cols: Dict[str, str], colors: Dict[str, str], title: str,
                y_title: str = "Headcount") -> go.Figure:
    """Annotated stacked bars: one bar per configuration, segment values inside, total on top."""
    x = xlabels(df)
    fig = go.Figure()
    for col, name in cols.items():
        vals = df[col].astype(float)
        fig.add_bar(x=x, y=vals, name=name, marker_color=colors[name],
                    text=[f"{int(v)}" if v > 0 else "" for v in vals], textposition="inside",
                    insidetextanchor="middle", hovertemplate=f"{name}: %{{y}}<extra>%{{x}}</extra>")
    totals = df[list(cols)].astype(float).sum(axis=1)
    fig.add_scatter(x=x, y=totals, mode="text", text=[f"<b>{int(t)}</b>" for t in totals],
                    textposition="top center", showlegend=False, hoverinfo="skip")
    fig.update_layout(barmode="stack", title=title, yaxis_title=y_title, height=460,
                      margin=dict(t=60, b=10), legend=dict(orientation="h", y=1.1, x=0))
    fig.update_yaxes(range=[0, float(totals.max()) * 1.12 if len(totals) else 1])
    return fig


def heatmap(matrix: np.ndarray, title: str, fmt: str = "{:.0f}") -> go.Figure:
    z = matrix
    text = [[fmt.format(v) if v else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=PATTERNS, y=[TIER_LABEL[t] for t in TIERS], text=text, texttemplate="%{text}",
        colorscale=[[0, "#f4f6f8"], [1, "#009983"]], showscale=True, hovertemplate="%{y} × %{x}: %{z}<extra></extra>"))
    fig.update_layout(title=title, height=330, margin=dict(t=50, b=10), yaxis=dict(autorange="reversed"))
    fig.update_xaxes(side="top")
    return fig


def peak_window(required: List[float], n: int) -> List[int]:
    """Indices of the n-interval window with the highest total demand (dashboard's computePeakIndices)."""
    ln = len(required)
    if ln == 0 or n <= 0:
        return []
    win = min(n, ln)
    s = sum(required[:win])
    best, start = s, 0
    for i in range(win, ln):
        s += required[i] - required[i - win]
        if s > best:
            best, start = s, i - win + 1
    return list(range(start, start + win)) if best > 0 else []


def net_staffing_chart(mins: List[int], required: List[float], on_phones: List[float],
                       on_shift: List[float], title: str, peak_idx: List[int], height: int = 420) -> go.Figure:
    """Plotly port of the dashboard's NetStaffingChart: Required / On phones / On shift lines with the
    break-area, surplus and deficit fills and the (optional) peak zone."""
    x = mins
    req, ph, sh = np.array(required), np.array(on_phones), np.array(on_shift)
    spl = dict(shape="spline", smoothing=0.8)
    fig = go.Figure()
    # break area: between on-phones and on-shift
    fig.add_scatter(x=x, y=ph, mode="lines", line=dict(width=0, **spl), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=sh, mode="lines", line=dict(width=0, **spl), fill="tonexty",
                    fillcolor="rgba(255,216,137,0.20)", name="Break area", hoverinfo="skip")
    # surplus: required -> max(phones, required)
    fig.add_scatter(x=x, y=req, mode="lines", line=dict(width=0, **spl), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=np.maximum(ph, req), mode="lines", line=dict(width=0, **spl), fill="tonexty",
                    fillcolor="rgba(0,229,144,0.30)", name="Surplus", hoverinfo="skip")
    # deficit: min(phones, required) -> required
    fig.add_scatter(x=x, y=np.minimum(ph, req), mode="lines", line=dict(width=0, **spl), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=req, mode="lines", line=dict(width=0, **spl), fill="tonexty",
                    fillcolor="rgba(239,68,68,0.28)", name="Deficit", hoverinfo="skip")
    fig.add_scatter(x=x, y=sh, mode="lines", name="On shift", line=dict(color=C_SHIFT, width=2, **spl))
    fig.add_scatter(x=x, y=ph, mode="lines", name="On phones", line=dict(color=C_PHONES, width=2.5, **spl))
    fig.add_scatter(x=x, y=req, mode="lines", name="Required",
                    line=dict(color=C_REQUIRED, width=1.5, dash="dot", **spl))
    if peak_idx:
        fig.add_vrect(x0=x[peak_idx[0]] - 7.5, x1=x[peak_idx[-1]] + 7.5, fillcolor="#7c3aed", opacity=0.10,
                      line_width=0, annotation_text="peak zone", annotation_position="top left")
    ticks = [m for m in x if m % 60 == 0]
    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=[fmt_time(m) for m in ticks], showgrid=True,
                     gridcolor="#eceef2")
    fig.update_yaxes(title="headcount", rangemode="tozero", gridcolor="#eceef2")
    fig.update_layout(title=title, height=height, hovermode="x unified", margin=dict(t=50, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), plot_bgcolor="white")
    return fig


def build_curve(sched: pd.DataFrame, forecast: Dict[str, Dict[str, Dict[int, float]]], day: str,
                demand_source: str):
    """(mins, required, on_phones, on_shift) for one day from the schedule CSV + forecast."""
    d = sched[sched["Day"] == day]
    slot_cols = [c for c in sched.columns if c not in ("Day", "AgentId", "AgentTier")]
    mins, prev, wrap = [], -1, 0
    for c in slot_cols:
        m = parse_time(c)
        m = (m or 0) + wrap
        if m < prev:
            wrap += 1440
            m += 1440
        prev = m
        mins.append(m)
    cells = d[slot_cols].to_numpy()
    on_phones = (cells == "i").sum(axis=0).astype(float)
    on_shift = ((cells == "i") | (cells == "b")).sum(axis=0).astype(float)
    if demand_source == "Overall column":
        req_map = forecast.get("Overall", {}).get(day, {})
        required = [req_map.get(m, 0.0) for m in mins]
    else:
        required = [sum(forecast.get(g, {}).get(day, {}).get(m, 0.0) for g in FORECAST_GROUPS) for m in mins]
    return mins, required, on_phones.tolist(), on_shift.tolist()


# ─── Sidebar ─────────────────────────────────────────────────────────────────────────────────

st.title("Christmas 2026 Grid Search Explorer")
st.caption("Christmas 2026 tiered waterfall grid search — filter, compare, and drill into configurations.")

FILTER_DEFAULTS = {"f_deficit": 0.0, "f_hc": 820, "f_flexi": 300, "f_topx": 20,
                   "f_rank": "Final headcount", "f_order": "Lowest first"}
for _k, _v in FILTER_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def reset_filters():
    for k, v in FILTER_DEFAULTS.items():
        st.session_state[k] = v
    for k in list(st.session_state):
        if k.startswith("f_p_"):
            del st.session_state[k]


with st.sidebar:
    with st.expander("📁 Data sources", expanded=False):
        candidates = sorted(
            [p for p in GS2.glob("results*.csv") if "_stageB" not in p.name],
            key=lambda p: p.stat().st_mtime, reverse=True)
        default_results = str(candidates[0]) if candidates else str(GS2 / "results.csv")
        results_path = st.text_input("Results CSV", default_results)
        fc_candidates = sorted(GS2.parent.glob("test_multi*.csv"), key=lambda p: ("v5_a" not in p.name, p.name))
        forecast_path = st.text_input("Forecast CSV (for the net staffing curve)",
                                      str(fc_candidates[0]) if fc_candidates else "")
        schedules_dir = st.text_input("Schedules folder", results_path[:-4] + "_schedules"
                                      if results_path.endswith(".csv") else "")
        demand_source = st.radio("Required demand", ["Optimiser groups (T1GD…T3ENT)", "Overall column"],
                                 help="The optimiser covers the six groups T1GD, T2GD, T3JPN, T3CJN, T3GD and T3ENT. "
                                      "The CSV's Overall column also includes other groups.")
        if st.button("🔄 Reload data", width="stretch"):
            st.cache_data.clear()
            st.rerun()

if not os.path.exists(results_path):
    st.error(f"Results file not found: {results_path}")
    st.stop()

raw = load_results(results_path, os.path.getmtime(results_path))
ok = raw[(raw["error"] == "") & (raw["hc_final"] > 0)].copy()
n_err = len(raw) - len(ok)
ok["HC_flexi"] = ok["HC_flexi"].astype(float)
sched_files = schedule_index(schedules_dir, os.path.getmtime(schedules_dir) if os.path.isdir(schedules_dir) else 0.0)

with st.sidebar:
    st.header("1 · Filter")
    st.number_input("Max total deficit (agent-intervals)", min_value=0.0, step=1.0, key="f_deficit",
                    help="Sum of shortfall over all intervals of the week (total_deficit_sum).")
    st.number_input("Max total headcount", min_value=0, step=5, key="f_hc")
    st.number_input("Max final flexi agents", min_value=0, step=10, key="f_flexi",
                    help="All patterns except 5x9 count as flexi.")
    with st.expander("Parameter filters", expanded=False):
        param_sel = {}
        for p, _ in PARAMS:
            vals = sorted(ok[p].astype(str).unique(), key=lambda s: (len(s), s))
            param_sel[p] = st.multiselect(PARAM_SHORT[p], vals, default=vals, key=f"f_p_{p}")

    st.header("2 · Rank")
    st.selectbox("Rank by", list(RANK_OPTIONS), key="f_rank",
                 help="Options marked '(from schedules)' count agents breaking the break-policy rules, measured "
                      "on the saved schedule files, so they only cover runs under the download threshold.")
    st.radio("Order", ["Lowest first", "Highest first"], horizontal=True, key="f_order")
    st.number_input("Show top X configurations", min_value=1, step=1, key="f_topx")

    st.header("3 · Rule check")
    with st.expander("Baseline ('default') parameters", expanded=False):
        st.caption("Used to highlight what each configuration changes, and as the rule values that the "
                   "'agents breaking' counts are measured against.")
        defaults = {p: st.text_input(PARAM_SHORT[p], d, key=f"d_{p}") for p, d in PARAMS}
    st.caption("Rules checked on each saved schedule: phone time before the first / after the last break, longest and "
               "shortest phone stretch, and phone time across a short break. As in the engine, the before/after "
               "rules use one buffer = the smaller of the two values.")
    check_mode = st.radio(
        "Count agents that break…", ["the baseline values", "each configuration's own values"],
        help="Baseline: how many agents of a configuration would break the default rules, even though the "
             "configuration was generated with looser ones. Own values: how many break the rules it was run with.")
    scan_limit = st.number_input("Max configurations to scan when ranking by rule breaks", min_value=20, value=300,
                                 step=20, help="Reading a schedule takes about a second; results are cached.")
    st.button("↺ Reset filters", on_click=reset_filters, width="stretch")

check_key = "own" if check_mode.startswith("each") else "baseline"
max_deficit, max_hc, max_flexi = st.session_state["f_deficit"], st.session_state["f_hc"], st.session_state["f_flexi"]
rank_name, rank_desc = st.session_state["f_rank"], st.session_state["f_order"] == "Highest first"
top_x = int(st.session_state["f_topx"])
rank_col = RANK_OPTIONS[rank_name]

# ─── Apply filters (with a funnel so empty results explain themselves) ───────────────────────

steps = [("Evaluated runs", len(raw)), ("Valid (no error)", len(ok))]
mask = ok["total_deficit_sum"] <= max_deficit
steps.append((f"Total deficit ≤ {max_deficit:g}", int(mask.sum())))
mask &= ok["hc_final"] <= max_hc
steps.append((f"Headcount ≤ {max_hc}", int(mask.sum())))
mask &= ok["HC_flexi"] <= max_flexi
steps.append((f"Flexi ≤ {max_flexi}", int(mask.sum())))
for p, sel_vals in param_sel.items():
    mask &= ok[p].astype(str).isin(sel_vals)
steps.append(("Parameter filters", int(mask.sum())))
flt = ok[mask].copy()


def schedule_path(task_idx: int) -> Optional[str]:
    return sched_files.get(int(task_idx))


def viol_for_row(row: pd.Series) -> Dict[str, float]:
    p = schedule_path(row["taskIdx"])
    if not p:
        return {}
    return violation_summary(p, os.path.getmtime(p), thresholds_for(row, check_key, defaults))


def attach_violations(df: pd.DataFrame, progress_label: str = "") -> pd.DataFrame:
    """Add v_* columns (NaN where no saved schedule)."""
    recs = []
    bar = st.progress(0.0, text=progress_label) if progress_label and len(df) > 5 else None
    for i, (_, r) in enumerate(df.iterrows()):
        recs.append(viol_for_row(r))
        if bar is not None and i % 5 == 0:
            bar.progress((i + 1) / len(df), text=f"{progress_label} {i + 1}/{len(df)}")
    if bar is not None:
        bar.empty()
    vdf = pd.DataFrame(recs, index=df.index)
    for c in ["v_agents_any", "v_pct_any"] + [f"v_agents_{k}" for k in RULE_KEYS] + \
             [f"v_events_{k}" for k in RULE_KEYS] + [f"v_days_{k}" for k in RULE_KEYS]:
        if c not in vdf:
            vdf[c] = np.nan
    return pd.concat([df, vdf], axis=1)


scan_note = ""
if rank_col in VIOLATION_RANKS:
    cand = flt.sort_values(["hc_final", "taskIdx"]).head(int(scan_limit))
    cand = cand[cand["taskIdx"].isin(sched_files.keys())]
    scanned = attach_violations(cand, "Scanning schedules")
    scanned = scanned.dropna(subset=[rank_col])
    n_unscanned = len(flt) - len(scanned)
    if n_unscanned > 0:
        scan_note = (f"Ranking by rule breaks covers {len(scanned):,} of {len(flt):,} matching configurations: "
                     f"the others have no saved schedule (headcount over the download threshold) or exceed the scan limit.")
    ranked = scanned.sort_values([rank_col, "hc_final", "taskIdx"], ascending=[not rank_desc, True, True])
    top = ranked.head(top_x).copy()
else:
    ranked = flt.sort_values([rank_col, "hc_final", "taskIdx"], ascending=[not rank_desc, True, True])
    top = attach_violations(ranked.head(top_x))

# ─── Header metrics + funnel ─────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Configurations evaluated", f"{len(raw):,}")
c2.metric("Matching the filters", f"{len(flt):,}")
c3.metric("Shown (top X)", f"{len(top):,}")
c4.metric("With a saved schedule", f"{int(flt['taskIdx'].isin(sched_files.keys()).sum()):,}",
          help="Needed for the net staffing curve and the rule-break counts.")
c5.metric("Evaluated with errors", f"{n_err:,}")

with st.expander("How the filters narrow the results", expanded=top.empty):
    funnel = pd.DataFrame(steps, columns=["Step", "Configurations left"])
    funnel["Removed by this step"] = (funnel["Configurations left"].shift(1) - funnel["Configurations left"]).fillna(0).astype(int)
    st.dataframe(funnel, hide_index=True, width="stretch")
    st.caption("The step that removes the most is the one to relax first.")
if scan_note:
    st.info(scan_note)

if top.empty:
    st.warning(
        "No configurations match these filters. Due to time constraints, only a **sample** of the full "
        f"parameter list was evaluated ({len(ok):,} valid runs), so a combination meeting every threshold may "
        "simply not have been run. Relax the thresholds (see the funnel above) or the parameter filters.")
    if len(ok):
        st.caption("Closest by headcount, ignoring all thresholds:")
        st.dataframe(ok.sort_values("hc_final").head(10)[["label", "workPatternConfig", "hc_final", "HC_flexi",
                                                          "total_deficit_sum"]], hide_index=True)
    st.stop()

tab_over, tab_cmp, tab_deep = st.tabs(["① Overview & ranking", "② Compare charts", "③ Deep dive"])

# ─── ① Overview ─────────────────────────────────────────────────────────────────────────────

with tab_over:
    st.subheader(f"Top {len(top)} by {rank_name.lower()} ({'highest' if rank_desc else 'lowest'} first)")
    st.caption("Click a row to open it in **③ Deep dive**.")
    overview = top[["label", "workPatternConfig", "hc_final", "HC_flexi", "HC_nonflexi", "total_deficit_sum",
                    "total_problem_intervals", "objective_base", "objective_flexi_surcharge",
                    "efficiency_efficiencyRatio", "efficiency_surplusStdevPct", "v_agents_any", "v_agents_adj"]].rename(columns={
        "label": "Task ID",
        "workPatternConfig": "Pattern Set",
        "hc_final": "Total Headcount", "HC_flexi": "Flexi",
        "HC_nonflexi": "Non-flexi (5x9)",
        "total_deficit_sum": "Total deficit",
        "total_problem_intervals": "Number of deficit intervals",
        "objective_base": "Cost-weighted (no flexi premium)",
        "objective_flexi_surcharge": "Cost-weighted (w flexi premium)",
        "efficiency_efficiencyRatio": "Efficiency",
        "efficiency_surplusStdevPct": "Surplus stdev % (higher is better)",
        "v_agents_any": "Agents breaking any rule",
        "v_agents_adj": f"Agents breaking {RULES['adj'][2].lower()}"})
    event = st.dataframe(
        overview, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key="overview_table",
        column_config={
            "Cost-weighted (no flexi premium)": st.column_config.NumberColumn(format="%.1f"),
            "Cost-weighted (w flexi premium)": st.column_config.NumberColumn(format="%.1f"),
            "Efficiency": st.column_config.NumberColumn(format="%.3f"),
            "Surplus stdev % (higher is better)": st.column_config.NumberColumn(format="%.2f"),
            "Agents breaking any rule": st.column_config.NumberColumn(
                format="%d", help=f"Agents with ≥1 violation, measured against {check_mode}. Empty = no saved schedule."),
            f"Agents breaking {RULES['adj'][2].lower()}": st.column_config.NumberColumn(format="%d")})
    sel_rows = event.selection.rows if event and event.selection else []
    if sel_rows:
        picked = top.iloc[sel_rows[0]]["label"]
        if st.session_state.get("_last_table_pick") != picked:
            st.session_state["_last_table_pick"] = picked
            st.session_state["_pending_pick"] = picked
        st.success(f"Selected {picked} — open the **③ Deep dive** tab.")
    d1, d2 = st.columns([1, 5])
    d1.download_button("⬇ Top X as CSV", top.drop(columns=["label"]).to_csv(index=False).encode(),
                       file_name="top_configurations.csv", mime="text/csv")
    d2.download_button("⬇ All matching as CSV", flt.drop(columns=["label"]).to_csv(index=False).encode(),
                       file_name="matching_configurations.csv", mime="text/csv")

# ─── ② Compare ──────────────────────────────────────────────────────────────────────────────

with tab_cmp:
    st.caption(f"All charts show the top {len(top)} configurations from the Overview.")
    (t_tier, t_pat, t_heat, t_par, t_viol, t_cost, t_eff) = st.tabs(
        ["Tier distribution", "Work patterns", "Pattern × tier heatmap", "Parameters vs default",
         "Rule breaks vs default", "Deficit & cost", "Staffing efficiency"])

    with t_tier:
        plot(stacked_bar(top, {f"HC_{t}": TIER_LABEL[t] for t in TIERS}, TIER_COLORS,
                                    "Headcount per tier"), width="stretch")

    with t_pat:
        plot(stacked_bar(
            top, {f"HC_pattern_{p}": p for p in PATTERNS}, dict(zip(PATTERNS, PATTERN_COLORS)),
            "Headcount per work pattern (5x9 is non-flexi; every other pattern is flexi)"), width="stretch")

    with t_heat:
        mode = st.radio("Show", ["Mean over shown configurations", "Sum over shown configurations",
                                 "A single configuration"], horizontal=True)
        cols = [[f"HC_{t}__{p}" for p in PATTERNS] for t in TIERS]
        if mode.startswith("A single"):
            pick = st.selectbox("Configuration", top["label"] + "  " + top["workPatternConfig"], key="heat_pick")
            row = top[top["label"] == pick.split()[0]].iloc[0]
            m = np.array([[row[c] for c in r] for r in cols], dtype=float)
            plot(heatmap(m, f"Agents: work pattern × tier — {pick}"), width="stretch")
        else:
            arr = np.array([[top[c].astype(float).mean() if mode.startswith("Mean") else top[c].astype(float).sum()
                             for c in r] for r in cols])
            plot(heatmap(arr, f"Agents: work pattern × tier — {mode.lower()}",
                                    "{:.1f}" if mode.startswith("Mean") else "{:.0f}"), width="stretch")

    with t_par:
        st.caption("Highlighted cells differ from the baseline in the sidebar.")
        pt = top[["label"] + [p for p, _ in PARAMS]].copy()
        pt["# changed"] = [sum(norm(r[p]) != norm(defaults[p]) for p, _ in PARAMS) for _, r in pt.iterrows()]
        base_row = pd.DataFrame([{"label": "DEFAULT", **{p: defaults[p] for p, _ in PARAMS}, "# changed": 0}])
        shown = pd.concat([base_row, pt], ignore_index=True)
        shown = shown.rename(columns={p: PARAM_SHORT[p] for p, _ in PARAMS})

        def hl(row):
            if row["label"] == "DEFAULT":
                return ["font-weight:bold;background:#eceef2"] * len(row)
            out = []
            for c in shown.columns:
                orig = next((p for p, _ in PARAMS if PARAM_SHORT[p] == c), None)
                out.append("background:#fff3cd" if orig and norm(row[c]) != norm(defaults[orig]) else "")
            return out

        st.dataframe(shown.astype(str).style.apply(hl, axis=1), hide_index=True, width="stretch")

    with t_viol:
        st.markdown(f"**How many agents would break each rule** — measured against **{check_mode}**.")
        th_txt = ", ".join(f"{RULES[k][2]} = {defaults.get(RULES[k][0], RULES[k][3])}" for k in RULE_KEYS) \
            if check_key == "baseline" else "each configuration's own parameters"
        st.caption(f"Rule values: {th_txt}. An agent counts once per rule if any day of their week breaks it. "
                   "Needs the saved schedule, so configurations above the download threshold are skipped.")
        vt = top.dropna(subset=["v_agents_any"])
        if vt.empty:
            st.info("None of the shown configurations has a saved schedule.")
        else:
            x = xlabels(vt)
            fig = go.Figure()
            colors = ["#009983", "#6c7382", "#e6b75c", "#7c3aed", C_BAD]
            for k, colr in zip(RULE_KEYS, colors):
                vals = vt[f"v_agents_{k}"].astype(int)
                fig.add_bar(x=x, y=vals, name=RULES[k][2], marker_color=colr,
                            text=[str(v) if v else "" for v in vals], textposition="outside")
            fig.add_scatter(x=x, y=vt["v_agents_any"], mode="markers+text", name="Any rule (unique agents)",
                            marker=dict(symbol="diamond", size=10, color=C_REQUIRED),
                            text=vt["v_agents_any"].astype(int), textposition="top center")
            fig.update_layout(barmode="group", height=480, yaxis_title="Agents breaking the rule",
                              legend=dict(orientation="h", y=1.12, x=0), margin=dict(t=60, b=10))
            plot(fig, width="stretch")
            tbl = vt[["label", "workPatternConfig", "hc_final", "v_agents_any", "v_pct_any"] +
                     [f"v_agents_{k}" for k in RULE_KEYS]].rename(columns={
                "label": "Task ID", "workPatternConfig": "Pattern Set", "hc_final": "Headcount",
                "v_agents_any": "Agents breaking any rule", "v_pct_any": "% of headcount",
                **{f"v_agents_{k}": RULES[k][2] for k in RULE_KEYS}})
            st.dataframe(tbl, hide_index=True, width="stretch",
                         column_config={"% of headcount": st.column_config.NumberColumn(format="%.1f%%")})

    with t_cost:
        x = xlabels(top)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=x, y=top["objective_base"], name="Cost-weighted", marker_color=C_PHONES,
                    text=top["objective_base"].round(0).astype(int), textposition="inside")
        fig.add_bar(x=x, y=top["objective_flexi_surcharge"] - top["objective_base"], name="Flexi surcharge (1.15×)",
                    marker_color=C_SHIFT, text=(top["objective_flexi_surcharge"] - top["objective_base"]).round(0).astype(int),
                    textposition="inside")
        fig.add_scatter(x=x, y=top["total_deficit_sum"], name="Total deficit", mode="lines+markers+text",
                        text=top["total_deficit_sum"].astype(int), textposition="top center",
                        line=dict(color=C_BAD), secondary_y=True)
        fig.update_layout(barmode="stack", height=460, title="Cost-weighted output and deficit",
                          legend=dict(orientation="h", y=1.1, x=0), margin=dict(t=60, b=10))
        fig.update_yaxes(title_text="cost-weighted headcount", secondary_y=False)
        fig.update_yaxes(title_text="deficit (agent-intervals)", secondary_y=True, rangemode="tozero")
        plot(fig, width="stretch")
        dcols = ["label", "total_deficit_sum", "total_problem_intervals", "coverage_worstDeficitAbsolute",
                 "coverage_worstDeficitPct"] + [f"deficits_{DAY_ABBR[d]}" for d in DAYS]
        st.dataframe(top[dcols].rename(columns={
            "label": "task", "total_deficit_sum": "deficit", "total_problem_intervals": "deficit intervals",
            "coverage_worstDeficitAbsolute": "worst deficit", "coverage_worstDeficitPct": "worst deficit %",
            **{f"deficits_{DAY_ABBR[d]}": f"{d[:3]} intervals" for d in DAYS}}), hide_index=True,
            width="stretch")

    with t_eff:
        top["lb_over_hc"] = top["efficiency_lowerBound"] / top["hc_final"]
        x = xlabels(top)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Efficiency ratio · lower-bound / headcount",
                                                            "Surplus spread (stdev %)"))
        fig.add_bar(x=x, y=top["efficiency_efficiencyRatio"], name="Efficiency ratio", marker_color=C_PHONES,
                    text=top["efficiency_efficiencyRatio"].round(3), textposition="outside", row=1, col=1)
        fig.add_bar(x=x, y=top["lb_over_hc"], name="Lower bound / headcount", marker_color=C_SHIFT,
                    text=top["lb_over_hc"].round(3), textposition="outside", row=1, col=1)
        fig.add_bar(x=x, y=top["efficiency_surplusStdevPct"], name="Surplus stdev %", marker_color="#7c3aed",
                    text=top["efficiency_surplusStdevPct"].round(2), textposition="outside", row=1, col=2)
        fig.update_layout(barmode="group", height=470, legend=dict(orientation="h", y=1.15, x=0), margin=dict(t=80, b=10))
        plot(fig, width="stretch")
        st.dataframe(top[["label", "hc_final", "efficiency_lowerBound", "lb_over_hc", "efficiency_efficiencyRatio",
                          "efficiency_readyToRequiredRatio", "efficiency_surplusStdevPct", "aqol_complianceRate"]].rename(
            columns={"label": "task", "hc_final": "headcount", "efficiency_lowerBound": "lower bound",
                     "lb_over_hc": "LB / headcount", "efficiency_efficiencyRatio": "efficiency ratio",
                     "efficiency_readyToRequiredRatio": "ready / required", "efficiency_surplusStdevPct": "surplus stdev %",
                     "aqol_complianceRate": "compliance"}), hide_index=True, width="stretch")

# ─── ③ Deep dive ────────────────────────────────────────────────────────────────────────────

with tab_deep:
    pool_all = st.checkbox("Choose from all evaluated configurations (not only the shown ones)", value=False)
    pool = ok.sort_values("hc_final") if pool_all else top
    options = (pool["label"] + "  ·  " + pool["workPatternConfig"] + "  ·  HC " +
               pool["hc_final"].astype(int).astype(str)).tolist()
    pending = st.session_state.pop("_pending_pick", None)
    if pending:
        match = [o for o in options if o.split("  ·  ")[0] == pending]
        if match:
            st.session_state["deep_pick"] = match[0]
    if st.session_state.get("deep_pick") not in options:
        st.session_state["deep_pick"] = options[0]
    choice = st.selectbox("Configuration", options, key="deep_pick")
    sel = pool.iloc[options.index(choice)]
    task_idx = int(sel["taskIdx"])
    sched_path = schedule_path(task_idx)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Final headcount", int(sel["hc_final"]),
              help=f"Stage B {int(sel['hc_stage_b'])} → after drop {int(sel['hc_after_greedy_drop'])}")
    m2.metric("Flexi / non-flexi", f"{int(sel['HC_flexi'])} / {int(sel['HC_nonflexi'])}")
    m3.metric("Total deficit", f"{sel['total_deficit_sum']:.0f}", help=f"{int(sel['total_problem_intervals'])} intervals")
    m4.metric("Cost-weighted", f"{sel['objective_base']:.1f}", help=f"with flexi surcharge: {sel['objective_flexi_surcharge']:.1f}")
    m5.metric("Efficiency ratio", f"{sel['efficiency_efficiencyRatio']:.3f}",
              help=f"surplus stdev {sel['efficiency_surplusStdevPct']:.2f}%")

    s_over, s_heat, s_viol, s_curve = st.tabs(["Set-up", "Pattern × tier", "Rule breaks", "Net staffing curve"])

    with s_over:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Parameters vs default**")
            ptab = pd.DataFrame([{"parameter": PARAM_SHORT[p], "value": str(sel[p]), "default": defaults[p],
                                  "changed": "●" if norm(sel[p]) != norm(defaults[p]) else ""} for p, _ in PARAMS])
            st.dataframe(ptab, hide_index=True, width="stretch")
        with right:
            st.markdown("**Headcount per tier × work pattern**")
            mat = pd.DataFrame([[int(sel[f"HC_{t}__{p}"]) for p in PATTERNS] for t in TIERS],
                               index=[TIER_LABEL[t] for t in TIERS], columns=PATTERNS)
            mat["total"] = mat.sum(axis=1)
            mat.loc["total"] = mat.sum(axis=0)
            st.dataframe(mat, width="stretch")
        st.markdown("**Deficits by day**")
        st.dataframe(pd.DataFrame({"day": DAYS,
                                   "deficit intervals": [int(sel[f"deficits_{DAY_ABBR[d]}"]) for d in DAYS],
                                   "worst deficit": [sel[f"worst_deficit_{DAY_ABBR[d]}"] for d in DAYS]}).astype(str).T,
                     width="stretch")

    with s_heat:
        hm = np.array([[sel[f"HC_{t}__{p}"] for p in PATTERNS] for t in TIERS], dtype=float)
        plot(heatmap(hm, f"Agents: work pattern × tier — {choice.split('·')[0].strip()} "
                         f"({choice.split('·')[1].strip()})"), width="stretch")
        st.caption(f"{int(hm.sum())} agents in total. 5x9 is the only non-flexi pattern; every other pattern "
                   "counts towards the flexi cap. Empty cells have no agents.")

    with s_viol:
        if not sched_path:
            st.info("No saved schedule for this configuration, so rule breaks can't be counted. Schedules are only "
                    "written for runs whose final headcount is below the download threshold given to `run.sh`.")
        else:
            thr = thresholds_for(sel, check_key, defaults)
            all_rows = violation_rows(sched_path, os.path.getmtime(sched_path), thr)
            days_avail = [d for d in DAYS if d in set(all_rows["Day"])]
            day_default = [d for d in ("Thursday", "Friday") if d in days_avail] or days_avail
            viol_days = st.multiselect(
                "Days to check", days_avail, default=day_default, key="viol_days",
                help="Only these days' schedules are checked. Defaults to Thursday and Friday; "
                     "select all seven for the whole week.")
            if not viol_days:
                st.warning("No day selected — showing the whole week.")
                viol_days = days_avail
            rows = all_rows[all_rows["Day"].isin(viol_days)]
            scope = "the whole week" if set(viol_days) == set(days_avail) else ", ".join(d[:3] for d in viol_days)
            summ = summarise_violations(rows)
            hc = int(summ.get("v_headcount", 0))
            st.markdown(f"Measured against **{check_mode}**, on **{scope}**. A rule is broken when …")
            a, b, c = st.columns(3)
            a.metric("Agents breaking any rule", f"{int(summ['v_agents_any'])} of {hc}",
                     help="Unique agents with at least one violation on the selected days. "
                          "'Of' counts the agents scheduled on those days.")
            b.metric("% of agents on these days", f"{summ['v_pct_any']:.1f}%")
            c.metric("Agent-days with a violation", f"{int(summ['v_days_any'])}")
            rtab = []
            for k, t in zip(RULE_KEYS, thr):
                worst = {"before": ("first_gap", "min"), "after": ("last_gap", "min"), "max": ("max_stretch", "max"),
                         "min": ("min_stretch", "min"), "adj": ("max_adj", "max")}[k]
                ser = rows[worst[0]].dropna()
                rtab.append({
                    "Rule": RULES[k][2], "Breaks when": f"{RULES[k][1]} {t if t is not None else '—'} min",
                    "Agents": int(summ[f"v_agents_{k}"]), "% of agents on these days": 100.0 * summ[f"v_agents_{k}"] / max(1, hc),
                    "Agent-days": int(summ[f"v_days_{k}"]), "Violating stretches": int(summ[f"v_events_{k}"]),
                    "Worst value (min)": (int(ser.min() if worst[1] == "min" else ser.max()) if len(ser) else None)})
            st.dataframe(pd.DataFrame(rtab), hide_index=True, width="stretch",
                         column_config={"% of agents on these days": st.column_config.NumberColumn(format="%.1f%%")})

            focus = st.selectbox("Look closer at", RULE_KEYS, index=RULE_KEYS.index("adj"),
                                 format_func=lambda k: RULES[k][2], key="viol_focus")
            fv = rows[rows[f"n_{focus}"] > 0]
            vc1, vc2 = st.columns(2)
            with vc1:
                st.markdown("**Agents breaking this rule, by tier and pattern**")
                if fv.empty:
                    st.success("No agent breaks this rule.")
                else:
                    by_t = fv.groupby("Tier")["AgentId"].nunique().rename(index=lambda t: TIER_LABEL.get(t, t))
                    by_p = fv.groupby("Pattern")["AgentId"].nunique()
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("by tier", "by pattern"))
                    fig.add_bar(x=by_t.index.tolist(), y=by_t.values, marker_color=C_BAD, text=by_t.values,
                                textposition="outside", showlegend=False, row=1, col=1)
                    fig.add_bar(x=by_p.index.tolist(), y=by_p.values, marker_color=C_SHIFT, text=by_p.values,
                                textposition="outside", showlegend=False, row=1, col=2)
                    fig.update_layout(height=320, margin=dict(t=40, b=10))
                    plot(fig, width="stretch")
            with vc2:
                meas = {"before": "first_gap", "after": "last_gap", "max": "max_stretch", "min": "min_stretch",
                        "adj": "max_adj"}[focus]
                st.markdown(f"**Distribution of {RULES[focus][2].lower()} over all agent-days**")
                series = rows[meas].dropna()
                if focus == "adj":
                    series = series[series > 0]
                fig = go.Figure(go.Histogram(x=series, xbins=dict(size=SLOT_MIN), marker_color=C_PHONES))
                t_val = thr[RULE_KEYS.index(focus)]
                if t_val is not None:
                    fig.add_vline(x=t_val, line_dash="dash", line_color=C_BAD, annotation_text=f"limit {t_val}")
                fig.update_layout(height=320, margin=dict(t=10, b=10), xaxis_title="minutes", yaxis_title="agent-days",
                                  bargap=0.05)
                plot(fig, width="stretch")
            if not fv.empty:
                with st.expander(f"Worst offenders — {RULES[focus][2].lower()}"):
                    meas_col = meas
                    asc = focus in ("before", "after", "min")
                    worst_rows = fv.sort_values(meas_col, ascending=asc).head(25)[
                        ["AgentId", "Tier", "Pattern", "Day", "first_gap", "last_gap", "max_stretch", "min_stretch", "max_adj"]]
                    worst_rows["Tier"] = worst_rows["Tier"].map(lambda t: TIER_LABEL.get(t, t))
                    st.dataframe(worst_rows.rename(columns={
                        "first_gap": "phone before 1st break", "last_gap": "phone after last break",
                        "max_stretch": "longest phone stretch", "min_stretch": "shortest phone stretch",
                        "max_adj": "largest adjacent total"}), hide_index=True, width="stretch")

    with s_curve:
        if not sched_path:
            st.info("No saved schedule for this configuration. Schedules are only written for runs whose final "
                    "headcount is below the download threshold given to `run.sh` (default 900). Pick a configuration "
                    "below that threshold, or re-run it with a higher threshold.")
        elif not forecast_path or not os.path.exists(forecast_path):
            st.warning("Set a valid forecast CSV under 📁 Data sources to draw the required-demand line.")
        else:
            forecast = load_forecast(forecast_path, os.path.getmtime(forecast_path))
            sched = load_schedule(sched_path, os.path.getmtime(sched_path))
            src = "Overall column" if demand_source == "Overall column" else "groups"
            view = st.radio("Days", ["All days", "One day"], horizontal=True)
            days_here = [d for d in DAYS if d in set(sched["Day"])]
            worst_day = max(days_here, key=lambda d: sel[f"deficits_{DAY_ABBR[d]}"]) if days_here else None
            peak_n = int(sel["peakIntervalCount"])
            if view == "One day":
                day = st.selectbox("Day", days_here, index=days_here.index(worst_day) if worst_day else 0,
                                   help="Defaults to the day with the most deficit intervals.")
                mins, req, ph, sh = build_curve(sched, forecast, day, src)
                plot(net_staffing_chart(mins, req, ph, sh, f"{day} — {choice.split('·')[0].strip()}",
                                                   peak_window(req, peak_n)), width="stretch")
            else:
                for day in days_here:
                    mins, req, ph, sh = build_curve(sched, forecast, day, src)
                    plot(net_staffing_chart(mins, req, ph, sh, day, peak_window(req, peak_n), height=300),
                                    width="stretch")
            st.caption(f"Schedule file: {os.path.basename(sched_path)} · peak zone = {peak_n} highest-demand consecutive "
                       "intervals (no breaks placed there).")
