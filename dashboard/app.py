"""
Live demo dashboard for the weather-arbitrage bot.

Run:
    .venv/bin/streamlit run dashboard/app.py

Auto-refreshes every 15s by default. Three tabs:
  • Overview — bot health, cumulative paper P&L, recent activity
  • Live Markets — per (city, date) Kalshi vs Polymarket vs forecast view
  • Performance — predicted vs realized edge, win-rate by city
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Make project imports work when streamlit imports this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import data
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range


REFRESH_SEC = 15


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
  page_title="Weather Arbitrage Bot",
  page_icon="🌤",
  layout="wide",
)

# In-place rerun on a timer — no browser reload, so the active tab and any
# selectbox state survive. The component returns an integer (refresh count)
# we don't need to use; setting `key` keeps it stable across reruns.
st_autorefresh(interval=REFRESH_SEC * 1000, key="dashboard_autorefresh")

st.markdown("""
<style>
  /* tighten things up for projector display */
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
  [data-testid="stMetricValue"] { font-size: 1.7rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────

def _humanize_age(iso_ts: str | None) -> str:
  if not iso_ts:
    return "—"
  ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
  if ts.tzinfo is None:
    ts = ts.replace(tzinfo=timezone.utc)
  delta = datetime.now(timezone.utc) - ts
  s = int(delta.total_seconds())
  if s < 60: return f"{s}s ago"
  if s < 3600: return f"{s // 60}m ago"
  if s < 86400: return f"{s // 3600}h ago"
  return f"{s // 86400}d ago"


def _bracket_bars(brackets: list[dict], venue: str) -> pd.DataFrame:
  """One row per bracket — preserves the venue's actual quote granularity
  rather than inventing per-degree structure. Returns center + width so a
  Plotly Bar trace can render each bracket as a single wide bar spanning
  its degree range. Height = implied prob / width (density per °F),
  which puts it in the same units as the forecast PDF for honest overlay."""
  rows = []
  for b in brackets:
    if venue == "kalshi":
      degrees = _parse_kalshi_range(b.get("subtitle", ""))
      label = b.get("subtitle", "")
    else:
      degrees = _parse_polymarket_range(b.get("question", ""))
      label = b.get("question", "")
    if not degrees:
      continue
    bid = b.get("best_yes_bid")
    ask = b.get("best_yes_ask")
    if bid is None and ask is None:
      continue
    if bid is not None and ask is not None:
      mid = (bid + ask) / 2.0
    else:
      mid = bid if bid is not None else ask
    width = len(degrees)
    center = (degrees[0] + degrees[-1]) / 2.0
    rows.append({"center": center, "width": width,
                 "density": mid / width, "implied_prob": mid,
                 "label": label, "bid": bid, "ask": ask, "venue": venue})
  return pd.DataFrame(rows)


# ── Header ───────────────────────────────────────────────────────────────

h = data.health()

left, right = st.columns([4, 1])
with left:
  st.title("Weather Arbitrage Bot")
  st.caption("Cross-venue hedged-pair strategy on Kalshi × Polymarket "
             "daily-high temperature markets.")
with right:
  st.write("")
  if st.button("Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
  st.caption(f"Auto-refresh every {REFRESH_SEC}s")

# Health row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cities live", len(h["cities"]))
m2.metric("Last poll", _humanize_age(h["last_snapshot_utc"]))
m3.metric("Last forecast", _humanize_age(h["last_forecast_utc"]))
m4.metric("Last fill", _humanize_age(h["last_fill_utc"]))
m5.metric("Last settlement", _humanize_age(h["last_settlement_utc"]))


# ── Tabs ─────────────────────────────────────────────────────────────────

tab_overview, tab_live, tab_perf, tab_back = st.tabs(
  ["Overview", "Live Markets", "Performance", "Backtest"])


# ──────────────────────────────  OVERVIEW  ───────────────────────────────

with tab_overview:
  pnl = data.pnl_frame()
  fills = data.fills_frame()

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Orders logged", f"{h['orders']:,}")
  c2.metric("Fills recorded", f"{h['fills']:,}")
  c3.metric("Settled positions", f"{len(pnl):,}")

  if not pnl.empty:
    realized = pnl["dollars_realized"].sum()
    win_rate = pnl["won_bool"].mean()
    c4.metric("Paper P&L", f"${realized:,.2f}",
              delta=f"{win_rate * 100:.1f}% wins")
  else:
    c4.metric("Paper P&L", "$0.00")

  st.divider()

  cL, cR = st.columns([3, 2], gap="large")

  with cL:
    st.subheader("Cumulative paper P&L")
    if pnl.empty:
      st.info("No settled positions yet — waiting on market resolutions.")
    else:
      df = pnl.sort_values("ts").copy()
      df["cumulative"] = df["dollars_realized"].cumsum()
      fig = px.line(
        df, x="ts", y="cumulative",
        labels={"ts": "", "cumulative": "Cumulative $ (paper)"},
      )
      fig.update_traces(line=dict(width=2.5))
      fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
      fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
      st.plotly_chart(fig, use_container_width=True)

  with cR:
    st.subheader("Recent fills")
    if fills.empty:
      st.info("No fills yet.")
    else:
      view = fills.head(15)[
        ["ts", "city", "venue", "side", "action", "fill_price",
         "fill_size", "expected_edge"]
      ].copy()
      view["ts"] = view["ts"].dt.strftime("%H:%M:%S")
      view = view.rename(columns={
        "ts": "time", "fill_price": "px", "fill_size": "sz",
        "expected_edge": "edge"})
      st.dataframe(view, use_container_width=True, hide_index=True,
                   height=420)


# ───────────────────────────  LIVE MARKETS  ──────────────────────────────

with tab_live:
  pairs = data.available_city_dates()
  if not pairs:
    st.warning("No (city, date) has snapshots from both venues yet.")
  else:
    options = [f"{c} — {d}" for c, d in pairs]
    pick = st.selectbox("Market", options, index=0)
    city, date = pick.split(" — ")

    k_brackets, k_ts = data.latest_brackets(city, date, "kalshi")
    p_brackets, p_ts = data.latest_brackets(city, date, "polymarket")
    pdf, fc_source, fc_ts = data.latest_forecast(city, date)
    k_settle, p_settle = data.settlement_for(city, date)

    cap = []
    if k_ts: cap.append(f"Kalshi snapshot: {_humanize_age(k_ts)}")
    if p_ts: cap.append(f"Polymarket snapshot: {_humanize_age(p_ts)}")
    if fc_ts: cap.append(f"Forecast ({fc_source}): {_humanize_age(fc_ts)}")
    if k_settle is not None or p_settle is not None:
      bits = []
      if k_settle is not None: bits.append(f"Kalshi {k_settle}°F")
      if p_settle is not None: bits.append(f"Polymarket {p_settle}°F")
      cap.append("Settled: " + " / ".join(bits))
    st.caption("  ·  ".join(cap))

    df_k = _bracket_bars(k_brackets, "kalshi")
    df_p = _bracket_bars(p_brackets, "polymarket")

    # Bracket-level density vs forecast PDF ────────────────────────────
    # Each bar spans its bracket's degree range. Height is probability
    # density per °F = implied_prob / bracket_width, matching the units
    # of the per-degree forecast PDF so overlay comparison is honest —
    # no flat-split fiction within brackets.
    fig = go.Figure()
    if not df_k.empty:
      fig.add_trace(go.Bar(
        x=df_k["center"], y=df_k["density"], width=df_k["width"],
        name="Kalshi YES (density)",
        marker_color="#3B82F6", opacity=0.55,
        marker_line=dict(width=1.5, color="#1E3A8A"),
        customdata=df_k[["label", "implied_prob", "width"]],
        hovertemplate=("<b>Kalshi</b> %{customdata[0]}<br>"
                       "Implied prob: %{customdata[1]:.3f}<br>"
                       "Width: %{customdata[2]:.0f}°F<br>"
                       "Density: %{y:.3f}/°F<extra></extra>"),
      ))
    if not df_p.empty:
      fig.add_trace(go.Bar(
        x=df_p["center"], y=df_p["density"], width=df_p["width"],
        name="Polymarket YES (density)",
        marker_color="#F59E0B", opacity=0.55,
        marker_line=dict(width=1.5, color="#92400E"),
        customdata=df_p[["label", "implied_prob", "width"]],
        hovertemplate=("<b>Polymarket</b> %{customdata[0]}<br>"
                       "Implied prob: %{customdata[1]:.3f}<br>"
                       "Width: %{customdata[2]:.0f}°F<br>"
                       "Density: %{y:.3f}/°F<extra></extra>"),
      ))
    if pdf:
      xs = sorted(pdf.keys())
      ys = [pdf[x] for x in xs]
      fig.add_trace(go.Scatter(
        x=xs, y=ys, name=f"Forecast PDF ({fc_source})",
        mode="lines+markers", line=dict(width=3, color="#10B981"),
        marker=dict(size=7),
      ))
    # Settled-high overlays — vertical lines for past resolutions so the
    # forecast and venue brackets can be evaluated against the outcome.
    if k_settle is not None:
      fig.add_vline(x=k_settle, line_color="#1E3A8A", line_width=2,
                    annotation_text=f"Kalshi settled {k_settle}°F",
                    annotation_position="top",
                    annotation_font_color="#1E3A8A")
    if p_settle is not None and p_settle != k_settle:
      fig.add_vline(x=p_settle, line_color="#92400E", line_width=2,
                    line_dash="dash",
                    annotation_text=f"Polymarket settled {p_settle}°F",
                    annotation_position="bottom",
                    annotation_font_color="#92400E")
    fig.update_layout(
      barmode="overlay",
      xaxis_title="Daily high (°F)",
      yaxis_title="Probability density per °F",
      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
      margin=dict(l=0, r=0, t=10, b=0),
      height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bracket-level book ────────────────────────────────────────────────
    cL, cR = st.columns(2)
    with cL:
      st.markdown("**Kalshi brackets**")
      if k_brackets:
        st.dataframe(
          pd.DataFrame([{
            "label": b.get("subtitle", ""),
            "yes_bid": b.get("best_yes_bid"),
            "yes_ask": b.get("best_yes_ask"),
            "bid_sz": b.get("best_yes_bid_size"),
            "ask_sz": b.get("best_yes_ask_size"),
          } for b in k_brackets]),
          use_container_width=True, hide_index=True)
      else:
        st.info("No Kalshi snapshot.")
    with cR:
      st.markdown("**Polymarket brackets**")
      if p_brackets:
        st.dataframe(
          pd.DataFrame([{
            "label": b.get("question", ""),
            "yes_bid": b.get("best_yes_bid"),
            "yes_ask": b.get("best_yes_ask"),
            "bid_sz": b.get("best_yes_bid_size"),
            "ask_sz": b.get("best_yes_ask_size"),
          } for b in p_brackets]),
          use_container_width=True, hide_index=True)
      else:
        st.info("No Polymarket snapshot.")


# ───────────────────────────  PERFORMANCE  ───────────────────────────────

with tab_perf:
  pnl = data.pnl_frame()
  if pnl.empty:
    st.info("No settled positions yet.")
  else:
    by_city = (pnl.groupby("city")
                  .agg(trades=("won_bool", "size"),
                       wins=("won_bool", "sum"),
                       realized=("dollars_realized", "sum"),
                       predicted=("dollars_predicted", "sum"))
                  .reset_index())
    by_city["win_rate"] = by_city["wins"] / by_city["trades"]

    cL, cR = st.columns(2)

    with cL:
      st.subheader("Realized P&L by city")
      fig = px.bar(
        by_city, x="city", y="realized",
        color="realized", color_continuous_scale="RdYlGn",
        labels={"realized": "Realized $ (paper)"},
      )
      fig.update_layout(coloraxis_showscale=False,
                        margin=dict(l=0, r=0, t=10, b=0), height=340)
      st.plotly_chart(fig, use_container_width=True)

    with cR:
      st.subheader("Win rate by city")
      fig = px.bar(
        by_city, x="city", y="win_rate", text="wins",
        labels={"win_rate": "Win rate"},
      )
      fig.update_traces(texttemplate="%{text}/%{customdata}",
                        customdata=by_city["trades"])
      fig.update_layout(yaxis_tickformat=".0%",
                        margin=dict(l=0, r=0, t=10, b=0), height=340)
      st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Predicted edge vs realized $")
    fig = px.scatter(
      pnl, x="dollars_predicted", y="dollars_realized",
      color="city", symbol="venue",
      hover_data=["side", "intended_price", "intended_size",
                  "settlement_date"],
      labels={"dollars_predicted": "Predicted edge ($)",
              "dollars_realized": "Realized ($)"},
    )
    lo = float(min(pnl["dollars_predicted"].min(),
                   pnl["dollars_realized"].min()))
    hi = float(max(pnl["dollars_predicted"].max(),
                   pnl["dollars_realized"].max()))
    fig.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi,
                  line=dict(dash="dot", color="gray"))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────  BACKTEST  ───────────────────────────────

with tab_back:
  bt = data.backtest_frame()
  if bt.empty:
    st.info(
      "No backtest results yet. Run `python -m bin.backtest_strategy` "
      "to replay the strategy against historical snapshots.")
  else:
    st.caption(
      "Replays `find_hedged_pairs` at every paired snapshot through each "
      "(city, date), with per-leg dedup mirroring live behaviour. "
      "Closes resolve against actual settlements where available; "
      "otherwise the highest-probability bracket in the day's final "
      "snapshot is used as a synthetic close.")

    total = bt["realized_total"].sum()
    n = len(bt)
    wr = bt["won"].mean()
    days = bt[["city", "date"]].drop_duplicates().shape[0]
    avg_per_pair = total / n if n else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs accepted", f"{n:,}",
              delta=f"{days} (city, date)s")
    c2.metric("Realized P&L", f"${total:,.2f}",
              delta=f"${avg_per_pair:.2f}/pair")
    c3.metric("Win rate", f"{wr * 100:.1f}%")
    c4.metric("Avg size", f"{bt['size'].mean():.0f} contracts")

    st.divider()
    cL, cR = st.columns([3, 2], gap="large")

    with cL:
      st.subheader("Cumulative simulated P&L")
      df = bt.sort_values("snapshot_ts").copy()
      df["cumulative"] = df["realized_total"].cumsum()
      fig = px.line(
        df, x="snapshot_ts", y="cumulative",
        labels={"snapshot_ts": "", "cumulative": "Cumulative $"},
      )
      fig.update_traces(line=dict(width=2.5))
      fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
      fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
      st.plotly_chart(fig, use_container_width=True)

    with cR:
      st.subheader("By city")
      by_city = (bt.groupby("city")
                   .agg(pairs=("won", "size"),
                        wins=("won", "sum"),
                        realized=("realized_total", "sum"))
                   .reset_index()
                   .sort_values("realized", ascending=False))
      by_city["win_rate"] = by_city["wins"] / by_city["pairs"]
      st.dataframe(
        by_city.style.format({"realized": "${:,.2f}",
                              "win_rate": "{:.0%}"}),
        use_container_width=True, hide_index=True, height=320)

    st.divider()

    cL, cR = st.columns(2, gap="large")
    with cL:
      st.subheader("Predicted edge vs realized $")
      fig = px.scatter(
        bt, x="predicted_edge", y="realized_per_pair",
        color="city", symbol="outcome_source",
        hover_data=["date", "kalshi_label", "poly_label",
                    "kalshi_side", "poly_side"],
        labels={"predicted_edge": "E[payoff] − cost ($/pair)",
                "realized_per_pair": "Realized ($/pair)"},
      )
      fig.add_shape(
        type="line",
        x0=float(bt["predicted_edge"].min()),
        x1=float(bt["predicted_edge"].max()),
        y0=float(bt["predicted_edge"].min()),
        y1=float(bt["predicted_edge"].max()),
        line=dict(dash="dot", color="gray"))
      fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
      st.plotly_chart(fig, use_container_width=True)

    with cR:
      st.subheader("Realized $ per pair (distribution)")
      fig = px.histogram(
        bt, x="realized_per_pair", color="city", nbins=30,
        labels={"realized_per_pair": "Realized ($/pair)"},
      )
      fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380,
                        bargap=0.05)
      st.plotly_chart(fig, use_container_width=True)

    with st.expander("All backtested pairs"):
      st.dataframe(
        bt[["city", "date", "snapshot_ts", "outcome_source",
            "kalshi_label", "kalshi_side", "kalshi_avg_fill",
            "poly_label", "poly_side", "poly_avg_fill",
            "cost_per_pair", "expected_payoff", "q05_payoff",
            "realized_per_pair", "realized_total", "won"]]
          .sort_values("snapshot_ts"),
        use_container_width=True, hide_index=True, height=320)


