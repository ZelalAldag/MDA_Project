from shiny import App, render, ui
import pandas as pd
import plotly.express as px
from htmltools import HTML
import numpy as np
import os

# --- 1. load data ---
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "data", "FinalDemoData.csv")

try:
    app_df = pd.read_csv(data_path)
    if 'datetime_hour' in app_df.columns:
        app_df['datetime_hour'] = app_df['datetime_hour'].astype(str)
    app_df['site_id'] = app_df['site_id'].astype(str)
except Exception:
    rng = np.random.default_rng(42)
    n = 120
    app_df = pd.DataFrame({
        'site_id':           [f"{i}" for i in range(1, n+1)],
        'datetime_hour':     [f"2024-01-01 {h:02d}:00" for h in np.tile(np.arange(24), 5)[:n]],
        'count':             rng.integers(50, 500, n),
        'year':              [2024] * n,
        'month':             [1] * n,
        'day':               rng.choice([1,2,3,4], n),
        'hour':              np.tile(np.arange(24), 5)[:n],
        'lat':               rng.uniform(51.15, 51.25, n),
        'lon':               rng.uniform(4.35,  4.45,  n),
        'municipality':      rng.choice(['Antwerp', 'Berchem', 'Deurne', 'Merksem'], n),
        'temperature_2m':    rng.uniform(5, 25, n),
        'precipitation':     rng.uniform(0, 10, n),
        'wind_speed_10m':    rng.uniform(0, 15, n),
        'actual':            rng.integers(50, 500, n),
        'predicted':         rng.uniform(1, 10, n),
        'signed_error':      rng.integers(-200, 200, n),
        'hourly Resilience': rng.uniform(0.5, 1.0, n),
        'avg_cum_km_per_pop':    rng.uniform(0.1, 5.0, n),
        'avg_cum_co2_per_pop':   rng.uniform(0.01, 1.0, n),
        'avg_cum_fuel_per_pop':  rng.uniform(0.01, 0.5, n),
    })

# ── 日期筛选：硬编码选择第几天（1=第一天, 2=第二天, ...）──────────────
# 修改这里的数字来切换日期
SELECTED_DAY_INDEX = 3   # 1 = 数据中最早的一天，2 = 第二天，以此类推

if 'day' in app_df.columns:
    available_days = sorted(app_df['day'].unique())
    selected_day = available_days[min(SELECTED_DAY_INDEX - 1, len(available_days) - 1)]
    app_df = app_df[app_df['day'] == selected_day].copy()
# ─────────────────────────────────────────────────────────────────────────────

SURGE_THRESHOLD    =  100
BLOCKAGE_THRESHOLD = -80

# ── CSS ─────────────────────────────────────────────────────────
GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

:root {
  --c0:      #0a0a0a;
  --c1:      #141414;
  --c2:      #1e1e1e;
  --c3:      #2a2a2a;
  --c4:      #404040;
  --accent:  #c8f560;
  --red:     #ff4d4d;
  --amber:   #ffb347;
  --text:    #e8e8e8;
  --muted:   #888888;
  --border:  rgba(200,245,96,0.15);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .shiny-app, #app { background: var(--c0) !important; }
body { font-family: 'IBM Plex Mono', monospace; color: var(--text); }
.container-fluid { padding: 0 !important; }
.shiny-output-error { color: var(--red) !important; }

.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 28px; background: var(--c1);
  border-bottom: 1px solid var(--border);
}
.topbar-logo {
  font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: 13px; letter-spacing: 4px; color: var(--accent);
}
.live-badge {
  display: flex; align-items: center; gap: 7px;
  font-size: 10px; letter-spacing: 2px; color: var(--accent);
}
.live-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
  animation: blink 1.8s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

.cmd-grid {
  display: grid; grid-template-columns: 260px 1fr;
  min-height: calc(100vh - 52px); /* ← 将 height 改为 min-height，让内容可以自由向下延伸 */
}

.cmd-sidebar {
  background: var(--c1); border-right: 1px solid var(--border);
  padding: 28px 22px; display: flex; flex-direction: column; gap: 30px;
  overflow-y: auto;
  position: sticky; top: 0; height: calc(100vh - 52px); align-self: start;
}
.sb-section { display: flex; flex-direction: column; gap: 10px; }
.sb-label   { font-size: 9px; letter-spacing: 3px; color: var(--muted); text-transform: uppercase; }

.clock-block {
  background: var(--c2); border: 1px solid var(--border);
  border-radius: 6px; padding: 20px 14px; text-align: center;
}
.clock-time {
  font-size: 30px; font-weight: 600; letter-spacing: 4px;
  color: var(--accent); line-height: 1;
}
.clock-sub { font-size: 9px; letter-spacing: 3px; color: var(--muted); margin-top: 5px; }

.irs--shiny .irs-bar           { background: var(--accent) !important; border-top: none !important; border-bottom: none !important; height: 3px !important; }
.irs--shiny .irs-line           { background: var(--c4) !important; height: 3px !important; border: none !important; }
.irs--shiny .irs-handle         { background: var(--accent) !important; border: none !important; box-shadow: 0 0 8px rgba(200,245,96,0.5) !important; width: 14px !important; height: 14px !important; top: 22px !important; }
.irs--shiny .irs-single         { background: transparent !important; color: var(--accent) !important; font-family: 'IBM Plex Mono',monospace !important; font-size: 11px !important; font-weight: 600 !important; }
.irs--shiny .irs-min,.irs--shiny .irs-max { color: var(--muted) !important; background: transparent !important; font-size: 9px !important; }
.irs--shiny .irs-grid-text      { color: var(--muted) !important; font-size: 8px !important; }
.irs--shiny .irs-grid-pol       { background: var(--c4) !important; }

.th-list { display: flex; flex-direction: column; gap: 7px; }
.th-item {
  display: flex; align-items: center; justify-content: space-between;
  background: var(--c2); border: 1px solid var(--c4); border-radius: 4px;
  padding: 9px 12px; font-size: 11px;
}
.th-item.surge { border-left: 2px solid var(--red) !important; }
.th-item.block  { border-left: 2px solid var(--amber) !important; }
.th-key  { color: var(--muted); }

.scale-box {
  background: var(--c2); border: 1px solid var(--border);
  border-radius: 4px; padding: 10px 14px;
  display: flex; flex-direction: column; gap: 6px;
}
.scale-ends  { display: flex; justify-content: space-between; font-size: 9px; color: var(--muted); }
.scale-bar   { height: 8px; border-radius: 2px; background: linear-gradient(90deg, #ffb347, #333, #ff4d4d); }
.scale-labels{ display: flex; justify-content: space-between; font-size: 8px; color: var(--muted); }

.cmd-content {
  background: var(--c0); overflow-y: auto;
  display: flex; flex-direction: column;
}

.stats-bar {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: var(--c3);
  border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.stat-card {
  background: var(--c1); padding: 20px 24px;
  display: flex; flex-direction: column; gap: 5px;
}
.stat-lbl { font-size: 9px; letter-spacing: 2.5px; color: var(--muted); }
.stat-num { font-size: 26px; font-weight: 600; color: var(--text); font-family: 'IBM Plex Mono',monospace; }
.stat-num.red   { color: var(--red); }
.stat-num.amber { color: var(--amber); }
.stat-num.green { color: var(--accent); }

.sec-hdr {
  display: flex; align-items: center; justify-content: space-between;
  padding: 18px 24px 10px;
}
.sec-title { font-size: 10px; letter-spacing: 3px; color: var(--muted); }
.legend    { display: flex; gap: 18px; font-size: 9px; color: var(--muted); letter-spacing: 1px; }
.leg-item  { display: flex; align-items: center; gap: 5px; }
.leg-dot   { width: 8px; height: 8px; border-radius: 50%; }

.map-wrap  { padding: 0 24px 16px; flex: 0 0 auto; }
.map-frame { border: 1px solid var(--border); border-radius: 4px; overflow: hidden; background: var(--c1); }

.table-wrap { padding: 0 24px 28px; }
.inv-table  { width: 100%; border-collapse: collapse; font-size: 11px; }
.inv-table thead th {
  text-align: left; padding: 8px 12px;
  font-size: 9px; letter-spacing: 2px; font-weight: 400;
  color: var(--muted); border-bottom: 1px solid var(--c3);
}
.inv-table tbody td { padding: 9px 12px; border-bottom: 1px solid var(--c2); }
.inv-table tbody tr:hover td { background: var(--c2); }
.badge {
  display: inline-block; padding: 2px 7px; border-radius: 2px;
  font-size: 9px; letter-spacing: 1.5px; font-weight: 600;
}
.badge-surge    { background: rgba(255,77,77,0.15);  color: var(--red); }
.badge-block    { background: rgba(255,179,71,0.15); color: var(--amber); }
.badge-priority { background: rgba(255,77,77,0.25);  color: #ff2020; border: 1px solid rgba(255,77,77,0.4); }
.nominal-row td { text-align: center; color: var(--accent); letter-spacing: 2px; padding: 20px !important; }

.charts-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 16px; padding: 0 24px 28px;
}
.chart-card {
  background: var(--c1); border: 1px solid var(--border);
  border-radius: 4px; overflow: hidden;
  min-height: 360px; /* ← 新增这一行，预留 360px 的安全高度给内部的标题和 320px 的图表 */
}
.chart-card-title {
  padding: 12px 16px 0;
  font-size: 9px; letter-spacing: 2px; color: var(--muted);
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.style(GLOBAL_CSS)),

    ui.div(
        ui.div("Flanders Government // CYCLING TRAFFIC MONITOR SYSTEM", class_="topbar-logo"),
        ui.div(ui.div(class_="live-dot"), "LIVE MONITORING", class_="live-badge"),
        class_="topbar"
    ),

    ui.div(
        # ── Sidebar ──────────────────────────────────────────────────────────
        ui.div(
            ui.div(
                ui.div("Selected Hour", class_="sb-label"),
                ui.div(
                    ui.output_ui("clock_face"),
                    ui.div("LOCAL TIME", class_="clock-sub"),
                    class_="clock-block"
                ),
                ui.input_slider("hour_sel", "", 0, 23, 8, step=1,
                                animate=ui.AnimationOptions(interval=900), width="100%"),
                class_="sb-section"
            ),
            ui.div(
                ui.div("Anomaly Thresholds", class_="sb-label"),
                ui.div(
                    ui.div(ui.span("SURGE ↑", class_="th-key"),
                           ui.span(f"> {SURGE_THRESHOLD}", style="color:var(--red);font-weight:600"),
                           class_="th-item surge"),
                    ui.div(ui.span("BLOCK ↓", class_="th-key"),
                           ui.span(f"< {BLOCKAGE_THRESHOLD}", style="color:var(--amber);font-weight:600"),
                           class_="th-item block"),
                    class_="th-list"
                ),
                class_="sb-section"
            ),
            ui.div(
                ui.div("Error Scale", class_="sb-label"),
                ui.div(
                    ui.div(ui.span("−200"), ui.span("0"), ui.span("+200"), class_="scale-ends"),
                    ui.div(class_="scale-bar"),
                    ui.div(ui.span("BLOCKAGE"), ui.span("NOMINAL"), ui.span("SURGE"), class_="scale-labels"),
                    class_="scale-box"
                ),
                style="margin-top:auto", class_="sb-section"
            ),
            class_="cmd-sidebar"
        ),

        # ── Main content ─────────────────────────────────────────────────────
        ui.div(

            # Stats bar
            ui.div(
                ui.div(ui.div("AVG SIGNED ERROR", class_="stat-lbl"), ui.output_ui("stat_avg"),   class_="stat-card"),
                ui.div(ui.div("CRITICAL SURGES",  class_="stat-lbl"), ui.output_ui("stat_surge"), class_="stat-card"),
                ui.div(ui.div("MAJOR BLOCKAGES",  class_="stat-lbl"), ui.output_ui("stat_block"), class_="stat-card"),
                ui.div(ui.div("AVG ACTUAL COUNT", class_="stat-lbl"), ui.output_ui("stat_actual"),class_="stat-card"),
                class_="stats-bar"
            ),

            # 1. Geospatial map
            ui.div(
                ui.div(
                    ui.span("GEOSPATIAL MONITORING MAP",
                            style="color:var(--accent);font-size:14px;font-weight:600;"),
                    ui.div(
                        ui.div(ui.div(class_="leg-dot", style="background:var(--red)"),   "SURGE",   class_="leg-item"),
                        ui.div(ui.div(class_="leg-dot", style="background:var(--amber)"), "BLOCK",   class_="leg-item"),
                        ui.div(ui.div(class_="leg-dot", style="background:#555"),         "NOMINAL", class_="leg-item"),
                        class_="legend"
                    ),
                    class_="sec-hdr"
                ),
                ui.div(ui.output_ui("map_plot"), class_="map-frame"),
                class_="map-wrap"
            ),

            # 2. Investigation queue
            ui.div(
                ui.div(
                    ui.span("INVESTIGATION QUEUE",
                            style="color:var(--accent);font-size:14px;font-weight:600;"),
                    ui.output_ui("queue_count"),
                    class_="sec-hdr", style="padding-bottom:10px"
                ),
                ui.output_ui("investigation_table"),
                class_="table-wrap"
            ),

            # 3. Resilience scatter (interactive/draggable via plotly)
            ui.div(
                ui.div(
                    ui.span("SITE RESILIENCE MONITORING",
                            style="color:var(--accent);font-size:14px;font-weight:600;"),
                    ui.div(
                        ui.div(ui.div(class_="leg-dot", style="background:var(--accent)"), "TOP 10% PREDICTED",  class_="leg-item"),
                        ui.div(ui.div(class_="leg-dot", style="background:var(--red)"),    "TOP 20% RESILIENCE", class_="leg-item"),
                        class_="legend"
                    ),
                    class_="sec-hdr"
                ),
                ui.div(ui.output_ui("resilience_plot"), class_="map-frame"),
                class_="map-wrap"
            ),

            # 4. Priority sites queue
            ui.div(
                ui.div(
                    ui.span("PRIORITIZATION SITES QUEUE — HIGH RESILIENCE",
                            style="color:var(--accent);font-size:14px;font-weight:600;"),
                    ui.span("TOP 5 BY RESILIENCE + PREDICTED COUNT",
                            style="font-size:9px;color:var(--muted)"),
                    class_="sec-hdr", style="padding-bottom:10px"
                ),
                ui.output_ui("priority_table"),
                class_="table-wrap"
            ),

            # 5. Three municipality ranking bar charts
            ui.div(
                ui.span("MUNICIPALITY RANKINGS",
                        style="color:var(--accent);font-size:14px;font-weight:600;padding:18px 24px 10px;display:block;"),
                ui.div(
                    ui.div(
                        ui.div("AVG KM PER POPULATION", class_="chart-card-title"),
                        ui.output_ui("chart_km"),
                        class_="chart-card"
                    ),
                    ui.div(
                        ui.div("AVG CO₂ SAVED PER POPULATION", class_="chart-card-title"),
                        ui.output_ui("chart_co2"),
                        class_="chart-card"
                    ),
                    ui.div(
                        ui.div("AVG FUEL SAVED PER POPULATION", class_="chart-card-title"),
                        ui.output_ui("chart_fuel"),
                        class_="chart-card"
                    ),
                    class_="charts-grid"
                ),
            ),

            class_="cmd-content"
        ),

        class_="cmd-grid"
    )
)


# ── Server ────────────────────────────────────────────────────────────────────
def server(input, output, session):

    def get_data():
        return app_df[app_df['hour'] == int(input.hour_sel())].copy()

    def make_bar_chart(col, title, color_scale, scale=1.0, unit=""):
        """按当前选择小时的 municipality 聚合均值，取 Top 10，横向柱状图，支持自定义渐变色和单位换算。"""
        df = get_data()   # ← 使用 sidebar 滑块选中小时过滤后的数据
        if col not in df.columns or 'municipality' not in df.columns:
            return HTML(f"<p style='color:var(--muted);padding:20px'>{col} not found.</p>")
        df_m = (df.groupby('municipality')[col]
                .mean()
                .reset_index()
                .rename(columns={col: 'value'})
                .nlargest(10, 'value')
                .sort_values('value', ascending=True))

        if df_m.empty:
            return HTML(f"<p style='color:var(--muted);padding:20px'>No data for {col}.</p>")

        # 单位换算
        df_m['value'] = df_m['value'] * scale

        unit_label = f" ({unit})" if unit else ""
        hover_fmt = '.4f' if df_m['value'].max() < 1 else '.2f'

        fig = px.bar(
            df_m, x='value', y='municipality', orientation='h',
            color='value',
            color_continuous_scale=color_scale,
        )
        fig.update_traces(
            hovertemplate=f'<b>%{{y}}</b><br>{title}: %{{x:{hover_fmt}}}{unit_label}<extra></extra>',
        )
        fig.update_layout(
            paper_bgcolor="#141414", plot_bgcolor="#1e1e1e",
            font=dict(family="IBM Plex Mono", color="#888"),
            xaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a",
                       title="", tickfont=dict(color="#888", size=9)),
            yaxis=dict(gridcolor="#2a2a2a", title="",
                       tickfont=dict(color="#e8e8e8", size=10)),
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                thickness=8, len=0.6, bgcolor="#141414", outlinecolor="#2a2a2a",
                tickfont=dict(color="#888", size=8, family="IBM Plex Mono"),
                title=dict(text="", font=dict(color="#888", size=8)),
            ),
            margin=dict(l=10, r=50, t=10, b=30),
            height=320,
            hoverlabel=dict(bgcolor="#1e1e1e", bordercolor="#c8f560",
                            font=dict(family="IBM Plex Mono", size=11, color="#e8e8e8")),
            dragmode='zoom',
        )
        return HTML(fig.to_html(include_plotlyjs='cdn', full_html=False,
                                config={'scrollZoom': True, 'displayModeBar': False}))

    # ── Clock ──
    @output
    @render.ui
    def clock_face():
        h = int(input.hour_sel())
        return ui.div(f"{h:02d}:00", class_="clock-time")

    # ── Stat cards ──
    @output
    @render.ui
    def stat_avg():
        val = get_data()['signed_error'].mean()
        sign = "+" if val >= 0 else ""
        return ui.div(f"{sign}{val:.1f}", class_="stat-num")

    @output
    @render.ui
    def stat_surge():
        count = len(get_data()[get_data()['signed_error'] > SURGE_THRESHOLD])
        return ui.div(str(count), class_="stat-num red")

    @output
    @render.ui
    def stat_block():
        count = len(get_data()[get_data()['signed_error'] < BLOCKAGE_THRESHOLD])
        return ui.div(str(count), class_="stat-num amber")

    @output
    @render.ui
    def stat_actual():
        val = get_data()['actual'].mean()
        return ui.div(f"{val:.0f}", class_="stat-num green")

    # ── Map ──
    @output
    @render.ui
    def map_plot():
        df = get_data()
        bubble_size = df['actual'].clip(10, 100)
        hover_extra = {}
        for col in ('municipality','temperature_2m','precipitation','wind_speed_10m','datetime_hour'):
            if col in df.columns:
                hover_extra[col] = True

        fig = px.scatter_mapbox(
            df, lat="lat", lon="lon", color="signed_error",
            color_continuous_scale=[[0.0,"#ffb347"],[0.5,"#b8d9a0"],[1.0,"#800000"]],
            range_color=[-150,150], size=bubble_size, zoom=11.5, height=340,
            hover_name="site_id",
            hover_data={"actual":True,"predicted":True,"signed_error":True,
                        "lat":False,"lon":False,**hover_extra}
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="#141414", plot_bgcolor="#141414",
            coloraxis_colorbar=dict(
                title=dict(text="ERROR", font=dict(color="#888",size=9,family="IBM Plex Mono")),
                tickfont=dict(color="#888",size=9,family="IBM Plex Mono"),
                thickness=10, len=0.7, bgcolor="#1e1e1e", outlinecolor="#2a2a2a",
            ),
            hoverlabel=dict(bgcolor="#1e1e1e", bordercolor="#c8f560",
                            font=dict(family="IBM Plex Mono",size=11,color="#e8e8e8"))
        )
        return HTML(fig.to_html(include_plotlyjs='cdn', full_html=False,
                                config={'scrollZoom': True, 'displayModeBar': False}))

    # ── Queue count ──
    @output
    @render.ui
    def queue_count():
        df = get_data()
        n = len(df[(df['signed_error'] > SURGE_THRESHOLD) | (df['signed_error'] < BLOCKAGE_THRESHOLD)])
        return ui.span(f"{n} ITEMS", style="font-size:9px;color:var(--muted)")

    # ── Investigation table ──
    @output
    @render.ui
    def investigation_table():
        df = get_data()
        anomalies = df[
            (df['signed_error'] > SURGE_THRESHOLD) | (df['signed_error'] < BLOCKAGE_THRESHOLD)
        ].copy().sort_values('signed_error', key=abs, ascending=False)

        has_muni = 'municipality' in df.columns
        has_dt   = 'datetime_hour' in df.columns
        extra_th = ""
        if has_muni: extra_th += "<th>MUNICIPALITY</th>"
        if has_dt:   extra_th += "<th>DATETIME</th>"

        if anomalies.empty:
            return HTML(f"""<table class="inv-table"><thead><tr>
              <th>SITE ID</th><th>ACTUAL</th><th>PREDICTED</th>
              <th>SIGNED ERROR</th>{extra_th}<th>STATUS</th>
            </tr></thead><tbody>
            <tr class="nominal-row"><td colspan="{5+has_muni+has_dt}">ALL SYSTEMS NOMINAL</td></tr>
            </tbody></table>""")

        rows = []
        for _, row in anomalies.iterrows():
            e = row['signed_error']
            is_surge = e > SURGE_THRESHOLD
            badge = '<span class="badge badge-surge">SURGE</span>' if is_surge else '<span class="badge badge-block">BLOCK</span>'
            color = "var(--red)" if is_surge else "var(--amber)"
            sign  = "+" if e > 0 else ""
            extra_td = ""
            if has_muni: extra_td += f"<td>{row['municipality']}</td>"
            if has_dt:   extra_td += f"<td>{row['datetime_hour']}</td>"
            rows.append(f"""<tr>
              <td>{row['site_id']}</td><td>{int(row['actual'])}</td><td>{row['predicted']:.3f}</td>
              <td style="color:{color};font-weight:600">{sign}{int(e)}</td>{extra_td}<td>{badge}</td>
            </tr>""")

        return HTML(f"""<table class="inv-table"><thead><tr>
          <th>SITE ID</th><th>ACTUAL</th><th>PREDICTED</th>
          <th>SIGNED ERROR</th>{extra_th}<th>STATUS</th>
        </tr></thead><tbody>{''.join(rows)}</tbody></table>""")

    # ── Resilience scatter (interactive / draggable) ──
    @output
    @render.ui
    def resilience_plot():
        res_col = 'hourly Resilience'
        if res_col not in app_df.columns:
            return HTML("<p style='color:var(--muted);padding:20px'>Column 'hourly Resilience' not found.</p>")

        df_plot = get_data().copy()
        df_plot = df_plot.rename(columns={res_col: 'resilience'})
        df_plot = df_plot[['site_id','predicted','resilience']].dropna()

        if df_plot.empty:
            return HTML("<p style='color:var(--muted);padding:20px'>No data for selected hour.</p>")

        df_plot['priority_score'] = df_plot['resilience'] + df_plot['predicted']

        p80_res  = df_plot['resilience'].quantile(0.80)   # top 20%
        p90_pred = df_plot['predicted'].quantile(0.90)    # top 10%

        x_min = df_plot['predicted'].min()
        x_max = df_plot['predicted'].max()
        y_min = df_plot['resilience'].min()
        y_max = df_plot['resilience'].max()

        x_center = (x_min + x_max) / 2
        x_half   = (x_max - x_min) / 2
        y_center = (y_min + y_max) / 2
        y_half   = (y_max - y_min) / 2

        # predicted x轴：视觉缩小为一半（range扩大2倍）
        x_range = [x_center - x_half * 2, x_center + x_half * 2]
        # resilience y轴：视觉放大2倍（range缩小为一半）
        y_range = [y_center - y_half / 2, y_center + y_half / 2]

        fig = px.scatter(
            df_plot, x='predicted', y='resilience', text='site_id',
            color='priority_score',
            color_continuous_scale=[[0.0,"#2a2a2a"],[0.5,"#ff9966"],[1.0,"#cc0000"]],
            hover_data={'site_id':True,'predicted':':.3f','resilience':':.3f','priority_score':':.3f'},
        )
        fig.update_traces(
            textposition='top center',
            textfont=dict(size=8, color='#888888', family='IBM Plex Mono'),
            marker=dict(size=9, line=dict(width=0)),
        )

        fig.add_vline(x=p90_pred, line_dash="dash", line_color="#c8f560", line_width=1.5,
                      annotation_text=f"Top 10% Predicted ({p90_pred:.3f})",
                      annotation_font=dict(color="#c8f560", size=9, family="IBM Plex Mono"),
                      annotation_position="top right")

        fig.add_hline(y=p80_res, line_dash="dash", line_color="#ff4d4d", line_width=1.5,
                      annotation_text=f"Top 20% Resilience ({p80_res:.3f})",
                      annotation_font=dict(color="#ff4d4d", size=9, family="IBM Plex Mono"),
                      annotation_position="top right")

        fig.add_annotation(
            x=p90_pred + (x_range[1] - p90_pred) * 0.5,
            y=p80_res  + (y_range[1] - p80_res)  * 0.3,
            text="HIGH PRIORITY ZONE", showarrow=False,
            font=dict(color="#ff4d4d", size=9, family="IBM Plex Mono"),
            bgcolor="rgba(255,77,77,0.08)", bordercolor="#ff4d4d", borderwidth=1, borderpad=6,
        )

        fig.update_layout(
            paper_bgcolor="#141414", plot_bgcolor="#1e1e1e",
            font=dict(family="IBM Plex Mono", color="#888888"),
            xaxis=dict(title="Predicted Count", range=x_range,
                       gridcolor="#2a2a2a", zerolinecolor="#2a2a2a",
                       title_font=dict(color="#888",size=10), tickfont=dict(color="#888",size=9)),
            yaxis=dict(title="Resilience Score", range=y_range,
                       gridcolor="#2a2a2a", zerolinecolor="#2a2a2a",
                       title_font=dict(color="#888",size=10), tickfont=dict(color="#888",size=9)),
            coloraxis_colorbar=dict(
                title=dict(text="PRIORITY", font=dict(color="#888",size=9,family="IBM Plex Mono")),
                tickfont=dict(color="#888",size=9,family="IBM Plex Mono"),
                thickness=10, len=0.6, bgcolor="#1e1e1e", outlinecolor="#2a2a2a",
            ),
            margin=dict(l=60,r=60,t=20,b=60), height=420,
            hoverlabel=dict(bgcolor="#1e1e1e", bordercolor="#c8f560",
                            font=dict(family="IBM Plex Mono",size=11,color="#e8e8e8")),
            dragmode='pan',   # ← 默认拖动模式
        )
        return HTML(fig.to_html(include_plotlyjs='cdn', full_html=False,
                                config={'scrollZoom': True, 'displayModeBar': True,
                                        'modeBarButtonsToRemove': ['select2d','lasso2d'],
                                        'displaylogo': False}))

    # ── Priority table (Top 5) ──
    @output
    @render.ui
    def priority_table():
        res_col = 'hourly Resilience'
        df = get_data().copy()
        if res_col not in df.columns:
            return HTML("<p style='color:var(--muted);padding:20px'>Column 'hourly Resilience' not found.</p>")

        df['priority_score'] = df[res_col] + df['predicted']
        top5 = df.nlargest(5, 'priority_score').reset_index(drop=True)
        has_muni = 'municipality' in df.columns

        score_min = top5['priority_score'].min()
        score_max = top5['priority_score'].max()
        score_range = score_max - score_min if score_max != score_min else 1

        def score_to_color(score):
            t = (score - score_min) / score_range
            r = int(0xff + (0xcc - 0xff) * t)
            g = int(0x99 + (0x00 - 0x99) * t)
            b = int(0x66 + (0x00 - 0x66) * t)
            return f"#{r:02x}{g:02x}{b:02x}"

        muni_th = "<th>MUNICIPALITY</th>" if has_muni else ""
        rows = []
        for i, row in top5.iterrows():
            rank = i + 1
            color = score_to_color(row['priority_score'])
            muni_td = f"<td>{row['municipality']}</td>" if has_muni else ""
            rows.append(f"""<tr>
              <td style="color:{color};font-weight:600;font-size:13px">{rank}</td>
              <td>{row['site_id']}</td>
              <td>{int(row['actual'])}</td>
              <td>{row['predicted']:.3f}</td>
              {muni_td}
              <td style="color:{color};font-weight:600">{row['priority_score']:.3f}</td>
              <td><span class="badge badge-priority">PRIORITY</span></td>
            </tr>""")

        return HTML(f"""<table class="inv-table"><thead><tr>
          <th>RANK</th><th>SITE ID</th><th>ACTUAL</th><th>PREDICTED</th>
          {muni_th}<th>PRIORITY SCORE</th><th>STATUS</th>
        </tr></thead><tbody>{''.join(rows)}</tbody></table>""")

    # 三个图统一使用同一渐变色主题：深色背景 → 青绿 → accent 黄绿
    CHART_COLOR_SCALE = [
        [0.0, "#1a2a1a"],
        [0.3, "#1a5c40"],
        [0.65, "#29a87a"],
        [1.0, "#c8f560"],
    ]

    # ── Municipality bar charts ──
    @output
    @render.ui
    def chart_km():
        # km/person × 1000 → m/person
        return make_bar_chart('avg_cum_km_per_pop', 'Avg m/person', CHART_COLOR_SCALE, scale=1000, unit='m/person')

    @output
    @render.ui
    def chart_co2():
        # kg CO₂/person × 1000 → g CO₂/person
        return make_bar_chart('avg_cum_co2_per_pop', 'Avg g CO₂/person', CHART_COLOR_SCALE, scale=1000, unit='g/person')

    @output
    @render.ui
    def chart_fuel():
        # L/person × 1000 → mL/person
        return make_bar_chart('avg_cum_fuel_per_pop', 'Avg mL/person', CHART_COLOR_SCALE, scale=1000, unit='mL/person')


app = App(app_ui, server)

if __name__ == "__main__":
    app.run(port=8003)