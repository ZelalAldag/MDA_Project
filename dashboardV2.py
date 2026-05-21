from shiny import App, render, ui
import pandas as pd
import plotly.express as px
from htmltools import HTML
import numpy as np
import os

# ── 1. Load data ─────────────────────────────────────────────────────────────
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "data", "FinalDemoData.csv")

try:
    app_df = pd.read_csv(data_path)
    if 'datetime_hour' in app_df.columns:
        app_df['datetime_hour'] = app_df['datetime_hour'].astype(str)
except Exception:
    rng = np.random.default_rng(42)
    n = 120
    app_df = pd.DataFrame({
        'site_id':           [f"ANT-{i:03d}" for i in range(1, n + 1)],
        'datetime_hour':     [f"2024-01-01 {h:02d}:00"
                              for h in np.tile(np.arange(24), 5)[:n]],
        'count':             rng.integers(50, 500, n),
        'year':              [2024] * n,
        'month':             [1] * n,
        'day':               [1] * n,
        'hour':              np.tile(np.arange(24), 5)[:n],
        'lat':               rng.uniform(51.15, 51.25, n),
        'lon':               rng.uniform(4.35,  4.45,  n),
        'municipality':      rng.choice(['Antwerp', 'Berchem', 'Deurne', 'Merksem'], n),
        'temperature_2m':    rng.uniform(5, 25, n),
        'precipitation':     rng.uniform(0, 10, n),
        'wind_speed_10m':    rng.uniform(0, 15, n),
        'actual':            rng.integers(50, 500, n),
        'predicted':         rng.integers(50, 500, n),
        'signed_error':      rng.integers(-200, 200, n),
        # ← FIX: added to fallback so resilience_plot never KeyErrors
        'hourly Resilience': rng.uniform(60, 130, n),
    })

SURGE_THRESHOLD    =  100
BLOCKAGE_THRESHOLD = -80

# ── CSS ───────────────────────────────────────────────────────────────────────
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
  padding: 16px 28px;
  background: var(--c1);
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
  display: grid;
  grid-template-columns: 260px 1fr;
  height: calc(100vh - 52px);
  min-height: 640px;
}

.cmd-sidebar {
  background: var(--c1);
  border-right: 1px solid var(--border);
  padding: 28px 22px;
  display: flex; flex-direction: column; gap: 30px;
  overflow-y: auto;
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
.irs--shiny .irs-line          { background: var(--c4) !important; height: 3px !important; border: none !important; }
.irs--shiny .irs-handle        { background: var(--accent) !important; border: none !important; box-shadow: 0 0 8px rgba(200,245,96,0.5) !important; width: 14px !important; height: 14px !important; top: 22px !important; }
.irs--shiny .irs-single        { background: transparent !important; color: var(--accent) !important; font-family: 'IBM Plex Mono',monospace !important; font-size: 11px !important; font-weight: 600 !important; }
.irs--shiny .irs-min, .irs--shiny .irs-max { color: var(--muted) !important; background: transparent !important; font-size: 9px !important; }
.irs--shiny .irs-grid-text     { color: var(--muted) !important; font-size: 8px !important; }
.irs--shiny .irs-grid-pol      { background: var(--c4) !important; }

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
.scale-ends   { display: flex; justify-content: space-between; font-size: 9px; color: var(--muted); }
.scale-bar    { height: 8px; border-radius: 2px; background: linear-gradient(90deg, #ffb347, #333, #ff4d4d); }
.scale-labels { display: flex; justify-content: space-between; font-size: 8px; color: var(--muted); }

.cmd-content {
  background: var(--c0);
  overflow-y: auto;
  display: flex; flex-direction: column;
}

.stats-bar {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: var(--c3);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
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

.map-wrap {
  padding: 0 24px 16px;
  flex: 0 0 auto;
}
.map-frame {
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  background: var(--c1);
}

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
.badge-surge { background: rgba(255,77,77,0.15);  color: var(--red); }
.badge-block { background: rgba(255,179,71,0.15); color: var(--amber); }
.nominal-row td { text-align: center; color: var(--accent); letter-spacing: 2px; padding: 20px !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.style(GLOBAL_CSS)),

    # Top bar
    ui.div(
        ui.div("Flanders Government // CYCLING TRAFFIC MONITOR SYSTEM",
               class_="topbar-logo"),
        ui.div(
            ui.div(class_="live-dot"),
            "LIVE MONITORING",
            class_="live-badge"
        ),
        class_="topbar"
    ),

    # Main grid
    ui.div(

        # ── Sidebar ──────────────────────────────────────────────────────────
        ui.div(
            # Clock / hour selector
            ui.div(
                ui.div("Selected Hour", class_="sb-label"),
                ui.div(
                    ui.output_ui("clock_face"),
                    ui.div("LOCAL TIME", class_="clock-sub"),
                    class_="clock-block"
                ),
                ui.input_slider("hour_sel", "", 0, 23, 8, step=1,
                                animate=ui.AnimationOptions(interval=900),
                                width="100%"),
                class_="sb-section"
            ),

            # Thresholds
            ui.div(
                ui.div("Anomaly Thresholds", class_="sb-label"),
                ui.div(
                    ui.div(
                        ui.span("SURGE ↑", class_="th-key"),
                        ui.span(f"> {SURGE_THRESHOLD}",
                                style="color:var(--red);font-weight:600"),
                        class_="th-item surge"
                    ),
                    ui.div(
                        ui.span("BLOCK ↓", class_="th-key"),
                        ui.span(f"< {BLOCKAGE_THRESHOLD}",
                                style="color:var(--amber);font-weight:600"),
                        class_="th-item block"
                    ),
                    class_="th-list"
                ),
                class_="sb-section"
            ),

            # Scale legend
            ui.div(
                ui.div("Error Scale", class_="sb-label"),
                ui.div(
                    ui.div(
                        ui.span("−200"), ui.span("0"), ui.span("+200"),
                        class_="scale-ends"
                    ),
                    ui.div(class_="scale-bar"),
                    ui.div(
                        ui.span("BLOCKAGE"), ui.span("NOMINAL"), ui.span("SURGE"),
                        class_="scale-labels"
                    ),
                    class_="scale-box"
                ),
                style="margin-top:auto",
                class_="sb-section"
            ),

            class_="cmd-sidebar"
        ),

        # ── Main content ─────────────────────────────────────────────────────
        ui.div(

            # Stats bar
            ui.div(
                ui.div(
                    ui.div("AVG SIGNED ERROR", class_="stat-lbl"),
                    ui.output_ui("stat_avg"),
                    class_="stat-card"
                ),
                ui.div(
                    ui.div("CRITICAL SURGES", class_="stat-lbl"),
                    ui.output_ui("stat_surge"),
                    class_="stat-card"
                ),
                ui.div(
                    ui.div("MAJOR BLOCKAGES", class_="stat-lbl"),
                    ui.output_ui("stat_block"),
                    class_="stat-card"
                ),
                ui.div(
                    ui.div("AVG ACTUAL COUNT", class_="stat-lbl"),
                    ui.output_ui("stat_actual"),
                    class_="stat-card"
                ),
                class_="stats-bar"
            ),

            # Map section
            ui.div(
                ui.div(
                    ui.span("GEOSPATIAL ERROR MAP — ANTWERP REGION",
                            class_="sec-title"),
                    ui.div(
                        ui.div(
                            ui.div(class_="leg-dot", style="background:var(--red)"),
                            "SURGE", class_="leg-item"
                        ),
                        ui.div(
                            ui.div(class_="leg-dot", style="background:var(--amber)"),
                            "BLOCK", class_="leg-item"
                        ),
                        ui.div(
                            ui.div(class_="leg-dot", style="background:#555"),
                            "NOMINAL", class_="leg-item"
                        ),
                        class_="legend"
                    ),
                    class_="sec-hdr"
                ),
                ui.div(
                    ui.output_ui("map_plot"),
                    class_="map-frame"
                ),
                class_="map-wrap"
            ),

            # Investigation queue table
            ui.div(
                ui.div(
                    ui.span("INVESTIGATION QUEUE", class_="sec-title"),
                    ui.output_ui("queue_count"),
                    class_="sec-hdr",
                    style="padding-bottom:10px"
                ),
                ui.output_ui("investigation_table"),
                class_="table-wrap"
            ),

            # ── FIX: Resilience scatter — UI block was missing; now mounted ──
            ui.div(
                ui.div(
                    ui.span("SITE RESILIENCE vs PREDICTED COUNT",
                            class_="sec-title"),
                    ui.div(
                        ui.div(
                            ui.div(class_="leg-dot",
                                   style="background:var(--accent)"),
                            "TOP 25% PREDICTED", class_="leg-item"
                        ),
                        ui.div(
                            ui.div(class_="leg-dot",
                                   style="background:var(--red)"),
                            "TOP 25% RESILIENCE", class_="leg-item"
                        ),
                        class_="legend"
                    ),
                    class_="sec-hdr"
                ),
                ui.div(
                    ui.output_ui("resilience_plot"),
                    class_="map-frame"
                ),
                class_="map-wrap"
            ),

            class_="cmd-content"
        ),

        class_="cmd-grid"
    )
)


# ── Server ────────────────────────────────────────────────────────────────────
def server(input, output, session):

    def get_data():
        """Filter rows matching the selected hour."""
        return app_df[app_df['hour'] == int(input.hour_sel())].copy()

    # ── Clock ────────────────────────────────────────────────────────────────
    @output
    @render.ui
    def clock_face():
        h = int(input.hour_sel())
        return ui.div(f"{h:02d}:00", class_="clock-time")

    # ── Stat cards ───────────────────────────────────────────────────────────
    @output
    @render.ui
    def stat_avg():
        val = get_data()['signed_error'].mean()
        sign = "+" if val >= 0 else ""
        return ui.div(f"{sign}{val:.1f}", class_="stat-num")

    @output
    @render.ui
    def stat_surge():
        df = get_data()
        count = len(df[df['signed_error'] > SURGE_THRESHOLD])
        return ui.div(str(count), class_="stat-num red")

    @output
    @render.ui
    def stat_block():
        df = get_data()
        count = len(df[df['signed_error'] < BLOCKAGE_THRESHOLD])
        return ui.div(str(count), class_="stat-num amber")

    @output
    @render.ui
    def stat_actual():
        val = get_data()['actual'].mean()
        return ui.div(f"{val:.0f}", class_="stat-num green")

    # ── Map ──────────────────────────────────────────────────────────────────
    @output
    @render.ui
    def map_plot():
        df = get_data()
        bubble_size = df['actual'].clip(10, 100)

        hover_extra = {}
        for col in ('municipality', 'temperature_2m', 'precipitation',
                    'wind_speed_10m', 'datetime_hour'):
            if col in df.columns:
                hover_extra[col] = True

        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color="signed_error",
            color_continuous_scale=[
                [0.0, "#ffb347"],
                [0.5, "#c8d8a8"],
                [1.0, "#cc0000"],
            ],
            range_color=[-200, 200],
            size=bubble_size,
            zoom=11.5, height=340,
            hover_name="site_id",
            hover_data={
                "actual":       True,
                "predicted":    True,
                "signed_error": True,
                "lat":          False,
                "lon":          False,
                **hover_extra,
            }
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            paper_bgcolor="#141414",
            plot_bgcolor="#141414",
            coloraxis_colorbar=dict(
                title=dict(text="ERROR",
                           font=dict(color="#888", size=9,
                                     family="IBM Plex Mono")),
                tickfont=dict(color="#888", size=9, family="IBM Plex Mono"),
                thickness=10, len=0.7,
                bgcolor="#1e1e1e", outlinecolor="#2a2a2a",
            ),
            hoverlabel=dict(
                bgcolor="#1e1e1e", bordercolor="#c8f560",
                font=dict(family="IBM Plex Mono", size=11, color="#e8e8e8")
            )
        )
        import json as _json
        fig_json = _json.loads(fig.to_json())
        html = f"""
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<div id="map-div" style="width:100%;height:340px;"></div>
<script>
(function(){{
  function tryRender(){{
    if(typeof Plotly==='undefined'){{setTimeout(tryRender,30);return;}}
    var fig={_json.dumps(fig_json)};
    Plotly.newPlot('map-div', fig.data, fig.layout, {{responsive:true}});
  }}
  tryRender();
}})();
</script>"""
        return HTML(html)

    # ── Queue count ──────────────────────────────────────────────────────────
    @output
    @render.ui
    def queue_count():
        df = get_data()
        n = len(df[(df['signed_error'] > SURGE_THRESHOLD) |
                   (df['signed_error'] < BLOCKAGE_THRESHOLD)])
        return ui.span(f"{n} ITEMS", style="font-size:9px;color:var(--muted)")

    # ── Investigation table ───────────────────────────────────────────────────
    @output
    @render.ui
    def investigation_table():
        df = get_data()
        anomalies = df[
            (df['signed_error'] > SURGE_THRESHOLD) |
            (df['signed_error'] < BLOCKAGE_THRESHOLD)
        ].copy().sort_values('signed_error', key=abs, ascending=False)

        has_muni = 'municipality' in df.columns
        has_dt   = 'datetime_hour' in df.columns

        extra_th = ""
        if has_muni: extra_th += "<th>MUNICIPALITY</th>"
        if has_dt:   extra_th += "<th>DATETIME</th>"

        if anomalies.empty:
            return HTML(f"""
            <table class="inv-table">
              <thead><tr>
                <th>SITE ID</th><th>ACTUAL</th><th>PREDICTED</th>
                <th>SIGNED ERROR</th>{extra_th}<th>STATUS</th>
              </tr></thead>
              <tbody>
                <tr class="nominal-row">
                  <td colspan="{5 + has_muni + has_dt}">ALL SYSTEMS NOMINAL</td>
                </tr>
              </tbody>
            </table>""")

        rows = []
        for _, row in anomalies.iterrows():
            e = row['signed_error']
            is_surge = e > SURGE_THRESHOLD
            badge = (
                '<span class="badge badge-surge">SURGE</span>'
                if is_surge else
                '<span class="badge badge-block">BLOCK</span>'
            )
            color = "var(--red)" if is_surge else "var(--amber)"
            sign  = "+" if e > 0 else ""

            extra_td = ""
            if has_muni: extra_td += f"<td>{row['municipality']}</td>"
            if has_dt:   extra_td += f"<td>{row['datetime_hour']}</td>"

            rows.append(f"""
            <tr>
              <td>{row['site_id']}</td>
              <td>{int(row['actual'])}</td>
              <td>{int(row['predicted'])}</td>
              <td style="color:{color};font-weight:600">{sign}{int(e)}</td>
              {extra_td}
              <td>{badge}</td>
            </tr>""")

        return HTML(f"""
        <table class="inv-table">
          <thead><tr>
            <th>SITE ID</th><th>ACTUAL</th><th>PREDICTED</th>
            <th>SIGNED ERROR</th>{extra_th}<th>STATUS</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>""")

    # ── Resilience vs Predicted scatter ───────────────────────────────────────
    @output
    @render.ui
    def resilience_plot():
        res_col = 'hourly Resilience'

        # Guard: column might be absent in some CSV versions
        if res_col not in app_df.columns:
            return HTML(
                "<p style='color:var(--muted);padding:20px;font-family:"
                "IBM Plex Mono'>Column 'hourly Resilience' not found.</p>"  # ← 删掉反斜杠，直接用单引号
            )

        # Show all sites across all hours (hour slider does not filter this chart)
        df_site = app_df.groupby('site_id').agg(
            predicted=(  'predicted', 'mean'),
            resilience=(res_col,      'mean'),
        ).reset_index()

        p75_pred = df_site['predicted'].quantile(0.75)
        p75_res  = df_site['resilience'].quantile(0.75)

        x_max = df_site['predicted'].max()
        y_max = df_site['resilience'].max()

        fig = px.scatter(
            df_site,
            x='predicted',
            y='resilience',
            text='site_id',
            color='resilience',
            color_continuous_scale=[
                [0.0, "#ffb347"],
                [0.5, "#c8f560"],
                [1.0, "#ff4d4d"],
            ],
            hover_data={'predicted': ':.1f', 'resilience': ':.1f'},
        )

        fig.update_traces(
            textposition='top center',
            textfont=dict(size=8, color='#aaaaaa', family='IBM Plex Mono'),
            marker=dict(size=11, opacity=0.9, line=dict(width=1, color='#444444')),
        )

        # Vertical reference line — Top 25% Predicted
        fig.add_vline(
            x=p75_pred,
            line_dash="dash", line_color="#c8f560", line_width=1.5,
            annotation_text=f"Top 25% Predicted ({p75_pred:.1f})",
            annotation_font=dict(color="#c8f560", size=9,
                                 family="IBM Plex Mono"),
            annotation_position="top right",
        )

        # Horizontal reference line — Top 25% Resilience
        fig.add_hline(
            y=p75_res,
            line_dash="dash", line_color="#ff4d4d", line_width=1.5,
            annotation_text=f"Top 25% Resilience ({p75_res:.1f})",
            annotation_font=dict(color="#ff4d4d", size=9,
                                 family="IBM Plex Mono"),
            annotation_position="top right",
        )

        # High-priority zone label in top-right quadrant
        fig.add_annotation(
            x=x_max * 0.72, y=y_max * 0.97,
            text="HIGH PRIORITY ZONE",
            showarrow=False,
            font=dict(color="#ff4d4d", size=9, family="IBM Plex Mono"),
            bgcolor="rgba(255,77,77,0.08)",
            bordercolor="#ff4d4d",
            borderwidth=1,
            borderpad=6,
        )

        fig.update_layout(
            paper_bgcolor="#141414",
            plot_bgcolor="#1e1e1e",
            font=dict(family="IBM Plex Mono", color="#888888"),
            xaxis=dict(
                title="Average Predicted Count",
                gridcolor="#2a2a2a", zerolinecolor="#2a2a2a",
                title_font=dict(color="#888", size=10),
                tickfont=dict(color="#888", size=9),
            ),
            yaxis=dict(
                title="Resilience Score",
                gridcolor="#2a2a2a", zerolinecolor="#2a2a2a",
                title_font=dict(color="#888", size=10),
                tickfont=dict(color="#888", size=9),
            ),
            coloraxis_showscale=False,
            margin=dict(l=60, r=40, t=20, b=60),
            height=420,
            hoverlabel=dict(
                bgcolor="#1e1e1e", bordercolor="#c8f560",
                font=dict(family="IBM Plex Mono", size=11, color="#e8e8e8")
            ),
        )

        import json as _json
        fig_json = _json.loads(fig.to_json())
        html = f"""
<div id="res-div" style="width:100%;height:420px;"></div>
<script>
(function(){{
  function tryRender(){{
    if(typeof Plotly==='undefined'){{setTimeout(tryRender,30);return;}}
    var fig={_json.dumps(fig_json)};
    Plotly.newPlot('res-div', fig.data, fig.layout, {{responsive:true}});
  }}
  tryRender();
}})();
</script>"""
        return HTML(html)


app = App(app_ui, server)

if __name__ == "__main__":
    app.run(port=8001)