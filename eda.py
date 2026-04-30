import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path

    return Path, mo, pd, px


@app.cell
def _(mo):
    mo.md("""
    # step 2 — exploratory data analysis
    """)
    return


@app.cell
def _(Path, pd):
    df = pd.read_parquet(Path("data/processed/fietstellingen_clean.parquet"))
    df["start_time"] = pd.to_datetime(df["start_time"])
    df

    return (df,)


@app.cell
def _(mo):
    mo.md("""
    ## average cyclists by hour of day
    """)
    return


@app.cell
def _(df, px):
    hourly = (
            df.groupby(["hour", "is_weekend"])["count"]
            .mean()
            .reset_index()
        )
    hourly["day_type"] = hourly["is_weekend"].map({True: "weekend", False: "weekday"})

    fig_hour = px.line(
            hourly,
            x="hour",
            y="count",
            color="day_type",
            markers=True,
            labels={"hour": "hour of day", "count": "avg cyclists per 15 min", "day_type": ""},
            color_discrete_map={"weekday": "#1f77b4", "weekend": "#ff7f0e"},
     )
    fig_hour.update_layout(xaxis=dict(tickmode="linear", dtick=1),hovermode="x unified",)
    fig_hour

    return


@app.cell
def _(mo):
    mo.md("""
    ## average cyclists by day of week
    """)
    return


@app.cell
def _(df, px):
    day_labels = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    daily = (
            df.groupby("day_of_week")["count"]
            .mean()
            .reset_index()
    )
    daily["day_name"] = daily["day_of_week"].map(dict(enumerate(day_labels)))

    fig_day = px.bar(
            daily,
            x="day_name",
            y="count",
            labels={"day_name": "", "count": "avg cyclists per 15 min"},
            color="count",
            color_continuous_scale="blues",)
    fig_day.update_layout(coloraxis_showscale=False)
    fig_day



    return


@app.cell
def _(mo):
    mo.md("""
    ## average cyclists by month
    """)
    return


@app.cell
def _(df, px):
    month_labels = {1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
                        7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"}

    monthly = (
            df.groupby(["year", "month"])["count"]
            .mean()
            .reset_index()
    )
    monthly["month_name"] = monthly["month"].map(month_labels)
    monthly["year"] = monthly["year"].astype(str)

    fig_month = px.line(
            monthly,
            x="month",
            y="count",
            color="year",
            markers=True,
            labels={"month": "", "count": "avg cyclists per 15 min", "year": "year"},)
    fig_month.update_layout(
            xaxis=dict(tickmode="array", tickvals=list(month_labels.keys()),
                       ticktext=list(month_labels.values())),
            hovermode="x unified",
    )
    fig_month


    return


@app.cell
def _(mo):
    mo.md("""
    ## traffic heatmap — hour of day vs day of week
    """)
    return


@app.cell
def _(df, px):
    day_labels_short = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}

    heatmap_data = (
            df.groupby(["day_of_week", "hour"])["count"]
            .mean()
            .reset_index()
    )
    heatmap_pivot = heatmap_data.pivot(index="day_of_week", columns="hour", values="count")
    heatmap_pivot.index = [day_labels_short[i] for i in heatmap_pivot.index]

    fig_heat = px.imshow(
            heatmap_pivot,
            labels={"x": "hour of day", "y": "", "color": "avg cyclists"},
            color_continuous_scale="blues",
            aspect="auto",)
    fig_heat



    return


@app.cell
def _(mo):
    mo.md("""
    ## top 15 busiest stations (daily average)
    """)
    return


@app.cell
def _(df, px):
    top_sites = (df.groupby(["site_id", "site_name", "municipality"])["count"].sum() .reset_index())
    
    top_sites["daily_avg"] = top_sites["count"] / df["start_time"].dt.date.nunique()
    top_sites = top_sites.nlargest(15, "daily_avg")

    fig_top = px.bar(
            top_sites.sort_values("daily_avg"),
            x="daily_avg",
            y="site_name",
            orientation="h",
            labels={"daily_avg": "avg cyclists per day", "site_name": ""},
            color="daily_avg",
            color_continuous_scale="blues", )
    fig_top.update_layout(coloraxis_showscale=False)
    fig_top


    return


@app.cell
def _(mo):
    mo.md("""
    ## station map — cycling intensity across flanders
    """)
    return


@app.cell
def _(df, px):
    site_totals = (
        df.groupby(["site_id", "site_name", "municipality", "lat", "lon"])["count"]
        .mean()
        .reset_index()
        .rename(columns={"count": "avg_count"})
    )
    # drop rows with missing coordinates or counts — scatter_map rejects NaN in size
    site_totals = site_totals.dropna(subset=["lat", "lon", "avg_count"])

    fig_map = px.scatter_map(
        site_totals,
        lat="lat",
        lon="lon",
        size="avg_count",
        color="avg_count",
        hover_name="site_name",
        hover_data={"municipality": True, "avg_count": ":.1f", "lat": False, "lon": False},
        color_continuous_scale="blues",
        size_max=30,
        zoom=7,
        center={"lat": 50.98, "lon": 4.1},
        map_style="carto-positron",
        labels={"avg_count": "avg cyclists per 15 min"},
    )
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig_map
    return


@app.cell
def _(mo):
    mo.md("""
    ## seasonal breakdown
    """)
    return


@app.cell
def _(df, px):
    season_map = {12: "winter", 1: "winter", 2: "winter",
                  3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer",
                  9: "autumn", 10: "autumn", 11: "autumn"}

    df_season = df.copy()
    df_season["season"] = df_season["month"].map(season_map)

    seasonal = (
        df_season.groupby(["season", "hour"])["count"]
        .mean()
        .reset_index()
    )

    fig_season = px.line(
        seasonal,
        x="hour",
        y="count",
        color="season",
        markers=True,
        labels={"hour": "hour of day", "count": "avg cyclists per 15 min", "season": ""},
        color_discrete_map={
            "spring": "#2ca02c",
            "summer": "#ff7f0e",
            "autumn": "#8c564b",
            "winter": "#1f77b4",
        },
        category_orders={"season": ["spring", "summer", "autumn", "winter"]},
    )
    fig_season.update_layout(
        xaxis=dict(tickmode="linear", dtick=1),
        hovermode="x unified",
    )
    fig_season
    return


if __name__ == "__main__":
    app.run()
