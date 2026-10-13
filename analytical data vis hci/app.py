from pathlib import Path
import math

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, dcc, html


DATA_PATH = Path(__file__).with_name("global_emissions.csv")
YEARS = list(range(1992, 2019))
SEVERITY_COLUMNS = [
    "Emissions.Production.CO2.Total",
    "Emissions.Production.N2O",
    "Emissions.Production.CH4",
]
STACK_COLUMNS = [
    "Emissions.Production.CH4",
    "Emissions.Production.N2O",
    "Emissions.Production.CO2.Cement",
    "Emissions.Production.CO2.Coal",
    "Emissions.Production.CO2.Gas",
    "Emissions.Production.CO2.Oil",
    "Emissions.Production.CO2.Flaring",
    "Emissions.Production.CO2.Other",
]
STACK_LABELS = ["CH4", "N2O", "CO2 cement", "CO2 coal", "CO2 gas", "CO2 oil", "CO2 flaring", "CO2 other"]
STACK_COLORS = ["#28634b", "#8baf55", "#8d99ae", "#343a40", "#238a91", "#df765c", "#e5a23b", "#577590"]

data = pd.read_csv(DATA_PATH)
data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
numeric_columns = list(dict.fromkeys([
    "Country.GDP",
    "Country.Population",
    *SEVERITY_COLUMNS,
    *STACK_COLUMNS,
]))
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")
data = data[data["Country.Code"].notna() & data["Year"].between(1992, 2018)].copy()
data["Severity"] = data[SEVERITY_COLUMNS].fillna(0).sum(axis=1)
MAX_SEVERITY = max(float(data["Severity"].max()), 1.0)

app = Dash(__name__, title="Global emissions | 1992-2018")
server = app.server


def format_value(value, suffix=""):
    if pd.isna(value):
        return "No data"
    return f"{value:,.2f}{suffix}"


def map_data_for_year(year):
    start_year = math.floor(year)
    end_year = min(start_year + 1, YEARS[-1])
    progress = year - start_year
    start_data = data[data["Year"] == start_year].set_index("Country.Code")
    end_data = data[data["Year"] == end_year].set_index("Country.Code")
    country_codes = start_data.index.intersection(end_data.index)
    year_data = start_data.loc[country_codes].copy().reset_index()
    year_data["Severity"] = (
        start_data.loc[country_codes, "Severity"].to_numpy() * (1 - progress)
        + end_data.loc[country_codes, "Severity"].to_numpy() * progress
    )
    return year_data


def build_map(year):
    year_data = map_data_for_year(year)
    figure = go.Figure(
        go.Choropleth(
            locations=year_data["Country.Code"],
            z=year_data["Severity"],
            locationmode="ISO-3",
            text=year_data["Country.Name"],
            customdata=year_data[["Country.Name", "Country.Code", "Severity"]].to_numpy(),
            colorscale=[[0, "#fde8e3"], [0.28, "#f4a18e"], [0.62, "#db5948"], [1, "#7b171b"]],
            zmin=0,
            zmax=MAX_SEVERITY,
            marker_line_color="#ffffff",
            marker_line_width=0.55,
            colorbar={"title": {"text": "Severity<br>score", "side": "right"}, "thickness": 11, "len": 0.7},
            hovertemplate="<b>%{customdata[0]}</b><br>ISO-3: %{customdata[1]}<br>Severity: %{customdata[2]:,.2f}<extra></extra>",
            showscale=True,
        )
    )
    figure.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=False,
        showcountries=True,
        countrycolor="#ffffff",
        showland=True,
        landcolor="#e8edeb",
        showocean=True,
        oceancolor="#f5f8f6",
        bgcolor="rgba(0,0,0,0)",
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        geo={"domain": {"x": [0, 1], "y": [0, 0.96]}},
        transition={"duration": 80, "easing": "linear"},
        uirevision="world-map",
    )
    return figure


def build_history(code):
    figure = go.Figure()
    country_data = data[data["Country.Code"] == code].set_index("Year") if code else pd.DataFrame()
    for column, label, color in zip(STACK_COLUMNS, STACK_LABELS, STACK_COLORS):
        values = [
            float(country_data.at[year, column]) if year in country_data.index and pd.notna(country_data.at[year, column]) else 0
            for year in YEARS
        ]
        figure.add_trace(go.Bar(name=label, x=YEARS, y=values, marker_color=color, hovertemplate=f"{label}: %{{y:,.2f}}<extra></extra>"))
    figure.update_layout(
        barmode="stack",
        margin={"l": 52, "r": 14, "t": 8, "b": 68},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "'IBM Plex Sans', sans-serif", "color": "#52615c", "size": 11},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#172b26", "font_color": "#ffffff"},
        legend={"orientation": "h", "y": -0.31, "x": 0, "font": {"size": 10}, "itemwidth": 88},
        xaxis={"tickmode": "array", "tickvals": YEARS, "ticktext": [str(year) for year in YEARS], "tickangle": -45, "showgrid": False, "fixedrange": True},
        yaxis={"title": "Emissions (dataset units)", "rangemode": "tozero", "gridcolor": "#e4e9e6", "zerolinecolor": "#cfd8d3", "fixedrange": True},
        bargap=0.16,
    )
    return figure


def interpolate_country_row(code, year):
    start_year = math.floor(year)
    end_year = min(start_year + 1, YEARS[-1])
    progress = year - start_year
    country_data = data[data["Country.Code"] == code].set_index("Year")
    if start_year not in country_data.index or end_year not in country_data.index:
        return None

    row = country_data.loc[start_year].copy()
    end_row = country_data.loc[end_year]
    for column in numeric_columns:
        start_value = row[column]
        end_value = end_row[column]
        if pd.notna(start_value) and pd.notna(end_value):
            row[column] = start_value + (end_value - start_value) * progress
    row["Severity"] = sum(row[column] for column in SEVERITY_COLUMNS if pd.notna(row[column]))
    return row


app.layout = html.Div(
    [
        dcc.Store(id="locked-country", data=None),
        dcc.Interval(id="year-timer", interval=100, n_intervals=0, disabled=True),
        html.Header(
            [
                html.Div([html.Div("CLIMATE DATA / EXPLORER", className="eyebrow"), html.H1("Global emissions")], className="brand"),
                html.Div([html.Span("Annual production", className="header-tag"), html.Span("1992 — 2018", className="header-period")], className="header-meta"),
            ],
            className="topbar",
        ),
        html.Main(
            [
                html.Section(
                    [
                        html.Div(
                            [
                                html.Div([html.Div("01 / GEOGRAPHY", className="eyebrow"), html.H2("Emissions by country")]),
                                html.Div([html.Span(id="year-label", className="year-readout"), html.Button("▶  Play", id="play-button", n_clicks=0, title="Play year animation", className="play-button")], className="map-controls-top"),
                            ],
                            className="panel-heading map-heading",
                        ),
                        dcc.Graph(id="world-map", figure=build_map(1992), config={"displayModeBar": False, "scrollZoom": False}, className="map-graph"),
                        dcc.Slider(
                            id="year-slider",
                            min=1992,
                            max=2018,
                            step=0.1,
                            value=1992,
                            marks={year: {"label": str(year), "style": {"fontSize": "10px", "color": "#77847f"}} for year in [1992, 1996, 2000, 2004, 2008, 2012, 2016, 2018]},
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="year-slider",
                        ),
                        html.Div("Combined severity = CO2 total + N2O + CH4. Countries without a value for the selected year are not selectable.", className="map-note"),
                    ],
                    className="panel map-panel",
                ),
                html.Section(
                    [
                        html.Div([html.Div("02 / COUNTRY PROFILE", className="eyebrow"), html.H2("Specified country")], className="panel-heading"),
                        html.Div(id="country-detail", className="country-detail"),
                    ],
                    className="panel detail-panel",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.Div([html.Div("03 / TIME SERIES", className="eyebrow"), html.H2("Historical emissions mix")]),
                                html.Div(id="chart-country-label", className="chart-country-label"),
                            ],
                            className="panel-heading chart-heading",
                        ),
                        dcc.Graph(id="history-chart", figure=build_history(None), config={"displayModeBar": False, "scrollZoom": False}, className="history-graph"),
                    ],
                    className="panel chart-panel",
                ),
            ],
            className="dashboard-grid",
        ),
        html.Footer("SOURCE FILE  global_emissions.csv   ·   COUNTRY GEOMETRY  ISO-3", className="footer"),
    ],
    className="app-shell",
)


@app.callback(Output("year-label", "children"), Output("world-map", "figure"), Input("year-slider", "value"))
def update_map(year):
    year_data = map_data_for_year(year)
    figure_patch = Patch()
    figure_patch["data"][0]["locations"] = year_data["Country.Code"].tolist()
    figure_patch["data"][0]["z"] = year_data["Severity"].tolist()
    figure_patch["data"][0]["customdata"] = year_data[["Country.Name", "Country.Code", "Severity"]].to_numpy().tolist()
    return f"{year:.1f}", figure_patch


@app.callback(Output("year-timer", "disabled"), Output("play-button", "children"), Input("play-button", "n_clicks"), State("year-timer", "disabled"), prevent_initial_call=True)
def toggle_animation(_, is_disabled):
    now_playing = is_disabled
    return not now_playing, "Ⅱ  Pause" if now_playing else "▶  Play"


app.clientside_callback(
    """function(tick, year) {
        if (tick === undefined || year === undefined) {
            return window.dash_clientside.no_update;
        }
        const nextYear = Math.round((year + 0.1) * 10) / 10;
        return nextYear >= 2018 ? 1992 : nextYear;
    }""",
    Output("year-slider", "value"),
    Input("year-timer", "n_intervals"),
    State("year-slider", "value"),
    prevent_initial_call=True,
)


@app.callback(
    Output("locked-country", "data"),
    Output("world-map", "clickData"),
    Input("world-map", "clickData"),
    State("locked-country", "data"),
    prevent_initial_call=True,
)
def lock_country(click_data, locked_code):
    if not click_data or not click_data.get("points"):
        return locked_code, None
    clicked_code = click_data["points"][0].get("location")
    return (None if clicked_code == locked_code else clicked_code), None


@app.callback(
    Output("country-detail", "children"),
    Input("world-map", "hoverData"),
    Input("locked-country", "data"),
    Input("year-slider", "value"),
)
def update_country_panel(hover_data, locked_code, year):
    hovered_code = None
    if hover_data and hover_data.get("points"):
        hovered_code = hover_data["points"][0].get("location")
    code = locked_code or hovered_code
    if not code:
        prompt = html.Div([html.Div("—", className="detail-placeholder-mark"), html.P("Hover a country on the map to inspect its profile and emissions history.")], className="empty-detail")
        return prompt

    row = interpolate_country_row(code, year)
    if row is None:
        return html.Div("No data for this country and year.", className="empty-detail")
    name = row["Country.Name"]
    locked = locked_code is not None
    severity = row["Severity"]
    detail = html.Div(
        [
            html.Div([html.Span("LOCKED SELECTION" if locked else "LIVE PREVIEW", className="selection-state"), html.Span(f"{year:.1f}", className="detail-year")], className="detail-state-row"),
            html.H3(name, className="country-name"),
            html.Div(code, className="country-code"),
            html.Div([html.Div("COMBINED SEVERITY", className="metric-label"), html.Div(format_value(severity), className="severity-value"), html.Div("CO2 total + N2O + CH4", className="metric-caption")], className="severity-block"),
            html.Div(
                [
                    html.Div([html.Div("CO2 production", className="metric-label"), html.Div(format_value(row["Emissions.Production.CO2.Total"]), className="small-metric-value")], className="small-metric"),
                    html.Div([html.Div("Population", className="metric-label"), html.Div(format_value(row["Country.Population"], " people"), className="small-metric-value")], className="small-metric"),
                    html.Div([html.Div("GDP", className="metric-label"), html.Div(format_value(row["Country.GDP"]), className="small-metric-value")], className="small-metric"),
                ],
                className="metadata-grid",
            ),
            html.Div("Click this country on the map to lock or release the selection.", className="detail-hint"),
        ],
        className="detail-content",
    )
    return detail


@app.callback(
    Output("history-chart", "figure"),
    Output("chart-country-label", "children"),
    Input("world-map", "hoverData"),
    Input("locked-country", "data"),
)
def update_history_chart(hover_data, locked_code):
    hovered_code = None
    if hover_data and hover_data.get("points"):
        hovered_code = hover_data["points"][0].get("location")
    code = locked_code or hovered_code
    if not code:
        return build_history(None), "Hover a country to preview its history"
    country = data[data["Country.Code"] == code].iloc[0]
    return build_history(code), f"{country['Country.Name']} · {code}"


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)