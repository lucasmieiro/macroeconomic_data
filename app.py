# app.py
# Streamlit macro dashboard (USA + Brazil) with Bloomberg-like look, CSV download
# Run locally: streamlit run app.py

from typing import Optional
from datetime import date, datetime
import io
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
import time
# -----------------------------
# Page / Theme
# -----------------------------
st.set_page_config(
    page_title="Macro Dashboard – USA & Brazil",
    page_icon="📈",
    layout="wide",
)

# Bloomberg-ish minimal CSS tweaks for a crisp dark look
st.markdown(
    """
    <style>
      :root {
        --bloom-font: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      }
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }
      h1, h2, h3 { letter-spacing: 0.2px; }
      .metric-row { gap: 1rem; }
      .download-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
      .stPlotlyChart { border-radius: 10px; }
      .css-zt5igj, .e1f1d6gn4 { font-family: var(--bloom-font) !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Helpers
# -----------------------------

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (StreamlitApp; +https://streamlit.io)",
    "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
}

@st.cache_data(show_spinner=False)
def fetch_fred(series_id: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Fetch US series from FRED.
    Priority:
      1) Official FRED API (requires free API key) if provided (st.secrets['FRED_API_KEY'] or sidebar).
      2) fredgraph.csv fallback (no key). Handles 'DATE' or 'observation_date'.
    """
    api_key = st.session_state.get("fred_api_key") or st.secrets.get("FRED_API_KEY", None)
    try:
        if api_key:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
            }
            if start:
                params["observation_start"] = pd.to_datetime(start).strftime("%Y-%m-%d")
            if end:
                params["observation_end"] = pd.to_datetime(end).strftime("%Y-%m-%d")
            url = "https://api.stlouisfed.org/fred/series/observations"
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            rows = js.get("observations", [])
            if not rows:
                return pd.DataFrame(columns=["value", "series"])
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "value"]].dropna()
            df = df.set_index("date").sort_index()
            df["series"] = series_id
            return df

        # Fallback: fredgraph.csv (no key)
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url, headers=UA_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text
        if "<!DOCTYPE html" in text.lower():
            raise ValueError("FRED returned HTML instead of CSV (blocked).")
        df = pd.read_csv(io.StringIO(text))
        # Accept several date header variants
        date_col = None
        for candidate in ["DATE", "date", "observation_date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            raise ValueError(f"Unexpected CSV columns: {df.columns.tolist()}")
        value_cols = [c for c in df.columns if c != date_col]
        if not value_cols:
            return pd.DataFrame(columns=["value", "series"])
        value_col = value_cols[0]
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df[["date", "value"]].dropna()
        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]
        df = df.set_index("date").sort_index()
        df["series"] = series_id
        return df
    except Exception as e:
        st.warning(f"FRED fetch failed for `{series_id}`: {e}")
        return pd.DataFrame(columns=["value", "series"])

@st.cache_data(show_spinner=False)
def fetch_bcb_sgs(series_id: int, start: Optional[str], end: Optional[str],
                  tries: int = 3, timeout: int = 60) -> pd.DataFrame:
    """
    Fetch a Banco Central do Brasil SGS series with retries, longer timeout, and
    a 10-year clamp for daily series (as required by SGS).
    """
    # compute start/end and clamp to 10 years (daily-series rule)
    end_dt = pd.to_datetime(end) if end else pd.Timestamp.today()
    start_dt = pd.to_datetime(start) if start else end_dt - pd.Timedelta(days=3650)
    if (end_dt - start_dt).days > 3650:
        start_dt = end_dt - pd.Timedelta(days=3650)
    if start_dt > end_dt:
        start_dt = end_dt - pd.Timedelta(days=30)

    start_d = start_dt.strftime("%d/%m/%Y")
    end_d   = end_dt.strftime("%d/%m/%Y")
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
        f"?formato=json&dataInicial={start_d}&dataFinal={end_d}"
    )

    last_err = None
    for i in range(tries):
        try:
            resp = requests.get(
                url,
                headers={
                    "Accept": "application/json; charset=utf-8",
                    "User-Agent": UA_HEADERS["User-Agent"],
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
            if df.empty or "data" not in df or "valor" not in df:
                return pd.DataFrame(columns=["value", "series"])
            df["date"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
            df["value"] = pd.to_numeric(df["valor"].str.replace(",", "."), errors="coerce")
            df = df[["date", "value"]].dropna().set_index("date").sort_index()
            df["series"] = str(series_id)
            return df
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            # exponential backoff: 1.5s, 3s, 6s...
            time.sleep(1.5 * (2 ** i))
        except Exception as e:
            st.warning(f"BCB SGS fetch failed for `{series_id}`: {e}")
            return pd.DataFrame(columns=["value", "series"])

    st.warning(f"BCB SGS fetch failed for `{series_id}` after {tries} attempts: {last_err}")
    return pd.DataFrame(columns=["value", "series"])


@st.cache_data(show_spinner=False)
def fetch_worldbank(indicator: str, country_code: str = "USA", start_year: int = 1990, end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch World Bank series: indicator (e.g., 'NY.GDP.MKTP.CN', 'SL.UEM.TOTL.ZS'), country code (e.g., 'USA','BRA').
    """
    try:
        url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page=20000"
        r = requests.get(url, headers={"Accept": "application/json", "User-Agent": UA_HEADERS["User-Agent"]}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list) or len(payload) < 2:
            raise ValueError("Unexpected World Bank response")
        rows = payload[1] or []
        df = pd.DataFrame(rows)
        if df.empty or "date" not in df or "value" not in df:
            return pd.DataFrame(columns=["value", "series"])
        df = df[["date", "value"]].dropna()
        df["year"] = pd.to_numeric(df["date"], errors="coerce")
        df = df.dropna(subset=["year"])
        if end_year is None:
            end_year = datetime.today().year
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
        # annual -> end-of-year timestamp
        df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-12-31", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date").set_index("date")
        df["series"] = indicator
        return df[["value", "series"]]
    except Exception as e:
        st.warning(f"World Bank fetch failed for `{country_code}:{indicator}`: {e}")
        return pd.DataFrame(columns=["value", "series"])

def px_line_common(df: pd.DataFrame, title: str, y_label: str):
    fig = px.line(
        df.reset_index(),
        x="date",
        y="value",
        title=title,
        template="plotly_dark",
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title=None,
        yaxis_title=y_label,
        font=dict(family="ui-monospace", size=13),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_traces(line=dict(width=2))
    return fig

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")

def scale_df(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale only the 'value' column by (1/factor); keep 'series' as-is."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "value" in out.columns:
        out["value"] = pd.to_numeric(out["value"], errors="coerce") / factor
    return out

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("**Date filter**")
default_start = st.sidebar.date_input("Start date", value=date(2010, 1, 1))
default_end = st.sidebar.date_input("End date", value=date.today())
start_str = default_start.isoformat()
end_str = default_end.isoformat()

st.sidebar.markdown("**Optional: FRED API key**")
fred_key = st.sidebar.text_input("Enter FRED API key (free)", value=st.secrets.get("FRED_API_KEY", ""), type="password")
if fred_key:
    st.session_state["fred_api_key"] = fred_key
st.sidebar.markdown("---")

st.sidebar.markdown("**Series (you can change these):**")

# USA (FRED) ids
us_series = {
    "Fed Funds (USA)": st.sidebar.text_input("FRED Fed Funds ID", value="FEDFUNDS"),
    "CPI (USA)": st.sidebar.text_input("FRED CPI ID", value="CPIAUCSL"),
    "GDP (USA, nominal)": st.sidebar.text_input("FRED GDP ID", value="GDP"),
    "Unemployment Rate (USA)": st.sidebar.text_input("FRED Unemployment ID", value="UNRATE"),
    "Retail Sales (USA)": st.sidebar.text_input("FRED Retail Sales ID", value="RSAFS"),
}

# Brazil (SGS / World Bank) defaults
brazil_sgs_selic = st.sidebar.text_input("BCB SGS Selic (target) ID", value="432")
brazil_sgs_ipca = st.sidebar.text_input("BCB SGS IPCA m/m % ID", value="433")
brazil_sgs_retail = st.sidebar.text_input("BCB SGS Retail Sales ID", value="1552")
wb_gdp_bra = st.sidebar.text_input("World Bank GDP (BRA) indicator", value="NY.GDP.MKTP.CN")
wb_unemp_bra = st.sidebar.text_input("World Bank Unemployment (BRA) indicator", value="SL.UEM.TOTL.ZS")

mobile_compact = st.sidebar.checkbox("Compact charts for mobile", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If any series fails, tweak IDs here. The app will keep running.")

# -----------------------------
# Title & Intro
# -----------------------------
st.title("📊 Macro Dashboard — USA & Brazil")
st.caption("Bloomberg-style dark theme • common chart design • CSV downloads • mobile-friendly")

# -----------------------------
# Load Data
# -----------------------------
with st.spinner("Fetching data..."):
    # USA (FRED via API or CSV)
    df_fedfunds = fetch_fred(us_series["Fed Funds (USA)"], start_str, end_str)
    df_cpi_us = fetch_fred(us_series["CPI (USA)"], start_str, end_str)
    df_gdp_us = fetch_fred(us_series["GDP (USA, nominal)"], start_str, end_str)
    df_unrate_us = fetch_fred(us_series["Unemployment Rate (USA)"], start_str, end_str)
    df_retail_us = fetch_fred(us_series["Retail Sales (USA)"], start_str, end_str)

    # Brazil (BCB SGS + World Bank)
    df_selic = fetch_bcb_sgs(int(brazil_sgs_selic), start_str, end_str) if brazil_sgs_selic.isdigit() else pd.DataFrame()
    df_ipca = fetch_bcb_sgs(int(brazil_sgs_ipca), start_str, end_str) if brazil_sgs_ipca.isdigit() else pd.DataFrame()
    df_retail_br = fetch_bcb_sgs(int(brazil_sgs_retail), start_str, end_str) if brazil_sgs_retail.isdigit() else pd.DataFrame()

    df_gdp_br = fetch_worldbank(wb_gdp_bra, "BRA", start_year=pd.to_datetime(start_str).year, end_year=pd.to_datetime(end_str).year)
    df_unemp_br = fetch_worldbank(wb_unemp_bra, "BRA", start_year=pd.to_datetime(start_str).year, end_year=pd.to_datetime(end_str).year)

# -----------------------------
# Layout
# -----------------------------
tabs = st.tabs([
    "Overview",
    "USA",
    "Brazil",
    "Downloads"
])

# -----------------------------
# Overview Tab
# -----------------------------
with tabs[0]:
    st.subheader("Overview")
    st.markdown("Latest prints (last available):")

    def latest_value(df: pd.DataFrame):
        if df is None or df.empty:
            return np.nan, "—"
        last_date = df.index.max()
        return float(df.loc[last_date, "value"]), last_date.strftime("%Y-%m-%d")

    # NOTE: This dict lives under the 'with tabs[0]:' indentation (8 spaces)
        metrics = {
        "Fed Funds (USA, %)": df_fedfunds,
        "CPI (USA, index)": df_cpi_us,
        "GDP (USA, $tn)": scale_df(df_gdp_us, 1e3) if not df_gdp_us.empty else df_gdp_us,
        "Unemp (USA, %)": df_unrate_us,
        "Retail (USA, $m)": df_retail_us,

        # Brazil – friendlier labels
        "Selic (meta, % a.a.)": df_selic,
        "IPCA mensal (% m/m)": df_ipca,
        "PIB Brasil (R$ correntes)": df_gdp_br,
        "Desemprego (%, IBGE)": df_unemp_br,
        "Varejo – índice (PMC)": df_retail_br,
    }


    # display metrics in 2 rows
    items = list(metrics.items())
    half = int(np.ceil(len(items) / 2))
    for row in [items[:half], items[half:]]:
        cols = st.columns(len(row))
        for (label, df_), c in zip(row, cols):
            with c:
                val, when = latest_value(df_)
                if np.isnan(val):
                    st.metric(label, "—", help="No data")
                else:
                    st.metric(label, f"{val:,.2f}", help=f"Last: {when}")

# -----------------------------
# USA Tab
# -----------------------------
with tabs[1]:
    st.subheader("United States")
    if not df_fedfunds.empty:
        st.plotly_chart(px_line_common(df_fedfunds, "Fed Funds Rate (Effective)", "%"), use_container_width=True)
    if not df_cpi_us.empty:
        st.plotly_chart(px_line_common(df_cpi_us, "CPI (1982–84=100)", "Index"), use_container_width=True)
    if not df_gdp_us.empty:
        st.plotly_chart(px_line_common(df_gdp_us, "GDP (Nominal)", "USD (Billions)"), use_container_width=True)
    if not df_unrate_us.empty:
        st.plotly_chart(px_line_common(df_unrate_us, "Unemployment Rate", "%"), use_container_width=True)
    if not df_retail_us.empty:
        st.plotly_chart(px_line_common(df_retail_us, "Retail & Food Services Sales", "USD (Millions)"), use_container_width=True)

# -----------------------------
# Brazil Tab
# -----------------------------
with tabs[2]:
    st.subheader("Brasil")
    if not df_selic.empty:
        st.plotly_chart(px_line_common(df_selic, "Selic – taxa meta (% a.a.)", "% a.a."), use_container_width=True)
    else:
        st.info("Selic: ajuste o ID da série SGS na barra lateral (ex.: 432 = meta diária).")

    if not df_ipca.empty:
        st.plotly_chart(px_line_common(df_ipca, "IPCA – variação mensal (%)", "% m/m"), use_container_width=True)
    else:
        st.info("IPCA: ajuste o ID da série SGS (ex.: 433 = var. mensal).")

    if not df_gdp_br.empty:
        st.plotly_chart(px_line_common(df_gdp_br, "PIB (preços correntes, anual)", "R$ correntes"), use_container_width=True)
    else:
        st.info("PIB (Brasil): verifique o indicador do World Bank (padrão: NY.GDP.MKTP.CN).")

    if not df_unemp_br.empty:
        st.plotly_chart(px_line_common(df_unemp_br, "Desemprego – taxa anual (%)", "% da força de trabalho"), use_container_width=True)
    else:
        st.info("Desemprego (Brasil): indicador padrão WB: SL.UEM.TOTL.ZS.")

    if not df_retail_br.empty:
        st.plotly_chart(px_line_common(df_retail_br, "Varejo (PMC) – índice", "Índice"), use_container_width=True)
    else:
        st.info("Varejo (PMC): ajuste o ID da série SGS (ex.: 1552).")

# -----------------------------
# Downloads Tab
# -----------------------------
with tabs[3]:
    st.subheader("Downloads")
    st.write("Download clean CSVs for each series, or a combined file with all columns aligned by date.")

    labeled = {
        "fedfunds_usa": df_fedfunds.rename(columns={"value": "fedfunds_usa"}),
        "cpi_usa": df_cpi_us.rename(columns={"value": "cpi_usa"}),
        "gdp_usa": df_gdp_us.rename(columns={"value": "gdp_usa"}),
        "unemp_usa": df_unrate_us.rename(columns={"value": "unemp_usa"}),
        "retail_usa": df_retail_us.rename(columns={"value": "retail_usa"}),

        "selic_bra": df_selic.rename(columns={"value": "selic_bra"}),
        "ipca_bra": df_ipca.rename(columns={"value": "ipca_bra"}),
        "gdp_bra": df_gdp_br.rename(columns={"value": "gdp_bra"}),
        "unemp_bra": df_unemp_br.rename(columns={"value": "unemp_bra"}),
        "retail_bra": df_retail_br.rename(columns={"value": "retail_bra"}),
    }

    # Individual downloads
    st.markdown("**Per-series CSVs**")
    for name, df in labeled.items():
        if df is not None and not df.empty:
            csv_bytes = df_to_csv_bytes(df[[df.columns[-1]]])  # ensure single data col
            st.download_button(
                label=f"⬇️ Download {name}.csv",
                data=csv_bytes,
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"dl_{name}"
            )

    # Combined download – safe concat using Series rename (no column overlap errors)
    st.markdown("---")
    st.markdown("**Combined CSV (all series, aligned by date)**")
    series_list = []
    for name, df in labeled.items():
        if df is not None and not df.empty:
            s = df.iloc[:, -1].rename(name)
            series_list.append(s)
    if not series_list:
        st.info("No data available to combine.")
    else:
        all_df = pd.concat(series_list, axis=1).sort_index()
        st.dataframe(all_df.tail(15), use_container_width=True, height=300)
        st.download_button(
            label="⬇️ Download combined.csv",
            data=df_to_csv_bytes(all_df),
            file_name="combined.csv",
            mime="text/csv",
            key="dl_combined"
        )

# -----------------------------
# Footer / Hints
# -----------------------------
st.markdown("---")
with st.expander("ℹ️ Sources & tips"):
    st.markdown(
        """
- **USA (FRED)** via official API (if key provided) or public **fredgraph.csv** fallback.
- **Brazil (BCB SGS)** for Selic & IPCA (10-year window clamp for daily series); **World Bank** for GDP & Unemployment; **SGS** for retail index.
- Use the Start/End date pickers (End defaults to **today**) to filter the charts and CSVs.
- The app caches results. Change a series ID or date to refresh that fetch.
"""
    )
