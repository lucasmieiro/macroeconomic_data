# 📊 Macro Dashboard — USA & Brazil (Streamlit)

Bloomberg‑style, mobile‑friendly macro dashboard that shows key indicators for the USA and Brazil and lets users download the underlying data as CSV.

**Live features**
- Common dark design across charts
- Date filter
- USA (FRED): Fed Funds, CPI, GDP, Unemployment, Retail Sales
- Brazil: Selic & IPCA (BCB SGS), Retail (SGS), GDP & Unemployment (World Bank)
- Per‑series and combined CSV downloads
- Works nicely on phones

> No API keys required. Data is fetched from public APIs (FRED via `pandas_datareader`, BCB SGS JSON, World Bank JSON).

---

## 🚀 Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL and a network URL. Open it in your browser (mobile responsive).

---

## 📦 Deploy on Streamlit Community Cloud

1. Push this repository to your GitHub account.
2. Go to https://streamlit.io/cloud and choose **New app**.
3. Select this repo, set **Main file path** to `app.py`, and deploy.
4. (Optional) In **Advanced settings**, set **Python version** to 3.10+.

---

## 🔧 Configuration

The sidebar lets you:
- Change the **start date** (filter)
- Override any **series ID** (FRED or BCB SGS) or **World Bank indicator**

**Defaults**
- USA (FRED): `FEDFUNDS`, `CPIAUCSL`, `GDP`, `UNRATE`, `RSAFS`
- Brazil (SGS): Selic `432` (target % a.a.), IPCA m/m `%` `433`, Retail index `1552`
- World Bank (BRA): GDP `NY.GDP.MKTP.CN` (current LCU), Unemployment `%` `SL.UEM.TOTL.ZS`

If a series fails, the app shows a warning but continues running.

---

## 🗃️ Data Sources

- **FRED** via `pandas_datareader`  
- **Banco Central do Brasil (SGS API)**  
- **World Bank (JSON API)**

---

## 🧰 Tech

- Python, Streamlit
- pandas, pandas-datareader, requests
- Plotly (dark theme) for charts

---

## 📁 Files

- `app.py` – Streamlit application
- `requirements.txt` – Python dependencies
- `README.md` – This file

---

## ✅ License

MIT – do whatever you want, no warranty.
