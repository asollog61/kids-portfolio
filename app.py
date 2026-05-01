import io
import os
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PCT_STYLE = None

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode

    HAS_AGGRID = True
except Exception:  # pragma: no cover
    HAS_AGGRID = False
    AgGrid = GridOptionsBuilder = JsCode = GridUpdateMode = None

st.set_page_config(
    page_title="Kids Portfolio Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
ALLOWED_ACCOUNTS = ["Eleanor Custodial", "Malcolm Custodial", "Mason Custodial"]
ACCOUNT_DISPLAY = {
    "Eleanor Custodial": "👩 Eleanor",
    "Malcolm Custodial": "👦 Malcolm",
    "Mason Custodial": "👦 Mason",
}
TICKER_ALIASES = {
    "MICROSOFT": "MSFT",
    "APPLE": "AAPL",
    "NVIDIA": "NVDA",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "META": "META",
    "FACEBOOK": "META",
    "TESLA": "TSLA",
    "NETFLIX": "NFLX",
    "AMD": "AMD",
    "CROWDSTRIKE": "CRWD",
    "STARBUCKS": "SBUX",
    "UNITED": "UAL",
    "UNITEDHEALTH": "UNH",
    "LUMENTUM": "LITE",
    "BITCOIN": "IBIT",
    "S&P": "SPY",
    "S AND P": "SPY",
    "SP500": "SPY",
    "SP 500": "SPY",
    "NASDAQ": "QQQ",
    "QQQ": "QQQ",
    "DISNEY": "DIS",
    "COSTCO": "COST",
    "WALMART": "WMT",
    "ROBLOX": "RBLX",
    "COINBASE": "COIN",
    "SHOPIFY": "SHOP",
    "SNAP": "SNAP",
    "SNAPCHAT": "SNAP",
    "SPOTIFY": "SPOT",
    "UBER": "UBER",
    "AIRBNB": "ABNB",
    "PALANTIR": "PLTR",
    "INTEL": "INTC",
    "IBM": "IBM",
    "ORACLE": "ORCL",
    "BOEING": "BA",
    "NIKE": "NKE",
    "MCDONALD'S": "MCD",
    "MCDONALDS": "MCD",
}
DATA_DIR = Path(__file__).resolve().parent / "data"

DARK_GRID_CSS = {
    ".ag-root-wrapper": {"background-color": "#0e1117 !important"},
    ".ag-body-viewport": {"background-color": "#0e1117 !important"},
    ".ag-center-cols-viewport": {"background-color": "#0e1117 !important"},
    ".ag-body-horizontal-scroll-viewport": {"background-color": "#0e1117 !important"},
    ".ag-header": {"background-color": "#16213e !important", "color": "#e0e0e0 !important"},
    ".ag-header-cell-text": {"color": "#e0e0e0 !important"},
    ".ag-row": {"background-color": "#1a1a2e !important", "color": "#e0e0e0 !important"},
    ".ag-row-odd": {"background-color": "#1f1f3a !important"},
    ".ag-cell": {"border-color": "#333 !important", "font-size": "12px !important"},
    ".ag-floating-top-container": {"background-color": "#0e1117 !important"},
    ".ag-floating-top-viewport": {"background-color": "#0e1117 !important"},
    ".ag-floating-top-container .ag-row": {
        "background-color": "#16213e !important",
        "font-weight": "bold !important",
    },
    ".ag-overlay": {"background-color": "#0e1117 !important"},
    ".ag-status-bar": {"background-color": "#0e1117 !important"},
}
if HAS_AGGRID:
    PCT_STYLE = JsCode(
        """
        function(params) {
            if (params.value === undefined || params.value === null || params.value === "") {
                return {};
            }
            const text = String(params.value);
            if (text.includes('(') || text.includes('-')) {
                return {color: '#ff6666', fontWeight: 'bold'};
            }
            if (text.match(/[0-9]/)) {
                return {color: '#32d74b', fontWeight: 'bold'};
            }
            return {};
        }
        """
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _latest_file(pattern: str) -> Path | None:
    files = glob.glob(str(DATA_DIR / pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return Path(files[-1])


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, [c for c in df.columns if not str(c).lower().startswith("unnamed")]]


def _normalize_account_label(raw: str | float | int | None) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    text = str(raw).strip()
    if "*" in text:
        text = text.split("*")[0].strip()
    return text


def _clean_money(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value)
    if text in ("--", ""):
        return np.nan
    text = text.replace("$", "").replace(",", "").replace("+", "").strip()
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        return float(text)
    except ValueError:
        return np.nan


def _clean_number(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).replace(",", "").strip()
    if text in ("", "--"):
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _fmt_money(val: float | int | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if val < 0:
        return f"(${abs(val):,.2f})"
    return f"${val:,.2f}"


def _fmt_pct(val: float | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return f"{val:+.1f}%"


def _fix_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col_map = {}
    new_cols: list[str | None] = []
    for col in df.columns:
        clean = str(col).strip()
        if clean.lower() == "date":
            new_cols.append("Date")
            continue
        if not clean or clean.lower().startswith("unnamed"):
            new_cols.append(None)
            continue
        clean = clean.upper()
        count = col_map.get(clean, 0)
        if count:
            clean = f"{clean}_{count}"
        col_map[str(col).strip().upper()] = count + 1
        new_cols.append(clean)
    df.columns = new_cols
    keep = [i for i, c in enumerate(new_cols) if c is not None]
    df = df.iloc[:, keep]
    return df


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_positions() -> pd.DataFrame:
    path = _latest_file("Portfolio_Positions_*.csv")
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, index_col=False)
    df = _drop_unnamed(df)
    df["Account Name"] = df["Account Name"].map(_normalize_account_label)
    df = df[df["Account Name"].isin(ALLOWED_ACCOUNTS)].copy()
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_activity() -> pd.DataFrame:
    path = _latest_file("Activity_All_Accounts*.csv")
    if not path:
        return pd.DataFrame()
    with path.open("r", encoding="utf-8-sig") as handle:
        lines = handle.readlines()
    header_idx = next((i for i, line in enumerate(lines) if line.lstrip().startswith("Date,")), None)
    if header_idx is None:
        return pd.DataFrame()
    csv_data = "".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_data), dtype=str, index_col=False)
    df = _drop_unnamed(df)
    df["Account"] = df["Account"].map(_normalize_account_label)
    rename_map = {
        "Date": "Run Date",
        "Description": "Action",
        "Security Description": "Description",
        "Price": "Price ($)",
        "Commission": "Commission ($)",
        "Fees": "Fees ($)",
        "Amount": "Amount ($)",
    }
    df = df.rename(columns=rename_map)
    df = df[df["Account"].isin(ALLOWED_ACCOUNTS)].copy()
    if "Run Date" in df.columns:
        df["Run Date"] = pd.to_datetime(df["Run Date"], errors="coerce")
        df = df.sort_values("Run Date", ascending=False)
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_price_book():
    excel_path = DATA_DIR / "prices_dividends_adjusted.xlsx"
    if not excel_path.exists():
        return None

    adj = pd.read_excel(excel_path, sheet_name="AdjClose")
    draw = pd.read_excel(excel_path, sheet_name="Drawdown")
    vol = pd.read_excel(excel_path, sheet_name="Volatility")
    meta = pd.read_excel(excel_path, sheet_name="Metadata")

    def _prep(df):
        df = _fix_price_columns(df)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date")
            df = df.set_index("Date")
        return df

    adj = _prep(adj)
    draw = _prep(draw)
    vol = _prep(vol)

    meta = meta.rename(
        columns={
            "Symbol": "Ticker",
            "info.shortName": "Name",
            "info.longName": "LongName",
            "info.sector": "Sector",
        }
    )
    meta["Ticker"] = meta["Ticker"].astype(str).str.upper()
    meta["Name"] = meta["Name"].fillna(meta["LongName"])
    meta = meta[["Ticker", "Name", "Sector"]].drop_duplicates("Ticker").set_index("Ticker")

    return {"adj": adj, "draw": draw, "vol": vol, "meta": meta}


PRICE_BOOK = load_price_book()


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding: 0.5rem 0.5rem !important; max-width: 100% !important; }
        h1, h2, h3 { margin: 0.25rem 0 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 0px; }
        .stTabs [data-baseweb="tab"] { padding: 6px 12px !important; font-size: 14px !important; }
        .stButton > button {
            font-size: 17px !important;
            padding: 12px !important;
            border-radius: 12px !important;
            min-height: 58px !important;
        }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        div[data-testid="stDataFrame"] table { font-size: 12px !important; }
        div[data-testid="stDataFrame"] th { padding: 3px 6px !important; }
        div[data-testid="stDataFrame"] td { padding: 2px 6px !important; }
        [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
        [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
        [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

        /* Mobile position cards */
        .pos-card {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-left: 4px solid #333;
        }
        .pos-card.positive { border-left-color: #44dd44; }
        .pos-card.negative { border-left-color: #ff4444; }
        .pos-card .sym { font-size: 1.1rem; font-weight: bold; color: #e0e0e0; }
        .pos-card .row { display: flex; justify-content: space-between; margin-top: 4px; }
        .pos-card .label { color: #888; font-size: 0.8rem; }
        .pos-card .val { color: #e0e0e0; font-size: 0.9rem; }
        .pos-card .green { color: #44dd44; }
        .pos-card .red { color: #ff4444; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Account selector
# ---------------------------------------------------------------------------
st.title("📊 Kids Portfolio Monitor")
positions_df = load_positions()
activity_df = load_activity()

if "kids_account" not in st.session_state:
    st.session_state.kids_account = ALLOWED_ACCOUNTS[0]

cols = st.columns(len(ALLOWED_ACCOUNTS))
for idx, account in enumerate(ALLOWED_ACCOUNTS):
    button_type = "primary" if st.session_state.kids_account == account else "secondary"
    if cols[idx].button(ACCOUNT_DISPLAY[account], use_container_width=True, type=button_type, key=f"acct_{idx}"):
        st.session_state.kids_account = account
        st.rerun()

selected_account = st.session_state.kids_account
st.caption(f"Showing data for {ACCOUNT_DISPLAY[selected_account]}")


# ---------------------------------------------------------------------------
# Holdings helpers
# ---------------------------------------------------------------------------
def _positions_for_account(account: str) -> pd.DataFrame:
    if positions_df.empty:
        return pd.DataFrame()
    df = positions_df[positions_df["Account Name"] == account].copy()
    return df


def _transactions_for_account(account: str) -> pd.DataFrame:
    if activity_df.empty:
        return pd.DataFrame()
    df = activity_df[activity_df["Account"] == account].copy()
    return df


def _build_positions_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol or symbol.upper() == "PENDING ACTIVITY":
            continue
        current_value = _clean_money(row.get("Current Value"))
        quantity = _clean_number(row.get("Quantity"))
        price = _clean_money(row.get("Last Price"))
        cost_total = _clean_money(row.get("Cost Basis Total"))
        gl_total = _clean_money(row.get("Total Gain/Loss Dollar"))
        gl_pct = _clean_number(str(row.get("Total Gain/Loss Percent", "")).replace("%", ""))
        today_pct = _clean_number(str(row.get("Today's Gain/Loss Percent", "")).replace("%", ""))

        is_cash = "**" in symbol
        if is_cash and (current_value is None or np.isnan(current_value)):
            continue

        display_symbol = "CASH" if is_cash else symbol
        rows.append(
            {
                "Symbol": display_symbol,
                "Qty": "" if is_cash or quantity is None or np.isnan(quantity) else f"{quantity:,.3f}".rstrip("0").rstrip("."),
                "Price": "" if price is None or np.isnan(price) else _fmt_money(price),
                "Cost": _fmt_money(cost_total),
                "Value": _fmt_money(current_value),
                "P&L": _fmt_money(gl_total),
                "P&L %": _fmt_pct(gl_pct),
                "Today": _fmt_pct(today_pct),
                "_value": current_value or 0,
                "_cost": cost_total or 0,
                "_pl": gl_total or 0,
            }
        )

    if not rows:
        return pd.DataFrame(), {"value": 0, "pl": 0, "cost": 0}

    table = pd.DataFrame(rows)
    totals = {
        "value": table["_value"].sum(),
        "pl": table["_pl"].sum(),
        "cost": table["_cost"].sum(),
    }
    table = table.drop(columns=["_value", "_cost", "_pl"])
    total_pct = (totals["pl"] / totals["cost"] * 100) if totals["cost"] else 0
    total_row = {
        "Symbol": "TOTAL",
        "Qty": "",
        "Price": "",
        "Cost": _fmt_money(totals["cost"]),
        "Value": _fmt_money(totals["value"]),
        "P&L": _fmt_money(totals["pl"]),
        "P&L %": _fmt_pct(total_pct),
        "Today": "",
    }
    return table, {"totals": totals, "total_row": total_row, "total_pct": total_pct}


def _render_table(
    df: pd.DataFrame,
    height: int,
    pinned_row: dict | None = None,
    key: str | None = None,
    highlight_cols: list[str] | None = None,
):
    if df.empty:
        st.info("No data available")
        return
    if HAS_AGGRID:
        builder = GridOptionsBuilder.from_dataframe(df)
        builder.configure_default_column(resizable=True, sortable=True, filter=False)
        builder.configure_grid_options(domLayout="normal", rowHeight=34, headerHeight=30)
        if highlight_cols and PCT_STYLE is not None:
            for col in highlight_cols:
                if col in df.columns:
                    builder.configure_column(col, cellStyle=PCT_STYLE)
        if pinned_row:
            builder.configure_grid_options(pinnedTopRowData=[pinned_row])
        grid_options = builder.build()
        grid_options["autoSizeStrategy"] = {"type": "fitCellContents"}
        AgGrid(
            df,
            gridOptions=grid_options,
            custom_css=DARK_GRID_CSS,
            theme="alpine-dark",
            allow_unsafe_jscode=True,
            update_mode=GridUpdateMode.NO_UPDATE,
            height=height,
            key=key,
        )
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
holdings_tab, charts_tab, performance_tab = st.tabs(["💼 Holdings", "📈 Charts", "📊 Performance"])

# Holdings tab ----------------------------------------------------------------
with holdings_tab:
    positions_subtab, transactions_subtab = st.tabs(["Positions", "Transactions"])

    with positions_subtab:
        acct_positions = _positions_for_account(selected_account)
        if acct_positions.empty:
            st.info("No positions yet. Let's keep saving! 💰")
        else:
            table, context = _build_positions_table(acct_positions)
            totals = context["totals"]
            total_pct = context["total_pct"]
            col_a, col_b = st.columns(2)
            col_a.metric("Account Value", _fmt_money(totals["value"]))
            col_b.metric("Total P&L", _fmt_money(totals["pl"]), delta=_fmt_pct(total_pct))
            st.divider()

            view_mode = st.radio("View", ["Cards", "Table"], horizontal=True, key="pos_view", label_visibility="collapsed")

            if view_mode == "Cards":
                for _, r in table.iterrows():
                    sym = r.get("Symbol", "")
                    if sym == "CASH" and r.get("Value", "") == "":
                        continue
                    pl_str = r.get("P&L", "")
                    is_pos = pl_str and not pl_str.startswith("(") and pl_str != "" and pl_str != "$0.00"
                    is_neg = pl_str.startswith("(") if pl_str else False
                    css_class = "positive" if is_pos else ("negative" if is_neg else "")
                    pl_color = "green" if is_pos else ("red" if is_neg else "val")
                    today_str = r.get("Today", "")
                    today_color = "green" if today_str.startswith("+") else ("red" if today_str.startswith("-") else "val")

                    st.markdown(f"""
                    <div class="pos-card {css_class}">
                        <div class="sym">{sym}</div>
                        <div class="row">
                            <div><span class="label">Qty</span> <span class="val">{r.get('Qty','')}</span></div>
                            <div><span class="label">Price</span> <span class="val">{r.get('Price','')}</span></div>
                        </div>
                        <div class="row">
                            <div><span class="label">Value</span> <span class="val">{r.get('Value','')}</span></div>
                            <div><span class="label">Cost</span> <span class="val">{r.get('Cost','')}</span></div>
                        </div>
                        <div class="row">
                            <div><span class="label">P&L</span> <span class="{pl_color}">{pl_str} {r.get('P&L %','')}</span></div>
                            <div><span class="label">Today</span> <span class="{today_color}">{today_str}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                _render_table(
                    table,
                    height=700,
                    pinned_row=context["total_row"],
                    key="positions_grid",
                    highlight_cols=["P&L", "P&L %", "Today"],
                )
            st.caption(f"{len(table)} holdings")

    with transactions_subtab:
        acct_txn = _transactions_for_account(selected_account)
        if acct_txn.empty:
            st.info("No transactions yet")
        else:
            display = pd.DataFrame(
                {
                    "Date": acct_txn["Run Date"].dt.strftime("%Y-%m-%d"),
                    "Action": acct_txn.get("Action"),
                    "Symbol": acct_txn.get("Symbol"),
                    "Description": acct_txn.get("Description"),
                    "Qty": acct_txn.get("Quantity").apply(_clean_number).apply(
                        lambda x: "" if pd.isna(x) else f"{x:,.3f}".rstrip("0").rstrip(".")
                    ),
                    "Price": acct_txn.get("Price ($)").apply(_clean_money).apply(_fmt_money),
                    "Amount": acct_txn.get("Amount ($)").apply(_clean_money).apply(_fmt_money),
                }
            )
            _render_table(display, height=800, key="txn_grid", highlight_cols=["Amount"])
            st.caption(f"{len(display)} transactions")


# Charts tab -------------------------------------------------------------------
def _resolve_tickers_from_text(text: str) -> list[str]:
    if not text:
        return []
    text = text.upper().strip()
    parts = re.split(r"\s+PLUS\s+|\s+AND\s+|,|/|\s+VS\.?\s+|\s+VERSUS\s+", text)
    tickers = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in TICKER_ALIASES:
            tickers.append(TICKER_ALIASES[part])
        elif part.isalpha() and len(part) <= 5:
            tickers.append(part)
    return list(dict.fromkeys(tickers))


with charts_tab:
    if not PRICE_BOOK:
        st.error("Price workbook missing")
    else:
        adjclose = PRICE_BOOK["adj"]
        if adjclose.empty:
            st.warning("No price data available yet")
        else:
            st.markdown("### Stock face-off")
            st.caption("Type names or tickers — we will normalize returns for easy comparison")
            col1, col2 = st.columns([3, 1])
            user_text = col1.text_input("Stocks", placeholder="Apple, Microsoft, Nvidia", label_visibility="collapsed")
            period = col2.selectbox("Period", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "Max"], index=4)

            selected = _resolve_tickers_from_text(user_text)
            if not selected:
                st.info("🎤 Try saying a company name or typing a ticker above")
            else:
                import plotly.graph_objects as go

                max_date = adjclose.index.max()
                if period == "1M":
                    start_date = max_date - pd.DateOffset(months=1)
                elif period == "3M":
                    start_date = max_date - pd.DateOffset(months=3)
                elif period == "6M":
                    start_date = max_date - pd.DateOffset(months=6)
                elif period == "YTD":
                    start_date = pd.Timestamp(max_date.year, 1, 1)
                elif period == "1Y":
                    start_date = max_date - pd.DateOffset(years=1)
                elif period == "3Y":
                    start_date = max_date - pd.DateOffset(years=3)
                else:
                    start_date = adjclose.index.min()

                fig = go.Figure()
                plotted = 0
                for ticker in selected:
                    if ticker not in adjclose.columns:
                        continue
                    series = adjclose[ticker].dropna()
                    series = series[series.index >= start_date]
                    if series.empty:
                        continue
                    norm = (series / series.iloc[0] - 1) * 100
                    fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode="lines", name=ticker))
                    plotted += 1

                if not plotted:
                    st.warning("No overlapping price history for those tickers")
                else:
                    fig.update_layout(
                        template="plotly_dark",
                        height=420,
                        margin=dict(l=0, r=12, t=40, b=30),
                        title=f"{' vs '.join(selected)} — {period}",
                        yaxis_title="% Return",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
                    )
                    st.plotly_chart(fig, use_container_width=True)


# Performance tab --------------------------------------------------------------
def _return_from_delta(series: pd.Series, delta_days: int) -> float:
    if series.empty:
        return np.nan
    end_date = series.index.max()
    start_date = end_date - pd.Timedelta(days=delta_days)
    history = series[series.index <= start_date]
    if history.empty:
        history = series.iloc[[0]]
    start_val = history.iloc[-1]
    last_val = series.iloc[-1]
    if start_val == 0 or pd.isna(start_val) or pd.isna(last_val):
        return np.nan
    return (last_val / start_val - 1) * 100


def _return_ytd(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    end_date = series.index.max()
    start_date = pd.Timestamp(end_date.year, 1, 1)
    history = series[series.index >= start_date]
    if history.empty:
        return np.nan
    base_val = history.iloc[0]
    last_val = series.iloc[-1]
    if base_val == 0 or pd.isna(base_val) or pd.isna(last_val):
        return np.nan
    return (last_val / base_val - 1) * 100


def _current_drawdown(series: pd.Series) -> tuple[float, int]:
    if series.empty:
        return np.nan, 0
    current = series.iloc[-1] * 100
    days = 0
    for value in reversed(series.dropna().tolist()):
        if value < 0:
            days += 1
        else:
            break
    return current, days


with performance_tab:
    if not PRICE_BOOK:
        st.error("Price workbook missing")
    else:
        adj = PRICE_BOOK["adj"]
        draw = PRICE_BOOK["draw"]
        vol = PRICE_BOOK["vol"]
        meta = PRICE_BOOK["meta"]
        if adj.empty:
            st.warning("No performance data")
        else:
            rows = []
            for ticker in adj.columns:
                series = adj[ticker].dropna()
                if series.empty:
                    continue
                returns = {
                    "Ticker": ticker,
                    "Name": meta.loc[ticker, "Name"] if ticker in meta.index else ticker,
                    "Sector": meta.loc[ticker, "Sector"] if ticker in meta.index else "",
                    "1D": _return_from_delta(series, 1),
                    "1W": _return_from_delta(series, 7),
                    "1M": _return_from_delta(series, 30),
                    "YTD": _return_ytd(series),
                    "1Y": _return_from_delta(series, 365),
                    "3Y": _return_from_delta(series, 365 * 3),
                    "5Y": _return_from_delta(series, 365 * 5),
                }
                dd_series = draw[ticker].dropna() if ticker in draw.columns else pd.Series(dtype=float)
                vol_series = vol[ticker].dropna() if ticker in vol.columns else pd.Series(dtype=float)
                cur_dd, days_dd = _current_drawdown(dd_series)
                returns.update(
                    {
                        "Cur DD": cur_dd,
                        "Days DD": days_dd,
                        "Volatility %": vol_series.iloc[-1] * 100 if not vol_series.empty else np.nan,
                    }
                )
                rows.append(returns)

            perf_df = pd.DataFrame(rows)
            perf_df = perf_df.sort_values("1Y", ascending=False)

            view_mode = st.radio("View", ["Performance", "Drawdown", "Volatility"], horizontal=True)

            if view_mode == "Performance":
                display_cols = [
                    "Ticker",
                    "Name",
                    "Sector",
                    "1D",
                    "1W",
                    "1M",
                    "YTD",
                    "1Y",
                    "3Y",
                    "5Y",
                    "Cur DD",
                    "Days DD",
                ]
            elif view_mode == "Drawdown":
                display_cols = ["Ticker", "Name", "Sector", "Cur DD", "Days DD"]
            else:
                display_cols = ["Ticker", "Name", "Sector", "Volatility %"]

            display = perf_df[display_cols].copy()
            pct_cols = [c for c in display.columns if c not in ("Ticker", "Name", "Sector", "Days DD")]
            for col in pct_cols:
                display[col] = display[col].apply(_fmt_pct)
            if "Days DD" in display.columns:
                display["Days DD"] = display["Days DD"].apply(lambda x: int(x) if not pd.isna(x) else "")
            _render_table(
                display,
                height=max(900, 40 + 28 * len(display)),
                key="perf_grid",
                highlight_cols=pct_cols,
            )
            st.caption(f"{len(display)} trackers")
