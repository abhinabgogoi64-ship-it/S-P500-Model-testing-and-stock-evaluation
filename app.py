"""
S&P 500 Stock Analysis (2014-2017) - Streamlit app
Converted from the S_P500.ipynb notebook (Alpha Quants project).
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="S&P 500 Stock Analysis", page_icon="📈", layout="wide")

DEFAULT_CSV = "S&P 500 Stock Prices 2014-2017.csv"
FEATURES = [
    "open", "high", "low", "volume", "moving_average_7", "moving_average_30",
    "lag_1", "lag_2", "lag_5", "daily_return",
]


# ----------------------------------------------------------------------------
# Data loading + feature engineering (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preparing data...")
def load_and_prepare(file_bytes: bytes | None, path: str, per_symbol_fill: bool):
    """Load the CSV, clean it, and add all engineered features."""
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # --- cleaning: missing open/high/low ---
    price_cols = ["open", "high", "low"]
    missing_before = df[price_cols].isnull().sum()
    if per_symbol_fill:
        # fill gaps from the same stock's neighbouring days (falls back to close)
        df[price_cols] = df.groupby("symbol")[price_cols].transform(lambda s: s.ffill().bfill())
        for c in price_cols:
            df[c] = df[c].fillna(df["close"])
    else:
        # original notebook behaviour: global median
        df[price_cols] = df[price_cols].fillna(df[price_cols].median())

    # --- features ---
    g = df.groupby("symbol")
    df["daily_return"] = df["close"] / df["open"] - 1
    df["rolling_std_5"] = g["daily_return"].transform(lambda x: x.rolling(5).std())
    df["moving_average_7"] = g["close"].transform(lambda x: x.rolling(7).mean())
    df["moving_average_30"] = g["close"].transform(lambda x: x.rolling(30).mean())
    df["lag_1"] = g["close"].shift(1)
    df["lag_2"] = g["close"].shift(2)
    df["lag_5"] = g["close"].shift(5)
    df["price_difference"] = df["close"] - df["open"]
    df["high_low_spread"] = df["high"] - df["low"]
    return df, missing_before


def fig_show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Sidebar: data source
# ----------------------------------------------------------------------------
st.sidebar.title("📈 S&P 500 Analysis")
uploaded = st.sidebar.file_uploader("Upload the stock prices CSV", type="csv")
per_symbol_fill = st.sidebar.checkbox(
    "Fill missing prices per stock",
    value=True,
    help="Unchecked = notebook behaviour (global median). The global median creates "
         "fake rows such as open=64.97 for stocks trading at $500+.",
)

if uploaded is None and not os.path.exists(DEFAULT_CSV):
    st.title("S&P 500 Stock Analysis")
    st.info(
        "Upload `S&P 500 Stock Prices 2014-2017.csv` in the sidebar to begin. "
        "Expected columns: symbol, date, open, high, low, close, volume."
    )
    st.stop()

df, missing_before = load_and_prepare(
    uploaded.getvalue() if uploaded is not None else None, DEFAULT_CSV, per_symbol_fill
)
symbols = sorted(df["symbol"].unique())

st.title("S&P 500 Stock Analysis (2014-2017)")
st.caption("Alpha Quants: returns, volatility, trading volume, trends, ML prediction and forecasting.")

tabs = st.tabs([
    "Overview", "Top Stocks", "Market Trends", "Returns & Statistics",
    "Stock Explorer", "Forecast", "ML Models", "Data Quality",
])

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Companies", df["symbol"].nunique())
    c3.metric("Trading days", df["date"].nunique())
    c4.metric("Date range", f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest close", f"${df['close'].max():,.2f}")
    c2.metric("Lowest close", f"${df['close'].min():,.2f}")
    c3.metric("Average close", f"${df['close'].mean():,.2f}")
    c4.metric("Total volume", f"{df['volume'].sum():,.0f}")

    st.subheader("Sample data")
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader("Descriptive statistics")
    st.dataframe(df[["open", "high", "low", "close", "volume"]].describe(), use_container_width=True)

# ----------------------------------------------------------------------------
# Top stocks
# ----------------------------------------------------------------------------
with tabs[1]:
    n = st.slider("How many stocks to show", 5, 50, 10, key="topn")
    g = df.groupby("symbol")

    first_close = g["close"].first()
    last_close = g["close"].last()
    total_return = ((last_close / first_close - 1) * 100).sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Highest total return (%)")
        st.caption("Last close ÷ first close − 1, per stock.")
        st.dataframe(total_return.head(n).round(2).rename("Total return %"), use_container_width=True)
    with col2:
        st.subheader("Most volatile (avg 5-day rolling std)")
        vol = g["rolling_std_5"].mean().sort_values(ascending=False)
        st.dataframe(vol.head(n).round(5).rename("Avg volatility"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Highest average closing price")
        st.dataframe(g["close"].mean().sort_values(ascending=False).head(n).round(2).rename("Avg close"),
                     use_container_width=True)
    with col2:
        st.subheader("Most traded (total volume)")
        st.dataframe(g["volume"].sum().sort_values(ascending=False).head(n).rename("Total volume"),
                     use_container_width=True)

    st.subheader("Highest single-day volume")
    st.dataframe(g["volume"].max().sort_values(ascending=False).head(n).rename("Max daily volume"),
                 use_container_width=True)

    best_day = df.loc[df["daily_return"].idxmax()]
    st.info(
        f"Highest single-day (open→close) return: **{best_day['symbol']}** on "
        f"{best_day['date']:%Y-%m-%d} ({best_day['daily_return']:.1%}). "
        "Check the Data Quality tab - extreme values like this are often data errors."
    )

# ----------------------------------------------------------------------------
# Market trends
# ----------------------------------------------------------------------------
with tabs[2]:
    freq_label = st.radio("Aggregation", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
    freq = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}[freq_label]
    trend = (
        df.assign(period=df["date"].dt.to_period(freq).astype(str))
        .groupby("period")
        .agg(volume=("volume", "sum"), close=("close", "mean"))
    )

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(trend.index, trend["close"], marker="o")
        ax.set_title(f"{freq_label} average closing price")
        ax.set_ylabel("Average close")
        step = max(1, len(trend) // 10)
        ax.set_xticks(range(0, len(trend), step))
        ax.set_xticklabels(trend.index[::step], rotation=45)
        ax.grid(True)
        fig_show(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(trend.index, trend["volume"], marker="o", color="tab:orange")
        ax.set_title(f"{freq_label} total trading volume")
        ax.set_ylabel("Volume")
        ax.set_xticks(range(0, len(trend), step))
        ax.set_xticklabels(trend.index[::step], rotation=45)
        ax.grid(True)
        fig_show(fig)

    st.dataframe(trend, use_container_width=True)

# ----------------------------------------------------------------------------
# Returns & statistics
# ----------------------------------------------------------------------------
with tabs[3]:
    ret = df["daily_return"].dropna()

    col1, col2, col3 = st.columns(3)
    col1.metric("Skewness", f"{ret.skew():,.2f}")
    col2.metric("Kurtosis", f"{ret.kurt():,.0f}")
    col3.metric("Max daily return", f"{ret.max():.1%}")
    st.caption("Very high skew/kurtosis means a few extreme (often erroneous) values dominate the tails.")

    zoom = st.checkbox("Zoom histogram/boxplot to ±10% daily return", value=True)
    plot_ret = ret[ret.between(-0.1, 0.1)] if zoom else ret

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(plot_ret, bins=50, kde=True, ax=ax)
        ax.set_title("Daily return distribution")
        fig_show(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(plot_ret, vert=False)
        ax.set_title("Daily return box plot")
        fig_show(fig)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[["open", "high", "low", "close"]].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation matrix")
        fig_show(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[["open", "high", "low", "close"]].cov(), annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
        ax.set_title("Covariance matrix")
        fig_show(fig)

    st.subheader("Open vs close (random 20,000-row sample)")
    sample = df.sample(min(20_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=sample["open"], y=sample["close"], s=8, ax=ax)
    fig_show(fig)

# ----------------------------------------------------------------------------
# Stock explorer
# ----------------------------------------------------------------------------
with tabs[4]:
    top_company = df.groupby("symbol")["volume"].sum().idxmax()
    sym = st.selectbox("Stock", symbols, index=symbols.index(top_company), key="explorer_sym")
    s = df[df["symbol"] == sym]

    dmin, dmax = s["date"].min().date(), s["date"].max().date()
    start, end = st.slider("Date range", dmin, dmax, (dmin, dmax), key="explorer_range")
    s = s[(s["date"].dt.date >= start) & (s["date"].dt.date <= end)]

    show_ma7 = st.checkbox("7-day moving average", True)
    show_ma30 = st.checkbox("30-day moving average", True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s["date"], s["close"], label="Closing price")
    if show_ma7:
        ax.plot(s["date"], s["moving_average_7"], label="MA 7")
    if show_ma30:
        ax.plot(s["date"], s["moving_average_30"], label="MA 30")
    ax.set_title(f"{sym} closing price")
    ax.grid(True)
    ax.legend()
    fig_show(fig)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(s["date"], s["volume"], width=2)
        ax.set_title(f"{sym} daily volume")
        fig_show(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(s["date"], s["rolling_std_5"])
        ax.set_title(f"{sym} 5-day rolling volatility")
        fig_show(fig)

    st.dataframe(s.tail(50), use_container_width=True)

# ----------------------------------------------------------------------------
# Forecast (decomposition + ARIMA)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Fitting ARIMA model...")
def run_arima(dates: pd.Series, closes: pd.Series, order: tuple, steps: int):
    from statsmodels.tsa.arima.model import ARIMA

    result = ARIMA(closes.reset_index(drop=True), order=order).fit()
    forecast = result.forecast(steps=steps)
    future_dates = pd.bdate_range(start=dates.iloc[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.DataFrame({"Forecast_Date": future_dates, "Predicted_Close": forecast.values})


with tabs[5]:
    sym_f = st.selectbox("Stock", symbols, index=symbols.index("AAPL") if "AAPL" in symbols else 0,
                         key="forecast_sym")
    s = df[df["symbol"] == sym_f].dropna(subset=["close"])

    st.subheader("Monthly average closing price & trend decomposition")
    monthly = s.set_index("date")["close"].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly, marker="o")
    ax.set_title(f"{sym_f} monthly average closing price")
    ax.grid(True)
    fig_show(fig)

    if len(monthly) >= 24:
        from statsmodels.tsa.seasonal import seasonal_decompose

        dec = seasonal_decompose(monthly, model="additive", period=12)
        fig = dec.plot()
        fig.set_size_inches(12, 7)
        fig_show(fig)
    else:
        st.warning("Need at least 24 months of data for seasonal decomposition.")

    st.subheader("ARIMA forecast")
    c1, c2, c3, c4 = st.columns(4)
    steps = c1.slider("Trading days ahead", 5, 90, 30)
    p = c2.number_input("p", 0, 10, 5)
    d = c3.number_input("d", 0, 2, 1)
    q = c4.number_input("q", 0, 10, 0)

    if st.button("Run forecast"):
        fc = run_arima(s["date"], s["close"], (int(p), int(d), int(q)), steps)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(s["date"].tail(200), s["close"].tail(200), label="Historical close (last 200 days)")
        ax.plot(fc["Forecast_Date"], fc["Predicted_Close"], color="red", marker="o", label=f"{steps}-day forecast")
        ax.set_title(f"{sym_f} - ARIMA({int(p)},{int(d)},{int(q)}) forecast")
        ax.grid(True)
        ax.legend()
        fig_show(fig)
        st.dataframe(fc, use_container_width=True)
        st.caption("A statistical projection from past prices only. It is not investment advice.")

# ----------------------------------------------------------------------------
# ML models
# ----------------------------------------------------------------------------
with tabs[6]:
    st.write("Predict the closing price from same-day open/high/low/volume, moving averages, and lagged closes.")
    st.warning(
        "Including same-day **high** and **low** makes the target almost a restatement of the inputs, "
        "which is why R² is ≈ 1. The random train/test split also mixes past and future days. "
        "Treat these scores as optimistic, not as proof of forecasting ability."
    )

    c1, c2 = st.columns(2)
    n_rows = c1.select_slider("Rows to use (random sample)", [20_000, 50_000, 100_000, 250_000, "All"], value=50_000)
    test_size = c2.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    chosen = st.multiselect(
        "Models",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
        default=["Linear Regression", "Decision Tree", "Random Forest"],
    )

    if st.button("Train models") and chosen:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                                     mean_squared_error, r2_score)
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor

        data = df.dropna(subset=FEATURES + ["close"])  # drops each stock's first 30 days (no lag/MA yet)
        if n_rows != "All" and n_rows < len(data):
            data = data.sample(n_rows, random_state=42)
        X, y = data[FEATURES], data["close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        def make(name):
            if name == "Linear Regression":
                return LinearRegression()
            if name == "Decision Tree":
                return DecisionTreeRegressor(max_depth=10, random_state=42)
            if name == "Random Forest":
                return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=30, max_depth=6, random_state=42)

        rows, trained = [], {}
        progress = st.progress(0.0)
        for i, name in enumerate(chosen):
            with st.spinner(f"Training {name}..."):
                model = make(name)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            trained[name] = model
            mse = mean_squared_error(y_test, pred)
            rows.append({
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2_score(y_test, pred),
                "MAPE": mean_absolute_percentage_error(y_test, pred),
            })
            progress.progress((i + 1) / len(chosen))

        summary = pd.DataFrame(rows).set_index("Model").round(4)
        st.subheader("Model comparison")
        st.dataframe(summary, use_container_width=True)
        st.success(f"Lowest MAE: **{summary['MAE'].idxmin()}**")

        if "Random Forest" in trained:
            imp = (pd.DataFrame({"Feature": FEATURES,
                                 "Importance": trained["Random Forest"].feature_importances_})
                   .sort_values("Importance", ascending=False))
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=imp, x="Importance", y="Feature", ax=ax)
            ax.set_title("Random Forest feature importance")
            fig_show(fig)

# ----------------------------------------------------------------------------
# Data quality
# ----------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Missing values in the raw file (open / high / low)")
    st.dataframe(missing_before.rename("Missing").to_frame().T, use_container_width=True)

    ohlc = df[["open", "high", "low", "close"]]
    invalid_high = df[df["high"] != ohlc.max(axis=1)]
    invalid_low = df[df["low"] != ohlc.min(axis=1)]

    c1, c2 = st.columns(2)
    c1.metric("Rows where High is not the max of OHLC", f"{len(invalid_high):,}")
    c2.metric("Rows where Low is not the min of OHLC", f"{len(invalid_low):,}")

    with st.expander("Rows with invalid High"):
        st.dataframe(invalid_high, use_container_width=True)
    with st.expander("Rows with invalid Low"):
        st.dataframe(invalid_low, use_container_width=True)

    st.subheader("Most extreme daily returns")
    st.dataframe(
        df.loc[df["daily_return"].abs().nlargest(20).index,
               ["symbol", "date", "open", "high", "low", "close", "volume", "daily_return"]],
        use_container_width=True,
    )
