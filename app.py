"""
S&P 500 Stock Analysis Dashboard
=================================
Converted from the "S_P500.ipynb" notebook into a deployable Streamlit app.

Run locally:
    streamlit run app.py

Deploy:
    Push this file + requirements.txt to a GitHub repo and deploy on
    https://share.streamlit.io (Streamlit Community Cloud), or any host
    that can run `streamlit run app.py`.
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

st.set_page_config(
    page_title="S&P 500 Stock Analysis",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"symbol", "date", "open", "high", "low", "close", "volume"}


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Load the raw CSV and apply the same cleaning steps as the notebook."""
    df = pd.read_csv(io.BytesIO(file_bytes))

    missing_cols = REQUIRED_COLUMNS - set(df.columns.str.lower())
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {missing_cols}. "
            f"Expected columns: {sorted(REQUIRED_COLUMNS)}"
        )
    df.columns = [c.lower() for c in df.columns]

    # Date conversion
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date", ascending=True)

    # Handle missing values in open/high/low with median (as in the notebook)
    df[["open", "high", "low"]] = df[["open", "high", "low"]].fillna(
        df[["open", "high", "low"]].median()
    )

    # Sort & reset index
    df = df.sort_values(by=["date", "symbol"], ascending=True)
    df.reset_index(inplace=True, drop=True)

    return df


@st.cache_data(show_spinner="Engineering features...")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate every derived column built in the notebook."""
    df = df.copy()

    # Daily return
    df["daily_return"] = (df["close"] / df["open"]) - 1
    df["daily_return_dist"] = df["daily_return"]

    # Calendar groupings
    df["year_month"] = df["date"].dt.to_period("M")
    df["year_quarter"] = df["date"].dt.to_period("Q")
    df["date_year"] = df["date"].dt.to_period("Y")

    # Rolling volatility
    df["rolling_std_5"] = df.groupby("symbol")["daily_return"].transform(
        lambda x: x.rolling(window=5).std()
    )

    # Moving averages
    df["moving_average_7"] = df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(window=7).mean()
    )
    df["moving_average_30"] = df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(window=30).mean()
    )

    # Cumulative return
    df["cumulative_return"] = (1 + df["daily_return"]).groupby(df["symbol"]).cumprod() - 1

    # Previous close / lag features
    df["previous_day_close"] = df.groupby("symbol")["close"].shift(1)
    df["lag_1"] = df["previous_day_close"]
    df["lag_2"] = df.groupby("symbol")["close"].shift(2)
    df["lag_5"] = df.groupby("symbol")["close"].shift(5)

    # Price difference & spread
    df["price_difference"] = df["close"] - df["open"]
    df["high_low_spread"] = df["high"] - df["low"]

    # Rolling max/min
    df["rolling_max"] = df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(window=5).max()
    )
    df["rolling_min"] = df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(window=5).min()
    )

    # 30-day moving average (Challenge 2)
    df["moving_average"] = df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(30).mean()
    )

    return df


@st.cache_data(show_spinner="Training regression models...")
def train_models(df: pd.DataFrame):
    """Train the 4 regression models from the notebook and return metrics + models."""
    feature_cols = [
        "open", "high", "low", "volume",
        "moving_average_7", "moving_average_30",
        "lag_1", "lag_2", "lag_5", "daily_return",
    ]
    X = df[feature_cols].fillna(0)
    y = df["close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    models = {}

    def evaluate(name, model, y_pred):
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        }

    # Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    evaluate("Linear Regression", model_lr, model_lr.predict(X_test))
    models["Linear Regression"] = model_lr

    # Decision Tree
    model_dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    model_dt.fit(X_train, y_train)
    evaluate("Decision Tree Regressor", model_dt, model_dt.predict(X_test))
    models["Decision Tree Regressor"] = model_dt

    # Random Forest
    model_rfr = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model_rfr.fit(X_train, y_train)
    evaluate("Random Forest Regressor", model_rfr, model_rfr.predict(X_test))
    models["Random Forest Regressor"] = model_rfr

    # XGBoost
    if XGBOOST_AVAILABLE:
        model_xgb = XGBRegressor(n_estimators=30, max_depth=6, random_state=42)
        model_xgb.fit(X_train, y_train)
        evaluate("XG Boost", model_xgb, model_xgb.predict(X_test))
        models["XG Boost"] = model_xgb

    summary = pd.DataFrame(results).T.round(4)
    return summary, models, feature_cols


# ---------------------------------------------------------------------------
# Sidebar - data input
# ---------------------------------------------------------------------------

st.sidebar.title("📈 S&P 500 Dashboard")
uploaded_file = st.sidebar.file_uploader(
    "Upload the S&P 500 CSV (symbol, date, open, high, low, close, volume)",
    type=["csv"],
)

st.title("S&P 500 Stock Price Analysis & Forecasting")
st.caption(
    "An end-to-end EDA, feature engineering, machine learning and "
    "time-series forecasting dashboard, converted from the original "
    "analysis notebook."
)

if uploaded_file is None:
    st.info(
        "👈 Upload the **S&P 500 Stock Prices** CSV in the sidebar to get started "
        "(the same file used in the original notebook: "
        "`S&P 500 Stock Prices 2014-2017.csv`)."
    )
    st.stop()

raw_df = load_data(uploaded_file.getvalue())
df = engineer_features(raw_df)

symbols = sorted(df["symbol"].dropna().unique().tolist())

tabs = st.tabs(
    [
        "Overview",
        "Exploratory Analysis",
        "Volatility & Features",
        "Statistical Analysis",
        "Machine Learning",
        "Time Series Forecast",
        "Business Insights",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1 - Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Unique stocks", df["symbol"].nunique())
    c4.metric("Trading days", df["date"].nunique())

    st.write(
        f"**Date range:** {df['date'].min().date()} → {df['date'].max().date()}"
    )

    st.subheader("Sample data")
    st.dataframe(df.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Descriptive statistics")
        st.dataframe(raw_df.describe(), use_container_width=True)
    with col_b:
        st.subheader("Missing values (raw)")
        st.dataframe(raw_df.isnull().sum().rename("missing"), use_container_width=True)

    st.subheader("Headline numbers")
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest closing price", f"${df['close'].max():,.2f}")
    c2.metric("Lowest closing price", f"${df['close'].min():,.2f}")
    c3.metric("Average closing price", f"${df['close'].mean():,.2f}")
    st.metric("Total trading volume", f"{df['volume'].sum():,.0f}")

# ---------------------------------------------------------------------------
# TAB 2 - Exploratory analysis
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Exploratory Data Analysis")

    st.subheader("Top 20 companies by average closing price")
    top_20_avg = (
        df.groupby("symbol")["close"].mean().sort_values(ascending=False).head(20)
    )
    st.plotly_chart(
        px.bar(top_20_avg, orientation="h", labels={"value": "Avg close", "symbol": "Symbol"})
        .update_yaxes(categoryorder="total ascending"),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 10 most traded companies (by total volume)")
        top_traded = df.groupby("symbol")["volume"].sum().sort_values(ascending=False).head(10)
        st.dataframe(top_traded.rename("total_volume"), use_container_width=True)
    with col_b:
        st.subheader("Top 10 highest single-day volume")
        highest_vol = df.groupby("symbol")["volume"].max().sort_values(ascending=False).head(10)
        st.dataframe(highest_vol.rename("max_volume"), use_container_width=True)

    st.subheader("Monthly / Quarterly / Yearly average closing price")
    freq = st.radio("Aggregation level", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
    period_col = {"Monthly": "year_month", "Quarterly": "year_quarter", "Yearly": "date_year"}[freq]
    trend = df.groupby(period_col).agg(volume=("volume", "sum"), close=("close", "mean"))
    trend.index = trend.index.astype(str)
    st.plotly_chart(
        px.line(trend, y="close", markers=True, title=f"{freq} average closing price"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(trend, y="volume", markers=True, title=f"{freq} trading volume"),
        use_container_width=True,
    )

    st.subheader("Daily return distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            px.histogram(df, x="daily_return", nbins=80, title="Daily return histogram"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            px.box(df, y="daily_return", title="Daily return box plot"),
            use_container_width=True,
        )

    st.subheader("Correlation between price columns")
    corr = df[["open", "high", "low", "close"]].corr()
    st.plotly_chart(
        px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation matrix"),
        use_container_width=True,
    )

    st.subheader("Open vs Close scatter")
    sample = df.sample(min(5000, len(df)), random_state=42)
    st.plotly_chart(px.scatter(sample, x="open", y="close", opacity=0.4), use_container_width=True)

    st.subheader("Moving averages for a selected stock")
    top_company_default = df.groupby("symbol")["volume"].sum().idxmax()
    stock_choice = st.selectbox(
        "Stock symbol", symbols, index=symbols.index(top_company_default) if top_company_default in symbols else 0,
        key="eda_stock"
    )
    company = df[df["symbol"] == stock_choice]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=company["date"], y=company["close"], name="Closing price"))
    fig.add_trace(go.Scatter(x=company["date"], y=company["moving_average_7"], name="7-day MA"))
    fig.add_trace(go.Scatter(x=company["date"], y=company["moving_average_30"], name="30-day MA"))
    fig.update_layout(title=f"{stock_choice} — Price & Moving Averages", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 - Volatility & feature engineering
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Volatility & Feature Engineering")

    st.markdown(
        "Derived features created: **daily return, 7/30-day moving average, "
        "rolling 5-day std (volatility), cumulative return, lag features "
        "(1/2/5 days), price difference, high-low spread, rolling max/min.**"
    )

    st.subheader("Highest single-day daily return")
    max_return_row = df.loc[df["daily_return"].idxmax()]
    st.write(
        f"**{max_return_row['symbol']}** on {max_return_row['date'].date()} "
        f"— return of **{max_return_row['daily_return']:.2%}**"
    )

    st.subheader("Rolling 5-day volatility (std of daily return) for a stock")
    vol_stock = st.selectbox("Stock symbol", symbols, key="vol_stock")
    vol_df = df[df["symbol"] == vol_stock]
    st.plotly_chart(
        px.line(vol_df, x="date", y="rolling_std_5", title=f"{vol_stock} — 5-day rolling volatility"),
        use_container_width=True,
    )

    st.subheader("Price difference (close − open) over time")
    st.plotly_chart(
        px.line(vol_df, x="date", y="price_difference", title=f"{vol_stock} — daily price difference"),
        use_container_width=True,
    )

    st.subheader("Cumulative return")
    st.plotly_chart(
        px.line(vol_df, x="date", y="cumulative_return", title=f"{vol_stock} — cumulative return"),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# TAB 4 - Statistical analysis
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Statistical Analysis")

    st.subheader("Covariance matrix (open, high, low, close)")
    cov_matrix = df[["open", "high", "low", "close"]].cov()
    st.plotly_chart(
        px.imshow(cov_matrix, text_auto=".1f", color_continuous_scale="RdBu_r", title="Covariance matrix"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Skewness (daily return)", f"{df['daily_return'].skew():.4f}")
    c2.metric("Kurtosis (daily return)", f"{df['daily_return'].kurt():.4f}")
    st.caption(
        "Positive skew → a small number of stocks post abnormally high returns. "
        "High kurtosis → fat tails / more outliers than a normal distribution."
    )

# ---------------------------------------------------------------------------
# TAB 5 - Machine learning
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Machine Learning — Predicting Closing Price")

    st.caption(
        "Models are trained on the full dataset using: open, high, low, volume, "
        "7/30-day moving averages, lag features (1/2/5) and daily return."
    )

    if st.button("Train models", type="primary"):
        summary, models, feature_cols = train_models(df)
        st.session_state["model_summary"] = summary
        st.session_state["models"] = models
        st.session_state["feature_cols"] = feature_cols

    if "model_summary" in st.session_state:
        st.subheader("Model comparison")
        st.dataframe(st.session_state["model_summary"], use_container_width=True)

        best_model_name = st.session_state["model_summary"]["RMSE"].idxmin()
        st.success(f"Best model by RMSE: **{best_model_name}**")

        if "Random Forest Regressor" in st.session_state["models"]:
            st.subheader("Random Forest feature importance")
            rf = st.session_state["models"]["Random Forest Regressor"]
            feat_imp = pd.DataFrame(
                {
                    "Feature": st.session_state["feature_cols"],
                    "Importance": rf.feature_importances_,
                }
            ).sort_values("Importance", ascending=False)
            st.plotly_chart(
                px.bar(feat_imp, x="Importance", y="Feature", orientation="h")
                .update_yaxes(categoryorder="total ascending"),
                use_container_width=True,
            )
    else:
        st.info("Click **Train models** to fit Linear Regression, Decision Tree, "
                 "Random Forest" + (" and XGBoost" if XGBOOST_AVAILABLE else "") + ".")
        if not XGBOOST_AVAILABLE:
            st.warning("xgboost is not installed in this environment — XGBoost will be skipped.")

# ---------------------------------------------------------------------------
# TAB 6 - Time series forecasting
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Time Series Analysis & Forecast")

    ts_stock = st.selectbox("Stock symbol", symbols, key="ts_stock")
    horizon = st.slider("Forecast horizon (trading days)", 5, 60, 30)

    stock_df = df[df["symbol"] == ts_stock].copy().set_index("date")
    monthly_stock = stock_df["close"].resample("ME").mean()

    st.subheader(f"{ts_stock} — Monthly average closing price")
    st.plotly_chart(px.line(monthly_stock, markers=True), use_container_width=True)

    if len(monthly_stock.dropna()) >= 24:
        st.subheader("Trend / seasonal decomposition")
        decomposition = seasonal_decompose(monthly_stock.dropna(), model="additive", period=12)
        dec_df = pd.DataFrame(
            {
                "observed": decomposition.observed,
                "trend": decomposition.trend,
                "seasonal": decomposition.seasonal,
                "resid": decomposition.resid,
            }
        )
        st.plotly_chart(
            px.line(dec_df, facet_col="variable", facet_col_wrap=1, height=700)
            .update_yaxes(matches=None),
            use_container_width=True,
        )
    else:
        st.info("Not enough monthly history for a seasonal decomposition (need ≥ 24 months).")

    st.subheader(f"ARIMA {horizon}-day forecast")
    with st.spinner("Fitting ARIMA model..."):
        daily_close = stock_df["close"].dropna()
        try:
            arima_model = ARIMA(daily_close, order=(5, 1, 0))
            arima_result = arima_model.fit()
            forecast = arima_result.forecast(steps=horizon)

            future_dates = pd.bdate_range(
                start=daily_close.index[-1] + pd.Timedelta(days=1), periods=horizon
            )
            forecast_df = pd.DataFrame(
                {"date": future_dates, "close": forecast.values}
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=daily_close.tail(200).index,
                    y=daily_close.tail(200).values,
                    name="Historical close",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"], y=forecast_df["close"],
                    name=f"{horizon}-day forecast", mode="lines+markers", line=dict(color="red"),
                )
            )
            fig.update_layout(title=f"{ts_stock} — ARIMA forecast", xaxis_title="Date", yaxis_title="Closing price")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(forecast_df, use_container_width=True)
        except Exception as e:
            st.error(f"ARIMA could not be fit for {ts_stock}: {e}")

# ---------------------------------------------------------------------------
# TAB 7 - Business insights & challenges
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("Business Insights")
    st.markdown(
        """
- The S&P 500 companies analyzed trend upward over the period, suggesting long-term growth.
- Daily returns cluster tightly around 0 — most stocks show low day-to-day volatility, with a handful of outliers.
- Open, high, low and close prices are highly correlated, so a move in one tends to move the others.
- 7-day moving average crossing above the 30-day moving average has historically lined up with upward price movement.
- Random Forest gave the strongest predictive performance among the models tested (lowest MAE/MSE/RMSE/MAPE).
- ARIMA forecasts provide a short-term (30-trading-day) view to support buy/sell timing decisions.
- Quarterly trading volume is a useful proxy for liquidity; it peaked in 2016 Q1 in the original dataset before declining, without derailing the overall uptrend.
        """
    )

    st.header("Challenges")

    st.subheader("Challenge 1 — Top 10 most volatile stocks")
    top_10_volatile = (
        df.groupby("symbol")["rolling_std_5"].mean().sort_values(ascending=False).head(10)
    )
    st.dataframe(top_10_volatile.rename("avg_rolling_std_5"), use_container_width=True)

    st.subheader("Challenge 2 — 30-day moving average for every stock")
    st.dataframe(
        df[["symbol", "date", "close", "moving_average"]].tail(20),
        use_container_width=True,
    )
    st.caption("Full column `moving_average` is available on every row of the processed dataset.")

st.sidebar.markdown("---")
st.sidebar.caption("Converted from the original S&P 500 analysis notebook.")
