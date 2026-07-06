"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         FinOptix - Financial Analytics Engine                 ║
║                 Institutional-Grade Portfolio Optimization Platform          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Built with: Streamlit | SciPy | Scikit-learn | Plotly | yFinance           ║
║  Author: AI Assistant | Version: 2.0.1 | License: MIT                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as pff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
from datetime import datetime, date
from typing import Tuple, Dict, Any

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────
# APPLICATION CONFIGURATION
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinOptix | Institutional Financial Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL ANALYTICS ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class FinancialAnalyticsEngine:
    """
    Production-grade Financial Analytics Engine.
    Fetches live data, engineers features, computes risk metrics,
    runs ML classification, and optimizes portfolios.
    """

    def __init__(self):
        self.tickers = ['AAPL', 'MSFT', 'AMZN', 'GLD', 'SPY']
        self.rf_rate = 0.04
        self.data_loaded = False

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data(_self):
        """Fetch 5+ years of data and engineer all features."""
        with st.spinner("📡 Fetching live market data and engineering features..."):
            end = pd.Timestamp.today()
            start = end - pd.DateOffset(years=6)

            raw = yf.download(_self.tickers, start=start, end=end, progress=False, auto_adjust=False)
            
            # Handle both old and new yfinance column layouts
            if 'Adj Close' in raw.columns.get_level_values(0):
                prices = raw['Adj Close'].copy()
            else:
                prices = raw['Close'].copy()

            # Ensure data is numeric and drop empty rows
            prices = prices.apply(pd.to_numeric, errors='coerce').dropna(how='all')

            # Safety check: Stop gracefully if download failed
            if prices.empty:
                st.error("⚠️ Market data download failed. This usually happens due to a temporary API limit. Please refresh the page in 1 minute.")
                st.stop()
                
            returns = np.log(prices / prices.shift(1))
            features = pd.DataFrame(index=prices.index)

            for ticker in _self.tickers:
                price = prices[ticker]
                ret = returns[ticker]
                features[f'{ticker}_logret'] = ret
                features[f'{ticker}_vol_21d'] = ret.rolling(21).std()
                features[f'{ticker}_vol_252d'] = ret.rolling(252).std()
                delta = price.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
                ema12 = price.ewm(span=12, adjust=False).mean()
                ema26 = price.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                features[f'{ticker}_macd'] = macd_line
                features[f'{ticker}_macd_signal'] = signal_line
                features[f'{ticker}_macd_hist'] = macd_line - signal_line
                features[f'{ticker}_lag_1d'] = ret.shift(1)
                features[f'{ticker}_lag_5d'] = ret.shift(5)
                features[f'{ticker}_lag_21d'] = ret.shift(21)

            features = features.ffill().dropna()
            returns = returns.loc[features.index]
            prices = prices.loc[features.index]

            return prices, returns, features

    def get_performance_table(self) -> pd.DataFrame:
        """Annualized returns, volatility, and Sharpe ratio for all assets."""
        ann_ret = self.returns.mean() * 252
        ann_vol = self.returns.std() * np.sqrt(252)
        sharpe = (ann_ret - self.rf_rate) / ann_vol
        max_dd = (self.returns.cumsum().expanding().max() - self.returns.cumsum()).min()

        df = pd.DataFrame({
            'Ann. Return': ann_ret,
            'Ann. Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        }).round(4)
        df['Trend'] = sharpe.apply(lambda x: '🟢 Strong' if x > 0.8 else ('🟡 Moderate' if x > 0.3 else '🔴 Weak'))
        return df

    def get_risk_metrics(self, portfolio_weights: np.ndarray = None) -> Dict[str, str]:
        if portfolio_weights is None:
            portfolio_weights = np.full(len(self.tickers), 1.0 / len(self.tickers))
        port_rets = (self.returns @ portfolio_weights)
        alpha = 0.05
        hist_var = -np.percentile(port_rets.dropna(), 5)
        tail_losses = port_rets[port_rets <= -hist_var]
        hist_cvar = -tail_losses.mean() if len(tail_losses) > 0 else hist_var
        mu_daily, sigma_daily = port_rets.mean(), port_rets.std()
        z = norm.ppf(alpha)
        param_var = -(mu_daily + z * sigma_daily)
        param_cvar = -(mu_daily + sigma_daily * norm.pdf(z) / alpha)
        return {
            "Historical VaR (95%)": f"{hist_var:.2%}",
            "Historical CVaR (95%)": f"{hist_cvar:.2%}",
            "Parametric VaR (95%)": f"{param_var:.2%}",
            "Parametric CVaR (95%)": f"{param_cvar:.2%}"
        }

    def get_rolling_correlation(self, window: int = 63) -> Tuple[go.Figure, pd.DataFrame]:
        """Compute rolling correlation matrix across all assets."""
        corr_df = self.returns.rolling(window).corr().dropna()
        latest_date = corr_df.index.get_level_values(0)[-1]
        latest_corr = corr_df.loc[latest_date]
        
        if isinstance(latest_corr, pd.Series):
            latest_corr = latest_corr.unstack()

        fig = px.imshow(
            latest_corr.round(3),
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect="auto",
            title=f"📊 Rolling Correlation Matrix ({window}-Day Window) | as of {latest_date.date()}"
        )
        fig.update_layout(height=600, template="plotly_dark", font=dict(size=13))
        fig.update_xaxes(side="top")
        return fig, latest_corr

    def get_return_distribution(self) -> go.Figure:
        eq_w = np.full(len(self.tickers), 1.0 / len(self.tickers))
        port_rets = (self.returns @ eq_w).dropna()
        fig = pff.create_distplot([port_rets.values], group_labels=['Portfolio Returns'], colors=['#22d3ee'], show_rug=False)
        fig.update_layout(title="Distribution of Daily Portfolio Returns", template="plotly_dark", height=450)
        return fig

    def run_ml_classification(self) -> Tuple[list, pd.Series, list, pd.DataFrame]:
        target = (self.returns['SPY'].shift(-5) > 0).astype(int)
        valid = target.notna()
        X = self.features.loc[valid]
        y = target.loc[valid]
        tscv = TimeSeriesSplit(n_splits=5)
        roc_aucs, importances, reports, all_preds = [], np.zeros(X.shape[1]), [], pd.DataFrame()

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=15, random_state=42, class_weight='balanced', n_jobs=-1)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            roc_aucs.append(auc)
            importances += model.feature_importances_
            rep = classification_report(y_test, model.predict(X_test), output_dict=True)
            rep['fold'] = fold
            rep['roc_auc'] = auc
            reports.append(rep)
        
        feat_imp = pd.Series(importances / 5, index=X.columns).sort_values(ascending=False)
        return roc_aucs, feat_imp, reports, all_preds

    def optimize_portfolios(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame]:
        mu = self.returns.mean() * 252
        cov = self.returns.cov() * 252
        n = len(self.tickers)
        def port_vol(w): return np.sqrt(w @ cov @ w)
        def neg_sharpe(w): return -(w @ mu - self.rf_rate) / port_vol(w)
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0.0, 1.0) for _ in range(n))
        res_sharpe = minimize(neg_sharpe, np.array([1/n]*n), method='SLSQP', bounds=bounds, constraints=cons)
        res_minvar = minimize(port_vol, np.array([1/n]*n), method='SLSQP', bounds=bounds, constraints=cons)
        return res_sharpe.x, res_minvar.x, mu, cov

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

engine = FinancialAnalyticsEngine()
prices, returns, features = engine.load_data()
engine.prices, engine.returns, engine.features = prices, returns, features

with st.sidebar:
    st.title("🏛️ FinOptix")
    page = st.radio("Navigation", ["🏠 Dashboard", "📈 Risk", "🧠 ML", "🎯 Optimization", "💰 Builder", "🔎 Explorer"])
    risk_free = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0) / 100
    engine.rf_rate = risk_free

if page == "🏠 Dashboard":
    st.header("Financial Dashboard")
    perf = engine.get_performance_table()
    cols = st.columns(5)
    for i, t in enumerate(engine.tickers):
        cols[i].metric(t, f"{perf.loc[t, 'Ann. Return']:.1%}", f"Sharpe: {perf.loc[t, 'Sharpe Ratio']:.2f}")
    
    # Safe chart rendering to prevent IndexError on Streamlit Cloud
    if not prices.empty and len(prices) > 0:
        st.line_chart(prices.div(prices.iloc[0], axis=1) * 100)
    else:
        st.warning("No price data available to display the chart.")

elif page == "📈 Risk":
    st.header("Risk Analytics")
    metrics = engine.get_risk_metrics()
    cols = st.columns(4)
    for i, (k, v) in enumerate(metrics.items()):
        cols[i].metric(k, v)
    window = st.slider("Rolling Window", 21, 126, 63)
    fig, _ = engine.get_rolling_correlation(window)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(engine.get_return_distribution(), use_container_width=True)

elif page == "🎯 Optimization":
    st.header("Portfolio Optimization")
    w_s, w_v, mu, cov = engine.optimize_portfolios()
    col1, col2 = st.columns(2)
    col1.subheader("Max Sharpe Weights")
    col1.write(pd.Series(w_s, index=engine.tickers).map('{:.2%}'.format))
    col2.subheader("Min Variance Weights")
    col2.write(pd.Series(w_v, index=engine.tickers).map('{:.2%}'.format))

elif page == "💰 Builder":
    st.header("Custom Portfolio Builder")
    mu = engine.returns.mean() * 252
    cov = engine.returns.cov() * 252
    u_weights = [st.slider(f"Weight {t}", 0.0, 1.0, 0.2) for t in engine.tickers]
    w_user = np.array(u_weights) / sum(u_weights)
    st.metric("Expected Return", f"{w_user @ mu:.2%}")
    st.metric("Volatility", f"{np.sqrt(w_user @ cov @ w_user):.2%}")

elif page == "🧠 ML":
    st.header("Machine Learning (Random Forest)")
    st.info("Predicting if SPY will be positive in 5 days...")
    aucs, feats, _, _ = engine.run_ml_classification()
    st.metric("Average ROC-AUC", f"{np.mean(aucs):.4f}")
    st.bar_chart(feats.head(10))

elif page == "🔎 Explorer":
    st.header("Data Explorer")
    st.dataframe(prices.tail(50))
    st.download_button("Download Prices CSV", prices.to_csv(), "prices.csv")

st.divider()
st.caption("FinOptix v2.0 | Institutional-Grade Financial Engine")