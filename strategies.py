"""
ML Trading Strategies Base Classes and Implementations
Object-oriented approach for Moving Average Crossover and Volatility Forecasting strategies
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

# ML and Technical Analysis
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor

# Technical indicators
try:
    from ta.trend import MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
except Exception:
    MACD = RSIIndicator = BollingerBands = None

# GARCH modeling
try:
    from arch import arch_model
except Exception:
    arch_model = None


@dataclass
class Metrics:
    """Performance metrics container"""
    cagr: float
    sharpe: float
    max_dd: float
    total_return: float
    realized_vol: Optional[float] = None
    trade_count: Optional[int] = None


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, ticker: str, start_date: str, risk_free: float = 0.015):
        self.ticker = ticker
        self.start_date = start_date
        self.risk_free = risk_free
        self.data = None
        self.results = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance"""
        df = yf.download(self.ticker, start=self.start_date, auto_adjust=True, progress=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        
        # Flatten MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        df["ret"] = df["Close"].pct_change()
        return df
    
    @staticmethod
    def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(equity) < 2:
            return np.nan
        total_ret = equity.iloc[-1] / equity.iloc[0] - 1
        years = len(equity) / periods_per_year
        return (1 + total_ret) ** (1/years) - 1 if years > 0 else np.nan
    
    def sharpe(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0 or returns.isna().all():
            return np.nan
        mean_excess = returns.mean() * periods_per_year - self.risk_free
        vol = returns.std() * np.sqrt(periods_per_year)
        return mean_excess / (vol + 1e-12)
    
    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        roll_max = equity.cummax()
        dd = equity / roll_max - 1.0
        return dd.min()
    
    def compute_metrics(self, df: pd.DataFrame) -> Metrics:
        """Compute performance metrics"""
        equity = df["equity"].dropna()
        rets = df["strat_ret"].dropna()
        
        # Count trades if position column exists
        trade_count = None
        if "position" in df.columns:
            trade_count = len(df[df["position"].diff().abs() > 0])
        elif "position_ml" in df.columns:
            trade_count = len(df[df["position_ml"].diff().abs() > 0])
        
        return Metrics(
            cagr=self.cagr(equity),
            sharpe=self.sharpe(rets),
            max_dd=self.max_drawdown(equity),
            total_return=(equity.iloc[-1] - 1.0) if len(equity) else np.nan,
            realized_vol=rets.std() * np.sqrt(252) if len(rets) > 1 else np.nan,
            trade_count=trade_count
        )
    
    @staticmethod
    def apply_slippage(returns: pd.Series, trade_flags: pd.Series, slippage_bps: float = 1.0) -> pd.Series:
        """Apply transaction costs"""
        cost_per_trade = slippage_bps / 1e4
        cost_series = trade_flags.astype(float) * cost_per_trade
        return returns - cost_series
    
    @abstractmethod
    def prepare_data(self, **kwargs) -> pd.DataFrame:
        """Prepare strategy-specific data"""
        pass
    
    @abstractmethod
    def run_backtest(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Run strategy backtest"""
        pass


class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy with ML Filter"""
    
    def __init__(self, ticker: str, start_date: str, short_window: int = 20, 
                 long_window: int = 100, ml_hold_days: int = 10, 
                 proba_threshold: float = 0.55, slippage_bps: float = 1.0,
                 risk_free: float = 0.015):
        super().__init__(ticker, start_date, risk_free)
        self.short_window = short_window
        self.long_window = long_window
        self.ml_hold_days = ml_hold_days
        self.proba_threshold = proba_threshold
        self.slippage_bps = slippage_bps
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        df = df.copy()
        
        # Moving averages
        df["sma_s"] = df["Close"].rolling(self.short_window).mean()
        df["sma_l"] = df["Close"].rolling(self.long_window).mean()
        df["sma_diff"] = (df["sma_s"] - df["sma_l"]) / df["sma_l"]
        
        # Realized volatility
        df["rv_10"] = df["ret"].rolling(10).std() * np.sqrt(252)
        df["rv_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
        
        # Momentum
        df["mom_5"] = df["Close"].pct_change(5)
        df["mom_10"] = df["Close"].pct_change(10)
        df["mom_20"] = df["Close"].pct_change(20)
        
        # Volume
        df["vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-9)
        
        # Additional indicators via 'ta'
        if MACD is not None:
            macd = MACD(df["Close"])
            df["macd"] = macd.macd()
            df["macd_sig"] = macd.macd_signal()
            df["macd_diff"] = macd.macd_diff()
        if RSIIndicator is not None:
            rsi = RSIIndicator(df["Close"], window=14)
            df["rsi"] = rsi.rsi()
        if BollingerBands is not None:
            bb = BollingerBands(df["Close"], window=20, window_dev=2)
            df["bb_high"] = bb.bollinger_hband()
            df["bb_low"] = bb.bollinger_lband()
            df["pct_bb"] = (df["Close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"] + 1e-9)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate crossover signals"""
        df = df.copy()
        df["signal"] = 0
        
        cross_up = (df["sma_s"] > df["sma_l"]) & (df["sma_s"].shift(1) <= df["sma_l"].shift(1))
        cross_down = (df["sma_s"] < df["sma_l"]) & (df["sma_s"].shift(1) >= df["sma_l"].shift(1))
        
        df.loc[cross_up, "signal"] = 1
        df.loc[cross_down, "signal"] = -1
        
        # Build position series (long-only)
        df["position"] = 0
        pos = 0
        for i in range(len(df)):
            s = df["signal"].iat[i]
            if s == 1:
                pos = 1
            elif s == -1:
                pos = 0
            df["position"].iat[i] = pos
            
        return df
    
    def train_ml_filter(self, df: pd.DataFrame) -> pd.Series:
        """Train ML filter for entry signals"""
        # Prepare features
        fcols = ["sma_diff", "rv_10", "rv_20", "mom_5", "mom_10", "mom_20", "vol_z"]
        if "macd" in df.columns:
            fcols += ["macd", "macd_sig", "macd_diff"]
        if "rsi" in df.columns:
            fcols += ["rsi"]
        if "pct_bb" in df.columns:
            fcols += ["pct_bb"]
        
        X_full = df[fcols].copy().fillna(method="ffill").fillna(method="bfill")
        
        # Create labels for entry events
        df_temp = df.copy()
        df_temp["fwd_ret_N"] = df_temp["Close"].pct_change(self.ml_hold_days).shift(-self.ml_hold_days)
        events = df_temp[df_temp["signal"] == 1].copy()
        events["label"] = (events["fwd_ret_N"] > 0).astype(int)
        
        if len(events) < 10:
            # Not enough events for ML training
            return pd.Series(index=df.index, dtype=float).fillna(0.5)
        
        X = X_full.loc[events.index]
        y = events["label"]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("gb", GradientBoostingClassifier(random_state=42))
        ])
        
        try:
            model.fit(X, y)
            
            # Generate probabilities
            proba = pd.Series(index=df.index, dtype=float)
            valid_idx = X_full.dropna().index
            proba.loc[valid_idx] = model.predict_proba(X_full.loc[valid_idx])[:, 1]
            
            return proba
        except Exception:
            return pd.Series(index=df.index, dtype=float).fillna(0.5)
    
    def backtest_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run baseline MA crossover backtest"""
        df = df.copy()
        df["strat_ret"] = df["position"].shift(1) * df["ret"]
        
        trade = df["position"].diff().abs().fillna(0) > 0
        df["strat_ret"] = self.apply_slippage(df["strat_ret"], trade, self.slippage_bps)
        df["equity"] = (1 + df["strat_ret"]).cumprod()
        
        return df
    
    def backtest_ml_filtered(self, df: pd.DataFrame, proba: pd.Series) -> pd.DataFrame:
        """Run ML-filtered backtest"""
        df = df.copy()
        enter = (df["signal"] == 1) & (proba > self.proba_threshold)
        exit_ = (df["signal"] == -1)
        
        df["position_ml"] = 0
        pos = 0
        for i in range(len(df)):
            if enter.iat[i]:
                pos = 1
            if exit_.iat[i]:
                pos = 0
            df["position_ml"].iat[i] = pos
        
        df["strat_ret"] = df["position_ml"].shift(1) * df["ret"]
        trade = df["position_ml"].diff().abs().fillna(0) > 0
        df["strat_ret"] = self.apply_slippage(df["strat_ret"], trade, self.slippage_bps)
        df["equity"] = (1 + df["strat_ret"]).cumprod()
        
        return df
    
    def prepare_data(self, **kwargs) -> pd.DataFrame:
        """Prepare data for MA crossover strategy"""
        self.data = self.fetch_data()
        self.data = self.add_indicators(self.data)
        self.data = self.generate_signals(self.data)
        self.data = self.data.dropna(subset=["sma_s", "sma_l"])
        return self.data
    
    def run_backtest(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Run complete MA crossover backtest"""
        if self.data is None:
            self.prepare_data()
        
        # Baseline strategy
        baseline_results = self.backtest_baseline(self.data)
        
        # ML-filtered strategy
        proba = self.train_ml_filter(self.data)
        ml_results = self.backtest_ml_filtered(self.data, proba)
        
        self.results = {
            "baseline": baseline_results,
            "ml_filtered": ml_results,
            "ml_probabilities": proba
        }
        
        return self.results


class VolatilityStrategy(BaseStrategy):
    """Volatility Forecasting Strategy with GARCH and ML"""
    
    def __init__(self, ticker: str, start_date: str, vol_target: float = 0.15,
                 max_leverage: float = 2.0, rebalance_freq: int = 5,
                 slippage_bps: float = 1.0, risk_free: float = 0.015):
        super().__init__(ticker, start_date, risk_free)
        self.vol_target = vol_target
        self.max_leverage = max_leverage
        self.rebalance_freq = rebalance_freq
        self.slippage_bps = slippage_bps
        
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-related features"""
        df = df.copy()
        
        # Realized volatility measures
        df["rv_1d"] = df["ret"].abs()
        df["rv_5d"] = df["ret"].rolling(5).std() * np.sqrt(252)
        df["rv_20d"] = df["ret"].rolling(20).std() * np.sqrt(252)
        df["rv_60d"] = df["ret"].rolling(60).std() * np.sqrt(252)
        
        # Intraday volatility proxies
        df["high_low"] = (df["High"] / df["Low"] - 1) * 100
        df["open_close"] = (df["Close"] / df["Open"] - 1) * 100
        
        # Volume features
        df["volume_ma"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / (df["volume_ma"] + 1e-9)
        df["volume_vol"] = np.log(df["Volume"]).rolling(20).std()
        
        # Return features
        df["ret_lag1"] = df["ret"].shift(1)
        df["ret_lag2"] = df["ret"].shift(2)
        df["ret_lag5"] = df["ret"].shift(5)
        df["ret_abs_ma"] = df["rv_1d"].rolling(10).mean()
        
        # Momentum and trend
        df["mom_5"] = df["Close"].pct_change(5)
        df["mom_20"] = df["Close"].pct_change(20)
        df["trend_20"] = df["Close"] / df["Close"].rolling(20).mean() - 1
        
        return df
    
    def fit_garch_model(self, returns: pd.Series) -> Tuple[Optional[object], Optional[object]]:
        """Fit GARCH(1,1) model"""
        if arch_model is None:
            return None, None
        
        try:
            clean_returns = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
            clean_returns = clean_returns * 100  # Convert to percentage
            
            if len(clean_returns) < 100:
                return None, None
            
            model = arch_model(clean_returns, vol='GARCH', p=1, q=1, rescale=False)
            fitted = model.fit(disp='off')
            return model, fitted
        except Exception:
            return None, None
    
    def forecast_garch_volatility(self, fitted_model, steps: int = 1) -> float:
        """Generate GARCH volatility forecast"""
        if fitted_model is None:
            return np.nan
        try:
            forecast = fitted_model.forecast(horizon=steps, reindex=False)
            vol_forecast = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            return vol_forecast
        except Exception:
            return np.nan
    
    def train_volatility_ml(self, df: pd.DataFrame) -> Tuple[Optional[object], pd.Series]:
        """Train ML model for volatility prediction"""
        feature_cols = [
            "rv_5d", "rv_20d", "rv_60d", "high_low", "open_close",
            "volume_ratio", "volume_vol", "ret_lag1", "ret_lag2", "ret_lag5",
            "ret_abs_ma", "mom_5", "mom_20", "trend_20"
        ]
        
        X = df[feature_cols].copy().fillna(method="ffill").fillna(method="bfill")
        y = df["rv_1d"].shift(-1)  # Predict next day volatility
        
        # Remove rows with missing target
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        if len(X) < 100:
            return None, pd.Series(index=df.index, dtype=float).fillna(np.nan)
        
        # Time series split for validation
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
        ])
        
        try:
            model.fit(X, y)
            
            # Generate predictions
            predictions = pd.Series(index=df.index, dtype=float)
            X_full = df[feature_cols].fillna(method="ffill").fillna(method="bfill")
            valid_full_idx = X_full.dropna().index
            predictions.loc[valid_full_idx] = model.predict(X_full.loc[valid_full_idx])
            
            return model, predictions
        except Exception:
            return None, pd.Series(index=df.index, dtype=float).fillna(np.nan)
    
    def calculate_position_size(self, predicted_vol: float) -> float:
        """Calculate position size based on predicted volatility"""
        if pd.isna(predicted_vol) or predicted_vol <= 0:
            return 1.0
        
        position_size = self.vol_target / predicted_vol
        position_size = np.clip(position_size, 1/self.max_leverage, self.max_leverage)
        
        return position_size
    
    def backtest_vol_strategy(self, df: pd.DataFrame, vol_predictions: pd.Series) -> pd.DataFrame:
        """Backtest volatility-based strategy"""
        df = df.copy()
        
        # Calculate position sizes
        df["position_size"] = vol_predictions.apply(self.calculate_position_size)
        
        # Rebalance only every N days
        df["rebalance"] = False
        df.iloc[::self.rebalance_freq, df.columns.get_loc("rebalance")] = True
        df["rebalance"].iloc[0] = True
        
        # Forward fill position sizes between rebalance dates
        df["active_position"] = np.nan
        df.loc[df["rebalance"], "active_position"] = df.loc[df["rebalance"], "position_size"]
        df["active_position"] = df["active_position"].fillna(method="ffill")
        
        # Calculate strategy returns
        df["strat_ret"] = df["active_position"].shift(1) * df["ret"]
        
        # Apply transaction costs on rebalance days
        trade_flags = df["rebalance"].shift(1).fillna(False)
        df["strat_ret"] = self.apply_slippage(df["strat_ret"], trade_flags, self.slippage_bps)
        
        # Calculate equity curve
        df["equity"] = (1 + df["strat_ret"]).cumprod()
        
        return df
    
    def prepare_data(self, **kwargs) -> pd.DataFrame:
        """Prepare data for volatility strategy"""
        self.data = self.fetch_data()
        self.data = self.add_volatility_features(self.data)
        self.data = self.data.dropna(subset=["rv_20d", "volume_ratio"])
        return self.data
    
    def run_backtest(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Run complete volatility forecasting backtest"""
        if self.data is None:
            self.prepare_data()
        
        # Baseline: Buy & Hold
        baseline = self.data.copy()
        baseline["strat_ret"] = baseline["ret"]
        baseline["equity"] = (1 + baseline["strat_ret"]).cumprod()
        
        results = {"buy_hold": baseline}
        
        # GARCH-based strategy
        garch_forecasts = pd.Series(index=self.data.index, dtype=float)
        for i in range(250, len(self.data)):
            try:
                train_data = self.data["ret"].iloc[:i]
                model, fitted = self.fit_garch_model(train_data)
                if fitted is not None:
                    forecast_vol = self.forecast_garch_volatility(fitted)
                    garch_forecasts.iloc[i] = forecast_vol
            except Exception:
                continue
        
        if garch_forecasts.notna().any():
            garch_results = self.backtest_vol_strategy(self.data, garch_forecasts)
            results["garch"] = garch_results
        
        # ML-based strategy
        ml_model, ml_predictions = self.train_volatility_ml(self.data)
        if ml_model is not None and ml_predictions.notna().any():
            ml_results = self.backtest_vol_strategy(self.data, ml_predictions)
            results["ml"] = ml_results
        
        # Store forecasts for visualization
        results["forecasts"] = {
            "garch": garch_forecasts,
            "ml": ml_predictions,
            "realized": self.data["rv_20d"]
        }
        
        self.results = results
        return self.results


class StrategyFactory:
    """Factory class to create strategy instances"""
    
    STRATEGIES = {
        "MA Crossover": MACrossoverStrategy,
        "Volatility Forecasting": VolatilityStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, **kwargs) -> BaseStrategy:
        """Create strategy instance"""
        if strategy_name not in cls.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return cls.STRATEGIES[strategy_name](**kwargs)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategies"""
        return list(cls.STRATEGIES.keys())
