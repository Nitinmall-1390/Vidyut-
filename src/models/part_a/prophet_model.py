"""
VIDYUT Part A — Prophet Forecasting Model
===========================================================================
Wraps Facebook Prophet for 15-minute zone-level demand forecasting with:
  - Karnataka holiday regressors
  - NASA POWER weather regressors
  - Fourier-mode seasonalities for 15-min / daily / weekly / yearly cycles
===========================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.metrics_demand import compute_all_demand_metrics

log = get_logger("vidyut.prophet_model")
warnings.filterwarnings("ignore", message=".*seasonality.*")

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    log.warning("Prophet not installed. ProphetModel will be non-functional.")

import holidays

_KA_HOLIDAYS_PROPHET = pd.DataFrame(
    [
        {"holiday": name, "ds": pd.Timestamp(dt), "lower_window": 0, "upper_window": 1}
        for dt, name in holidays.India(state="KA", years=range(2013, 2027)).items()
    ]
)


class ProphetModel:
    """
    Prophet-based 15-minute electricity demand forecaster.

    One instance per feeder/zone. Fitted on a 'ds'+'y' format DataFrame
    with optional external regressors.
    """

    WEATHER_REGRESSORS = [
        "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M",
        "PRECTOTCORR", "ALLSKY_SFC_SW_DWN",
    ]

    def __init__(
        self,
        feeder_id: str,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        interval_width: float = 0.90,
        add_weather: bool = True,
    ) -> None:
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet package is not installed.")
        self.feeder_id = feeder_id
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        self.add_weather = add_weather
        self.model: Optional[Prophet] = None
        self._regressor_cols: List[str] = []
        self.is_fitted: bool = False

    def _build_model(self, regressor_cols: List[str]) -> "Prophet":
        m = Prophet(
            holidays=_KA_HOLIDAYS_PROPHET,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        # 15-minute intra-day seasonality
        m.add_seasonality(
            name="intraday",
            period=1,
            fourier_order=8,
            prior_scale=self.seasonality_prior_scale,
        )
        # Monthly seasonality
        m.add_seasonality(
            name="monthly",
            period=30.5,
            fourier_order=5,
        )
        for col in regressor_cols:
            m.add_regressor(col, standardize=True)
        return m

    def fit(
        self,
        df: pd.DataFrame,
        datetime_col: str = "timestamp",
        value_col: str = "demand_kw",
    ) -> "ProphetModel":
        """
        Fit the Prophet model.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain datetime_col and value_col.
            Weather columns (optional) must also be present.
        """
        df_prophet = df.rename(
            columns={datetime_col: "ds", value_col: "y"}
        )[["ds", "y"]].copy()
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
        df_prophet["y"] = df_prophet["y"].clip(lower=0)

        # Attach weather regressors if available
        self._regressor_cols = []
        if self.add_weather:
            for col in self.WEATHER_REGRESSORS:
                if col in df.columns:
                    df_prophet[col] = df[col].values
                    self._regressor_cols.append(col)

        log.info(
            "Fitting Prophet for feeder %s | %d rows | regressors: %s",
            self.feeder_id, len(df_prophet), self._regressor_cols,
        )
        self.model = self._build_model(self._regressor_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df_prophet)
        self.is_fitted = True
        return self

    def predict(
        self,
        future_df: pd.DataFrame,
        datetime_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Generate demand forecasts for a future horizon DataFrame.

        Parameters
        ----------
        future_df : pd.DataFrame
            Must contain datetime_col. Include weather columns if model
            was fitted with them.

        Returns
        -------
        pd.DataFrame
            Columns: ds, yhat, yhat_lower, yhat_upper
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        future = future_df.rename(columns={datetime_col: "ds"})[["ds"]].copy()
        future["ds"] = pd.to_datetime(future["ds"])

        for col in self._regressor_cols:
            if col in future_df.columns:
                future[col] = future_df[col].values
            else:
                future[col] = 0.0  # safe fallback

        forecast = self.model.predict(future)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def evaluate(
        self,
        test_df: pd.DataFrame,
        datetime_col: str = "timestamp",
        value_col: str = "demand_kw",
    ) -> Dict[str, float]:
        """Evaluate on held-out test set. Returns metrics dict."""
        forecast = self.predict(test_df, datetime_col=datetime_col)
        y_true = test_df[value_col].values
        y_pred = forecast["yhat"].values[: len(y_true)]
        y_low = forecast["yhat_lower"].values[: len(y_true)]
        y_high = forecast["yhat_upper"].values[: len(y_true)]
        return compute_all_demand_metrics(y_true, y_pred, y_low, y_high)

    def get_components(self) -> Optional[pd.DataFrame]:
        """Return component DataFrame (trend, seasonalities) for decomposition plot."""
        if self.model is None:
            return None
        # Returns last predicted components
        return None  # components available after predict via model.plot_components()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "meta": self.__dict__}, path)
        log.info("ProphetModel saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "ProphetModel":
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.__dict__.update(data["meta"])
        instance.model = data["model"]
        return instance
