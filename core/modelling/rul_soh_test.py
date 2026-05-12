import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import curve_fit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================================================
# Metrics
# =========================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return None

    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


# =========================================================
# Config
# =========================================================

@dataclass
class RULSOHExperimentConfig:
    eol_threshold: float = 0.7
    train_ratio: float = 0.7
    window_size: int = 10
    epochs: int = 250
    batch_size: int = 16
    learning_rate: float = 1e-3
    hidden_size: int = 64
    random_seed: int = 42


# =========================================================
# Feature Engineering
# =========================================================

class BatteryFeatureEngineer:
    def __init__(self, eol_threshold: float = 0.7):
        self.eol_threshold = eol_threshold

    def prepare_aging_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Input aging dataframe is empty.")

        if "cycle_index" not in df.columns:
            raise ValueError("Missing required column: cycle_index")

        if "capacity" not in df.columns:
            raise ValueError("Missing required column: capacity")

        df = df.copy()
        df = df.sort_values("cycle_index").reset_index(drop=True)

        df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
        df = df.dropna(subset=["cycle_index", "capacity"])

        q0 = df["capacity"].iloc[0]
        df["soh"] = df["capacity"] / q0

        eol_candidates = df[df["soh"] <= self.eol_threshold]

        if not eol_candidates.empty:
            eol_cycle = int(eol_candidates["cycle_index"].iloc[0])
        else:
            eol_cycle = int(df["cycle_index"].iloc[-1])

        df["rul"] = eol_cycle - df["cycle_index"]
        df["rul"] = df["rul"].clip(lower=0)

        df["capacity_fade"] = 1 - df["soh"]
        df["delta_soh"] = df["soh"].diff().fillna(0)
        df["fade_rate"] = -df["soh"].diff().fillna(0)

        return df

    def get_feature_columns(self, df: pd.DataFrame):
        candidates = [
            "cycle_index",
            "capacity",
            "soh",
            "capacity_fade",
            "delta_soh",
            "fade_rate",
            "voltage_mean",
            "voltage_min",
            "voltage_max",
            "temperature_mean",
            "temperature_max",
            "time_duration",
            "physical_soh",
            "physical_rul",
        ]

        return [c for c in candidates if c in df.columns]

    def make_sequence_data(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        window_size: int,
    ):
        values = df[feature_cols].values.astype(np.float32)
        targets = df[target_cols].values.astype(np.float32)

        X, y = [], []

        for i in range(window_size, len(df)):
            X.append(values[i - window_size:i])
            y.append(targets[i])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# =========================================================
# Physical Model
# SOH(k) = 1 - a*sqrt(k) - b*k
# =========================================================

class CapacityDegradationPhysicalModel:
    name = "Physical degradation model"

    def __init__(self, eol_threshold: float = 0.7):
        self.eol_threshold = eol_threshold
        self.params = None

    @staticmethod
    def degradation_function(k, a, b):
        k = np.maximum(k, 1)
        return 1 - a * np.sqrt(k) - b * k

    def fit(self, train_df: pd.DataFrame):
        k = train_df["cycle_index"].values.astype(float)
        soh = train_df["soh"].values.astype(float)

        k = np.maximum(k, 1)

        params, _ = curve_fit(
            self.degradation_function,
            k,
            soh,
            p0=[0.001, 0.0005],
            bounds=([0, 0], [1, 1]),
            maxfev=20000,
        )

        self.params = params
        return self

    def predict_soh(self, cycle_index):
        if self.params is None:
            raise RuntimeError("Physical model is not fitted.")

        k = np.asarray(cycle_index, dtype=float)
        return self.degradation_function(k, *self.params)

    def predict_rul_single(self, current_cycle: int, max_search_cycle: int = 3000):
        future_cycles = np.arange(current_cycle, max_search_cycle + 1)
        future_soh = self.predict_soh(future_cycles)

        below = np.where(future_soh <= self.eol_threshold)[0]

        if len(below) == 0:
            return max_search_cycle - current_cycle

        eol_cycle = future_cycles[below[0]]
        return max(int(eol_cycle - current_cycle), 0)

    def predict_rul(self, cycle_index):
        return np.array([
            self.predict_rul_single(int(k))
            for k in cycle_index
        ])

    def predict(self, test_df: pd.DataFrame):
        cycle_index = test_df["cycle_index"].values

        pred_soh = self.predict_soh(cycle_index)
        pred_rul = self.predict_rul(cycle_index)

        return pd.DataFrame({
            "cycle_index": cycle_index,
            "true_soh": test_df["soh"].values,
            "true_rul": test_df["rul"].values,
            "pred_soh": pred_soh,
            "pred_rul": pred_rul,
            "model": self.name,
        })


# =========================================================
# Optional Exponential Physical Model
# =========================================================

class ExponentialPhysicalModel:
    name = "Exponential degradation model"

    def __init__(self, eol_threshold: float = 0.7):
        self.eol_threshold = eol_threshold
        self.params = None

    @staticmethod
    def degradation_function(k, a, b):
        k = np.maximum(k, 1)
        return 1 - a * (1 - np.exp(-b * k))

    def fit(self, train_df: pd.DataFrame):
        k = train_df["cycle_index"].values.astype(float)
        soh = train_df["soh"].values.astype(float)

        self.params, _ = curve_fit(
            self.degradation_function,
            k,
            soh,
            p0=[0.3, 0.001],
            bounds=([0, 0], [1, 1]),
            maxfev=20000,
        )

        return self

    def predict_soh(self, cycle_index):
        if self.params is None:
            raise RuntimeError("Exponential physical model is not fitted.")

        return self.degradation_function(
            np.asarray(cycle_index, dtype=float),
            *self.params,
        )

    def predict_rul_single(self, current_cycle: int, max_search_cycle: int = 3000):
        future_cycles = np.arange(current_cycle, max_search_cycle + 1)
        future_soh = self.predict_soh(future_cycles)

        below = np.where(future_soh <= self.eol_threshold)[0]

        if len(below) == 0:
            return max_search_cycle - current_cycle

        eol_cycle = future_cycles[below[0]]
        return max(int(eol_cycle - current_cycle), 0)

    def predict_rul(self, cycle_index):
        return np.array([
            self.predict_rul_single(int(k))
            for k in cycle_index
        ])

    def predict(self, test_df: pd.DataFrame):
        cycle_index = test_df["cycle_index"].values

        return pd.DataFrame({
            "cycle_index": cycle_index,
            "true_soh": test_df["soh"].values,
            "true_rul": test_df["rul"].values,
            "pred_soh": self.predict_soh(cycle_index),
            "pred_rul": self.predict_rul(cycle_index),
            "model": self.name,
        })


# =========================================================
# GRU Network
# =========================================================

class GRUNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()

        if not isinstance(input_size, int):
            raise TypeError(f"input_size should be int, got {type(input_size)}")

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        output, _ = self.gru(x)
        last = output[:, -1, :]
        return self.head(last)


# =========================================================
# GRU AI Model
# =========================================================

class GRUBatteryModel:
    name = "GRU model"

    def __init__(self, config: RULSOHExperimentConfig):
        self.config = config
        self.model = None
        self.feature_cols = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_engineer = BatteryFeatureEngineer(config.eol_threshold)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train_df: pd.DataFrame):
        df = train_df.copy()

        self.feature_cols = self.feature_engineer.get_feature_columns(df)

        if not self.feature_cols:
            raise ValueError("No valid feature columns found for GRU model.")

        target_cols = ["soh", "rul"]

        df[self.feature_cols] = self.scaler_x.fit_transform(df[self.feature_cols])
        df[target_cols] = self.scaler_y.fit_transform(df[target_cols])

        X, y = self.feature_engineer.make_sequence_data(
            df=df,
            feature_cols=self.feature_cols,
            target_cols=target_cols,
            window_size=self.config.window_size,
        )

        if len(X) == 0:
            raise ValueError("Not enough data to build GRU sequences.")

        torch.manual_seed(self.config.random_seed)

        input_size = int(len(self.feature_cols))

        self.model = GRUNetwork(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
        ).to(self.device)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        loss_fn = nn.MSELoss()

        self.model.train()

        for _ in range(self.config.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, test_df: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("GRU model is not fitted.")

        df = test_df.copy()
        target_cols = ["soh", "rul"]

        df[self.feature_cols] = self.scaler_x.transform(df[self.feature_cols])

        X, _ = self.feature_engineer.make_sequence_data(
            df=df,
            feature_cols=self.feature_cols,
            target_cols=target_cols,
            window_size=self.config.window_size,
        )

        if len(X) == 0:
            raise ValueError("Not enough test data to build GRU sequences.")

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred_scaled = self.model(X_tensor).cpu().numpy()

        pred = self.scaler_y.inverse_transform(pred_scaled)

        aligned_df = test_df.iloc[self.config.window_size:].copy()

        return pd.DataFrame({
            "cycle_index": aligned_df["cycle_index"].values,
            "true_soh": aligned_df["soh"].values,
            "true_rul": aligned_df["rul"].values,
            "pred_soh": pred[:, 0],
            "pred_rul": pred[:, 1],
            "model": self.name,
        })


# Backward compatibility
GRURULSOHModel = GRUBatteryModel


# =========================================================
# Hybrid: Physical features + GRU
# =========================================================

class HybridPhysicalGRUModel(GRUBatteryModel):
    name = "Physical + GRU hybrid model"

    def __init__(
        self,
        config: RULSOHExperimentConfig,
        physical_model_class=CapacityDegradationPhysicalModel,
        ai_model_class=None,
    ):
        super().__init__(config)
        self.physical_model = physical_model_class(
            eol_threshold=config.eol_threshold
        )

    def _add_physical_features(self, df: pd.DataFrame):
        df = df.copy()

        df["physical_soh"] = self.physical_model.predict_soh(
            df["cycle_index"].values
        )

        df["physical_rul"] = self.physical_model.predict_rul(
            df["cycle_index"].values
        )

        return df

    def fit(self, train_df: pd.DataFrame):
        self.physical_model.fit(train_df)
        train_df_with_physics = self._add_physical_features(train_df)
        return super().fit(train_df_with_physics)

    def predict(self, test_df: pd.DataFrame):
        test_df_with_physics = self._add_physical_features(test_df)
        result = super().predict(test_df_with_physics)
        result["model"] = self.name
        return result


# =========================================================
# Experiment Runner
# =========================================================

class RULSOHExperimentRunner:
    def __init__(self, config: RULSOHExperimentConfig):
        self.config = config
        self.feature_engineer = BatteryFeatureEngineer(config.eol_threshold)

    def train_test_split_by_time(self, df: pd.DataFrame):
        split_index = int(len(df) * self.config.train_ratio)

        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()

        return train_df, test_df

    def evaluate_prediction(self, pred_df: pd.DataFrame):
        return {
            "SOH RMSE": rmse(pred_df["true_soh"], pred_df["pred_soh"]),
            "SOH MAE": mae(pred_df["true_soh"], pred_df["pred_soh"]),
            "RUL RMSE": rmse(pred_df["true_rul"], pred_df["pred_rul"]),
            "RUL MAE": mae(pred_df["true_rul"], pred_df["pred_rul"]),
            "RUL MAPE (%)": safe_mape(
                pred_df["true_rul"],
                pred_df["pred_rul"],
            ),
        }

    def build_health_summary(self, df: pd.DataFrame):
        first_capacity = df["capacity"].iloc[0]
        last_capacity = df["capacity"].iloc[-1]
        last_soh = df["soh"].iloc[-1]
        total_cycles = int(df["cycle_index"].max())

        eol_rows = df[df["soh"] <= self.config.eol_threshold]

        if eol_rows.empty:
            eol_cycle = None
            eol_status = "EOL not reached"
        else:
            eol_cycle = int(eol_rows["cycle_index"].iloc[0])
            eol_status = "EOL reached"

        capacity_loss_pct = (1 - last_capacity / first_capacity) * 100

        recent = df.tail(min(20, len(df)))
        recent_fade_rate = float(recent["fade_rate"].mean())

        return {
            "Initial Capacity": round(float(first_capacity), 4),
            "Latest Capacity": round(float(last_capacity), 4),
            "Latest SOH": round(float(last_soh), 4),
            "Capacity Loss (%)": round(float(capacity_loss_pct), 2),
            "Total Observed Cycles": total_cycles,
            "EOL Threshold": self.config.eol_threshold,
            "EOL Status": eol_status,
            "EOL Cycle": eol_cycle,
            "Recent Avg Fade Rate": round(recent_fade_rate, 6),
        }

    def prepare_feature_preview(
        self,
        aging_df: pd.DataFrame,
        dataset_id: str,
        unit_id: str,
    ):
        df = self.feature_engineer.prepare_aging_dataframe(aging_df)

        df["dataset_id"] = dataset_id
        df["unit_id"] = unit_id

        if "cycle_type" not in df.columns:
            df["cycle_type"] = "discharge"

        return df

    def run(
        self,
        aging_df: pd.DataFrame,
        dataset_id: str,
        unit_id: str,
        physical_model_class,
        ai_model_class,
        hybrid_model_class,
    ):
        df = self.feature_engineer.prepare_aging_dataframe(aging_df)

        df["dataset_id"] = dataset_id
        df["unit_id"] = unit_id

        if "cycle_type" not in df.columns:
            df["cycle_type"] = "discharge"

        if df.empty:
            raise ValueError("Prepared aging dataframe is empty.")

        train_df, test_df = self.train_test_split_by_time(df)

        if train_df.empty or test_df.empty:
            raise ValueError("Train or test data is empty. Please check train_ratio.")

        physical_model = physical_model_class(
            eol_threshold=self.config.eol_threshold
        )

        ai_model = ai_model_class(self.config)

        hybrid_model = hybrid_model_class(
            config=self.config,
            physical_model_class=physical_model_class,
            ai_model_class=ai_model_class,
        )

        models = [
            physical_model,
            ai_model,
            hybrid_model,
        ]

        predictions = []
        metrics = []

        for model in models:
            model.fit(train_df)
            pred_df = model.predict(test_df)

            if pred_df.empty:
                raise ValueError(f"{model.name} returned empty prediction dataframe.")

            predictions.append(pred_df)

            model_metrics = self.evaluate_prediction(pred_df)
            model_metrics["Model"] = model.name
            metrics.append(model_metrics)

        prediction_df = pd.concat(predictions, ignore_index=True)
        metrics_df = pd.DataFrame(metrics)

        if "Model" in metrics_df.columns:
            cols = ["Model"] + [c for c in metrics_df.columns if c != "Model"]
            metrics_df = metrics_df[cols]

        health_summary = self.build_health_summary(df)

        return {
            "prepared_data": df,
            "train_data": train_df,
            "test_data": test_df,
            "predictions": prediction_df,
            "metrics": metrics_df,
            "health_summary": health_summary,
        }