import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import curve_fit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


INPUT_FEATURES = [
    "cycle_index",
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_max",
    "voltage_drop",
    "current_mean",
    "current_std",
    "temperature_mean",
    "temperature_max",
    "temperature_rise",
    "time_duration",
]

OUTPUT_LABELS = ["soh", "rul"]


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

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


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


class BatteryFeatureEngineer:

    def prepare_model_feature_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("battery_model_features is empty.")

        required = INPUT_FEATURES + OUTPUT_LABELS + ["cycle_index"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"battery_model_features missing columns: {missing}")

        df = df.copy()
        df = df.sort_values("cycle_index").reset_index(drop=True)

        numeric_cols = list(set(INPUT_FEATURES + OUTPUT_LABELS + ["capacity"]))

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=INPUT_FEATURES + OUTPUT_LABELS)
        df = df.sort_values("cycle_index").reset_index(drop=True)

        return df

    def get_input_features(self, df: pd.DataFrame, include_physics: bool = False):
        features = [c for c in INPUT_FEATURES if c in df.columns]

        if include_physics:
            for col in ["physical_soh", "physical_rul"]:
                if col in df.columns:
                    features.append(col)

        return features

    def make_sequence_data(self, df, feature_cols, target_cols, window_size):
        values = df[feature_cols].values.astype(np.float32)
        targets = df[target_cols].values.astype(np.float32)

        X, y = [], []

        for i in range(window_size, len(df)):
            X.append(values[i - window_size:i])
            y.append(targets[i])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class CapacityDegradationPhysicalModel:
    name = "Physical degradation model"

    def __init__(self, eol_threshold: float = 0.7):
        self.eol_threshold = eol_threshold
        self.params = None

    @staticmethod
    def degradation_function(k, a, b):
        k = np.maximum(k, 1)
        return 1 - a * np.sqrt(k) - b * k

    def fit(self, train_df):
        k = train_df["cycle_index"].values.astype(float)
        soh = train_df["soh"].values.astype(float)

        self.params, _ = curve_fit(
            self.degradation_function,
            k,
            soh,
            p0=[0.001, 0.0005],
            bounds=([0, 0], [1, 1]),
            maxfev=20000,
        )
        return self

    def predict_soh(self, cycle_index):
        if self.params is None:
            raise RuntimeError("Physical model is not fitted.")
        return self.degradation_function(np.asarray(cycle_index, dtype=float), *self.params)

    def predict_rul_single(self, current_cycle: int, max_search_cycle: int = 3000):
        future_cycles = np.arange(current_cycle, max_search_cycle + 1)
        future_soh = self.predict_soh(future_cycles)
        below = np.where(future_soh <= self.eol_threshold)[0]

        if len(below) == 0:
            return max_search_cycle - current_cycle

        return max(int(future_cycles[below[0]] - current_cycle), 0)

    def predict_rul(self, cycle_index):
        return np.array([self.predict_rul_single(int(k)) for k in cycle_index])

    def predict(self, test_df):
        cycle_index = test_df["cycle_index"].values

        return pd.DataFrame({
            "cycle_index": cycle_index,
            "true_soh": test_df["soh"].values,
            "true_rul": test_df["rul"].values,
            "pred_soh": self.predict_soh(cycle_index),
            "pred_rul": self.predict_rul(cycle_index),
            "model": self.name,
        })


class ExponentialPhysicalModel(CapacityDegradationPhysicalModel):
    name = "Exponential degradation model"

    @staticmethod
    def degradation_function(k, a, b):
        k = np.maximum(k, 1)
        return 1 - a * (1 - np.exp(-b * k))

    def fit(self, train_df):
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


class MLPBatteryModel:
    name = "MLP model"

    def __init__(self, config: RULSOHExperimentConfig):
        self.config = config
        self.feature_engineer = BatteryFeatureEngineer()
        self.feature_cols = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=config.random_seed,
        )

    def fit(self, train_df):
        df = train_df.copy()
        self.feature_cols = self.feature_engineer.get_input_features(df, include_physics=False)

        X = df[self.feature_cols].values
        y = df[OUTPUT_LABELS].values

        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        self.model.fit(X_scaled, y_scaled)
        return self

    def predict_values(self, df):
        X = df[self.feature_cols].values
        X_scaled = self.scaler_x.transform(X)
        pred_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(pred_scaled)

    def predict(self, test_df):
        pred = self.predict_values(test_df)

        return pd.DataFrame({
            "cycle_index": test_df["cycle_index"].values,
            "true_soh": test_df["soh"].values,
            "true_rul": test_df["rul"].values,
            "pred_soh": pred[:, 0],
            "pred_rul": pred[:, 1],
            "model": self.name,
        })


class ResidualMLPNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h = torch.relu(self.input_layer(x))
        h = torch.relu(h + self.block1(h))
        h = torch.relu(h + self.block2(h))
        return self.output_layer(h)


class ResidualMLPBatteryModel:
    name = "Residual MLP model"

    def __init__(self, config: RULSOHExperimentConfig):
        self.config = config
        self.feature_engineer = BatteryFeatureEngineer()
        self.feature_cols = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train_df):
        df = train_df.copy()
        self.feature_cols = self.feature_engineer.get_input_features(df, include_physics=False)

        X = df[self.feature_cols].values.astype(np.float32)
        y = df[OUTPUT_LABELS].values.astype(np.float32)

        X_scaled = self.scaler_x.fit_transform(X).astype(np.float32)
        y_scaled = self.scaler_y.fit_transform(y).astype(np.float32)

        torch.manual_seed(self.config.random_seed)

        self.model = ResidualMLPNetwork(
            input_size=len(self.feature_cols),
            hidden_size=self.config.hidden_size,
        ).to(self.device)

        dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_scaled, dtype=torch.float32),
        )

        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
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

    def predict_values(self, df):
        X = df[self.feature_cols].values.astype(np.float32)
        X_scaled = self.scaler_x.transform(X).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(
                torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            ).cpu().numpy()

        return self.scaler_y.inverse_transform(pred_scaled)

    def predict(self, test_df):
        pred = self.predict_values(test_df)

        return pd.DataFrame({
            "cycle_index": test_df["cycle_index"].values,
            "true_soh": test_df["soh"].values,
            "true_rul": test_df["rul"].values,
            "pred_soh": pred[:, 0],
            "pred_rul": pred[:, 1],
            "model": self.name,
        })


class GRUNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()

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


class GRUBatteryModel:
    name = "GRU model"

    def __init__(self, config: RULSOHExperimentConfig):
        self.config = config
        self.feature_engineer = BatteryFeatureEngineer()
        self.feature_cols = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train_df):
        df = train_df.copy()
        self.feature_cols = self.feature_engineer.get_input_features(df, include_physics=False)

        df[self.feature_cols] = self.scaler_x.fit_transform(df[self.feature_cols])
        df[OUTPUT_LABELS] = self.scaler_y.fit_transform(df[OUTPUT_LABELS])

        X, y = self.feature_engineer.make_sequence_data(
            df=df,
            feature_cols=self.feature_cols,
            target_cols=OUTPUT_LABELS,
            window_size=self.config.window_size,
        )

        if len(X) == 0:
            raise ValueError("Not enough data to build GRU sequences.")

        torch.manual_seed(self.config.random_seed)

        self.model = GRUNetwork(
            input_size=len(self.feature_cols),
            hidden_size=self.config.hidden_size,
        ).to(self.device)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
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

    def predict_values_with_aligned_df(self, df):
        test_df = df.copy()
        test_df[self.feature_cols] = self.scaler_x.transform(test_df[self.feature_cols])

        X, _ = self.feature_engineer.make_sequence_data(
            df=test_df,
            feature_cols=self.feature_cols,
            target_cols=OUTPUT_LABELS,
            window_size=self.config.window_size,
        )

        if len(X) == 0:
            raise ValueError("Not enough test data to build GRU sequences.")

        self.model.eval()

        with torch.no_grad():
            pred_scaled = self.model(
                torch.tensor(X, dtype=torch.float32).to(self.device)
            ).cpu().numpy()

        pred = self.scaler_y.inverse_transform(pred_scaled)
        aligned_df = df.iloc[self.config.window_size:].copy()

        return aligned_df, pred

    def predict(self, test_df):
        aligned_df, pred = self.predict_values_with_aligned_df(test_df)

        return pd.DataFrame({
            "cycle_index": aligned_df["cycle_index"].values,
            "true_soh": aligned_df["soh"].values,
            "true_rul": aligned_df["rul"].values,
            "pred_soh": pred[:, 0],
            "pred_rul": pred[:, 1],
            "model": self.name,
        })


class HybridPhysicalAIModel:
    def __init__(self, config, physical_model_class, ai_model_class):
        self.config = config
        self.physical_model = physical_model_class(
            eol_threshold=config.eol_threshold
        )
        self.ai_model = ai_model_class(config)
        self.name = f"Physical + {self.ai_model.name}"

    def _add_physical_features(self, df):
        df = df.copy()
        df["physical_soh"] = self.physical_model.predict_soh(df["cycle_index"].values)
        df["physical_rul"] = self.physical_model.predict_rul(df["cycle_index"].values)
        return df

    def fit(self, train_df):
        self.physical_model.fit(train_df)
        train_df = self._add_physical_features(train_df)

        self.ai_model.feature_engineer = BatteryFeatureEngineer()

        if isinstance(self.ai_model, GRUBatteryModel):
            self.ai_model.feature_cols = None
            self.ai_model.fit(train_df)
            self.ai_model.feature_cols = self.ai_model.feature_engineer.get_input_features(
                train_df,
                include_physics=True,
            )
        else:
            self.ai_model.fit(train_df)
            self.ai_model.feature_cols = self.ai_model.feature_engineer.get_input_features(
                train_df,
                include_physics=True,
            )

            X = train_df[self.ai_model.feature_cols].values
            y = train_df[OUTPUT_LABELS].values

            self.ai_model.scaler_x = StandardScaler()
            self.ai_model.scaler_y = StandardScaler()

            X_scaled = self.ai_model.scaler_x.fit_transform(X)
            y_scaled = self.ai_model.scaler_y.fit_transform(y)

            if isinstance(self.ai_model, MLPBatteryModel):
                self.ai_model.model.fit(X_scaled, y_scaled)

            elif isinstance(self.ai_model, ResidualMLPBatteryModel):
                X_scaled = X_scaled.astype(np.float32)
                y_scaled = y_scaled.astype(np.float32)

                torch.manual_seed(self.config.random_seed)

                self.ai_model.model = ResidualMLPNetwork(
                    input_size=len(self.ai_model.feature_cols),
                    hidden_size=self.config.hidden_size,
                ).to(self.ai_model.device)

                dataset = TensorDataset(
                    torch.tensor(X_scaled, dtype=torch.float32),
                    torch.tensor(y_scaled, dtype=torch.float32),
                )

                loader = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                )

                optimizer = torch.optim.Adam(
                    self.ai_model.model.parameters(),
                    lr=self.config.learning_rate,
                )
                loss_fn = nn.MSELoss()

                self.ai_model.model.train()

                for _ in range(self.config.epochs):
                    for xb, yb in loader:
                        xb = xb.to(self.ai_model.device)
                        yb = yb.to(self.ai_model.device)

                        optimizer.zero_grad()
                        pred = self.ai_model.model(xb)
                        loss = loss_fn(pred, yb)
                        loss.backward()
                        optimizer.step()

        return self

    def predict(self, test_df):
        test_df = self._add_physical_features(test_df)
        result = self.ai_model.predict(test_df)
        result["model"] = self.name
        return result


class RULSOHExperimentRunner:
    def __init__(self, config: RULSOHExperimentConfig):
        self.config = config
        self.feature_engineer = BatteryFeatureEngineer()

    def prepare_feature_preview(self, feature_df):
        return self.feature_engineer.prepare_model_feature_dataframe(feature_df)

    def train_test_split_by_time(self, df):
        split_index = int(len(df) * self.config.train_ratio)
        return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()

    def evaluate_prediction(self, pred_df):
        return {
            "SOH RMSE": rmse(pred_df["true_soh"], pred_df["pred_soh"]),
            "SOH MAE": mae(pred_df["true_soh"], pred_df["pred_soh"]),
            "RUL RMSE": rmse(pred_df["true_rul"], pred_df["pred_rul"]),
            "RUL MAE": mae(pred_df["true_rul"], pred_df["pred_rul"]),
            "RUL MAPE (%)": safe_mape(pred_df["true_rul"], pred_df["pred_rul"]),
        }

    def build_health_summary(self, df):
        latest_soh = float(df["soh"].iloc[-1])
        eol_rows = df[df["soh"] <= self.config.eol_threshold]

        if eol_rows.empty:
            eol_cycle = None
            eol_status = "EOL not reached"
        else:
            eol_cycle = int(eol_rows["cycle_index"].iloc[0])
            eol_status = "EOL reached"

        return {
            "Latest SOH": round(latest_soh, 4),
            "Total Observed Cycles": int(df["cycle_index"].max()),
            "EOL Threshold": self.config.eol_threshold,
            "EOL Status": eol_status,
            "EOL Cycle": eol_cycle,
        }

    def run(self, feature_df, physical_model_class, ai_model_classes, hybrid_model_class):
        df = self.feature_engineer.prepare_model_feature_dataframe(feature_df)

        train_df, test_df = self.train_test_split_by_time(df)

        if train_df.empty or test_df.empty:
            raise ValueError("Train or test data is empty.")

        models = [
            physical_model_class(eol_threshold=self.config.eol_threshold)
        ]

        for ai_model_class in ai_model_classes:
            models.append(ai_model_class(self.config))
            models.append(
                hybrid_model_class(
                    config=self.config,
                    physical_model_class=physical_model_class,
                    ai_model_class=ai_model_class,
                )
            )

        predictions = []
        metrics = []

        for model in models:
            model.fit(train_df)
            pred_df = model.predict(test_df)

            predictions.append(pred_df)

            model_metrics = self.evaluate_prediction(pred_df)
            model_metrics["Model"] = model.name
            metrics.append(model_metrics)

        metrics_df = pd.DataFrame(metrics)

        if "Model" in metrics_df.columns:
            cols = ["Model"] + [c for c in metrics_df.columns if c != "Model"]
            metrics_df = metrics_df[cols]

        return {
            "prepared_data": df,
            "train_data": train_df,
            "test_data": test_df,
            "input_features": INPUT_FEATURES,
            "output_labels": OUTPUT_LABELS,
            "predictions": pd.concat(predictions, ignore_index=True),
            "metrics": metrics_df,
            "health_summary": self.build_health_summary(df),
        }