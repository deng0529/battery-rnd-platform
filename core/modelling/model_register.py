from core.modelling.rul_soh_test import (
    CapacityDegradationPhysicalModel,
    ExponentialPhysicalModel,
    GRUBatteryModel,
    MLPBatteryModel,
    ResidualMLPBatteryModel,
    HybridPhysicalAIModel,
)


PHYSICAL_MODEL_REGISTRY = {
    "capacity_degradation": {
        "display_name": "Capacity degradation curve",
        "class": CapacityDegradationPhysicalModel,
    },
    "exponential_degradation": {
        "display_name": "Exponential degradation model",
        "class": ExponentialPhysicalModel,
    },
}


AI_MODEL_REGISTRY = {
    "mlp": {
        "display_name": "MLP",
        "class": MLPBatteryModel,
    },
    "residual_mlp": {
        "display_name": "Residual MLP",
        "class": ResidualMLPBatteryModel,
    },
    "gru": {
        "display_name": "GRU",
        "class": GRUBatteryModel,
    },
}


HYBRID_MODEL_CLASS = HybridPhysicalAIModel