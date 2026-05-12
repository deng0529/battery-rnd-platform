from core.modelling.rul_soh_test import (
    CapacityDegradationPhysicalModel,
    # ExponentialPhysicalModel,
    GRUNetwork,
    HybridPhysicalGRUModel,
)


PHYSICAL_MODEL_REGISTRY = {
    "capacity_degradation": {
        "display_name": "Capacity degradation curve",
        "class": CapacityDegradationPhysicalModel,
    },
    # "exponential_degradation": {
    #     "display_name": "Exponential degradation model",
    #     "class": ExponentialPhysicalModel,
    # },
}


AI_MODEL_REGISTRY = {
    "gru": {
        "display_name": "GRU neural network",
        "class": GRUNetwork,
    },
}


HYBRID_MODEL_CLASS = HybridPhysicalGRUModel