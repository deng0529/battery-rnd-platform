from core.modelling.rul_soh_test import (
    CapacityDegradationPhysicalModel,
    GRURULSOHModel,
    HybridPhysicalGRUModel,
)


MODEL_REGISTRY = {
    "physical_capacity": {
        "display_name": "Capacity Degradation Physical Model",
        "class": CapacityDegradationPhysicalModel,
    },

    "gru": {
        "display_name": "GRU Model",
        "class": GRURULSOHModel,
    },

    "hybrid_physical_gru": {
        "display_name": "Physical + GRU Hybrid",
        "class": HybridPhysicalGRUModel,
    },
}