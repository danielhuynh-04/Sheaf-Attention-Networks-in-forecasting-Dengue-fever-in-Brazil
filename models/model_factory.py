# models/model_factory.py
# ----------------------------------------------------------
# Centralized factory for dynamically initializing any of the
# 5 benchmarked graph neural network architectures by name.
# ----------------------------------------------------------


def build_model(name, **kwargs):
    """
    Build and return a graph neural network model by name.

    Supported names: 'gnn', 'gcn', 'gat', 'sheaf', 'sheaf_conn'
    """

    if name == "gnn":
        from .simple_gnn import SimpleGNN
        return SimpleGNN(**kwargs)

    elif name == "gcn":
        from .gcn_model import GCNModel
        return GCNModel(**kwargs)

    elif name == "gat":
        from .temporal_gat import TemporalGAT
        return TemporalGAT(**kwargs)

    elif name == "sheaf":
        # Sheaf Neural Network baseline (edge MLP restriction maps)
        from .sheaf_model import SheafTemporal
        return SheafTemporal(**kwargs)

    elif name in ["sheaf_conn", "sheaf_connection"]:
        # Sheaf-Connection: improved variant with 2D rotation restriction maps
        from .sheaf_connection import SheafConnectionTemporal
        return SheafConnectionTemporal(**kwargs)

    else:
        raise ValueError(f"Unknown model {name}")