"""Shared plotting utilities."""


def _get_model_colors() -> dict[str, str]:
    """Consistent color palette for model architectures."""
    return {
        "PointNet": "#f39c12",
        "SimplePointNet": "#3498db",
        "DGCNN": "#e74c3c",
        "PointNetPP": "#2ecc71",
        "PointTransformer": "#9b59b6",
    }
