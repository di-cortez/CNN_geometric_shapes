"""Formatting helpers shared by GUI widgets."""

def format_weight(value: float) -> str:
    """Render a kernel weight with four decimal places and aligned width."""
    return f"{value:>7.4f}"
