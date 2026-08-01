"""Reporting and visualization package exporting table rendering and matplotlib plotting utilities."""

from hdlnn.report.plots import plot_comparison_bars, plot_trajectory_divergence
from hdlnn.report.tables import load_comparison_table, render_csv, render_markdown_table

__all__ = [
    "load_comparison_table",
    "render_markdown_table",
    "render_csv",
    "plot_trajectory_divergence",
    "plot_comparison_bars",
]
