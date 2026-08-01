"""Streamlit interactive local dashboard for HDC-LNN evaluation results and trajectory plots.

Launch command:
    streamlit run hdlnn/report/dashboard.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from hdlnn.report.plots import plot_comparison_bars, plot_trajectory_divergence
from hdlnn.report.tables import load_comparison_table, render_markdown_table


def run_dashboard(runs_dir: str = "runs") -> None:
    """Streamlit dashboard entrypoint."""
    st.set_page_config(page_title="HDC-LNN Anomaly Detection Dashboard", layout="wide")

    st.title("🛡️ Hybrid HDC-LNN Anomaly Detection Dashboard")
    st.caption("Reporting & Visualization Layer (L5) — Strict disk artifact inspection mode")

    runs_path = Path(runs_dir)
    if not runs_path.exists():
        st.warning(f"Runs directory '{runs_dir}' does not exist yet. Run an experiment or ablation suite first.")
        return

    # Section 1: Latest Ablation Suite Comparison Table
    st.header("📊 Multi-Config Ablation Comparison Suite")

    suite_dirs = sorted(runs_path.glob("ablation_suite_*"), reverse=True)
    if suite_dirs:
        latest_suite_json = suite_dirs[0] / "comparison_table.json"
        if latest_suite_json.exists():
            table = load_comparison_table(str(latest_suite_json))

            st.markdown(render_markdown_table(table))

            st.subheader("Metric Comparison Chart")
            selected_metric = st.selectbox(
                "Select Comparison Metric",
                options=["f1", "precision", "recall", "auroc", "fpr_at_95_tpr", "mean_latency_ms", "p95_latency_ms", "peak_rss_mb", "model_size_mb", "oov_rate"],
                index=0,
            )
            fig_bar = plot_comparison_bars(table, metric=selected_metric)
            st.pyplot(fig_bar)
        else:
            st.info("No comparison_table.json found in the latest ablation suite folder.")
    else:
        st.info("No ablation suite runs found in 'runs/'.")

    st.divider()

    # Section 2: Entity Trajectory Divergence Viewer
    st.header("📈 Entity Trajectory Drift Viewer")

    run_dirs = [d.name for d in runs_path.iterdir() if d.is_dir() and not d.name.startswith("ablation_suite_")]
    if not run_dirs:
        st.info("No individual experiment run folders found under 'runs/'.")
        return

    selected_run = st.selectbox("Select Experiment Run ID", options=sorted(run_dirs), index=0)

    decisions_path = runs_path / selected_run / "decisions.parquet"
    if not decisions_path.exists():
        st.error(f"decisions.parquet artifact missing for run '{selected_run}'.")
        return

    df_decisions = pd.read_parquet(decisions_path)
    if df_decisions.empty:
        st.info(f"No decisions recorded in run '{selected_run}'.")
        return

    entities = sorted(df_decisions["entity_id"].unique())
    selected_entity = st.selectbox("Select Entity ID (e.g. Source IP)", options=entities, index=0)

    try:
        fig_traj = plot_trajectory_divergence(selected_run, selected_entity, runs_dir=runs_dir)
        st.pyplot(fig_traj)
    except Exception as e:
        st.error(f"Failed to plot trajectory divergence for entity '{selected_entity}': {e}")


if __name__ == "__main__":
    run_dashboard()
