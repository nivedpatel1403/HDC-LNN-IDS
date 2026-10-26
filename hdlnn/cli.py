"""Command-Line Interface (CLI) for HDC-LNN Anomaly Detection pipeline.

CLI Commands:
    hdlnn train --config <path>
    hdlnn eval --config <path> [--run-id <id>]
    hdlnn stream --config <path> --run-id <id> [--replay-rate <float>] [--check-consistency]
    hdlnn ablate [--configs-dir <path>]
    hdlnn report [--run-id <id>] [--ablation-dir <path>] [--format markdown|csv]
"""

from pathlib import Path
import click

from hdlnn.common.config import load_settings
from hdlnn.common.seed import set_global_seed
from hdlnn.eval import ComparisonTable
from hdlnn.pipeline import run_ablate, run_eval, run_stream, run_train
from hdlnn.report import load_comparison_table, render_csv, render_markdown_table


@click.group()
def main() -> None:
    """HDC-LNN Architecture Command-Line Interface."""
    pass


@main.command(name="train")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
def train_cmd(config: str) -> None:
    """Run model training and validation pipeline."""
    cfg = load_settings(config)
    set_global_seed(cfg.seed)

    click.echo(f"🚀 Starting Training (run_id='{cfg.run_id}', seed={cfg.seed})...")
    res = run_train(cfg)

    click.echo("✅ Training Complete!")
    click.echo(f"   • Run ID: {res.run_id}")
    click.echo(f"   • F1 Score: {res.detection_metrics.get('f1', 0.0):.4f}")
    click.echo(f"   • AUROC: {res.detection_metrics.get('auroc', 0.0):.4f}")
    click.echo(f"   • Artifacts saved under: runs/{res.run_id}/")


@main.command(name="eval")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--run-id", "-r", default=None, help="Run ID of trained checkpoint to load.")
def eval_cmd(config: str, run_id: str | None) -> None:
    """Evaluate pipeline performance on test split."""
    cfg = load_settings(config)
    set_global_seed(cfg.seed)

    click.echo(f"🔍 Starting Evaluation (run_id='{cfg.run_id}', checkpoint='{run_id or 'none'}')...")
    res = run_eval(cfg, run_id_to_load=run_id)

    click.echo("✅ Evaluation Complete!")
    click.echo(f"   • Run ID: {res.run_id}")
    click.echo(f"   • F1 Score: {res.detection_metrics.get('f1', 0.0):.4f}")
    click.echo(f"   • Precision: {res.detection_metrics.get('precision', 0.0):.4f}")
    click.echo(f"   • Recall: {res.detection_metrics.get('recall', 0.0):.4f}")
    click.echo(f"   • AUROC: {res.detection_metrics.get('auroc', 0.0):.4f}")
    click.echo(f"   • Artifacts saved under: runs/{res.run_id}/")


@main.command(name="stream")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--run-id", "-r", required=True, help="Run ID of trained checkpoint to load.")
@click.option("--replay-rate", default=1.0, type=float, help="Stream pacing simulation replay rate factor.")
@click.option("--check-consistency", is_flag=True, help="Verify decision equivalence against batch forward().")
def stream_cmd(config: str, run_id: str, replay_rate: float, check_consistency: bool) -> None:
    """Benchmark streaming step-by-step inference."""
    cfg = load_settings(config)
    set_global_seed(cfg.seed)

    click.echo(f"⚡ Starting Streaming Benchmark (replay_rate={replay_rate}, consistency_check={check_consistency})...")
    res = run_stream(cfg, run_id_to_load=run_id, replay_rate=replay_rate, check_consistency=check_consistency)

    click.echo("✅ Streaming Complete!")
    click.echo(f"   • Run ID: {res.run_id}")
    click.echo(f"   • Mean Latency:       {res.latency_stats.get('mean_ms', 0.0):.3f} ms")
    click.echo(f"   • P50 Latency:        {res.latency_stats.get('p50_ms', 0.0):.3f} ms")
    click.echo(f"   • P95 Latency:        {res.latency_stats.get('p95_ms', 0.0):.3f} ms")
    click.echo(f"   • P99 Latency:        {res.latency_stats.get('p99_ms', 0.0):.3f} ms")
    click.echo(f"   • Max Latency:        {res.latency_stats.get('max_ms', 0.0):.3f} ms")
    click.echo(f"   • Throughput:         {res.latency_stats.get('throughput_eps', 0.0):.1f} events/sec")
    click.echo(f"   • Peak RSS:           {res.memory_stats.get('peak_rss_mb', 0.0):.1f} MB")
    click.echo(f"   • Device:             {res.latency_stats.get('device', 'cpu')}")
    click.echo(f"   • Warm-up N:          {res.latency_stats.get('warmup_n', 0)} events discarded")
    click.echo(f"   • Artifacts saved under: runs/{res.run_id}/")


@main.command(name="ablate")
@click.option("--configs-dir", "-d", default="configs", type=click.Path(exists=True), help="Directory containing config YAMLs.")
def ablate_cmd(configs_dir: str) -> None:
    """Execute multi-configuration ablation suite."""
    yaml_files = sorted(Path(configs_dir).glob("*.yaml"))
    if not yaml_files:
        click.echo(f"❌ Error: No YAML config files found in '{configs_dir}'", err=True)
        return

    first_cfg = load_settings(str(yaml_files[0]))
    set_global_seed(first_cfg.seed)

    click.echo(f"🔬 Starting Ablation Suite ({len(yaml_files)} configs from '{configs_dir}')...")
    table = run_ablate(configs_dir=configs_dir)

    click.echo("✅ Ablation Suite Complete!")
    click.echo(f"   • Timestamp: {table.timestamp}")
    click.echo(f"   • Config Rows Evaluated: {len(table.rows)}")
    click.echo(f"   • Comparison Table saved under: runs/ablation_suite_{table.timestamp}/")


@main.command(name="report")
@click.option("--run-id", "-r", default=None, help="Run ID to render trajectory divergence for.")
@click.option("--ablation-dir", "-a", default=None, help="Path to ablation suite output directory.")
@click.option("--format", "-f", "fmt", type=click.Choice(["markdown", "csv"]), default="markdown", help="Output format.")
def report_cmd(run_id: str | None, ablation_dir: str | None, fmt: str) -> None:
    """Render markdown/CSV comparison tables and reporting artifacts."""
    set_global_seed(42)

    if ablation_dir:
        json_path = Path(ablation_dir) / "comparison_table.json"
    else:
        suite_dirs = sorted(Path("runs").glob("ablation_suite_*"), reverse=True)
        if not suite_dirs:
            click.echo("❌ Error: No ablation suite directories found under 'runs/'", err=True)
            return
        json_path = suite_dirs[0] / "comparison_table.json"

    if not json_path.exists():
        click.echo(f"❌ Error: comparison_table.json not found at '{json_path}'", err=True)
        return

    table = load_comparison_table(str(json_path))

    if fmt == "csv":
        output = render_csv(table)
    else:
        output = render_markdown_table(table)

    click.echo("\n" + output + "\n")


if __name__ == "__main__":
    main()
