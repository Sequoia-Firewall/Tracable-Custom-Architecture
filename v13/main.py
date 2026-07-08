"""
v13/main.py  —  DragonChild entry point.

All configuration lives in settings.json (or a custom path you pass).
Run mode can be chosen interactively at startup or set via --mode.

Usage
-----
    python main.py                         # interactive mode selection
    python main.py my_settings.json        # custom settings file
    python main.py --mode train            # skip mode prompt, go straight to train
    python main.py --mode compare
    python main.py --mode infer
"""
import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Settings import Settings
from SystemHandler import SystemHandler

# ─────────────────────────────────────────────────────────────────────────────
# Interactive helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confirm(console, message: str, default: bool = True) -> bool:
    from rich.prompt import Confirm
    return Confirm.ask(message, default=default, console=console)


def _prompt_int(console, message: str, default: int) -> int:
    from rich.prompt import IntPrompt
    return IntPrompt.ask(message, default=default, console=console)


def _prompt_str(console, message: str, default: str, choices=None) -> str:
    from rich.prompt import Prompt
    return Prompt.ask(message, default=default, choices=choices, console=console)


def _nexseg_files(nexseg_dir: str, dimensions: int) -> tuple[list[str], list[str]]:
    """Return (found, missing) .nexseg paths for the given dimension count."""
    seg_count = 2 ** dimensions
    found, missing = [], []
    for i in range(seg_count):
        path = os.path.join(nexseg_dir, f"segment_{i}.nexseg")
        (found if os.path.exists(path) else missing).append(path)
    return found, missing


# ─────────────────────────────────────────────────────────────────────────────
# Shared graph / metric helpers
# ─────────────────────────────────────────────────────────────────────────────

SEG_COLORS = ["steelblue", "tomato", "mediumseagreen", "darkorchid",
              "darkorange", "hotpink", "teal", "saddlebrown"]


def _make_logger(settings: Settings):
    import Components.RichConsole as RC
    cfg = settings.logging
    prefix = cfg.get("filename_prefix", "v13_run")
    return RC.RichLogger(
        filename=f"{prefix}_{int(time.time())}.log",
        log_level=cfg.get("log_level", 4),
        console_level=cfg.get("console_level", 4),
    )


def _save_system_graph(system: SystemHandler, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)

    for seg in system.segments:
        sc = SEG_COLORS[seg.segment_id % len(SEG_COLORS)]
        comp = seg.segmentComponents
        if comp is None:
            continue
        splitter   = comp["splitter"]
        reviewers  = comp["reviewer"]
        processors = comp["processing_nodes"]

        for node in processors:
            ax.scatter(node.position[0], node.position[1], color=sc, s=12, alpha=0.6, zorder=2)
            for tgt in node.connected_nodes:
                ax.plot([node.position[0], tgt.position[0]],
                        [node.position[1], tgt.position[1]],
                        color=sc, linewidth=0.4, alpha=0.25, zorder=1)
        for tgt in splitter.connected_nodes:
            ax.plot([splitter.position[0], tgt.position[0]],
                    [splitter.position[1], tgt.position[1]],
                    color=sc, linewidth=0.5, alpha=0.35, zorder=1, linestyle="--")
        ax.scatter(splitter.position[0], splitter.position[1],
                   color=sc, s=120, marker="D", zorder=4, label=f"Seg {seg.segment_id} splitter")
        for rev in reviewers:
            ax.scatter(rev.position[0], rev.position[1], color=sc, s=150, marker="*", zorder=4)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _err_pct(pred, actual):
    if pred is None or actual is None or actual == 0:
        return None
    return abs(pred - actual) / abs(actual) * 100.0


def _fmt(val, d=4):
    return f"{val:.{d}f}" if val is not None else "—"


def _fv(val, d=4):
    return f"{val:.{d}f}" if not math.isnan(val) else "—"


# ─────────────────────────────────────────────────────────────────────────────
# Mode: train
# ─────────────────────────────────────────────────────────────────────────────

def run_train(settings: Settings, logger) -> None:
    import pandas as pd
    from rich.rule import Rule
    from rich.table import Table
    from rich import box
    from Components.PreProcessingNode import PreProcesingNode

    console = logger.console
    d   = settings.dataset
    m   = settings.model
    t   = settings.training
    out = settings.output

    console.print(Rule("[bold cyan]DragonChild v13 — Train Mode[/bold cyan]"))

    # ── Show full config before asking anything ───────────────────────────
    cfg = Table(title="Training Configuration", box=box.ROUNDED,
                border_style="cyan", show_lines=True)
    cfg.add_column("Setting",  style="bold white")
    cfg.add_column("Value",    style="bright_cyan", justify="right")
    cfg.add_row("Dataset",          d["csv_path"])
    cfg.add_row("Target",           d["target_column"])
    cfg.add_row("Ignored columns",  str(d.get("ignored_columns") or "none"))
    cfg.add_row("Shuffle",          str(d.get("shuffle", True)))
    cfg.add_row("Max X",            str(m["max_x"]))
    cfg.add_row("Dimensions",       str(m["dimensions"]))
    cfg.add_row("Connection %",     str(m["connection_percentage"]))
    cfg.add_row("Density",          str(m["density"]))
    cfg.add_row("Training mode",    m["training_mode"])
    cfg.add_row("Aggregation mode", m["aggregation_mode"])
    cfg.add_row("Selection %",      str(m["selection_percentage"]))
    cfg.add_row("Epochs",           str(t["epoch_count"]))
    cfg.add_row("Judge iterations", str(t["judge_iterations"]))
    cfg.add_row("Judge clusters",   f"{t.get('judge_min_clusters', '?')} – {t.get('judge_max_clusters', '?')}")
    cfg.add_row("Test split",       f"{int(t.get('test_split', 0.2) * 100)}%")
    console.print(cfg)

    # ── Warn about existing .nexseg files ─────────────────────────────────
    found, _ = _nexseg_files(".", m["dimensions"])
    if found:
        console.print(f"\n[yellow]Warning:[/yellow] {len(found)} trained segment file(s) already exist in this directory:")
        for p in found:
            console.print(f"  [dim]{p}[/dim]")
        console.print("[dim]Training will overwrite them.[/dim]")
        if not _confirm(console, "Overwrite existing segments and retrain?", default=False):
            console.print("[dim]Tip: set run_mode = 'infer' in settings.json to run inference on the existing segments.[/dim]")
            return

    # ── Final go/no-go ────────────────────────────────────────────────────
    if not _confirm(console, "\nBegin training?", default=True):
        console.print("[dim]Aborted.[/dim]")
        return

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = pd.read_csv(d["csv_path"])
    if d.get("shuffle", True):
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    sample_raw = dataset.iloc[0].to_dict()
    actual     = sample_raw.get(d["target_column"])
    logger.log(f"Dataset: {len(dataset)} rows  |  sample[0] actual={actual}", 4, True)

    _pre = PreProcesingNode(Logger=logger, logger_classification=4,
                            removable_columns=d.get("ignored_columns") or None)
    _pre.process_dataset(dataset.copy())

    # ── Build system ──────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Initialising System[/bold cyan]"))
    system = SystemHandler.from_settings(settings, logger)
    system.initializeAllSegments(Loud=True)

    if out.get("save_pretrain_graph", True):
        _save_system_graph(system, out.get("pretrain_graph_path", "nexus_pretrain.png"),
                           "Nexus — Pre-Training Structure")
        logger.log("Pre-training graph saved.", 4, True)

    # ── Pre-train inference ───────────────────────────────────────────────
    console.print(Rule("[bold yellow]Pre-Training Inference[/bold yellow]"))
    pre_pred = system.runInfer(sample_raw.copy(), loud=False,
                               aggregation_mode=m["aggregation_mode"],
                               selection_percentage=m["selection_percentage"])

    # ── Train ─────────────────────────────────────────────────────────────
    console.print(Rule("[bold green]Training[/bold green]"))
    if m["training_mode"] == "full":
        system.train_full(dataset, epoch_count=t["epoch_count"], loud=True)
    else:
        system.train(dataset,
                     epoch_count=t["epoch_count"],
                     judge_iterations=t["judge_iterations"],
                     loud=True,
                     judge_min_clusters=t.get("judge_min_clusters"),
                     judge_max_clusters=t.get("judge_max_clusters"))

    if out.get("save_posttrain_graph", True):
        _save_system_graph(system, out.get("posttrain_graph_path", "nexus_posttrain.png"),
                           "Nexus — Post-Training Structure")
        logger.log("Post-training graph saved.", 4, True)

    # ── Post-train inference ──────────────────────────────────────────────
    console.print(Rule("[bold yellow]Post-Training Inference[/bold yellow]"))
    post_pred = system.runInfer(sample_raw.copy(), loud=True,
                                aggregation_mode=m["aggregation_mode"],
                                selection_percentage=m["selection_percentage"])

    pre_err  = _err_pct(pre_pred,  actual)
    post_err = _err_pct(post_pred, actual)

    infer_table = Table(title="Inference Results", box=box.ROUNDED,
                        border_style="yellow", show_lines=True)
    infer_table.add_column("Stage",      style="bold white")
    infer_table.add_column("Prediction", style="yellow",     justify="right")
    infer_table.add_column("Actual",     style="bold white", justify="right")
    infer_table.add_column("Error %",    style="red",        justify="right")
    infer_table.add_row("Pre-Training",  _fmt(pre_pred),  _fmt(actual),
                        f"{pre_err:.2f}%"  if pre_err  is not None else "—")
    infer_table.add_row("Post-Training", _fmt(post_pred), _fmt(actual),
                        f"{post_err:.2f}%" if post_err is not None else "—")
    if pre_err is not None and post_err is not None:
        imp  = pre_err - post_err
        sign = "[green]▼[/green]" if imp > 0 else "[red]▲[/red]"
        infer_table.add_section()
        infer_table.add_row("Improvement", "—", "—", f"{sign} {abs(imp):.2f}%")
    console.print(infer_table)

    # ── Test evaluation (optional) ────────────────────────────────────────
    if _confirm(console, "\nRun full test-set evaluation?", default=True):
        console.print(Rule("[bold green]System Test Evaluation[/bold green]"))
        split_idx = int(len(dataset) * (1.0 - t.get("test_split", 0.2)))
        test_ds   = dataset.iloc[split_idx:].reset_index(drop=True)
        _evaluate_system(system, test_ds, d["target_column"], m, logger)
        _segment_summary(system, logger)

    _config_table(system, settings, dataset, logger)
    console.print(Rule("[bold green]Training complete[/bold green]"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode: infer
# ─────────────────────────────────────────────────────────────────────────────

def run_infer(settings: Settings, logger) -> None:
    import pandas as pd
    from rich.rule import Rule
    from rich.table import Table
    from rich import box
    from Components.PreProcessingNode import PreProcesingNode

    console = logger.console
    d   = settings.dataset
    m   = settings.model
    inf = settings.infer

    console.print(Rule("[bold cyan]DragonChild v13 — Infer Mode[/bold cyan]"))

    # ── Check .nexseg files exist ─────────────────────────────────────────
    nexseg_dir = inf.get("nexseg_dir", ".")
    found, missing = _nexseg_files(nexseg_dir, m["dimensions"])

    seg_table = Table(title=f"Segment Files  ({nexseg_dir})", box=box.ROUNDED,
                      border_style="cyan", show_lines=True)
    seg_table.add_column("File",   style="bold white")
    seg_table.add_column("Status", style="bold white")
    for p in found:
        seg_table.add_row(os.path.basename(p), "[green]found[/green]")
    for p in missing:
        seg_table.add_row(os.path.basename(p), "[red]MISSING[/red]")
    console.print(seg_table)

    if missing:
        console.print(f"[bold red]{len(missing)} segment file(s) missing.[/bold red] Train the system first.")
        if _confirm(console, "Switch to train mode instead?", default=True):
            run_train(settings, logger)
        return

    # ── Pick sample index ─────────────────────────────────────────────────
    dataset = pd.read_csv(d["csv_path"])
    console.print(f"[dim]Dataset: {len(dataset)} rows  |  target: {d['target_column']}[/dim]")

    idx = _prompt_int(console, "Sample index to evaluate", default=inf.get("sample_index", 0))
    if idx < 0 or idx >= len(dataset):
        console.print(f"[red]Index {idx} out of range (0–{len(dataset) - 1}).[/red]")
        return

    sample_raw = dataset.iloc[idx].to_dict()
    actual     = sample_raw.get(d["target_column"])

    # Show the sample being evaluated
    sample_table = Table(title=f"Sample [{idx}]", box=box.ROUNDED,
                         border_style="dim", show_lines=False)
    sample_table.add_column("Feature", style="bold white", no_wrap=True)
    sample_table.add_column("Value",   style="dim white",  justify="right")
    for k, v in sample_raw.items():
        bold = k == d["target_column"]
        val_str = f"[bold yellow]{v}[/bold yellow]" if bold else str(v)
        sample_table.add_row(f"[bold]{k}[/bold]" if bold else k, val_str)
    console.print(sample_table)

    if not _confirm(console, f"Run inference on this sample (actual {d['target_column']} = {actual})?",
                    default=True):
        console.print("[dim]Aborted.[/dim]")
        return

    # ── Load segments & run ───────────────────────────────────────────────
    _pre = PreProcesingNode(Logger=logger, logger_classification=4,
                            removable_columns=d.get("ignored_columns") or None)
    _pre.process_dataset(dataset.copy())

    console.print(Rule("[bold cyan]Loading Segments[/bold cyan]"))
    system = SystemHandler.from_settings(settings, logger)
    system.load_segments(nexseg_dir)
    logger.log("JudgeNode not restored — all segments weighted equally.", 3, True)

    console.print(Rule("[bold yellow]Inference[/bold yellow]"))
    pred = system.runInfer(sample_raw.copy(),
                           loud=inf.get("loud", True),
                           aggregation_mode=m["aggregation_mode"],
                           selection_percentage=m["selection_percentage"])

    err = _err_pct(pred, actual)

    result_table = Table(title=f"Result  (sample[{idx}])", box=box.ROUNDED,
                         border_style="yellow", show_lines=True)
    result_table.add_column("Field",  style="bold white")
    result_table.add_column("Value",  style="bright_yellow", justify="right")
    result_table.add_row("Prediction",  _fmt(pred))
    result_table.add_row("Actual",      _fmt(actual))
    result_table.add_row("Abs Error",   _fmt(abs(pred - actual) if pred is not None and actual is not None else None))
    result_table.add_row("Error %",     f"{err:.2f}%" if err is not None else "—")
    result_table.add_row("Agg Mode",    m["aggregation_mode"])
    result_table.add_row("Segs Used",   str(system.getNumberSegmentsUsed()))
    console.print(result_table)

    # ── Offer to run another sample ───────────────────────────────────────
    while _confirm(console, "\nEvaluate another sample?", default=False):
        idx = _prompt_int(console, "Sample index", default=idx + 1)
        if idx < 0 or idx >= len(dataset):
            console.print(f"[red]Index {idx} out of range.[/red]")
            break
        sample_raw = dataset.iloc[idx].to_dict()
        actual     = sample_raw.get(d["target_column"])
        pred = system.runInfer(sample_raw.copy(), loud=False,
                               aggregation_mode=m["aggregation_mode"],
                               selection_percentage=m["selection_percentage"])
        err  = _err_pct(pred, actual)
        console.print(f"  sample[{idx}]  pred=[bold yellow]{_fmt(pred)}[/bold yellow]  "
                      f"actual=[bold]{_fmt(actual)}[/bold]  "
                      f"err=[red]{f'{err:.2f}%' if err is not None else '—'}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: compare
# ─────────────────────────────────────────────────────────────────────────────

def _compare_job_list(c: dict) -> list[tuple[str, str]]:
    """Return (model_name, config_str) for every job that will run."""
    jobs = []
    if c.get("run_linear"):  jobs.append(("LinearRegression",  "—"))
    if c.get("run_knn"):
        jobs.append(("KNN", "k=5"))
        jobs.append(("KNN", "k=10"))
    if c.get("run_rf"):      jobs.append(("RandomForest",       "—"))
    if c.get("run_xgb"):     jobs.append(("XGBoost",            "—"))
    if c.get("run_mlp"):     jobs.append(("MLP",                "—"))
    if c.get("run_cnn"):     jobs.append(("CNN",                "—"))
    if c.get("run_segment"):
        for mx in c.get("segment_max_x", []):
            jobs.append(("SegmentHandler", f"max_x={mx}"))
    if c.get("run_system"):
        for mx in c.get("system_max_x", []):
            for agg in ("bma", "simple_mean", "relevance_weighted"):
                jobs.append(("SystemHandler", f"max_x={mx}  agg={agg}"))
    return jobs


def run_compare(settings: Settings, logger) -> None:
    import pandas as pd
    from rich.rule import Rule
    from rich.table import Table
    from rich.prompt import Prompt
    from rich import box
    import comparisons.ComparisonManager as CompMgr

    console = logger.console
    d = settings.dataset
    c = settings.comparison

    console.print(Rule("[bold cyan]DragonChild v13 — Compare Mode[/bold cyan]"))

    # ── Show every planned job ────────────────────────────────────────────
    jobs = _compare_job_list(c)
    job_table = Table(title=f"Planned Jobs  ({len(jobs)} total)",
                      box=box.ROUNDED, border_style="cyan", show_lines=True)
    job_table.add_column("#",       style="dim white",   justify="right")
    job_table.add_column("Model",   style="bold white")
    job_table.add_column("Config",  style="bright_cyan")
    for i, (name, cfg_str) in enumerate(jobs, 1):
        job_table.add_row(str(i), name, cfg_str)
    console.print(job_table)
    console.print(f"[dim]Epochs per job: {c.get('epoch_count', 20)}  |  "
                  f"Dataset: {d['csv_path']}  |  "
                  f"Output: {c.get('output_csv', 'comparison_results.csv')}[/dim]\n")

    # ── Warn if output CSV already exists ─────────────────────────────────
    csv_path = c.get("output_csv", "comparison_results.csv")
    if os.path.exists(csv_path):
        console.print(f"[yellow]Note:[/yellow] [bold]{csv_path}[/bold] already exists — new results will be appended.")
        if not _confirm(console, "Continue and append to existing CSV?", default=True):
            console.print("[dim]Aborted.[/dim]")
            return

    # ── Time warning + hard confirm ───────────────────────────────────────
    nexus_jobs = sum(1 for n, _ in jobs if "Handler" in n)
    if nexus_jobs > 0:
        console.print(f"\n[bold red]Time warning:[/bold red] {nexus_jobs} Nexus job(s) × "
                      f"{c.get('epoch_count', 20)} epochs each. This can take several hours.")

    answer = Prompt.ask(
        "\nType [bold green]yes[/bold green] to start, or anything else to cancel",
        default="no",
        console=console,
    )
    if answer.strip().lower() != "yes":
        console.print("[dim]Aborted.[/dim]")
        return

    # ── Run ───────────────────────────────────────────────────────────────
    console.print(Rule("[bold green]Running Comparisons[/bold green]"))
    dataset = pd.read_csv(d["csv_path"])
    manager = CompMgr.ComparisonManager(logger=logger)
    manager.run_all(
        dataset        = dataset,
        target         = d["target_column"],
        epoch_count    = c.get("epoch_count", 20),
        segment_max_x  = c.get("segment_max_x", [10, 15, 20, 25]),
        system_max_x   = c.get("system_max_x", [10, 15, 20, 25]),
        run_segment    = c.get("run_segment", True),
        run_cnn        = c.get("run_cnn", False),
        run_linear     = c.get("run_linear", True),
        run_knn        = c.get("run_knn", True),
        run_rf         = c.get("run_rf", True),
        run_xgb        = c.get("run_xgb", True),
        run_mlp        = c.get("run_mlp", True),
        run_system     = c.get("run_system", True),
        output_csv     = csv_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_system(system: SystemHandler, test_ds, target_col: str, m: dict, logger) -> None:
    from rich.table import Table
    from rich import box

    preds, actuals = [], []
    with logger.make_progress() as prog:
        task = prog.add_task(f"Evaluating {len(test_ds)} test rows…", total=len(test_ds))
        for _, row in test_ds.iterrows():
            row_dict = row.to_dict()
            a = row_dict.get(target_col)
            if a is None:
                prog.update(task, advance=1)
                continue
            p = system.runInfer(row_dict, loud=False,
                                aggregation_mode=m["aggregation_mode"],
                                selection_percentage=m["selection_percentage"])
            if p is not None:
                preds.append(float(p))
                actuals.append(float(a))
            prog.update(task, advance=1)

    n = len(preds)
    if n == 0:
        logger.log("No test predictions produced.", 2, True)
        return

    mae  = sum(abs(p - a) for p, a in zip(preds, actuals)) / n
    rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / n)
    mean_a = sum(actuals) / n
    ss_tot = sum((a - mean_a) ** 2 for a in actuals)
    ss_res = sum((p - a) ** 2 for p, a in zip(preds, actuals))
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mape = sum(abs(p - a) / abs(a) * 100.0
               for p, a in zip(preds, actuals) if a != 0) / n

    def _within(pct):
        return sum(1 for p, a in zip(preds, actuals)
                   if a != 0 and abs(p - a) / abs(a) <= pct) / n * 100.0

    acc5, acc10, acc20 = _within(0.05), _within(0.10), _within(0.20)
    sorted_a = sorted(actuals)
    med = sorted_a[n // 2]
    ap  = [1 if a >= med else 0 for a in actuals]
    pp  = [1 if p >= med else 0 for p in preds]
    tp  = sum(1 for a, p in zip(ap, pp) if a == 1 and p == 1)
    fp  = sum(1 for a, p in zip(ap, pp) if a == 0 and p == 1)
    fn  = sum(1 for a, p in zip(ap, pp) if a == 1 and p == 0)
    tn  = sum(1 for a, p in zip(ap, pp) if a == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = (2 * prec * rec / (prec + rec)
            if not (math.isnan(prec) or math.isnan(rec)) and (prec + rec) > 0
            else float("nan"))
    dir_acc = (tp + tn) / n * 100.0

    t = Table(title=f"System Test Evaluation  (n={n})",
              box=box.ROUNDED, border_style="green", show_lines=True)
    t.add_column("Metric", style="bold white")
    t.add_column("Value",  style="bright_green", justify="right")
    t.add_row("MAE",                _fv(mae))
    t.add_row("RMSE",               _fv(rmse))
    t.add_row("R²",                 _fv(r2))
    t.add_row("MAPE",               f"{mape:.2f}%" if not math.isnan(mape) else "—")
    t.add_section()
    t.add_row("Within  5%",         f"{acc5:.1f}%")
    t.add_row("Within 10%",         f"{acc10:.1f}%")
    t.add_row("Within 20%",         f"{acc20:.1f}%")
    t.add_section()
    t.add_row("Direction accuracy", f"{dir_acc:.1f}%")
    t.add_row("Precision (median)", _fv(prec))
    t.add_row("Recall    (median)", _fv(rec))
    t.add_row("F1        (median)", _fv(f1))
    t.add_section()
    t.add_row("TP / FP / FN / TN",  f"{tp} / {fp} / {fn} / {tn}")
    logger.console.print(t)


def _segment_summary(system: SystemHandler, logger) -> None:
    from rich.table import Table
    from rich import box
    from rich.rule import Rule

    seg_metrics = [s.best_epoch_metrics for s in system.segments
                   if s.best_epoch_metrics is not None]
    if not seg_metrics:
        return

    logger.console.print(Rule("[bold magenta]Per-Segment Best-Epoch Metrics[/bold magenta]"))
    t = Table(box=box.ROUNDED, border_style="magenta", show_lines=True)
    t.add_column("Seg",        style="bold white",     justify="right")
    t.add_column("Best Epoch", style="bright_cyan",    justify="right")
    t.add_column("Train Rows", style="bright_cyan",    justify="right")
    t.add_column("MAE",        style="bright_magenta", justify="right")
    t.add_column("R²",         style="bright_magenta", justify="right")
    t.add_column("F1",         style="bright_green",   justify="right")
    t.add_column("Acc@10%",    style="bright_green",   justify="right")
    for m in seg_metrics:
        t.add_row(
            str(m["segment_id"]),
            str(m["best_epoch"]),
            str(m["n_train"]),
            f"{m['mae']:.4f}",
            _fv(m["r2"]),
            _fv(m["f1"]),
            f"{m['acc_10']:.1f}%",
        )
    logger.console.print(t)


def _config_table(system: SystemHandler, settings: Settings, dataset, logger) -> None:
    from rich.table import Table
    from rich import box
    from rich.rule import Rule

    m = settings.model
    t = settings.training

    logger.console.print(Rule("[bold white]Run Summary[/bold white]"))
    cfg = Table(box=box.ROUNDED, border_style="cyan", show_lines=True)
    cfg.add_column("Parameter", style="bold white")
    cfg.add_column("Value",     style="bright_cyan", justify="right")
    cfg.add_row("Target",        settings.dataset["target_column"])
    cfg.add_row("Dataset rows",  str(len(dataset)))
    cfg.add_row("Max X",         str(m["max_x"]))
    cfg.add_row("Dimensions",    str(m["dimensions"]))
    cfg.add_row("Conn %",        str(m["connection_percentage"]))
    cfg.add_row("Density",       str(m["density"]))
    cfg.add_row("Segments",      str(len(system.segments)))
    cfg.add_row("Epochs",        str(t["epoch_count"]))
    cfg.add_row("Judge iters",   str(t["judge_iterations"]))
    cfg.add_row("Train mode",    m["training_mode"])
    cfg.add_row("Agg mode",      m["aggregation_mode"])
    clusters = system.JudgeNode.segment_weights.get("clusters", [])
    cfg.add_row("Clusters used", str(len(clusters)))
    logger.console.print(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from rich.panel import Panel

    settings_path = "settings.json"
    mode_override = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--mode" and i + 1 < len(args):
            mode_override = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            settings_path = args[i]
            i += 1
        else:
            i += 1

    settings = Settings(settings_path)
    logger   = _make_logger(settings)
    console  = logger.console

    console.print(Panel(
        f"[bold]Settings:[/bold] {os.path.abspath(settings_path)}\n"
        f"[bold]Configured mode:[/bold] {settings.run_mode}",
        title="[bold cyan]DragonChild v13[/bold cyan]",
        border_style="cyan",
        expand=False,
    ))

    dispatch = {
        "train":   run_train,
        "infer":   run_infer,
        "compare": run_compare,
    }

    # Let user pick mode if not forced via --mode
    if mode_override:
        settings.override("run_mode", mode_override)
    else:
        mode = _prompt_str(
            console,
            "Select mode",
            default=settings.run_mode,
            choices=["train", "infer", "compare"],
        )
        settings.override("run_mode", mode)

    mode = settings.run_mode
    if mode not in dispatch:
        console.print(f"[red]Unknown mode '{mode}'.[/red]")
        sys.exit(1)

    dispatch[mode](settings, logger)


if __name__ == "__main__":
    main()
