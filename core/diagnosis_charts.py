"""Diagnosis Charts — matplotlib-based visualization for model diagnosis reports.

Charts are built with the object-oriented Figure / FigureCanvasAgg API rather
than the pyplot global state machine. WHY: the diagnose flow renders five charts
and we want to generate them concurrently (see generate_all_charts). pyplot's
plt.* API keeps a process-global figure registry and mutates global rcParams,
neither of which is thread-safe — concurrent builders would corrupt each other's
state. A standalone Figure + Agg canvas owns all of its state, so each chart can
be rendered on its own thread safely. Per-figure styling replaces global rcParams
for the same reason.
"""
import base64
import io
from concurrent.futures import ThreadPoolExecutor

# Dark theme palette shared by every chart. Applied per-Figure (not via global
# rcParams) so concurrent builders don't race on shared matplotlib state.
_FIG_FACE = "#1e1e2e"
_AXES_FACE = "#2a2a3e"
_AXES_EDGE = "#444"
_TEXT = "#ccc"
_TICK = "#999"
_GRID = "#333"


def _blank_figure(figsize):
    """Create a standalone Figure backed by an Agg canvas, no axes attached.

    The figure carries an explicit FigureCanvasAgg so it is fully self-contained
    and safe to build/render off the main thread (no pyplot global registry).
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=figsize, facecolor=_FIG_FACE)
    FigureCanvasAgg(fig)
    return fig


def _new_figure(figsize, *, polar: bool = False):
    """Blank figure + one styled subplot. Returns (figure, axes)."""
    fig = _blank_figure(figsize)
    subplot_kw = dict(polar=True) if polar else {}
    ax = fig.add_subplot(111, **subplot_kw)
    _style_axes(ax)
    return fig, ax


def _style_axes(ax) -> None:
    """Apply the dark theme to one Axes via OO setters (no global rcParams)."""
    ax.set_facecolor(_AXES_FACE)
    for spine in ax.spines.values():
        spine.set_edgecolor(_AXES_EDGE)
    ax.title.set_color("#eee")
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.tick_params(colors=_TICK)
    ax.grid(color=_GRID)


def _colormap(name: str):
    """Resolve a colormap without touching pyplot.

    matplotlib.colormaps is the modern registry (>=3.5); fall back to the
    matplotlib.cm module for older installs.
    """
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        return getattr(cm, name)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    # savefig drives the figure's own Agg canvas — no pyplot global registry,
    # so no plt.close() bookkeeping is required.
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=_FIG_FACE)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _empty_chart(msg="No data available") -> str:
    try:
        fig, ax = _new_figure((6, 2))
        ax.text(0.5, 0.5, msg, ha="center", va="center", color="#888", fontsize=12)
        ax.axis("off")
        return _fig_to_base64(fig)
    except ImportError:
        return ""


def generate_weight_distribution_chart(weight_analysis: list) -> str:
    if not weight_analysis:
        return _empty_chart("No weight data")
    try:
        import numpy as np
        n = min(len(weight_analysis), 8)
        fig = _blank_figure((3 * n, 3))
        axes = fig.subplots(1, n)
        if n == 1:
            axes = [axes]
        for i, w in enumerate(weight_analysis[:n]):
            ax = axes[i]
            _style_axes(ax)
            # Simulate distribution from stats
            data = np.random.normal(w["mean"], max(w["std"], 1e-6), 1000)
            ax.hist(data, bins=30, color="#7c3aed", alpha=0.8, edgecolor="none")
            name = w["name"][-20:] if len(w["name"]) > 20 else w["name"]
            ax.set_title(name, fontsize=8, color="#aaa")
            ax.tick_params(labelsize=6)
        fig.suptitle("Weight Distribution per Layer", fontsize=11, color="#eee")
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return _empty_chart("Chart generation failed")


def generate_op_time_chart(op_summary: list) -> str:
    if not op_summary:
        return _empty_chart("No op data")
    try:
        fig, ax = _new_figure((6, 4))
        labels = [o["op_type"] for o in op_summary[:10]]
        counts = [o["count"] for o in op_summary[:10]]
        colors = _colormap("Set3")([i / max(len(labels), 1) for i in range(len(labels))])
        ax.barh(labels, counts, color=colors, edgecolor="none")
        ax.set_xlabel("Count")
        ax.set_title("Op Type Distribution", color="#eee")
        ax.invert_yaxis()
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return _empty_chart("Chart generation failed")


def generate_quantization_heatmap(quant_analysis: dict) -> str:
    nodes = quant_analysis.get("sensitive_nodes", [])
    if not nodes:
        return _empty_chart("No quantization sensitivity data")
    try:
        fig, ax = _new_figure((8, max(3, len(nodes[:15]) * 0.4)))
        names = [n["name"][-25:] for n in nodes[:15]]
        scores = [n["sensitivity"] for n in nodes[:15]]
        max_s = max(scores) if scores else 1
        norm_scores = [s / max_s for s in scores]
        colors = _colormap("RdYlGn_r")(norm_scores)
        ax.barh(names, scores, color=colors, edgecolor="none")
        ax.set_xlabel("Sensitivity Score")
        ax.set_title("Quantization Sensitivity (higher = more sensitive)", color="#eee")
        ax.invert_yaxis()
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return _empty_chart("Chart generation failed")


def generate_channel_importance_chart(pruning_analysis: dict) -> str:
    convs = pruning_analysis.get("conv_layers", [])
    if not convs:
        return _empty_chart("No Conv layer data")
    try:
        fig, ax = _new_figure((8, max(3, len(convs[:12]) * 0.4)))
        names = [c["name"][-25:] for c in convs[:12]]
        total = [c["out_channels"] for c in convs[:12]]
        low = [c["low_importance_channels"] for c in convs[:12]]
        high = [t - l for t, l in zip(total, low)]
        ax.barh(names, high, color="#22c55e", label="Important", edgecolor="none")
        ax.barh(names, low, left=high, color="#ef4444", label="Low importance", edgecolor="none")
        ax.set_xlabel("Channels")
        ax.set_title("Channel Importance per Conv Layer", color="#eee")
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return _empty_chart("Chart generation failed")


def generate_model_overview_chart(diagnosis: dict) -> str:
    """Radar-style overview of model characteristics."""
    try:
        import numpy as np
        summary = diagnosis["summary"]
        quant = diagnosis["quantization_analysis"]
        pruning = diagnosis["pruning_analysis"]
        graph = diagnosis["graph_analysis"]

        categories = ["Quantizable", "Sparsity", "Graph\nEfficiency", "Size\nEfficiency", "Health"]
        values = [
            quant["quantizable_ratio"],
            1 - pruning["overall_sparsity"],
            1 - graph["memory_bound_ratio"],
            min(1.0, 10 / max(summary["file_size_mb"], 0.1)),
            summary["health_score"] / 100,
        ]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = _new_figure((5, 5), polar=True)
        ax.fill(angles, values, color="#7c3aed", alpha=0.25)
        ax.plot(angles, values, color="#7c3aed", linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, color="#ccc")
        ax.set_ylim(0, 1)
        ax.set_title("Model Health Overview", pad=20, color="#eee", fontsize=12)
        ax.set_facecolor(_AXES_FACE)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return _empty_chart("Chart generation failed")


# Chart key -> (builder, key into the diagnosis dict). "model_overview" gets the
# whole diagnosis dict, so its source key is None.
_CHART_BUILDERS = (
    ("weight_distribution", generate_weight_distribution_chart, "weight_analysis"),
    ("op_distribution", generate_op_time_chart, "op_summary"),
    ("quantization_sensitivity", generate_quantization_heatmap, "quantization_analysis"),
    ("channel_importance", generate_channel_importance_chart, "pruning_analysis"),
    ("model_overview", generate_model_overview_chart, None),
)


def generate_all_charts(diagnosis: dict, *, max_workers: int = 5) -> dict:
    """Render all diagnosis charts concurrently and return them keyed by chart.

    Returns the same dict shape the diagnose route built serially:
        {"weight_distribution": <png>, "op_distribution": <png>,
         "quantization_sensitivity": <png>, "channel_importance": <png>,
         "model_overview": <png>}

    WHY concurrent: the five matplotlib charts cost ~3-5s end-to-end when built
    one after another. Each builder owns a private Figure + Agg canvas (no pyplot
    global state), so they render safely in parallel on a small thread pool,
    cutting the diagnose flow's chart latency to roughly the slowest single chart.

    If matplotlib is unavailable, each builder degrades to "" / a graceful empty
    chart exactly as in the serial path, so the dict shape is preserved.
    """
    def _build(spec: tuple) -> tuple:
        key, builder, source_key = spec
        arg = diagnosis if source_key is None else diagnosis.get(source_key)
        try:
            return key, builder(arg)
        except Exception as build_error:
            # A single failing chart must not sink the whole batch — surface an
            # empty chart for it and keep the others.
            return key, _empty_chart(f"Chart generation failed: {build_error}")

    charts: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers,
                            thread_name_prefix="ssook-chart") as pool:
        for key, image in pool.map(_build, _CHART_BUILDERS):
            charts[key] = image
    return charts
