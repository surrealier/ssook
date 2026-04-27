"""Diagnosis Charts — matplotlib-based visualization for model diagnosis reports."""
import base64
import io


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#1e1e2e")
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _empty_chart(msg="No data available") -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, msg, ha="center", va="center", color="#888", fontsize=12)
        ax.set_facecolor("#1e1e2e")
        ax.axis("off")
        return _fig_to_base64(fig)
    except ImportError:
        return ""


def _setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#1e1e2e",
        "axes.facecolor": "#2a2a3e",
        "axes.edgecolor": "#444",
        "text.color": "#ccc",
        "axes.labelcolor": "#ccc",
        "xtick.color": "#999",
        "ytick.color": "#999",
        "grid.color": "#333",
    })
    return plt


def generate_weight_distribution_chart(weight_analysis: list) -> str:
    if not weight_analysis:
        return _empty_chart("No weight data")
    try:
        plt = _setup_style()
        n = min(len(weight_analysis), 8)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]
        for i, w in enumerate(weight_analysis[:n]):
            ax = axes[i]
            # Simulate distribution from stats
            import numpy as np
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
        plt = _setup_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [o["op_type"] for o in op_summary[:10]]
        counts = [o["count"] for o in op_summary[:10]]
        colors = plt.cm.Set3([i / max(len(labels), 1) for i in range(len(labels))])
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
        plt = _setup_style()
        import numpy as np
        fig, ax = plt.subplots(figsize=(8, max(3, len(nodes[:15]) * 0.4)))
        names = [n["name"][-25:] for n in nodes[:15]]
        scores = [n["sensitivity"] for n in nodes[:15]]
        max_s = max(scores) if scores else 1
        norm_scores = [s / max_s for s in scores]
        colors = plt.cm.RdYlGn_r(norm_scores)
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
        plt = _setup_style()
        fig, ax = plt.subplots(figsize=(8, max(3, len(convs[:12]) * 0.4)))
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
        plt = _setup_style()
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

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color="#7c3aed", alpha=0.25)
        ax.plot(angles, values, color="#7c3aed", linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, color="#ccc")
        ax.set_ylim(0, 1)
        ax.set_title("Model Health Overview", pad=20, color="#eee", fontsize=12)
        ax.set_facecolor("#2a2a3e")
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        return _empty_chart("Chart generation failed")
