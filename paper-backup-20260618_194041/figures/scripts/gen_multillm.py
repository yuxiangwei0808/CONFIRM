"""Generate multi-LLM agentic benchmark figure."""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, load_json, save_pdf, setup_matplotlib


MODEL_LABELS = {
    "openai:gpt-5-mini": "GPT-5\nmini",
    "openai:gpt-4o": "GPT-4o",
    "anthropic:claude-sonnet-4-6": "Claude\nSonnet 4.6",
    "anthropic:claude-haiku-4-5-20251001": "Claude\nHaiku 4.5",
    "openrouter:deepseek/deepseek-chat": "DeepSeek\nChat",
    "openrouter:qwen/qwen-2.5-72b-instruct": "Qwen\n2.5-72B",
}

METRICS = [
    ("draft_success_rate", "Draft success", COLORS["blue"]),
    ("estimand_match_rate", "Estimand match", COLORS["orange"]),
    ("gate_success_rate", "Gate success", COLORS["green"]),
]


def generate() -> list:
    payload = load_json("multillm")
    per_model = payload["per_model"]
    model_keys = list(per_model.keys())
    x = np.arange(len(model_keys))
    width = 0.24

    plt = setup_matplotlib()
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.55))
    for idx, (metric_key, label, color) in enumerate(METRICS):
        values = [float(per_model[model][metric_key]) for model in model_keys]
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            label=label,
            zorder=3,
        )

    ax.text(
        0.985,
        0.09,
        (
            f"Verdict agreement: {payload['cross_model_verdict_agreement_count']}/"
            f"{payload['cross_model_verdict_agreement_denominator']}\n"
            f"Anti-hallucination catches: {payload['anti_hallucination_catch_count']}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#777777", "linewidth": 0.4, "pad": 2.0},
    )
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(model, model) for model in model_keys])
    ax.grid(axis="y", color="#cccccc", alpha=0.55, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.0, 1.10))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.25, top=0.84)
    out_path = save_pdf(fig, "fig_multillm.pdf")
    plt.close(fig)
    return [out_path]


def main() -> int:
    for path in generate():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
