import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update(
    {
        # "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "font.size": 7,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.figsize": (3.5, 1.8),
        "pdf.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


def plot(
    parameters: list[str],
    results: list[list[tuple[float, float]]],
    baselines: list[float],
):
    random_type = [res[0] for res in results]
    swap_type = [res[1] for res in results]
    cycle_type = [res[2] for res in results]
    derange_type = [res[3] for res in results]

    x = range(len(parameters))
    # colors = ["#ffb3ba","#ffdfba","#baffc9","#bae1ff"]
    # colors = ["#ff71ce", "#01cdfe", "#05ffa1", "#b967ff"]
    # colors = ["#011f4b", "#03396c", "#005b96", "#6497b1"]

    width = 0.15
    colors = ["#66545e", "#a39193", "#aa6f73", "#eea990"]

    plt.bar(
        [p - 1.5 * width for p in x],
        [rec[0] for rec in random_type],
        width,
        label="Uniform",
        color=colors[0],
        yerr=[1.96 * rec[1] for rec in random_type],
        error_kw={
            "elinewidth": 0.5,
        },
    )
    plt.bar(
        [p - 0.5 * width for p in x],
        [rec[0] for rec in swap_type],
        width,
        label="Swap",
        color=colors[1],
        yerr=[1.96 * rec[1] for rec in swap_type],
        error_kw={
            "elinewidth": 0.5,
        },
    )
    plt.bar(
        [p + 0.5 * width for p in x],
        [rec[0] for rec in cycle_type],
        width,
        label="Cycle",
        color=colors[2],
        yerr=[1.96 * rec[1] for rec in cycle_type],
        error_kw={
            "elinewidth": 0.5,
        },
    )
    plt.bar(
        [p + 1.5 * width for p in x],
        [rec[0] for rec in derange_type],
        width,
        label="Derangement",
        color=colors[3],
        yerr=[1.96 * rec[1] for rec in derange_type],
        error_kw={
            "elinewidth": 0.5,
        },
    )

    plt.axhline(
        y=baselines[0],
        xmin=0.3 * width,
        xmax=1.3 * width,
        color="grey",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.7,
    )
    plt.axhline(
        y=baselines[1],
        xmin=1.98 * width,
        xmax=4.68 * width,
        color="grey",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.7,
    )
    plt.axhline(
        y=baselines[2],
        xmin=5.36 * width,
        xmax=6.36 * width,
        color="grey",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.7,
    )

    # plt.xlabel("Parameter setting")
    plt.ylabel(r"Estimated probability ($\times 10^{-7}$)")

    plt.xticks(x, parameters)
    plt.yscale("log")

    plt.plot(
        [],
        [],
        color="grey",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.7,
        label="Baseline",
    )
    plt.legend()

    plt.savefig("probability.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def main():
    # parameters = [
    #     "$h=16$",
    #     "$h_k=8, g=4$, groups",
    #     "$h_k=8, g=4$, one group",
    #     "$h_k=8, g=4$, both",
    # ]
    # results = [
    #     [(7.8, 0.832), (0.0, 0.0), (4.3, 0.6557), (6.2, 0.7874)],
    #     [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    #     [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    #     [(26.5, 1.628), (0.0, 0.0), (18.5, 1.36), (19.9, 1.411)],
    # ]
    # baselines = [21.8, 262.8]
    # plot(parameters, results, baselines)
    #
    # parameters = [
    #     "$h=32$",
    #     "$h_k=8, g=3$, groups",
    #     "$h_k=8, g=3$, one group",
    #     "$h_k=8, g=3$, both",
    # ]
    # results = [
    #     [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    #     [(1.27, 0.1127), (0.0, 0.0), (0.0, 0.0), (0.82, 0.09055)],
    #     [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    #     [(5.08, 2.254), (0.0, 0.0), (98.40, 3.137), (10.86, 0.3295)],
    # ]
    # baselines = [0.0, 67.42]
    # plot(parameters, results, baselines)

    parameters = [
        "$h=16$",
        "$8\\times 3$, groups",
        "$8\\times 3$, both",
        "$8\\times 4$, both",
    ]
    results = [
        [(7.8, 0.832), (0.0, 0.0), (4.3, 0.6557), (6.2, 0.7874)],
        [(12.7, 1.127), (0.0, 0.0), (0.0, 0.0), (8.2, 0.9055)],
        [(50.8, 2.254), (0.0, 0.0), (984.0, 31.37), (108.6, 3.295)],
        [(26.5, 1.628), (0.0, 0.0), (18.5, 1.36), (19.9, 1.411)],
    ]
    baselines = [21.8, 674.2, 262.8]
    plot(parameters, results, baselines)


if __name__ == "__main__":
    main()
