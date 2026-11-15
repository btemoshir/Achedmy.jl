# Shared setup for the paper figures.
#
# The figures are redrawn from the stored artifacts in data/ (read-only) with
# every piece of text scaled so that it prints at the size of the journal's main
# text font. Nothing here writes to data/.
#
# Font scaling
# ------------
# A figure of width `W` inches included at width `W_print` inches is shrunk by
# s = W / W_print, so a label set at `BODY_PT * s` in the figure comes out at
# `BODY_PT` on the page. Every figure keeps its original figsize (hence its
# aspect ratio); only the point sizes and line widths change.
#
# `printed_width_in` is the knob: COL_IN for a single-column figure, PAGE_IN for
# one spanning both columns. The wide multi-panel figures (figsize width >= 10in,
# 4-9 panels side by side) are set to PAGE_IN; everything else to COL_IN.

ENV["MPLBACKEND"] = "Agg"   # no display needed; must be set before PyPlot loads

using PyCall
using PyPlot
using Serialization
using LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA = joinpath(ROOT, "data")
const OUT = joinpath(ROOT, "plots", "paper")

# Journal geometry (REVTeX two-column, e.g. Phys. Rev. *).
const BODY_PT = 10.0   # main text font size
const COL_IN = 3.4     # one column
const PAGE_IN = 7.0    # both columns

# The .jls artifacts were written by a session in which Achedmy was a top-level
# module, so they name it as Base.PkgId(nothing, "Achedmy"). Loading the source
# here and registering it under that key is what lets deserialize find the types.
if !isdefined(Main, :Achedmy)
    include(joinpath(ROOT, "src", "Achedmy.jl"))
    Base.register_root_module(Achedmy)
end

# master.jls / emre.jls hold pickled objects of the python modules in
# extras/other_dynamics, so that directory has to be importable before they are
# deserialized.
py"""
import sys
sys.path.append($(joinpath(ROOT, "extras", "other_dynamics")))
"""

rcdict() = PyDict(PyPlot.matplotlib."rcParams")

"""
    paper_rc!(fig_width_in; printed_width_in=COL_IN, body_pt=BODY_PT) -> s

Set the rcParams for a figure `fig_width_in` inches wide that will be printed
`printed_width_in` inches wide, and return the scale factor `s`. Fonts, line
widths, ticks and spines are all multiplied by `s`, so use `s` for any size
passed explicitly to a plotting call (e.g. `linewidth=1.2s`).
"""
const CURRENT_GEOM = Ref((COL_IN, COL_IN))   # (figure width, printed width) of the figure being drawn
const PAD_IN = 0.02                          # padding added by the tight crop on save

function paper_rc!(fig_width_in; printed_width_in = COL_IN, body_pt = BODY_PT)
    s = fig_width_in / printed_width_in
    CURRENT_GEOM[] = (fig_width_in, printed_width_in)
    merge!(rcdict(), Dict(
        "text.usetex" => false,
        "font.family" => "sans-serif",
        # math set in the body font: the STIX glyphs are metrically smaller than
        # the sans-serif text, so at the same point size the $...$ labels came
        # out visibly smaller than the surrounding text
        "mathtext.fontset" => "dejavusans",
        # text: everything at the main-text size
        "font.size" => body_pt * s,
        "axes.titlesize" => body_pt * s,
        "axes.labelsize" => body_pt * s,
        "xtick.labelsize" => body_pt * s,
        "ytick.labelsize" => body_pt * s,
        "legend.fontsize" => body_pt * s,
        "figure.titlesize" => body_pt * s,
        # lines and frames: printed widths of ~1.2pt / 0.8pt
        "lines.linewidth" => 1.2 * s,
        "lines.markersize" => 3.0 * s,
        "axes.linewidth" => 0.8 * s,
        "xtick.major.width" => 0.8 * s,
        "ytick.major.width" => 0.8 * s,
        "xtick.minor.width" => 0.6 * s,
        "ytick.minor.width" => 0.6 * s,
        "xtick.major.size" => 3.5 * s,
        "ytick.major.size" => 3.5 * s,
        "xtick.minor.size" => 2.0 * s,
        "ytick.minor.size" => 2.0 * s,
        "xtick.direction" => "in",
        "ytick.direction" => "in",
        "legend.frameon" => false,
        "figure.autolayout" => false,
        "savefig.dpi" => 600,
    ))
    return s
end

"""
    match_printed_size!(fig)

Correct the text sizes for the tight crop. The figures are saved with
`bbox_inches="tight"`, which trims empty margins but also keeps whatever sticks
out of the nominal figure box (axis labels of figures whose axes are placed with
`add_axes`), so the saved width is not the `fig_width_in` that `paper_rc!` scaled
the fonts to, and the text ends up a few percent off the main-text size. Here the
actual saved width is measured and every piece of text rescaled by the ratio, so
all figures really do print at `BODY_PT`. (Line widths are left alone: the same
few-percent error on a 1.2pt line is invisible.)
"""
function match_printed_size!(fig)
    fig_width_in = CURRENT_GEOM[][1]
    for _ in 1:3                           # rescaling moves the bbox a little; converges at once
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD_IN
        f = w / fig_width_in
        abs(f - 1) < 0.005 && break
        for t in fig.findobj(PyPlot.matplotlib.text.Text)
            t.set_fontsize(t.get_fontsize() * f)
        end
    end
end

"""
    align_ylabels!(fig)

Line up the y labels of the panels that sit in the same column. Matplotlib
places each y label just left of *its own* tick labels, so rows whose tick
labels have different widths ("0.04" vs "0.75") end up with their y labels at
different horizontal positions. Here the leftmost tick-label edge of a column is
found and every y label of that column is put there.

Panels are grouped by the left edge of their axes box, so the narrow/wide panels
of the lag figures (different widths, hence different columns) are left alone.
"""
function align_ylabels!(fig)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    columns = Dict{Int,Vector{Any}}()
    for ax in fig.axes
        ax.get_ylabel() == "" && continue
        push!(get!(columns, round(Int, 1000 * ax.get_position().x0), []), ax)
    end
    for (_, axs) in columns
        length(axs) < 2 && continue
        # leftmost edge of the tick labels of the column, in display coordinates
        x0 = minimum(axs) do ax
            labels = [t for t in ax.get_yticklabels() if t.get_text() != ""]
            isempty(labels) ? ax.get_window_extent(r).x0 :
                minimum(t.get_window_extent(r).x0 for t in labels)
        end
        for ax in axs
            box = ax.get_window_extent(r)
            pad = 2 * fig.dpi / 72                  # 2pt of air between label and ticks
            ax.yaxis.set_label_coords((x0 - pad - box.x0) / box.width, 0.5)
        end
    end
end

"""
    savefig_both(system, name)

Write the current figure to plots/paper/<system>/<name>.{png,svg} and close it.
"""
function savefig_both(system, name)
    dir = joinpath(OUT, system)
    mkpath(dir)
    match_printed_size!(PyPlot.gcf())
    align_ylabels!(PyPlot.gcf())
    for ext in ("png", "svg")
        PyPlot.savefig(joinpath(dir, "$name.$ext"), bbox_inches = "tight", pad_inches = PAD_IN)
    end
    PyPlot.close("all")
    println("  wrote $system/$name.{png,svg}")
end

"Indices of the grid points closest to each of `times`."
find_closed_timegrid_indices(timegrid, times) =
    [findmin(abs.(timegrid .- X))[2] for X in times]

"Deserialize `path` under data/ (read-only)."
read_artifact(path...) = open(deserialize, joinpath(DATA, path...))

"Default colour cycle."
cycle_colors() = PyPlot.plt.rcParams["axes.prop_cycle"].by_key()["color"]

"""
    thin_xticks!(ax, n)

Keep at most `n` x tick labels. Needed on the narrow panels of the multi-panel
figures: at main-text size the default number of ticks runs into each other.
"""
function thin_xticks!(ax, n; drop_last = false)
    ax.xaxis.set_major_locator(
        PyPlot.matplotlib.ticker.MaxNLocator(nbins = n, steps = [1, 2, 5, 10]))
    if drop_last
        # panels that sit flush against the next one: the rightmost label would
        # otherwise run into the neighbour's leftmost one
        lo, hi = ax.get_xlim()
        ticks = [t for t in ax.get_xticks() if lo <= t <= hi]
        length(ticks) > 1 && ax.set_xticks(ticks[1:end-1])
    end
end

"A plain line handle for hand-built legends."
line_handle(; kwargs...) = PyPlot.matplotlib.lines.Line2D([0], [0]; kwargs...)

"""
    pick_diverse_traces(traces, n; keep=t->true)

Pick `n` Gillespie trajectories that start from distinct (Poisson-sampled)
initial states and that actually move, so the plotted traces show visibly
different realizations rather than n copies of the same flat line.

One trajectory is taken per distinct initial state, the liveliest of that group
(most total variation); groups are then ordered by liveliness and the n
liveliest distinct starts are kept.
"""
function pick_diverse_traces(traces, n; keep = t -> true)
    candidates = [t for t in traces if keep(t)]
    isempty(candidates) && error("no trajectories left after filtering")
    liveliness(t) = sum(abs.(diff(t.y, dims = 2)))   # total variation over all species

    groups = Dict{Vector{Float64},Any}()
    for t in candidates
        x0 = collect(Float64, t.y[:, 1])
        if !haskey(groups, x0) || liveliness(t) > liveliness(groups[x0])
            groups[x0] = t
        end
    end
    reps = collect(values(groups))
    sort!(reps, by = liveliness, rev = true)
    return reps[1:min(n, length(reps))]
end
