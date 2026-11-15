# Paper figures for the gene regulation system.
# Redraws the figures of examples/gene_regulation.ipynb from the stored
# artifacts, with text at main-text size. Run from anywhere:
#   julia --project=. scripts/paper_plots/gene.jl

include(joinpath(@__DIR__, "common.jl"))

const SYS = "gene"
const ALPHA = exp10.(range(-2, stop = 1, length = 10))   # k_u values
const NSPEC = 3                                          # G, G*, P

println("loading gene artifacts ...")
SOL = read_artifact("gene", "sol_gSBR.jls")
VAR = read_artifact("gene", "var_gSBR.jls")
SOL_mak = read_artifact("gene", "sol_mak.jls")
VAR_mak = read_artifact("gene", "var_mak.jls")
SOL_crossMCA = read_artifact("gene", "sol_crossMCA.jls")
VAR_crossMCA = read_artifact("gene", "var_crossMCA.jls")
SOL_singleMCA = read_artifact("gene", "sol_singleMCA.jls")
VAR_singleMCA = read_artifact("gene", "var_singleMCA.jls")
SOL_singleSBR = read_artifact("gene", "sol_singleSBR.jls")
VAR_singleSBR = read_artifact("gene", "var_singleSBR.jls")
SOL_single_gSBR = read_artifact("gene", "sol_single_gSBR.jls")
VAR_single_gSBR = read_artifact("gene", "var_single_gSBR.jls")
Master = read_artifact("gene", "master.jls")
EMRE = read_artifact("gene", "emre.jls")

# ---------------------------------------------------------------- time steps

"""
    power_law_fit(t, dt, lo, hi) -> (alpha, A)

Least-squares fit of log10(dt) against log10(t) over `lo <= t <= hi`, i.e.
dt = A * t^alpha.
"""
function power_law_fit(t, dt, lo, hi)
    m = (t .>= lo) .& (t .<= hi)
    X = log10.(t[m])
    M = hcat(ones(length(X)), X)
    b, alpha = M \ log10.(dt[m])
    return alpha, 10^b
end

# ranges the two power laws are fitted over: the startup ramp, where the grid is
# essentially logarithmic and the same for every k_u, and the relaxed regime
const FIT_EARLY = (1e-5, 1e-2)
const FIT_LATE = (1e-1, 3.5)

function fig_time_step(; logscale::Bool)
    s = paper_rc!(5.0)                       # single column
    figure(figsize = (5, 4))
    ax = PyPlot.gca()
    sat = [1, 4, 7, 10]
    handles, labels, early = [], [], NTuple{2,Float64}[]

    for i in sat
        # the step size itself, Delta_tau_j = t_{j+1} - t_j, plotted against the
        # time the step ends at. (The notebook plotted sol.w[end], the last row
        # of Gregory quadrature weights; those track the step size but are not
        # equal to it, and their slope on the log-log plot is ~0.1 shallower.)
        # The final step is truncated so that the grid lands exactly on tmax,
        # so it is left out.
        dt = diff(SOL[i].t)[1:end-1]
        t = SOL[i].t[2:end-1]
        h = scatter(t, dt, s = 5 * s^2)
        lab = string(round(ALPHA[i], digits = 3))
        if logscale
            alpha, A = power_law_fit(t, dt, FIT_LATE...)
            push!(early, power_law_fit(t, dt, FIT_EARLY...))
            # the fit, drawn a little past the data at both ends
            x = exp10.(range(log10(0.7 * FIT_LATE[1]), log10(1.3 * FIT_LATE[2]), length = 50))
            plot(x, A .* x .^ alpha, "--", color = h.get_facecolor()[1, :], linewidth = 1.0s)
            lab *= ",  " * string(round(alpha, digits = 2))
        end
        push!(handles, h)
        push!(labels, lab)
    end

    if logscale
        # the startup ramp is the same for every k_u, so one guide line for all
        alpha = sum(first, early) / length(early)
        A = exp10(sum(log10(f[2]) for f in early) / length(early))
        x = exp10.(range(log10(0.5 * FIT_EARLY[1]), log10(3 * FIT_EARLY[2]), length = 50))
        g, = plot(x, A .* x .^ alpha, "-", color = "0.35", linewidth = 1.0s)
        # the slope of each late-time fit goes in the legend next to its k_u
        leg = ax.legend(handles, labels, title = L"$k_u$,  $\alpha$", loc = "upper left",
                        frameon = false)
        ax.add_artist(leg)
        ax.legend([g], [L"$\Delta\tau \propto \tau^{%$(round(alpha, digits = 2))}$"],
                  loc = "lower right", frameon = false)
    else
        ax.legend(handles, labels, title = L"$k_u$", frameon = false)
    end

    title("gSBR time-step size")
    if logscale
        yscale("log")
        xscale("log")
    end
    xlabel(L"$\tau$")
    ylabel(L"$\Delta \tau$")
    savefig_both(SYS, logscale ? "time_step_loglog" : "time_step")
end

# ------------------------------------------------------------ means/variances

function fig_means_variance()
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    fig1 = figure(figsize = (15, 8))
    ax_1 = fig1.add_axes([0.0, 0.525, 0.3, 0.4])
    ax_2 = fig1.add_axes([0.35, 0.525, 0.3, 0.4])
    ax_5 = fig1.add_axes([0.7, 0.525, 0.3, 0.4])
    ax_3 = fig1.add_axes([0.0, 0.0, 0.3, 0.4])
    ax_4 = fig1.add_axes([0.35, 0.0, 0.3, 0.4])
    ax_6 = fig1.add_axes([0.7, 0.0, 0.3, 0.4])
    axs = [ax_1, ax_2, ax_5, ax_3, ax_4, ax_6]
    sat = [0, 3, 6] .+ 1
    cOL = cycle_colors()

    for i in 1:length(sat)
        for j in 1:NSPEC
            axs[i].plot(SOL[sat[i]].t, VAR[sat[i]].μ[j, :], linestyle = "-", c = cOL[j], linewidth = lw)
            axs[i].plot(SOL_mak[sat[i]].t, VAR_mak[sat[i]].μ[j, :], linestyle = (0, (1, 1)), c = cOL[j], linewidth = lw)
            axs[i].plot(Master[sat[i]].timeGrid, Master[sat[i]].y[j, :]', "--", c = cOL[j], linewidth = lw)
        end
        axs[i].spines["right"].set_visible(false)
        axs[i].spines["top"].set_visible(false)
        axs[i].set_title("\$k_u\$ = " * string(round(ALPHA[sat[i]], digits = 2)))
        axs[i].set_yscale("log")
        axs[i].set_ylim((0.002, 2.50))
        axs[i].set(xticklabels = [])
        i != 1 && axs[i].set(yticklabels = [])
    end
    axs[1].set_ylabel(L"$ \langle n(\tau) \rangle $")

    for i in 1:length(sat)
        for j in 1:NSPEC
            axs[i+3].plot(SOL[sat[i]].t, VAR[sat[i]].μ[j, :] .+ diag(VAR[sat[i]].C[j, j, :, :]),
                          linestyle = "-", c = cOL[j], linewidth = lw)
            axs[i+3].plot(EMRE[sat[i]].timeGrid, EMRE[sat[i]].lna_var[j, j, :]', linestyle = "dotted", c = cOL[j], linewidth = lw)
            axs[i+3].plot(Master[sat[i]].timeGrid, Master[sat[i]].variance[j, :]', "--", c = cOL[j], linewidth = lw)
        end
        axs[i+3].spines["right"].set_visible(false)
        axs[i+3].spines["top"].set_visible(false)
        axs[i+3].set_xlabel(L"$ \tau $")
        axs[i+3].set_yscale("log")
        axs[i+3].set_ylim((0.002, 5.0))
        i != 1 && axs[i+3].set(yticklabels = [])
    end
    axs[4].set_ylabel(L"$\langle \delta n(\tau)\delta n(\tau) \rangle $")

    # At main-text size a 9-entry legend no longer fits in a panel, so colour
    # (species) and linestyle (method) are listed separately: 3 + 3 entries.
    species_h = [line_handle(color = cOL[j], lw = lw) for j in 1:NSPEC]
    style_h(styles) = [line_handle(color = "0.3", lw = lw, linestyle = ls) for ls in styles]
    # in the third column the curves all sit above n ~ 0.25 after the transient,
    # so the lower right corner is free
    axs[3].legend(vcat(species_h, style_h(["-", (0, (1, 1)), "--"])),
                  ["G", "G*", "P", "gSBR", "MAK", "Master"],
                  loc = "lower right", frameon = false, ncol = 2)
    axs[6].legend(vcat(species_h, style_h(["-", "dotted", "--"])),
                  ["G", "G*", "P", "gSBR", "LNA", "Master"],
                  loc = "lower right", frameon = false, ncol = 2)
    savefig_both(SYS, "gene_means_variance")
end

# ------------------------------------------------- two-time number correlator

function fig_num_corr_P()
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    sat = 4
    time_grid_all = EMRE[sat].timeGrid
    time_grid_gSBR = SOL[sat].t
    plotting_tau = [0.75, 1.5, 2.25, 3.0]
    tau = find_closed_timegrid_indices(time_grid_all, plotting_tau)
    tau_SBR = find_closed_timegrid_indices(time_grid_gSBR, plotting_tau)

    fig = figure(figsize = (15, 3.5))
    ax1 = fig.add_axes([0.0, 0.0, 0.1, 1])
    ax2 = fig.add_axes([0.11, 0.0, 0.19, 1])
    ax3 = fig.add_axes([0.31, 0.0, 0.29, 1])
    ax4 = fig.add_axes([0.61, 0.0, 0.38, 1])
    axs = [ax4, ax3, ax2, ax1]

    for j in 1:length(tau)
        axs[j].plot(time_grid_gSBR[tau_SBR[j]:end] .- time_grid_gSBR[tau_SBR[j]],
                    VAR[sat].N[3, 3, tau_SBR[j]:end, tau_SBR[j]], "-", alpha = 0.75, linewidth = lw)
        axs[j].plot(time_grid_all[1:end-tau[j]], EMRE[sat].corr[3, 3, tau[j]+1:end, tau[j]],
                    linestyle = "dashdot", alpha = 0.75, linewidth = lw)
        axs[j].plot(Master[sat].timeGrid[1:end-tau[j]], Master[sat].correlator_tau[3, 3, 1:end-tau[j], tau[j]],
                    "--", alpha = 0.75, color = "gray", linewidth = lw)
        axs[j].set_title(L"\tau^\prime = " * string(plotting_tau[j]))
        axs[j].spines["right"].set_visible(false)
        axs[j].spines["top"].set_visible(false)
        axs[j].set_xlabel(L"\tau-\tau\prime")
        axs[j].set_ylim((-0.1, 2))
    end
    ax1.set_ylabel(L"\langle \delta n_P(\tau) \delta n_P(\tau^\prime) \rangle ")
    for a in (ax2, ax3, ax4)
        a.set(yticklabels = [])
    end
    # panels get wider left to right; prune the top tick so neighbouring
    # panels' labels do not run into each other
    for (a, n) in zip((ax1, ax2, ax3), (2, 3, 4))
        thin_xticks!(a, n; drop_last = true)
    end
    thin_xticks!(ax4, 5)
    ax4.legend(["gSBR", "LNA", "Master"], bbox_to_anchor = (0.5, 0.45), loc = 0, frameon = false)
    savefig_both(SYS, "gene_num_corr_P")
end

# ------------------------------------------------- equal-time cross-correlators

function fig_equal_time_corr()
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    # a little taller than the notebook version: the long y labels of the three
    # rows would otherwise run into each other at main-text size
    fig1 = figure(figsize = (15, 9))
    ax_1 = fig1.add_axes([0.0, 0.35, 0.3, 0.3])
    ax_2 = fig1.add_axes([0.35, 0.35, 0.3, 0.3])
    ax_5 = fig1.add_axes([0.7, 0.35, 0.3, 0.3])
    ax_3 = fig1.add_axes([0.0, 0.0, 0.3, 0.3])
    ax_4 = fig1.add_axes([0.35, 0.0, 0.3, 0.3])
    ax_6 = fig1.add_axes([0.7, 0.0, 0.3, 0.3])
    ax_11 = fig1.add_axes([0.0, 0.7, 0.3, 0.3])
    ax_22 = fig1.add_axes([0.35, 0.7, 0.3, 0.3])
    ax_55 = fig1.add_axes([0.7, 0.7, 0.3, 0.3])
    axs = [ax_1, ax_2, ax_5, ax_3, ax_4, ax_6, ax_11, ax_22, ax_55]
    sat = [0, 3, 6] .+ 1
    cOL = cycle_colors()

    # (a, b) index pair, y-limits and label for each row of three panels
    rows = [(1, 3, (-0.055, 0.75), L"\langle \delta n_G(\tau)\delta n_P(\tau) \rangle", 0),
            (2, 3, (-0.055, 0.65), L"\langle \delta n_{G^*}(\tau)\delta n_P(\tau) \rangle", 3),
            (1, 2, (-0.027, 0.04), L"\langle \delta n_G(\tau)\delta n_{G^*}(\tau) \rangle", 6)]

    for (a, b, ylim, ylab, off) in rows
        for i in 1:length(sat)
            ax = axs[i+off]
            ax.plot(SOL[sat[i]].t, diag(VAR[sat[i]].N[a, b, :, :]), linestyle = "-", c = cOL[1], linewidth = lw)
            ax.plot(EMRE[sat[i]].timeGrid, EMRE[sat[i]].lna_var[a, b, :], linestyle = "dashdot", c = cOL[2], linewidth = lw)
            ax.plot(Master[sat[i]].timeGrid, Master[sat[i]].correlator_tau[a, b, 1, :], "--", c = "gray", linewidth = lw)
            ax.spines["right"].set_visible(false)
            ax.spines["top"].set_visible(false)
            ax.set_ylim(ylim)
            i != 1 && ax.set(yticklabels = [])
            if off == 3              # middle row carries the x labels
                ax.set_xlabel(L"\tau")
            else
                ax.set(xticklabels = [])
            end
            if off == 6              # top row carries the titles
                ax.set_title("\$k_u\$ = " * string(round(ALPHA[sat[i]], digits = 2)))
            end
        end
        axs[1+off].set_ylabel(ylab)
    end
    ax_1.legend(["gSBR", "LNA", "Master"], bbox_to_anchor = (1.0, 1.0), loc = 1, frameon = false, ncol = 1)
    savefig_both(SYS, "gene_equal_time_corr")
end

# ------------------------------------------------------------ response function

function fig_response_P()
    # NOTE: the Master-equation curve of the notebook version is omitted here:
    # data/gene/Master_response.jls is truncated (EOFError on deserialize) and
    # recomputing it is the expensive cheMASTER response run of cell 68.
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    sat = 6
    time_grid_all = EMRE[sat].timeGrid
    time_grid_gSBR = SOL[sat].t
    plotting_tau = [0.75, 1.5, 2.25, 3.0]
    tau = find_closed_timegrid_indices(time_grid_all, plotting_tau)
    tau_SBR = find_closed_timegrid_indices(time_grid_gSBR, plotting_tau)

    fig = figure(figsize = (15, 3.5))
    ax1 = fig.add_axes([0.0, 0.0, 0.1, 1])
    ax2 = fig.add_axes([0.11, 0.0, 0.19, 1])
    ax3 = fig.add_axes([0.31, 0.0, 0.29, 1])
    ax4 = fig.add_axes([0.61, 0.0, 0.38, 1])
    axs = [ax4, ax3, ax2, ax1]

    for j in 1:length(plotting_tau)
        axs[j].plot(time_grid_gSBR[tau_SBR[j]:end] .- time_grid_gSBR[tau_SBR[j]],
                    VAR[sat].R[3, 3, tau_SBR[j]:end, tau_SBR[j]], "-", alpha = 0.75, linewidth = lw)
        axs[j].plot(time_grid_all[1:end-tau[j]], EMRE[sat].resp[3, 3, tau[j]+1:end, tau[j]],
                    linestyle = "dashdot", alpha = 0.75, linewidth = lw)
        axs[j].set_title(L"\tau^\prime = " * string(plotting_tau[j]))
        axs[j].spines["right"].set_visible(false)
        axs[j].spines["top"].set_visible(false)
        axs[j].set_xlabel(L"\tau-\tau\prime")
        axs[j].set_ylim((-0.1, 1.0))
    end
    ax1.set_ylabel(L"R_{PP}(\tau,\tau^\prime)")
    for a in (ax2, ax3, ax4)
        a.set(yticklabels = [])
    end
    for (a, n) in zip((ax1, ax2, ax3), (2, 3, 4))
        thin_xticks!(a, n; drop_last = true)
    end
    thin_xticks!(ax4, 5)
    ax4.legend(["gSBR", "LNA"], bbox_to_anchor = (0.45, 0.95), loc = 0, frameon = false)
    savefig_both(SYS, "gene_response_P")
end

# ----------------------------------------------- comparison of all the methods

function fig_all_methods(; variance::Bool, with_mak::Bool)
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    idx = 1
    color = cycle_colors()
    fig = figure(figsize = (15, 5), constrained_layout = true)
    for j in 1:NSPEC
        subplot(1, 3, j)
        if variance
            plot(SOL[idx].t, diag(VAR[idx].N[j, j, :, :]), label = "gSBR cross response", color = color[1], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_single_gSBR[idx].t, diag(VAR_single_gSBR[idx].N[j, :, :]), label = "gSBR single response", color = color[2], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_singleSBR[idx].t, diag(VAR_singleSBR[idx].N[j, :, :]), label = "SBR single response", color = color[3], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_crossMCA[idx].t, diag(VAR_crossMCA[idx].N[j, j, :, :]), label = "MCA cross response", color = color[4], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_singleMCA[idx].t, diag(VAR_singleMCA[idx].N[j, :, :]), label = "MCA single response", color = color[5], alpha = 0.75, linestyle = "--", linewidth = lw)
            if with_mak
                plot(SOL_mak[idx].t, diag(VAR_mak[idx].N[j, :, :]), label = "MAK", color = color[6], alpha = 0.75, linestyle = "--", linewidth = lw)
                plot(EMRE[idx].timeGrid, EMRE[idx].lna_var[j, j, :], label = "EMRE", color = color[7], alpha = 0.75, linestyle = "--", linewidth = lw)
            end
            plot(Master[idx].timeGrid, Master[idx].variance[j, :], label = "Master Eq.", color = color[8], alpha = 0.75, linestyle = "-", linewidth = lw)
            ylabel(L"$\langle \delta n(\tau) \delta n(\tau) \rangle$")
        else
            plot(SOL[idx].t, VAR[idx].μ[j, :], label = "gSBR cross response", color = color[1], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_single_gSBR[idx].t, VAR_single_gSBR[idx].μ[j, :], label = "gSBR single response", color = color[2], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_singleSBR[idx].t, VAR_singleSBR[idx].μ[j, :], label = "SBR single response", color = color[3], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_crossMCA[idx].t, VAR_crossMCA[idx].μ[j, :], label = "MCA cross response", color = color[4], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_singleMCA[idx].t, VAR_singleMCA[idx].μ[j, :], label = "MCA single response", color = color[5], alpha = 0.75, linestyle = "--", linewidth = lw)
            if with_mak
                plot(SOL_mak[idx].t, VAR_mak[idx].μ[j, :], label = "MAK", color = color[6], alpha = 0.75, linestyle = "--", linewidth = lw)
                plot(EMRE[idx].timeGrid, EMRE[idx].y[j, :], label = "EMRE", color = color[7], alpha = 0.75, linestyle = "--", linewidth = lw)
            end
            plot(Master[idx].timeGrid, Master[idx].y[j, :], label = "Master Eq.", color = color[8], alpha = 0.75, linestyle = "-", linewidth = lw)
            ylabel(L"$\langle n(\tau)\rangle$")
        end
        title(["G", "G*", "P"][j])
        yscale("log")
        xlabel(L"$\tau$")
    end
    # one legend for the whole figure, below the panels: at main-text size it no
    # longer fits inside a panel
    handles, labels = PyPlot.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc = "outside lower center", ncol = 4, frameon = false)
    name = (variance ? "gene_variances" : "gene_means") * "_comparisons_all_methods" * (with_mak ? "" : "_noMAK")
    savefig_both(SYS, name)
end

# ---------------------------------------------------------- Gillespie traces

function fig_gillespie(; n_traces = 5, n_runs = 40, seed = 0)
    py"""
    import sys
    sys.path.append($(joinpath(ROOT, "extras", "other_dynamics")))
    """
    cheMASTER = pyimport("cheMASTER")
    np = pyimport("numpy")
    np.random.seed(seed)

    num_species, num_interaction = 3, 3
    k1 = np.array([0.0, 0.0, 0.0])
    k2 = np.array([0.0, 0.0, 2.0])
    k3 = np.array([10.0, 1.0, 0.1])
    s_i, r_i = np.zeros([num_interaction, num_species]), np.zeros([num_interaction, num_species])
    rxn_par = [k1, k2, k3]
    s_i[1, :] = [1, 0, 1]; r_i[1, :] = [1, 0, 0]
    s_i[2, :] = [0, 1, 0]; r_i[2, :] = [1, 0, 1]
    s_i[3, :] = [1, 0, 1]; r_i[3, :] = [0, 1, 0]
    initial_values = np.array([0.01, 1.0, 0.1])
    dt, tf = 0.05, 3.5

    gill = []
    for _ in 1:n_runs
        g = cheMASTER.chemical_system_class(num_interaction, num_species, rxn_par, r_i, s_i)
        cheMASTER.gillespie_avg(g, 1, initial_values, 0, tf, dt, 10000)
        push!(gill, g)
    end
    traces = pick_diverse_traces(gill, n_traces; keep = g -> all(g.y[1, :] .- 3 .< 0.0))

    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    fig1 = figure(figsize = (15, 8))
    ax_1 = fig1.add_axes([0.0, 0.525, 0.3, 0.4])
    ax_2 = fig1.add_axes([0.35, 0.525, 0.3, 0.4])
    ax_3 = fig1.add_axes([0.7, 0.525, 0.3, 0.4])
    axs = [ax_1, ax_2, ax_3]
    # a sequential map sampled over its full range: the traces then differ in
    # colour as clearly as the map allows (the top ~10% is dropped, it is too
    # pale on white)
    cmap = PyPlot.plt.get_cmap("viridis")
    col = [cmap(x) for x in range(0, 0.9, length = length(traces))]

    for i in 1:num_species
        for (j, g) in enumerate(traces)
            axs[i].plot(g.timeGrid, g.y[i, :], "--", c = col[j], linewidth = lw)
        end
        axs[i].spines["right"].set_visible(false)
        axs[i].spines["top"].set_visible(false)
        axs[i].set_xlabel(L"${\tau}$")
    end
    axs[1].set_ylabel(L"$n_G$")
    axs[2].set_ylabel(L"$n_{G^*}$")
    axs[3].set_ylabel(L"$n_P$")
    savefig_both(SYS, "gene_gillespie_new")
end

# --------------------------------------------------------------------- run all

println("drawing gene figures ...")
fig_time_step(logscale = true)
fig_time_step(logscale = false)
fig_means_variance()
fig_num_corr_P()
fig_equal_time_corr()
fig_response_P()
fig_all_methods(variance = false, with_mak = false)
fig_all_methods(variance = false, with_mak = true)
fig_all_methods(variance = true, with_mak = false)
fig_all_methods(variance = true, with_mak = true)
fig_gillespie()
println("done: ", joinpath(OUT, SYS))
