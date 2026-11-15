# Paper figures for the enzyme kinetics system.
# Redraws the figures of examples/enzyme_kinetics.ipynb from the stored
# artifacts, with text at main-text size. Run from anywhere:
#   julia --project=. scripts/paper_plots/enzyme.jl

include(joinpath(@__DIR__, "common.jl"))

using CSV
using DataFrames
using OrdinaryDiffEq

const SYS = "enzyme"
const ALPHA_SBR = [0.01, 0.1, 1.0, 10.0]                  # k_f for the gSBR/MAK runs
const ALPHA = exp10.(range(-2, stop = 1, length = 10))    # k_f for Master/EMRE
const NSPEC = 4                                           # S, E, C, X
const SPECIES = ["S", "E", "C", "X"]

println("loading enzyme artifacts ...")
SOL = read_artifact("enzyme", "sol_gSBR.jls")
VAR = read_artifact("enzyme", "var_gSBR.jls")
SOL_mak = read_artifact("enzyme", "sol_mak.jls")
VAR_mak = read_artifact("enzyme", "var_mak.jls")
SOL_crossMCA = read_artifact("enzyme", "sol_crossMCA.jls")
VAR_crossMCA = read_artifact("enzyme", "var_crossMCA.jls")
SOL_single_gSBR = read_artifact("enzyme", "sol_single_gSBR.jls")
VAR_single_gSBR = read_artifact("enzyme", "var_single_gSBR.jls")
Master = read_artifact("enzyme", "master.jls")
EMRE = read_artifact("enzyme", "emre.jls")
Master_response = read_artifact("enzyme", "Master_response_kf1_fine_res_script.jls")
df = CSV.read(joinpath(DATA, "enzyme", "enzyme_means_variances.csv"), DataFrame)

# data/enzyme/QSS_MM.jls cannot be deserialized here (it stores an ODESolution
# whose solver cache types do not round-trip), so the two-variable
# Michaelis-Menten QSS ODE of the notebook is re-solved; it is deterministic and
# costs milliseconds.
function MM_MAK!(dy, y, p, t)
    E_total, k_f, k_b, k_d, k_1S, k_2S, k_2X = p
    k_m = (k_d + k_b) / k_f
    k_max = k_d * E_total
    dy[1] = k_1S - k_max * y[1] / (y[1] + k_m) - k_2S * y[1]
    dy[2] = k_max * y[1] / (y[1] + k_m) - k_2X * y[2]
end

function solve_qss(k_f; tstart = 0.0, tmax = 10.0, atol = 1e-3, rtol = 1e-2)
    initial_vals = [1.0, 0.9, 0.1, 0.1]                    # S, E, C, X
    p = [initial_vals[1] + initial_vals[2], k_f, 0.1, 1.0, 1.0, 1.0, 1.0]
    prob = ODEProblem(MM_MAK!, [initial_vals[1], initial_vals[4]], (tstart, tmax), p)
    return solve(prob, Tsit5(), abstol = atol, reltol = rtol)
end
QSS = solve_qss(1.0)

# ------------------------------------------------------- means and variances

function fig_means(; mak::Bool)
    # four panels: text at main-text size only leaves room if this one spans
    # both columns
    s = paper_rc!(10.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    figure(figsize = (10, 9), constrained_layout = true)
    for idx in 1:4
        subplot(2, 2, idx)
        for j in 1:NSPEC
            sol, var = mak ? (SOL_mak[idx], VAR_mak[idx]) : (SOL[idx], VAR[idx])
            plot(sol.t, var.μ[j, :], linewidth = lw)
        end
        legend(SPECIES, ncol = 2)
        title(L"$k_f$ = " * string(ALPHA_SBR[idx]))
        xlabel(L"$\tau$")
        ylabel(L"$\langle n(\tau)\rangle$")
    end
    savefig_both(SYS, mak ? "enzyme_mak_means" : "enzyme_means")
end

# ------------------------------------------- gSBR-only equal-time cross-variance

function fig_gSBR_cross_var()
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    fig1 = figure(figsize = (15, 8))
    ax_1 = fig1.add_axes([0.0, 0.35, 0.3, 0.3])
    ax_2 = fig1.add_axes([0.35, 0.35, 0.3, 0.3])
    ax_5 = fig1.add_axes([0.7, 0.35, 0.3, 0.3])
    ax_3 = fig1.add_axes([0.0, 0.0, 0.3, 0.3])
    ax_4 = fig1.add_axes([0.35, 0.0, 0.3, 0.3])
    ax_6 = fig1.add_axes([0.7, 0.0, 0.3, 0.3])
    axs = [ax_1, ax_2, ax_5, ax_3, ax_4, ax_6]
    sat = [2, 3, 4]
    cOL = cycle_colors()

    for i in 1:length(sat)
        axs[i].plot(SOL[sat[i]].t, diag(VAR[sat[i]].N[1, 4, :, :]), linestyle = "-", c = cOL[1], linewidth = lw)
        axs[i].spines["right"].set_visible(false)
        axs[i].spines["top"].set_visible(false)
        axs[i].set_ylim((-0.2, 0.01))
        axs[i].set(xticklabels = [])
        i != 1 && axs[i].set(yticklabels = [])
        axs[i].set_title("\$k_f\$ = " * string(round(ALPHA_SBR[sat[i]], digits = 2)))
    end
    axs[1].set_ylabel(L"\langle \delta n_S(\tau)\delta n_X(\tau) \rangle")

    for i in 1:length(sat)
        axs[i+3].plot(SOL[sat[i]].t, diag(VAR[sat[i]].N[2, 4, :, :]), linestyle = "-", c = cOL[1], linewidth = lw)
        axs[i+3].spines["right"].set_visible(false)
        axs[i+3].spines["top"].set_visible(false)
        axs[i+3].set_xlabel(L"\tau")
        axs[i+3].set_ylim((-0.01, 0.3))
        i != 1 && axs[i+3].set(yticklabels = [])
    end
    axs[4].set_ylabel(L"\langle \delta n_{E}(\tau)\delta n_X(\tau) \rangle")
    ax_1.legend(["gSBR"], bbox_to_anchor = (1.0, 0.5), loc = 1, frameon = false, ncol = 1)
    savefig_both(SYS, "enzyme_gSBR_cross_var")
end

# --------------------------------------- gSBR-only lag correlator / response

"Four panels of increasing width sharing a y axis, as used for the lag plots."
function lag_axes(fig; y0 = 0.0, h = 1.0)
    ax1 = fig.add_axes([0.0, y0, 0.1, h])
    ax2 = fig.add_axes([0.11, y0, 0.19, h])
    ax3 = fig.add_axes([0.31, y0, 0.29, h])
    ax4 = fig.add_axes([0.61, y0, 0.38, h])
    return ax1, ax2, ax3, ax4
end

"Tidy up the shared-y lag panels: no repeated y labels, no colliding x ticks."
function finish_lag_axes!(ax1, ax2, ax3, ax4)
    for a in (ax2, ax3, ax4)
        a.set(yticklabels = [])
    end
    for (a, n) in zip((ax1, ax2, ax3), (2, 3, 4))
        thin_xticks!(a, n; drop_last = true)
    end
    thin_xticks!(ax4, 5)
end

function fig_gSBR_lag(; response::Bool)
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    sat = 4
    tg = SOL[sat].t
    plotting_tau = [2.0, 4.0, 6.0, 8.0]
    tau_SBR = find_closed_timegrid_indices(tg, plotting_tau)

    fig = figure(figsize = (15, 3.5))
    ax1, ax2, ax3, ax4 = lag_axes(fig)
    axs = [ax4, ax3, ax2, ax1]
    for j in 1:length(plotting_tau)
        y = response ? VAR[sat].R[4, 4, tau_SBR[j]:end, tau_SBR[j]] :
                       VAR[sat].N[4, 4, tau_SBR[j]:end, tau_SBR[j]]
        axs[j].plot(tg[tau_SBR[j]:end] .- tg[tau_SBR[j]], y, "-", alpha = 0.75, linewidth = lw)
        axs[j].set_title(L"\tau^\prime = " * string(plotting_tau[j]))
        axs[j].spines["right"].set_visible(false)
        axs[j].spines["top"].set_visible(false)
        axs[j].set_xlabel(L"\tau-\tau\prime")
        axs[j].set_ylim(response ? (-0.1, 1.1) : (0.075, 0.35))
    end
    ax1.set_ylabel(response ? L"R_{XX}(\tau,\tau^\prime)" :
                              L"\langle \delta n_X(\tau) \delta n_X(\tau^\prime) \rangle ")
    finish_lag_axes!(ax1, ax2, ax3, ax4)
    ax4.legend(["gSBR"], bbox_to_anchor = (0.45, 0.95), loc = 0, frameon = false)
    savefig_both(SYS, response ? "enzyme_response_gSBR_X" : "enzyme_gSBR_lag_correlator_X")
end

function fig_gSBR_cross_response()
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    sat = 4
    tg = SOL[sat].t
    plotting_tau = [2.0, 4.0, 6.0, 8.0]
    tau_SBR = find_closed_timegrid_indices(tg, plotting_tau)

    fig = figure(figsize = (15, 7))
    ax1, ax2, ax3, ax4 = lag_axes(fig; y0 = 0.55, h = 0.45)
    ax11, ax22, ax33, ax44 = lag_axes(fig; y0 = 0.0, h = 0.45)
    top = [ax4, ax3, ax2, ax1]
    bot = [ax44, ax33, ax22, ax11]

    for j in 1:length(plotting_tau)
        x = tg[tau_SBR[j]:end] .- tg[tau_SBR[j]]
        top[j].plot(x, VAR[sat].R[4, 1, tau_SBR[j]:end, tau_SBR[j]], "-", alpha = 0.75, linewidth = lw)
        top[j].set_title(L"\tau^\prime = " * string(plotting_tau[j]))
        top[j].spines["right"].set_visible(false)
        top[j].spines["top"].set_visible(false)
        top[j].set_ylim((-0.05, 0.15))
        top[j].set(xticklabels = [])

        bot[j].plot(x, VAR[sat].R[2, 1, tau_SBR[j]:end, tau_SBR[j]], "-", alpha = 0.75, linewidth = lw)
        bot[j].spines["right"].set_visible(false)
        bot[j].spines["top"].set_visible(false)
        bot[j].set_xlabel(L"\tau-\tau\prime")
        bot[j].set_ylim((-0.225, 0.01))
    end
    ax1.set_ylabel(L"R_{XS}(\tau,\tau^\prime)")
    ax11.set_ylabel(L"R_{ES}(\tau,\tau^\prime)")
    for a in (ax2, ax3, ax4, ax22, ax33, ax44)
        a.set(yticklabels = [])
    end
    finish_lag_axes!(ax11, ax22, ax33, ax44)
    ax4.legend(["gSBR"], bbox_to_anchor = (0.45, 0.95), loc = 0, frameon = false)
    savefig_both(SYS, "enzyme_gSBR_cross_response")
end

# ------------------------------------------ means/variances against LNA, Master

function fig_means_variance_comparison()
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
    sat_sbr = [2, 3, 4]
    sat = [3, 6, 9] .+ 1
    cOL = cycle_colors()
    plotted = [1, 2, 4]                 # C is left out, as in the notebook

    for i in 1:length(sat)
        for j in plotted
            axs[i].plot(SOL[sat_sbr[i]].t, VAR[sat_sbr[i]].μ[j, :], linestyle = "-", c = cOL[j], linewidth = lw)
            axs[i].plot(SOL_mak[sat_sbr[i]].t, VAR_mak[sat_sbr[i]].μ[j, :], linestyle = (0, (1, 1)), c = cOL[j], linewidth = lw)
            axs[i].plot(Master[sat[i]].timeGrid, Master[sat[i]].y[j, :]', "--", c = cOL[j], linewidth = lw)
        end
        axs[i].spines["right"].set_visible(false)
        axs[i].spines["top"].set_visible(false)
        axs[i].set_title("\$k_f\$ = " * string(round(ALPHA[sat[i]], digits = 2)))
        axs[i].set_xlim((0.001, 10))
        axs[i].set_xscale("log")
        axs[i].set(xticklabels = [])
        i != 1 && axs[i].set(yticklabels = [])
    end
    axs[1].set_ylabel(L"$ \langle n(\tau) \rangle $")

    for i in 1:length(sat)
        for j in plotted
            axs[i+3].plot(SOL[sat_sbr[i]].t, diag(VAR[sat_sbr[i]].N[j, j, :, :]), linestyle = "-", c = cOL[j], linewidth = lw)
            axs[i+3].plot(EMRE[sat[i]].timeGrid, EMRE[sat[i]].lna_var[j, j, :]', linestyle = "dotted", c = cOL[j], linewidth = lw)
            axs[i+3].plot(Master[sat[i]].timeGrid, Master[sat[i]].variance[j, :]', "--", c = cOL[j], linewidth = lw)
        end
        axs[i+3].spines["right"].set_visible(false)
        axs[i+3].spines["top"].set_visible(false)
        axs[i+3].set_xlim((0.001, 10))
        axs[i+3].set_xlabel(L"$ \tau $")
        axs[i+3].set_xscale("log")
        i != 1 && axs[i+3].set(yticklabels = [])
    end
    axs[4].set_ylabel(L"$\langle \delta n(\tau)\delta n(\tau) \rangle $")

    # colour (species) and linestyle (method) listed separately: a 9-entry
    # legend does not fit in a panel at main-text size
    species_h = [line_handle(color = cOL[j], lw = lw) for j in plotted]
    style_h(styles) = [line_handle(color = "0.3", lw = lw, linestyle = ls) for ls in styles]
    # one legend per row: the second curve is MAK in the top row and LNA in the
    # bottom one, and the two are drawn with different dashes
    axs[1].legend(vcat(species_h, style_h(["-", (0, (1, 1)), "--"])),
                  ["S", "E", "X", "gSBR", "MAK", "Master"],
                  loc = "center left", frameon = false, ncol = 2)
    axs[4].legend(vcat(species_h, style_h(["-", "dotted", "--"])),
                  ["S", "E", "X", "gSBR", "LNA", "Master"],
                  loc = "center left", frameon = false, ncol = 2)
    savefig_both(SYS, "enzyme_means_variance_comparisons_log_time-3")
end

# ------------------------------------------------- two-time number correlator

function fig_num_corr_X()
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    sat_sbr, sat = 3, 7
    time_grid_all = EMRE[sat].timeGrid
    tg = SOL[sat_sbr].t
    plotting_tau = [0.75, 1.5, 2.25, 3.0]
    tau = find_closed_timegrid_indices(time_grid_all, plotting_tau)
    tau_SBR = find_closed_timegrid_indices(tg, plotting_tau)

    fig = figure(figsize = (15, 3.5))
    ax1, ax2, ax3, ax4 = lag_axes(fig)
    axs = [ax4, ax3, ax2, ax1]
    for j in 1:length(tau)
        axs[j].plot(tg[tau_SBR[j]:end] .- tg[tau_SBR[j]],
                    VAR[sat_sbr].N[4, 4, tau_SBR[j]:end, tau_SBR[j]], "-", alpha = 0.75, linewidth = lw)
        axs[j].plot(time_grid_all[1:end-tau[j]], EMRE[sat].corr[4, 4, tau[j]+1:end, tau[j]],
                    linestyle = "dashdot", alpha = 0.75, linewidth = lw)
        axs[j].plot(Master[sat].timeGrid[1:end-tau[j]], Master[sat].correlator_tau[4, 4, 1:end-tau[j], tau[j]],
                    "--", alpha = 0.75, color = "gray", linewidth = lw)
        axs[j].set_title(L"\tau^\prime = " * string(plotting_tau[j]))
        axs[j].spines["right"].set_visible(false)
        axs[j].spines["top"].set_visible(false)
        axs[j].set_xlabel(L"\tau-\tau\prime")
        axs[j].set_ylim((-0.1, 0.75))
    end
    ax1.set_ylabel(L"\langle \delta n_X(\tau) \delta n_X(\tau^\prime) \rangle ")
    finish_lag_axes!(ax1, ax2, ax3, ax4)
    ax4.legend(["gSBR", "LNA", "Master"], bbox_to_anchor = (0.5, 0.45), loc = 0, frameon = false)
    # the notebook writes this same figure under both names
    savefig_both(SYS, "enzyme_num_corr_X_kf1")
    savefig_both(SYS, "enzyme_num_corr_X")
end

# ------------------------------------------------- equal-time cross-correlators

function fig_equal_time_corr()
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    fig1 = figure(figsize = (15, 8))
    ax_1 = fig1.add_axes([0.0, 0.35, 0.3, 0.3])
    ax_2 = fig1.add_axes([0.35, 0.35, 0.3, 0.3])
    ax_5 = fig1.add_axes([0.7, 0.35, 0.3, 0.3])
    ax_3 = fig1.add_axes([0.0, 0.0, 0.3, 0.3])
    ax_4 = fig1.add_axes([0.35, 0.0, 0.3, 0.3])
    ax_6 = fig1.add_axes([0.7, 0.0, 0.3, 0.3])
    axs = [ax_1, ax_2, ax_5, ax_3, ax_4, ax_6]
    sat = [3, 6, 9] .+ 1
    sat_sbr = [2, 3, 4]
    cOL = cycle_colors()

    for i in 1:length(sat)
        axs[i].plot(SOL[sat_sbr[i]].t, diag(VAR[sat_sbr[i]].N[1, 4, :, :]), linestyle = "-", c = cOL[1], linewidth = lw)
        axs[i].plot(EMRE[sat[i]].timeGrid, EMRE[sat[i]].lna_var[1, 4, :], linestyle = "dashdot", c = cOL[2], linewidth = lw)
        axs[i].plot(Master[sat[i]].timeGrid, Master[sat[i]].correlator_tau[1, 4, 1, :], "--", c = "gray", linewidth = lw)
        axs[i].spines["right"].set_visible(false)
        axs[i].spines["top"].set_visible(false)
        axs[i].set_ylim((-0.25, 0.01))
        axs[i].set(xticklabels = [])
        i != 1 && axs[i].set(yticklabels = [])
        axs[i].set_title("\$k_f\$ = " * string(round(ALPHA[sat[i]], digits = 2)))
    end
    axs[1].set_ylabel(L"\langle \delta n_S(\tau)\delta n_X(\tau) \rangle")

    for i in 1:length(sat)
        axs[i+3].plot(SOL[sat_sbr[i]].t, diag(VAR[sat_sbr[i]].N[2, 4, :, :]), linestyle = "-", c = cOL[1], linewidth = lw)
        axs[i+3].plot(EMRE[sat[i]].timeGrid, EMRE[sat[i]].lna_var[2, 4, :], linestyle = "dashdot", c = cOL[2], linewidth = lw)
        axs[i+3].plot(Master[sat[i]].timeGrid, Master[sat[i]].correlator_tau[2, 4, 1, :], "--", c = "gray", linewidth = lw)
        axs[i+3].spines["right"].set_visible(false)
        axs[i+3].spines["top"].set_visible(false)
        axs[i+3].set_xlabel(L"\tau")
        axs[i+3].set_ylim((-0.055, 0.4))
        i != 1 && axs[i+3].set(yticklabels = [])
    end
    axs[4].set_ylabel(L"\langle \delta n_{E}(\tau)\delta n_X(\tau) \rangle")
    ax_1.legend(["gSBR", "LNA", "Master"], bbox_to_anchor = (1.0, 0.75), loc = 1, frameon = false, ncol = 1)
    savefig_both(SYS, "enzyme_equal_time_corr")
end

# ------------------------------------------- response against LNA and Master

function fig_response_kf1(; both_rows::Bool)
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    sat, sat_sbr = 7, 3
    time_grid_all = EMRE[sat].timeGrid
    time_grid_Master = Master_response.timeGrid
    tg = SOL[sat_sbr].t
    plotting_tau = [2.0, 4.0, 6.0, 8.0]
    tau = find_closed_timegrid_indices(time_grid_all, plotting_tau)
    tau_SBR = find_closed_timegrid_indices(tg, plotting_tau)
    tau_Master = find_closed_timegrid_indices(time_grid_Master, plotting_tau)

    fig = figure(figsize = (15, 7))
    ax1, ax2, ax3, ax4 = lag_axes(fig; y0 = both_rows ? 0.55 : 0.0, h = both_rows ? 0.45 : 1.0)
    top = [ax4, ax3, ax2, ax1]

    "one row of panels: species pair (a, b), y-limits and label"
    function row!(axsrow, a, b, ylim, ylab, xlabels::Bool)
        for j in 1:length(plotting_tau)
            axsrow[j].plot(tg[tau_SBR[j]:end] .- tg[tau_SBR[j]],
                           VAR[sat_sbr].R[a, b, tau_SBR[j]:end, tau_SBR[j]], "-", alpha = 0.75, linewidth = lw)
            axsrow[j].plot(time_grid_all[1:end-tau[j]], EMRE[sat].resp[a, b, tau[j]+1:end, tau[j]],
                           linestyle = "dashdot", alpha = 0.75, linewidth = lw)
            axsrow[j].plot(time_grid_Master[tau_Master[j]:end] .- time_grid_Master[tau_Master[j]],
                           Master_response.Response[a, b, tau_Master[j]:end, tau_Master[j]],
                           "--", alpha = 0.75, color = "gray", linewidth = lw)
            axsrow[j].spines["right"].set_visible(false)
            axsrow[j].spines["top"].set_visible(false)
            axsrow[j].set_ylim(ylim)
            if xlabels
                axsrow[j].set_xlabel(L"\tau-\tau\prime")
            else
                axsrow[j].set(xticklabels = [])
            end
        end
        axsrow[4].set_ylabel(ylab)   # axsrow[4] is the leftmost (narrowest) panel
    end

    for j in 1:length(plotting_tau)
        top[j].set_title(L"\tau^\prime = " * string(plotting_tau[j]))
    end
    row!(top, 4, 1, (-0.005, 0.25), L"R_{XS}(\tau,\tau^\prime)", !both_rows)
    for a in (ax2, ax3, ax4)
        a.set(yticklabels = [])
    end

    if both_rows
        ax11, ax22, ax33, ax44 = lag_axes(fig; y0 = 0.0, h = 0.45)
        bot = [ax44, ax33, ax22, ax11]
        row!(bot, 2, 1, (-0.14, 0.005), L"R_{ES}(\tau,\tau^\prime)", true)
        for a in (ax22, ax33, ax44)
            a.set(yticklabels = [])
        end
        finish_lag_axes!(ax11, ax22, ax33, ax44)
    else
        finish_lag_axes!(ax1, ax2, ax3, ax4)
    end
    ax4.legend(["gSBR", "LNA", "Master"], bbox_to_anchor = (0.45, 0.95), loc = 0, frameon = false)
    savefig_both(SYS, both_rows ? "enzyme_response_kf1" : "enzyme_response_kf1_XS")
end

# ----------------------------------------------- comparison of all the methods

function fig_all_methods(; variance::Bool)
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    idx, idx_sbr, idx_master = 1, 3, 7
    sub = df[df.k_f.==1.0, :]
    color = cycle_colors()
    fig = figure(figsize = (15, 5), constrained_layout = true)
    for (panel, j) in enumerate([1, 2, 4])          # S, E, X
        subplot(1, 3, panel)
        if variance
            plot(SOL[idx_sbr].t, diag(VAR[idx_sbr].N[j, j, :, :]), label = "gSBR cross response", color = color[1], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_single_gSBR[idx].t, diag(VAR_single_gSBR[idx].N[j, :, :]), label = "gSBR single response", color = color[2], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_crossMCA[idx].t, diag(VAR_crossMCA[idx].N[j, j, :, :]), label = "MCA cross response", color = color[4], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_mak[idx_sbr].t, diag(VAR_mak[idx_sbr].N[j, :, :]), label = "MAK", color = color[6], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(EMRE[idx_master].timeGrid, EMRE[idx_master].lna_var[j, j, :], label = "EMRE", color = color[7], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(sub.time, [sub.S_var, sub.E_var, sub.X_var][panel], label = "Normal Moment Closure", color = color[9], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(Master[idx_master].timeGrid, Master[idx_master].variance[j, :], label = "Master Eq.", color = color[8], alpha = 0.75, linestyle = "-", linewidth = lw)
            ylabel(L"$\langle \delta n(\tau) \delta n(\tau) \rangle$")
        else
            plot(SOL[idx_sbr].t, VAR[idx_sbr].μ[j, :], label = "gSBR cross response", color = color[1], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_single_gSBR[idx].t, VAR_single_gSBR[idx].μ[j, :], label = "gSBR single response", color = color[2], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_crossMCA[idx].t, VAR_crossMCA[idx].μ[j, :], label = "MCA cross response", color = color[4], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(SOL_mak[idx_sbr].t, VAR_mak[idx_sbr].μ[j, :], label = "MAK", color = color[6], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(EMRE[idx_master].timeGrid, EMRE[idx_master].y[j, :], label = "EMRE", color = color[7], alpha = 0.75, linestyle = "--", linewidth = lw)
            plot(sub.time, [sub.S_mean, sub.E_mean, sub.X_mean][panel], label = "Normal Moment Closure", color = color[9], alpha = 0.75, linestyle = "--", linewidth = lw)
            if j == 1 || j == 4      # the QSS reduction only has S and X
                qss = [u[j == 1 ? 1 : 2] for u in QSS.u]
                plot(QSS.t, qss, label = "MM QSS", color = color[10], alpha = 0.75, linestyle = "--", linewidth = lw)
            end
            plot(Master[idx_master].timeGrid, Master[idx_master].y[j, :], label = "Master Eq.", color = color[8], alpha = 0.75, linestyle = "-", linewidth = lw)
            ylabel(L"$\langle n(\tau)\rangle$")
        end
        title(SPECIES[j])
        yscale("log")
        xlabel(L"$\tau$")
    end
    handles, labels = PyPlot.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc = "outside lower center", ncol = 4, frameon = false)
    savefig_both(SYS, variance ? "enzyme_variances_comparisons_all_methods" : "enzyme_means_comparisons_all_methods")
end

# ---------------------------------------------------------- Gillespie traces

function fig_gillespie(; n_traces = 5, n_runs = 40, seed = 0)
    cheMASTER = pyimport("cheMASTER")
    np = pyimport("numpy")
    np.random.seed(seed)

    num_species, num_interaction = 4, 3
    k1 = np.array([1.0, 0.0, 0.0, 0.0])
    k2 = np.array([1.0, 0.0, 0.0, 1.0])
    k3 = np.array([1.0, 0.1, 1.0])
    s_i, r_i = np.zeros([num_interaction, num_species]), np.zeros([num_interaction, num_species])
    rxn_par = [k1, k2, k3]
    s_i[1, :] = [0, 0, 1, 0]; r_i[1, :] = [1, 1, 0, 0]
    s_i[2, :] = [1, 1, 0, 0]; r_i[2, :] = [0, 0, 1, 0]
    s_i[3, :] = [0, 1, 0, 1]; r_i[3, :] = [0, 0, 1, 0]
    initial_values = np.array([1.0, 0.9, 0.1, 0.1])
    dt, tf = 0.025, 10.0

    gill = []
    for _ in 1:n_runs
        g = cheMASTER.chemical_system_class(num_interaction, num_species, rxn_par, r_i, s_i)
        cheMASTER.gillespie_avg(g, 1, initial_values, 0, tf, dt, 10000)
        push!(gill, g)
    end
    traces = pick_diverse_traces(gill, n_traces)

    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    fig1 = figure(figsize = (15, 8))
    ax_1 = fig1.add_axes([0.0, 0.525, 0.3, 0.4])
    ax_2 = fig1.add_axes([0.35, 0.525, 0.3, 0.4])
    ax_3 = fig1.add_axes([0.7, 0.525, 0.3, 0.4])
    axs = Dict(1 => ax_1, 2 => ax_2, 4 => ax_3)      # C is left out, as in the notebook
    cmap = PyPlot.plt.get_cmap("tab10")
    col = [cmap(x) for x in range(0, 1, length = length(traces))]

    for (i, ax) in sort(collect(axs), by = first)
        for (j, g) in enumerate(traces)
            ax.plot(g.timeGrid, g.y[i, :], "--", c = col[j], linewidth = lw)
        end
        ax.spines["right"].set_visible(false)
        ax.spines["top"].set_visible(false)
        ax.set_xlabel(L"${\tau}$")
    end
    ax_1.set_ylabel(L"$n_S$")
    ax_2.set_ylabel(L"$n_{E}$")
    ax_3.set_ylabel(L"$n_X$")
    savefig_both(SYS, "enzyme_gillespie_new")
end

# --------------------------------------------------------------------- run all

println("drawing enzyme figures ...")
fig_means(mak = false)
fig_means(mak = true)
fig_gSBR_cross_var()
fig_gSBR_lag(response = false)
fig_gSBR_lag(response = true)
fig_gSBR_cross_response()
fig_means_variance_comparison()
fig_num_corr_X()
fig_equal_time_corr()
fig_response_kf1(both_rows = true)
fig_response_kf1(both_rows = false)
fig_all_methods(variance = false)
fig_all_methods(variance = true)
fig_gillespie()
println("done: ", joinpath(OUT, SYS))
