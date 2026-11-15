# Paper figures for the SIR infection dynamics system.
# Redraws the figures of examples/SIR_infection_dynamics.ipynb from the stored
# artifacts, with text at main-text size. Run from anywhere:
#   julia --project=. scripts/paper_plots/SIR.jl

include(joinpath(@__DIR__, "common.jl"))

using Catalyst

const SYS = "SIR"
const NSPEC = 2          # S, I
const DT = 0.05          # master equation time step
const TF = 10.0

println("loading SIR artifacts ...")
# single parameter value (k_i = 0.75), stored as one-element vectors
SOL = read_artifact("SIR", "sol_gSBR.jls")[1]
VAR = read_artifact("SIR", "var_gSBR.jls")[1]
EMRE = read_artifact("SIR", "emre.jls")

# data/SIR/sol_mak.jls is empty (0 bytes), so the MAK run is redone here; it is
# a deterministic few-second solve of the same system and nothing is written
# back to data/.
println("re-running the MAK solve (data/SIR/sol_mak.jls is empty) ...")
SIR_system = @reaction_network begin
    @species S(t) = 5.0 I(t) = 1.0
    @parameters k_i = 0.75 k_2I = 1.0
    (k_i), S + I --> I + I
    (k_2I), I --> 0
end
let structure = Achedmy.ReactionStructure(SIR_system)
    global VAR_mak = Achedmy.ReactionVariables(structure, "single")
    global SOL_mak = Achedmy.solve_dynamics!(structure, VAR_mak, selfEnergy = "MAK",
                                             tmax = TF, tstart = 0.0, atol = 1e-3, rtol = 1e-2)
end

# The stored data/SIR/master.jls was run without `return_probability` and
# without the S-I cross correlator, so it carries neither `probability` nor
# `crossC`. The master equation for this 300-state system takes well under a
# second, so it is rerun here with those switches on (again, nothing is written
# back to data/).
println("re-running the SIR master equation (stored one lacks probability/crossC) ...")
Master = let
    cheMASTER = pyimport("cheMASTER")
    np = pyimport("numpy")
    num_species, num_interaction = 2, 1
    k1 = np.array([0.0, 0.0])
    k2 = np.array([0.0, 1.0])
    k3 = np.array([0.75])
    s_i, r_i = np.zeros([num_interaction, num_species]), np.zeros([num_interaction, num_species])
    rxn_par = [k1, k2, k3]
    s_i[1, :] = [0, 2]; r_i[1, :] = [1, 1]
    m = cheMASTER.chemical_system_class(num_interaction, num_species, rxn_par, r_i, s_i)
    cheMASTER.initialize_dynamics(m, np.array([5.0, 1.0]), 0, TF, DT)
    cheMASTER.runDynamics(m, max_num = [20, 15], method = "RK4", crossCorrelator_tau = "connected",
                          variance = true, crossCorrelator = [0, 1], return_probability = true)
    m
end

# ----------------------------------------------------- gSBR means / variances

function fig_means()
    # the notebook draws this into one quadrant of a 10x9 grid; drawn directly
    # as the single panel it is cropped to, so the font scaling is exact
    s = paper_rc!(5.0)
    lw = 1.2s
    figure(figsize = (5, 4.5), constrained_layout = true)
    for j in 1:NSPEC
        plot(SOL.t, VAR.μ[j, :], linewidth = lw)
    end
    legend(["S", "I"])
    xlabel(L"$\tau$")
    ylabel(L"$\langle n(\tau)\rangle$")
    title(L"$k_i$ = " * string(0.75))
    savefig_both(SYS, "SIR_means")
end

function fig_variances()
    s = paper_rc!(12.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    figure(figsize = (12, 4), constrained_layout = true)
    panels = [(1, 1, "Var(SS)", L"$\langle \delta n_S(\tau) \delta n_S(\tau) \rangle$"),
              (2, 2, "Var(II)", L"$\langle \delta n_I(\tau) \delta n_I(\tau) \rangle$"),
              (1, 2, "Cov(SI)", L"$\langle \delta n_S(\tau) \delta n_I(\tau) \rangle$")]
    for (k, (a, b, lab, ylab)) in enumerate(panels)
        subplot(1, 3, k)
        plot(SOL.t, diag(VAR.N[a, b, :, :]), linewidth = lw)
        legend([lab])
        xlabel(L"$\tau$")
        ylabel(ylab)
        title(L"$k_i$ = " * string(0.75))
    end
    savefig_both(SYS, "SIR_variances")
end

# -------------------------------------------------- two-time maps (pcolor 2x2)

function fig_two_time(field, name, titlefmt)
    # four panels plus colour bars: only legible at main-text size across both
    # columns
    paper_rc!(6.0, printed_width_in = PAGE_IN)
    fig = figure(figsize = (6, 6), constrained_layout = true)
    pairs = [(1, 1, "SS"), (2, 1, "IS"), (1, 2, "SI"), (2, 2, "II")]
    for (k, (a, b, lab)) in enumerate(pairs)
        ax = subplot(2, 2, k)
        im = ax.pcolor(SOL.t, SOL.t, field[a, b, :, :])
        ax.set_title(titlefmt(lab))
        ax.set_xlabel(L"$t$")
        ax.set_ylabel(L"$t'$")
        fig.colorbar(im, ax = ax)
    end
    savefig_both(SYS, name)
end

# ------------------------------------------- comparison against LNA and Master

function fig_means_comparison()
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    fig1 = figure(figsize = (15, 8))
    ax_1 = fig1.add_axes([0.0, 0.525, 0.4, 0.4])
    ax_2 = fig1.add_axes([0.5, 0.525, 0.4, 0.4])
    axs = [ax_1, ax_2]
    cOL = cycle_colors()

    for j in 1:NSPEC
        axs[j].plot(SOL.t, VAR.μ[j, :], linestyle = "-", c = cOL[1], linewidth = lw)
        axs[j].plot(SOL_mak.t, VAR_mak.μ[j, :], linestyle = (0, (1, 1)), c = cOL[2], linewidth = lw)
        axs[j].plot(EMRE[1].timeGrid, EMRE[1].y[j, :], linestyle = (0, (1, 0.5)), c = cOL[3], linewidth = lw)
        axs[j].plot(Master.timeGrid, Master.y[j, :], "--", c = "gray", linewidth = lw)
        axs[j].spines["right"].set_visible(false)
        axs[j].spines["top"].set_visible(false)
        axs[j].set_xlabel(L"$\tau$")
    end
    axs[1].set_ylabel(L"$\mu_S(\tau)$")
    axs[2].set_ylabel(L"$\mu_I(\tau)$")
    ax_1.legend(["gSBR", "MAK", "EMRE", "Master"], bbox_to_anchor = (1.0, 1.02), loc = 1,
                frameon = false, ncol = 2)
    savefig_both(SYS, "SIR_means_comparison")
end

function fig_variances_comparison()
    s = paper_rc!(15.0, printed_width_in = PAGE_IN)
    lw = 1.2s
    fig1 = figure(figsize = (15, 8))
    ax_1 = fig1.add_axes([0.0, 0.525, 0.3, 0.4])
    ax_2 = fig1.add_axes([0.35, 0.525, 0.3, 0.4])
    ax_5 = fig1.add_axes([0.7, 0.525, 0.3, 0.4])
    axs = [ax_1, ax_2, ax_5]
    cOL = cycle_colors()

    # (a, b, master curve, y label)
    panels = [(1, 1, Master.variance[1, :], L"$\langle \delta n_S(\tau) \delta n_S(\tau) \rangle$"),
              (2, 2, Master.variance[2, :], L"$\langle \delta n_I(\tau) \delta n_I(\tau) \rangle$"),
              (1, 2, Master.crossC, L"$\langle \delta n_S(\tau) \delta n_I(\tau) \rangle$")]
    for (k, (a, b, master_y, ylab)) in enumerate(panels)
        axs[k].plot(SOL.t, diag(VAR.N[a, b, :, :]), linestyle = "-", c = cOL[1], linewidth = lw)
        axs[k].plot(EMRE[1].timeGrid, EMRE[1].lna_var[a, b, :], linestyle = (0, (1, 0.5)), c = cOL[3], linewidth = lw)
        axs[k].plot(Master.timeGrid, master_y, "--", c = "gray", linewidth = lw)
        axs[k].spines["right"].set_visible(false)
        axs[k].spines["top"].set_visible(false)
        axs[k].set_xlabel(L"$\tau$")
        axs[k].set_ylabel(ylab)
    end
    ax_1.legend(["gSBR", "LNA", "Master"], bbox_to_anchor = (0.95, 0.5), loc = 1, frameon = false, ncol = 1)
    savefig_both(SYS, "SIR_variances_comparison")
end

# --------------------------------------- probability distribution of the master

function fig_probability()
    paper_rc!(15.0, printed_width_in = PAGE_IN)
    sat = [1, 21, 101]
    maxNum = Int.(Master.master_maxNum)
    stateSpace = Master.master_stateSpace
    prob = Master.probability

    probability = [zeros(maxNum...) for _ in sat]
    for (t, col) in enumerate(sat)
        norm = sum(prob[:, col])
        for i in 1:maxNum[1], j in 1:maxNum[2]
            idx = findfirst(k -> stateSpace[k, 1] == i - 1 && stateSpace[k, 2] == j - 1,
                            1:size(stateSpace, 1))
            idx === nothing && continue
            probability[t][i, j] = max(prob[idx, col], 0.0) / norm
        end
    end

    # laid out as a 1x3 grid (what the notebook's figure crops down to anyway):
    # at main-text size the hand-placed panels left no room for the colour bars
    fig1, axs = subplots(1, 3, figsize = (15, 4), constrained_layout = true)
    crops = [(1:12, 1:7), (1:12, 1:9), (1:9, 1:3)]

    for i in 1:length(sat)
        rows, cols = crops[i]
        im = axs[i].contourf(probability[i][rows, cols])
        # one tick per few contour levels: the default puts a label on every
        # level, which is far too dense at main-text size
        fig1.colorbar(im, ax = axs[i],
                      ticks = PyPlot.matplotlib.ticker.MaxNLocator(nbins = 4))
        axs[i].set_title(L"$\tau = $ " * string((sat[i] - 1) * DT))
        axs[i].set_xlabel(L"$n_I$")
        axs[i].set_ylabel(L"$n_S$")
    end
    savefig_both(SYS, "SIR_probability")
end

# --------------------------------------------------------------------- run all

println("drawing SIR figures ...")
fig_means()
fig_variances()
fig_two_time(VAR.R, "SIR_response_functions", lab -> L"Response $R_{%$lab}(t,t')$")
fig_two_time(VAR.C, "SIR_correlation_functions", lab -> L"Correlation $C_{%$lab}(t,t')$")
fig_two_time(VAR.N, "SIR_number_correlations", lab -> L"Number correlation $N_{%$lab}(t,t')$")
fig_means_comparison()
fig_variances_comparison()
fig_probability()
println("done: ", joinpath(OUT, SYS))
