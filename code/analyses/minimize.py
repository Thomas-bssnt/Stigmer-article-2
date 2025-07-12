import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.optimize import minimize

# import sys

# Parameters of the minimization (plot parameters are in the correponding sections)
n_var = 2  # Number of variables
var = "PIF"  # Order of variables (adapt the figure x-axis-legends, if necessary)
RelativeError = -1  # -1 = Same weight; 0 = Each normalized; 1 = Relative error; 2 = Sqrt(weight)
Err_N = 1  # Include the error of Neutrals


# Initialisation
if True:
    dim = 2 + 2 * n_var
    x0 = [1.0] * dim

    var = var.upper()
    name_var = ""
    if n_var == 1 and var[:1] == "P":
        name_var = r"$P(20)$"
    if n_var == 1 and var[:1] == "I":
        name_var = r"$\mathrm{IPR}(\mathbf{P}(20))$"
    if n_var == 1 and var[:1] == "F":
        name_var = "Rank"

    if n_var == 2 and (var[:2] == "PI" or var[:2] == "IP"):
        name_var = r"$P(20)$ / $\mathrm{IPR}(\mathbf{P}(20))$"
    if n_var == 2 and (var[:2] == "IF" or var[:2] == "FI"):
        name_var = r"$\mathrm{IPR}(\mathbf{P}(20))$ / Rank"
    if n_var == 2 and (var[:2] == "PF" or var[:2] == "FP"):
        name_var = r"$P(20)$ / F($P(20)$)"

    if n_var == 3:
        name_var = r"$P(20)$ / $\mathrm{IPR}(\mathbf{P}(20))$ / Rank"

    Case = 10
    i_min = 0
    i_norm = 0

    C = np.zeros(Case)
    D = np.zeros(Case)
    N = np.zeros(Case)
    P = np.zeros(Case)
    I = np.zeros(Case)
    F = np.zeros(Case)

    # Condition: 5 humans
    i = 0
    C[i] = 0.181529
    D[i] = 0.38639
    P[i] = 0.258069
    I[i] = 19.421961
    F[i] = 0.418779

    # Condition: 4-0 Col/Def
    i = 1
    C[i] = 0.169301
    D[i] = 0.425374
    P[i] = 0.529950
    I[i] = 9.656562
    F[i] = 0.535284

    # Condition: 3-1 Col/Def
    i = 2
    C[i] = 0.153461
    D[i] = 0.337527
    P[i] = 0.428312
    I[i] = 14.442556
    F[i] = 0.520483

    # Condition: 2-2 Col/Def
    i = 3
    C[i] = 0.186569
    D[i] = 0.383909
    P[i] = 0.290746
    I[i] = 26.367217
    F[i] = 0.489706

    # Condition: 1-3 Col/Def
    i = 4
    C[i] = 0.315223
    D[i] = 0.287939
    P[i] = 0.185436
    I[i] = 40.219668
    F[i] = 0.462791

    # Condition: 0-4 Col/Def
    i = 5
    C[i] = 0.259719
    D[i] = 0.149016
    P[i] = 0.103585
    I[i] = 51.450846
    F[i] = 0.417346

    # Condition: Neutral 1
    i = 6
    C[i] = 0.151462
    D[i] = 0.252009
    P[i] = 0.278531
    I[i] = 33.272805
    F[i] = 0.554375

    # Condition: Neutral 3
    i = 7
    C[i] = 0.4308
    D[i] = 0.247092
    P[i] = 0.299298
    I[i] = 36.311312
    F[i] = 0.576063

    # Condition: Neutral 5
    i = 8
    C[i] = 0.259552
    D[i] = 0.235256
    P[i] = 0.279910
    I[i] = 38.323815
    F[i] = 0.571300

    # Condition: Opt
    i = 9
    C[i] = 0.424015
    D[i] = 0.265285
    P[i] = 0.616553
    I[i] = 4.252325
    F[i] = 0.284411

    F = [3.0, 1.457633, 1.824071, 1.822442, 2.0711, 2.312462, 1.810701, 2.015742, 1.739582, 2.067175]

    N = 1 - C - D

    Mean_C = np.mean(C[i_min:])
    Mean_D = np.mean(D[i_min:])
    Mean_N = np.mean(N[i_min:])
    Std_C = np.std(C[i_min:])
    Std_D = np.std(D[i_min:])
    Std_N = np.std(N[i_min:])

    P_Mean = np.mean(P[i_norm:])
    I_Mean = np.mean(I[i_norm:])
    F_Mean = np.mean(F[i_norm:])
    P_std = np.std(P[i_norm:])
    I_std = np.std(I[i_norm:])
    F_std = np.std(F[i_norm:])

    C_norm = np.sum(C[i_norm:] ** 2)
    D_norm = np.sum(D[i_norm:] ** 2)
    N_norm = np.sum(N[i_norm:] ** 2)


# Compute the error
def error(x):
    x1 = (P - P_Mean) / P_std
    x2 = (I - I_Mean) / I_std
    x3 = (F - F_Mean) / F_std

    if var == "PFI":
        x1 = (P - P_Mean) / P_std
        x2 = (F - F_Mean) / F_std
        x3 = (I - I_Mean) / I_std
    if var == "IPF":
        x1 = (I - I_Mean) / I_std
        x2 = (P - P_Mean) / P_std
        x3 = (F - F_Mean) / F_std
    if var == "IFP":
        x1 = (I - I_Mean) / I_std
        x2 = (F - F_Mean) / F_std
        x3 = (P - P_Mean) / P_std
    if var == "FPI":
        x1 = (F - F_Mean) / F_std
        x2 = (P - P_Mean) / P_std
        x3 = (I - I_Mean) / I_std
    if var == "FIP":
        x1 = (F - F_Mean) / F_std
        x2 = (I - I_Mean) / I_std
        x3 = (P - P_Mean) / P_std

    C_th = np.zeros(Case)
    D_th = np.zeros(Case)

    # if RelativeError == -1:
    #     x[0] = 0
    #     x[1] = 0

    for i in range(Case):
        C_th[i] = x[0]
        D_th[i] = x[1]
        if n_var >= 1:
            C_th[i] = C_th[i] + x[2] * x1[i]
            D_th[i] = D_th[i] + x[3] * x1[i]
        if n_var >= 2:
            C_th[i] = C_th[i] + x[4] * x2[i]
            D_th[i] = D_th[i] + x[5] * x2[i]
        if n_var >= 3:
            C_th[i] = C_th[i] + x[6] * x3[i]
            D_th[i] = D_th[i] + x[7] * x3[i]

    C_th = Mean_C + Std_C * C_th
    D_th = Mean_D + Std_D * D_th
    N_th = 1 - C_th - D_th

    if RelativeError == 1:
        err = np.sum((C_th[i_min:] / C[i_min:] - 1) ** 2)
        err = err + np.sum((D_th[i_min:] / D[i_min:] - 1) ** 2)
        if Err_N == 1:
            err = err + np.sum((N_th[i_min:] / N[i_min:] - 1) ** 2)
        err = err / 3 / Case
    elif RelativeError == 0:
        err = np.sum((C_th[i_min:] - C[i_min:]) ** 2) / C_norm
        err = err + np.sum((D_th[i_min:] - D[i_min:]) ** 2) / D_norm
        if Err_N == 1:
            err = err + np.sum((N_th[i_min:] - N[i_min:]) ** 2) / N_norm
        err = err / 3

    elif RelativeError == -1:
        err = np.sum((C_th[i_min:] - C[i_min:]) ** 2)
        err = err + np.sum((D_th[i_min:] - D[i_min:]) ** 2)
        if Err_N == 1:
            err = err + np.sum((N_th[i_min:] - N[i_min:]) ** 2)
        err = err / (C_norm + D_norm + N_norm)

    else:
        err = np.sum((C_th[i_min:] / C[i_min:] - 1) ** 2 * C[i_min:])
        err = err + np.sum((D_th[i_min:] / D[i_min:] - 1) ** 2 * D[i_min:])
        if Err_N == 1:
            err = err + np.sum((N_th[i_min:] / N[i_min:] - 1) ** 2 * N[i_min:])
        err = err / 3 / Case
    return err


# Prediction of the model
def Prediction(x):
    x1 = (P - P_Mean) / P_std
    x2 = (I - I_Mean) / I_std
    x3 = (F - F_Mean) / F_std
    if var == "PFI":
        x1 = (P - P_Mean) / P_std
        x2 = (F - F_Mean) / F_std
        x3 = (I - I_Mean) / I_std
    if var == "IPF":
        x1 = (I - I_Mean) / I_std
        x2 = (P - P_Mean) / P_std
        x3 = (F - F_Mean) / F_std
    if var == "IFP":
        x1 = (I - I_Mean) / I_std
        x2 = (F - F_Mean) / F_std
        x3 = (P - P_Mean) / P_std
    if var == "FPI":
        x1 = (F - F_Mean) / F_std
        x2 = (P - P_Mean) / P_std
        x3 = (I - I_Mean) / I_std
    if var == "FIP":
        x1 = (F - F_Mean) / F_std
        x2 = (I - I_Mean) / I_std
        x3 = (P - P_Mean) / P_std

    C_th = np.zeros(Case)
    D_th = np.zeros(Case)
    for i in range(Case):
        C_th[i] = x[0]
        D_th[i] = x[1]
        if n_var >= 1:
            C_th[i] = C_th[i] + x[2] * x1[i]
            D_th[i] = D_th[i] + x[3] * x1[i]
        if n_var >= 2:
            C_th[i] = C_th[i] + x[4] * x2[i]
            D_th[i] = D_th[i] + x[5] * x2[i]
        if n_var >= 3:
            C_th[i] = C_th[i] + x[6] * x3[i]
            D_th[i] = D_th[i] + x[7] * x3[i]

    C_th = Mean_C + Std_C * C_th
    D_th = Mean_D + Std_D * D_th
    N_th = 1 - C_th - D_th

    return C_th, N_th, D_th


# Compute parameters by minimizing the error
if True:
    res = minimize(error, x0, method="BFGS", tol=1e-8)
    Param = res.x
    print(res)
    print("***************************************************************")
    print(var, f"({n_var} variables);     Weights = ", RelativeError)
    print("***************************************************************")
    print("Error =", np.sqrt(error(Param)))
    C_th, N_th, D_th = Prediction(Param)

    print("C parameters:", Param[i_min::2])
    print("D parameters:", Param[i_min + 1 :: 2])

    Mean_C_th = np.mean(C_th[i_min:])
    Mean_D_th = np.mean(D_th[i_min:])
    Mean_N_th = np.mean(N_th[i_min:])
    Std_C_th = np.std(C_th[i_min:])
    Std_D_th = np.std(D_th[i_min:])
    Std_N_th = np.std(N_th[i_min:])
    print("<C> (th/exp) =", Mean_C_th, Mean_C)
    print("<N> (th/exp) =", Mean_N_th, Mean_N)
    print("<D> (th/exp) =", Mean_D_th, Mean_D)
    print("STD C (th/exp) =", Std_C_th, Std_C)
    print("STD N (th/exp) =", Std_N_th, Std_N)
    print("STD D (th/exp) =", Std_D_th, Std_D)
    print("Norme C/N/D (exp)=", C_norm, D_norm, N_norm)


# Plots of the quality of the fit
if True:
    plt.close("all")
    Size = 65
    L = 5
    Width = 1.5
    labelsize = 18
    ticksize = 14
    legend1size = 10
    legend2size = 10
    x_min = 0.1
    x_max = 0.65
    marker_N = "o"

    Col = [""] * Case
    Col[0] = COLOR_REF = "#8e8e85"
    Col[1] = COLOR_4COL_0DEF = "#0CC122"
    Col[2] = COLOR_3COL_1DEF = "#91B955"
    Col[3] = COLOR_2COL_2DEF = "#E8B27D"
    Col[4] = COLOR_1COL_3DEF = "#F06E50"
    Col[5] = COLOR_0COL_4DEF = "#F41818"
    Col[6] = COLOR_CONST1 = "#64BFFF"
    Col[7] = COLOR_CONST3 = "#3264AA"
    Col[8] = COLOR_CONST5 = "#030059"
    Col[9] = COLOR_OPT = "#FF7CF7"
    if n_var == 1:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(L * 2, L),
            # layout="constrained"
        )
    elif n_var == 2:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(L * 3.2, L),
            # ↨layout="constrained"
        )
    else:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=4,
            figsize=(L * 4.3, L),
            # layout="constrained"
        )
    for ax in axs.flat:
        ax.set_box_aspect(1)

    for ax in axs.flat:
        ax.tick_params(axis="both", labelsize=ticksize)

    axs[0].set_aspect("equal")
    axs[0].set_xlabel("Experimental %", fontsize=labelsize)
    axs[0].set_ylabel(f"Predicted % ({name_var})", fontsize=labelsize)
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(x_min, x_max)

    axs[0].scatter(0, 0, color="black", marker="^", s=Size)
    axs[0].scatter(0, 0, color="black", marker=marker_N, s=Size)
    axs[0].scatter(0, 0, color="black", marker="v", s=Size)
    legend_profiles = axs[0].legend(["Collaborators", "Neutrals", "Defectors"], loc="lower right", fontsize=legend2size)
    axs[1].scatter(0, 0, color="black", marker="s", s=Size)
    axs[1].scatter(0, 0, color="white", edgecolor="black", marker="s", s=Size)
    legend_profiles = axs[0].legend(["Collaborators", "Neutrals", "Defectors"], loc="lower right", fontsize=legend2size)
    legend_exp = axs[1].legend(["Model", "Experiment"], loc="upper right", fontsize=legend2size)

    axs[0].scatter(C, C_th, color=Col, marker="^", s=Size)
    axs[0].scatter(N, N_th, color=Col, marker=marker_N, s=Size)
    axs[0].scatter(D, D_th, color=Col, marker="v", s=Size)

    if n_var == 1:
        # axs[1].scatter(P, C, color = "w", edgecolors = Col, marker="^", s=Size, lw=Width)
        # axs[1].scatter(P, N, color = "w", edgecolors = Col, marker="s", s=Size, lw=Width)
        # axs[1].scatter(P, D, color = "w", edgecolors = Col, marker="v", s=Size, lw=Width)
        # axs[1].scatter(P, C_th, color = Col, marker="^", s=Size)
        # axs[1].scatter(P, N_th, color = Col, marker=marker_N, s=Size)
        # axs[1].scatter(P, D_th, color = Col, marker="v", s=Size)
        # axs[1].set_ylabel("Predicted & Experimental %", fontsize = labelsize)
        # axs[1].set_xlabel(r"$P(20)$", fontsize = labelsize)
        # axs[1].set_ylim(x_min,x_max)

        axs[1].scatter(I, C, color="w", edgecolors=Col, marker="^", s=Size, lw=Width)
        axs[1].scatter(I, N, color="w", edgecolors=Col, marker="s", s=Size, lw=Width)
        axs[1].scatter(I, D, color="w", edgecolors=Col, marker="v", s=Size, lw=Width)
        axs[1].scatter(I, C_th, color=Col, marker="^", s=Size)
        axs[1].scatter(I, N_th, color=Col, marker=marker_N, s=Size)
        axs[1].scatter(I, D_th, color=Col, marker="v", s=Size)
        axs[1].set_ylabel("Predicted & Experimental %", fontsize=labelsize)
        axs[1].set_xlabel(r"$\mathrm{IPR}(\mathbf{P}(20))$", fontsize=labelsize)
        axs[1].set_ylim(x_min, x_max)

        # axs[1].scatter(F, C, color = "w", edgecolors = Col, marker="^", s=Size, lw=Width)
        # axs[1].scatter(F, N, color = "w", edgecolors = Col, marker="s", s=Size, lw=Width)
        # axs[1].scatter(F, D, color = "w", edgecolors = Col, marker="v", s=Size, lw=Width)
        # axs[1].scatter(F, C_th, color = Col, marker="^", s=Size)
        # axs[1].scatter(F, N_th, color = Col, marker=marker_N, s=Size)
        # axs[1].scatter(F, D_th, color = Col, marker="v", s=Size)
        # axs[1].set_ylabel("Predicted & Experimental %", fontsize = labelsize)
        # axs[1].set_xlabel(r"F($P(20)$)", fontsize = labelsize)
        # axs[1].set_ylim(x_min,x_max)

    if n_var == 2:
        axs[1].scatter(P, C, color="w", edgecolors=Col, marker="^", s=Size, lw=Width)
        axs[1].scatter(P, N, color="w", edgecolors=Col, marker=marker_N, s=Size, lw=Width)
        axs[1].scatter(P, D, color="w", edgecolors=Col, marker="v", s=Size, lw=Width)
        axs[2].scatter(I, C, color="w", edgecolors=Col, marker="^", s=Size, lw=Width)
        axs[2].scatter(I, N, color="w", edgecolors=Col, marker=marker_N, s=Size, lw=Width)
        axs[2].scatter(I, D, color="w", edgecolors=Col, marker="v", s=Size, lw=Width)

        axs[1].scatter(P, C_th, color=Col, marker="^", s=Size)
        axs[1].scatter(P, N_th, color=Col, marker=marker_N, s=Size)
        axs[1].scatter(P, D_th, color=Col, marker="v", s=Size)
        axs[2].scatter(I, C_th, color=Col, marker="^", s=Size)
        axs[2].scatter(I, N_th, color=Col, marker=marker_N, s=Size)
        axs[2].scatter(I, D_th, color=Col, marker="v", s=Size)

        axs[1].set_ylabel("Predicted & Experimental %", fontsize=labelsize)
        axs[1].set_xlabel(r"$P(20)$", fontsize=labelsize)
        axs[2].set_xlabel(r"$\mathrm{IPR}(\mathbf{P}(20))$", fontsize=labelsize)
        axs[1].set_ylim(x_min, x_max)
        axs[2].set_ylim(x_min, x_max)

    if n_var > 2:
        axs[1].scatter(P, C, color="w", edgecolors=Col, marker="^", s=Size, lw=Width)
        axs[1].scatter(P, N, color="w", edgecolors=Col, marker=marker_N, s=Size, lw=Width)
        axs[1].scatter(P, D, color="w", edgecolors=Col, marker="v", s=Size, lw=Width)
        axs[2].scatter(I, C, color="w", edgecolors=Col, marker="^", s=Size, lw=Width)
        axs[2].scatter(I, N, color="w", edgecolors=Col, marker=marker_N, s=Size, lw=Width)
        axs[2].scatter(I, D, color="w", edgecolors=Col, marker="v", s=Size, lw=Width)
        axs[3].scatter(F, C, color="w", edgecolors=Col, marker="^", s=Size, lw=Width)
        axs[3].scatter(F, N, color="w", edgecolors=Col, marker=marker_N, s=Size, lw=Width)
        axs[3].scatter(F, D, color="w", edgecolors=Col, marker="v", s=Size, lw=Width)

        axs[1].scatter(P, C_th, color=Col, marker="^", s=Size)
        axs[1].scatter(P, N_th, color=Col, marker=marker_N, s=Size)
        axs[1].scatter(P, D_th, color=Col, marker="v", s=Size)
        axs[2].scatter(I, C_th, color=Col, marker="^", s=Size)
        axs[2].scatter(I, N_th, color=Col, marker=marker_N, s=Size)
        axs[2].scatter(I, D_th, color=Col, marker="v", s=Size)
        axs[3].scatter(F, C_th, color=Col, marker="^", s=Size)
        axs[3].scatter(F, N_th, color=Col, marker=marker_N, s=Size)
        axs[3].scatter(F, D_th, color=Col, marker="v", s=Size)

        axs[1].set_ylabel("Predicted & Experimental %", fontsize=labelsize)
        axs[1].set_xlabel(r"$P(20)$", fontsize=labelsize)
        axs[2].set_xlabel(r"$\mathrm{IPR}(\mathbf{P}(20))$", fontsize=labelsize)
        axs[3].set_xlabel(r"Rank", fontsize=labelsize)
        axs[1].set_ylim(x_min, x_max)
        axs[2].set_ylim(x_min, x_max)
        axs[3].set_ylim(x_min, x_max)

    legend_case = axs[0].legend(
        handles=[
            Patch(facecolor=Col[0], label="Humans"),
            Patch(facecolor=Col[1], label="4 Col-0 Def"),
            Patch(facecolor=Col[2], label="3 Col-1 Def"),
            Patch(facecolor=Col[3], label="2 Col-2 Def"),
            Patch(facecolor=Col[4], label="1 Col-3 Def"),
            Patch(facecolor=Col[5], label="0 Col-4 Def"),
            Patch(facecolor=Col[6], label="Neu-1"),
            Patch(facecolor=Col[7], label="Neu-3"),
            Patch(facecolor=Col[8], label="Neu-5"),
            Patch(facecolor=Col[9], label="Opt"),
        ],
        loc="upper left",
        fontsize=legend1size,
        handlelength=1,
        handleheight=1,
    )
    axs[0].add_artist(legend_profiles)

    axs[0].plot([x_min, x_max], [x_min, x_max], color="black", linestyle="--")
    plt.tight_layout()
    if n_var == 1:
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
    if n_var == 2:
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
    if n_var > 2:
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
    # plt.tight_layout()

    if n_var == 2 and var == "PIF":
        plt.savefig("Figure_PI.pdf", format="pdf")
    elif n_var == 3 and var == "PIF":
        plt.savefig("Figure_PIF.pdf", format="pdf")
    else:
        plt.savefig(var + "-" + str(n_var) + ".pdf", format="pdf")


# Color maps
if n_var == 2 and var == "PIF":

    def S(u, a):
        s = 0.5 * np.tanh(a * (2 * u - 1)) / np.tanh(a) + 0.5
        return s

    def Profiles(P0, I0):
        x1 = (P0 - P_Mean) / P_std
        x2 = (I0 - I_Mean) / I_std
        C_f = Param[0] + Param[2] * x1 + Param[4] * x2
        D_f = Param[1] + Param[3] * x1 + Param[5] * x2

        C_f = Mean_C + Std_C * C_f
        D_f = Mean_D + Std_D * D_f
        N_f = 1 - C_f - D_f
        return C_f, N_f, D_f

    # Plot parameters
    a = 0.001  # Nonlinearity of the S function
    Grid = 500
    P_min = 0.05
    P_max = 0.65
    I_min = 3
    I_max = 55

    labelsize = 13
    titlesize = 16
    legendsize = 16
    ticksize = 13
    Contour = 1
    color_contour = "white"
    Manual_Contour = False
    lc = 1
    Show_Error = 0
    error_color = "yellow"
    Err = 25
    shrink_cbar = 1
    Size = 60
    mask_black = 1
    # æshader ="gouraud"
    shader = "auto"
    edge = "none"

    # Create the mesh
    x = np.linspace(P_min, P_max, Grid)  # Generate N equally spaced values between 0 and 1
    y = np.linspace(I_min, I_max, Grid)  # Generate N equally spaced values between 0 and 1
    X, Y = np.meshgrid(x, y)

    # Create the collaborator/defectors/neutrals maps
    C_map, N_map, D_map = Profiles(X, Y)

    C_min = C_map.min()
    C_max = C_map.max()
    N_min = N_map.min()
    N_max = N_map.max()
    D_min = D_map.min()
    D_max = D_map.max()
    print(f"Plotting color map for IPR/P in [{I_min},{I_max}] x [{P_min},{P_max}]")
    print("C min/max", C_min, C_max)
    print("C min/max", C_min, C_max)
    print("N min/max", N_min, N_max)
    print("N min/max", N_min, N_max)
    print("D min/max", D_min, D_max)
    print("D min/max", D_min, D_max)
    if C_min < 0 or C_max > 1:
        print("Alert C not in [0, 1]!")
    if N_min < 0 or N_max > 1:
        print("Alert N not in [0, 1]!")
    if D_min < 0 or D_max > 1:
        print("Alert D not in [0, 1]!")

    Err_C = np.abs((C_th - C)) * Err
    Err_N = np.abs((N_th - N)) * Err
    Err_D = np.abs((D_th - D)) * Err
    if Show_Error == 0:
        error_color = "black"

    # Plot the maps
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(L * 3.3, L * 0.8),
        # layout="constrained"
    )
    for ax in axs.flat:
        ax.set_box_aspect(1)
        ax.set_xlim(P_min, P_max)
        ax.set_ylim(I_min, I_max)
        ax.set_xlabel("$P(20)$", fontsize=titlesize)
        ax.set_ylabel(r"$\mathrm{IPR}(\mathbf{P}(20))$", fontsize=titlesize)
        ax.tick_params(labelsize=ticksize)

    N_map_Contour = N_map.copy()
    # Mask the region where the predicted C, D, N are <0 or >1
    mask = (C_map < 0) | (C_map > 1) | (N_map < 0) | (N_map > 1) | (D_map < 0) | (D_map > 1)

    alpha_mask = np.zeros((Grid, Grid))
    alpha_mask[mask] = 1

    if mask_black == 0:
        C_map[mask] = 0
        N_map[mask] = 0
        D_map[mask] = 0
    elif mask_black == 1:
        mask_C = C_map < 0
        mask_D = D_map < 0
        C_map[mask_C] = 0
        N_map[mask_C] = 1
        D_map[mask_C] = 1
        C_map[mask_D] = 1
        N_map[mask_D] = 1
        D_map[mask_D] = 0
    else:
        C_map = np.clip(C_map, 0, 1)
        N_map = np.clip(N_map, 0, 1)
        D_map = np.clip(D_map, 0, 1)

    # Create a color plot for collaborators with custom colormap (last is entry alpha)
    C_min_Plot = 0
    C_max_Plot = 0.6
    # colors_green = [(1, 1, 1), (0, 0.8, 0, 0.8), (0, 0.5, 0, 0.5), (0, 0, 0)]
    colors_green = [(1, 1, 1), (0, 0.8, 0, 1), (0, 0, 0)]
    green_cmap = LinearSegmentedColormap.from_list("custom_green", colors_green, N=256)
    cmap0 = axs[0].pcolormesh(
        X, Y, C_map, shading=shader, cmap=green_cmap, vmin=C_min_Plot, vmax=C_max_Plot, edgecolor=edge
    )
    cbar0 = plt.colorbar(cmap0, ax=axs[0], ticks=np.linspace(0, 1, 11), shrink=shrink_cbar)
    cbar0.ax.tick_params(labelsize=labelsize)
    cbar0.set_label("% Collaborators", fontsize=legendsize)

    # Create a color plot for neutrals with custom colormap
    N_min_Plot = 0.3
    N_max_Plot = 0.6
    colors_maroon = [(1, 1, 1), (0.6, 0.3, 0, 1), (0, 0, 0)]
    maroon_cmap = LinearSegmentedColormap.from_list("custom_maroon", colors_maroon, N=256)
    cmap1 = axs[1].pcolormesh(
        X, Y, N_map, shading=shader, cmap=maroon_cmap, vmin=N_min_Plot, vmax=N_max_Plot, edgecolor=edge
    )
    cbar1 = plt.colorbar(cmap1, ax=axs[1], ticks=np.linspace(0, 1, 11), shrink=shrink_cbar)
    cbar1.ax.tick_params(labelsize=labelsize)
    cbar1.set_label("% Neutrals", fontsize=legendsize)

    # Create a color plot for defectors with custom colormap
    D_min_Plot = 0
    D_max_Plot = 0.6
    colors_red = [(1, 1, 1), (1, 0, 0, 1), (0, 0, 0)]
    red_cmap = LinearSegmentedColormap.from_list("custom_red", colors_red, N=256)
    cmap2 = axs[2].pcolormesh(
        X, Y, D_map, shading=shader, cmap=red_cmap, vmin=D_min_Plot, vmax=D_max_Plot, edgecolor=edge
    )
    cbar2 = plt.colorbar(cmap2, ax=axs[2], ticks=np.linspace(0, 1, 11), shrink=shrink_cbar)
    cbar2.ax.tick_params(labelsize=labelsize)
    cbar2.set_label("% Defectors", fontsize=legendsize)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.17, right=0.95, top=0.85, bottom=0.15, wspace=0.3)

    # Add the legend
    legend_case = axs[0].legend(
        handles=[
            Patch(facecolor=Col[0], label="Ref", edgecolor=error_color),
            Patch(facecolor=Col[1], label="4 Col – 0 Def", edgecolor=error_color),
            Patch(facecolor=Col[2], label="3 Col – 1 Def", edgecolor=error_color),
            Patch(facecolor=Col[3], label="2 Col – 2 Def", edgecolor=error_color),
            Patch(facecolor=Col[4], label="1 Col – 3 Def", edgecolor=error_color),
            Patch(facecolor=Col[5], label="0 Col – 4 Def", edgecolor=error_color),
            Patch(facecolor=Col[6], label="Const-1", edgecolor=error_color),
            Patch(facecolor=Col[7], label="Const-3", edgecolor=error_color),
            Patch(facecolor=Col[8], label="Const-5", edgecolor=error_color),
            Patch(facecolor=Col[9], label="Opt", edgecolor=error_color),
        ],
        bbox_to_anchor=(-0.35, 1.04),
        fontsize=13.5,
        handlelength=1,
        handleheight=1,
    )

    # Add the positions of the experimental data
    if Show_Error == 1:
        axs[0].scatter(P, I, color=Col, edgecolors=error_color, lw=Err_C, marker="s", s=Size, alpha=1, zorder=9)
        axs[1].scatter(P, I, color=Col, edgecolors=error_color, lw=Err_D, marker="s", s=Size, alpha=1, zorder=9)
        axs[2].scatter(P, I, color=Col, edgecolors=error_color, lw=Err_N, marker="s", s=Size, alpha=1, zorder=9)
    else:
        for ax in axs.flat:
            ax.scatter(P, I, color=Col, edgecolors="black", marker="s", s=Size, alpha=1, zorder=9)

    if Contour == 1:
        contour_levels = [0.15, 0.3, 0.45]
        contour = axs[0].contour(
            X,
            Y,
            C_map,
            levels=contour_levels,
            colors=color_contour,
            linestyles="dashed",
            alpha=1,
            linewidths=[lc, lc, lc],
        )
        contour_labels = axs[0].clabel(contour, fontsize=10, inline=True, fmt="%1.2f", manual=Manual_Contour)

        contour_levels = [0.4, 0.45, 0.5]
        contour = axs[1].contour(
            X,
            Y,
            N_map_Contour,
            levels=contour_levels,
            colors=color_contour,
            linestyles="dashed",
            alpha=1,
            linewidths=[lc, lc, lc],
        )
        contour_labels = axs[1].clabel(contour, fontsize=10, inline=True, fmt="%1.2f", manual=Manual_Contour)
        cmap1bis = axs[1].pcolormesh(
            X,
            Y,
            alpha_mask,
            shading=shader,
            cmap=maroon_cmap,
            vmin=N_min_Plot,
            vmax=N_max_Plot,
            edgecolor=edge,
            alpha=alpha_mask,
            zorder=10,
        )

        contour_levels = [0.15, 0.3, 0.45]
        contour = axs[2].contour(
            X,
            Y,
            D_map,
            levels=contour_levels,
            colors=color_contour,
            linestyles="dashed",
            alpha=1,
            linewidths=[lc, lc, lc],
        )
        contour_labels = axs[2].clabel(contour, fontsize=10, inline=True, fmt="%1.2f", manual=Manual_Contour)

    plt.show()

    # Save the plot of the color maps
    if Contour == 1:
        if Show_Error == 1:
            plt.savefig("Figure_Color-Map-Contour-Err.pdf", format="pdf")
        else:
            plt.savefig("Figure_Color-Map-Contour.pdf", format="pdf")
    else:
        if Show_Error == 1:
            plt.savefig("Figure_Color-Map-Err.pdf", format="pdf")
        else:
            plt.savefig("Figure_Color-Map.pdf", format="pdf")
    # plt.savefig("Figure_Color-Map.jpg", format='jpg', dpi=300)
