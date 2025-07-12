from itertools import chain, combinations

import numpy as np
from scipy.optimize import minimize

COLOR_REF = "grey"
COLOR_4COL_0DEF = "#0CC122"
COLOR_3COL_1DEF = "#91B955"
COLOR_2COL_2DEF = "#E8B27D"
COLOR_1COL_3DEF = "#F06E50"
COLOR_0COL_4DEF = "#F41818"
COLOR_CONST1 = "#64BFFF"
COLOR_CONST3 = "#3264AA"
COLOR_CONST5 = "#030059"
COLOR_OPT = "#FF7CF7"

np.set_printoptions(suppress=True)


def main(path_data_figures, experiments):
    observables = ["F", "F'", "P", "I", "R"]
    for observables_names in chain(
        combinations(observables, 1),
        combinations(observables, 2),
        combinations(observables, 3),
    ):
        X = get_parameters_model(path_data_figures, experiments, observables_names)

        observables, C_exp, N_exp, D_exp = get_observables(path_data_figures, experiments)
        observables_values = np.array([observables[obs_name] for obs_name in observables_names])
        E = error(X, observables_values, C_exp, N_exp, D_exp)

        print(f"({', '.join(observables_names)}) E: {E:.3f}")
        print(f"    {X[: len(X) // 2]}")
        print(f"    {X[len(X) // 2 :]}")

        observables_std = [standardize(obs) for obs in observables_values]
        C_theo = model(observables_std, X[: len(X) // 2]) * np.std(C_exp) + np.mean(C_exp)
        D_theo = model(observables_std, X[len(X) // 2 :]) * np.std(D_exp) + np.mean(D_exp)
        N_theo = 1 - C_theo - D_theo

        for (session_type, game_type, _), c, n, d in zip(experiments, C_theo, N_theo, D_theo):
            np.savetxt(
                path_data_figures / session_type / game_type / f"model/pred_profile/{'_'.join(observables_names)}.txt",
                [c, n, d],
                fmt="%f",
            )


def get_parameters_model(path_data_figures, experiments, observables_names):
    observables, C_exp, N_exp, D_exp = get_observables(path_data_figures, experiments)
    observables_values = np.array([observables[obs_name] for obs_name in observables_names])
    res = minimize(
        error,
        np.ones(2 * (1 + len(observables_names))),
        args=(observables_values, C_exp, N_exp, D_exp),
        method="BFGS",
        tol=1e-8,
    )
    return res.x


def get_observables(path_data_figures, experiments):
    observables = {
        "F": [],
        "F'": [],
        "P": [],
        "I": [],
        "R": [],
    }
    C_exp = []
    N_exp = []
    D_exp = []
    for session_type, game_type, _ in experiments:
        _, mean, *_ = np.loadtxt(path_data_figures / session_type / game_type / "exp/observables/F_P.txt", unpack=True)
        observables["F"].append(mean[-1])

        _, mean, *_ = np.loadtxt(path_data_figures / session_type / game_type / "exp/observables/F'_P.txt", unpack=True)
        observables["F'"].append(mean[-1])

        _, mean, *_ = np.loadtxt(path_data_figures / session_type / game_type / "exp/observables/P.txt", unpack=True)
        observables["P"].append(mean[-1])

        _, mean, *_ = np.loadtxt(
            path_data_figures / session_type / game_type / "exp/observables/IPR_P.txt", unpack=True
        )
        observables["I"].append(mean[-1])

        mean = np.loadtxt(
            path_data_figures / session_type / game_type / "exp/observables_hum/mean_rank.txt", unpack=True
        )
        observables["R"].append(mean[0])

        fraction_col, fraction_neu, fraction_def = np.loadtxt(
            path_data_figures / session_type / game_type / "model/parameters/players_profiles.txt"
        )
        C_exp.append(fraction_col)
        N_exp.append(fraction_neu)
        D_exp.append(fraction_def)

    observables = {k: np.array(v) for k, v in observables.items()}
    C_exp = np.array(C_exp)
    N_exp = np.array(N_exp)
    D_exp = np.array(D_exp)

    return observables, C_exp, N_exp, D_exp


def standardize(X):
    return (X - np.mean(X)) / np.std(X)


def model(observables, X):
    return X[0] + np.dot(X[1:], observables)


def error(X, observables, C_exp, N_exp, D_exp):
    observables_std = [standardize(obs) for obs in observables]

    C_theo = model(observables_std, X[: len(X) // 2]) * np.std(C_exp) + np.mean(C_exp)
    D_theo = model(observables_std, X[len(X) // 2 :]) * np.std(D_exp) + np.mean(D_exp)
    N_theo = 1 - C_theo - D_theo

    return np.sqrt(
        np.sum((C_exp - C_theo) ** 2 + (N_exp - N_theo) ** 2 + (D_exp - D_theo) ** 2)
        / np.sum(C_exp**2 + N_exp**2 + D_exp**2)
    )


if __name__ == "__main__":
    from modules.constants import PATH_DATA_FIGURES

    main(
        PATH_DATA_FIGURES,
        [
            # Ref
            ("R2_intra/", "R2_intra", COLOR_REF),
            # vs col-def
            ("R2_vs_col_def/4_col_0_def/", "4_col_0_def", COLOR_4COL_0DEF),
            ("R2_vs_col_def/3_col_1_def/", "3_col_1_def", COLOR_3COL_1DEF),
            ("R2_vs_col_def/2_col_2_def/", "2_col_2_def", COLOR_2COL_2DEF),
            ("R2_vs_col_def/1_col_3_def/", "1_col_3_def", COLOR_1COL_3DEF),
            ("R2_vs_col_def/0_col_4_def/", "0_col_4_def", COLOR_0COL_4DEF),
            # vs const
            ("R2_vs_const/4_const1/", "4_const1", COLOR_CONST1),
            ("R2_vs_const/4_const3/", "4_const3", COLOR_CONST3),
            ("R2_vs_const/4_const5/", "4_const5", COLOR_CONST5),
            # vs opt
            ("R2_vs_opt/4_opt/", "4_opt", COLOR_OPT),
        ],
    )
