import numpy as np
from modules.constants import S_MAX
from modules.games import Games


def main(path_data, path_data_figures, session_type, game_type, player_type, bootstrap_reps):
    dx = 0.15
    n_bins = 25

    games = Games(path_data / session_type, game_type, player_type)
    x_values, player_scores_dict, team_scores_dict = bootstrap(games, bootstrap_reps, dx, n_bins)

    obs_folder = "observables" if player_type is None else f"observables_{player_type}"
    path_folder = path_data_figures / session_type / game_type / "exp" / obs_folder

    for filename, dict_ in (("S", player_scores_dict), ("S_group", team_scores_dict)):
        np.savetxt(
            path_folder / f"{filename}.txt",
            np.column_stack(
                (
                    x_values,
                    dict_["pdf"][0],
                    dict_["pdf"][1].T,
                )
            ),
            fmt=("%f", "%f", "%f", "%f"),
        )

    for type_ in ["mean", "median", "std"]:
        np.savetxt(
            path_folder / f"S_{type_}.txt",
            [[player_scores_dict[type_][0]] + list(player_scores_dict[type_][1])],
            fmt="%f",
        )
        np.savetxt(
            path_folder / f"S_group_{type_}.txt",
            [[team_scores_dict[type_][0]] + list(team_scores_dict[type_][1])],
            fmt="%f",
        )


def bootstrap(games, bootstrap_reps, dx, n_bins):
    scores_grouped = [
        [np.array(list(game.scores_R2.values())) / S_MAX for game in games_session] for games_session in games.session
    ]

    player_scores_pdf = []
    player_scores_means = []
    player_scores_medians = []
    player_scores_stds = []
    team_scores_pdf = []
    team_scores_means = []
    team_scores_medians = []
    team_scores_stds = []
    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(len(scores_grouped), replace=True, size=len(scores_grouped))
        bs_scores_grouped = [scores for i in bs_indices for scores in scores_grouped[i]]

        bs_player_scores = [score for scores in bs_scores_grouped for score in scores]
        bs_team_scores = [np.mean(scores) for scores in bs_scores_grouped]

        x_values, counts = get_hist(bs_player_scores, dx, n_bins)
        player_scores_pdf.append(counts)
        player_scores_means.append(np.mean(bs_player_scores))
        player_scores_medians.append(np.median(bs_player_scores))
        player_scores_stds.append(np.std(bs_player_scores))

        x_values, counts = get_hist(bs_team_scores, dx, n_bins)
        team_scores_pdf.append(counts)
        team_scores_means.append(np.mean(bs_team_scores))
        team_scores_medians.append(np.median(bs_team_scores))
        team_scores_stds.append(np.std(bs_team_scores))

    def _get_mean_err(list_):
        mean = np.mean(list_, axis=0)
        err = np.abs(np.percentile(list_, [50 - 34.13, 50 + 34.13], axis=0) - mean)
        return mean, err

    player_scores_dict = {
        "pdf": _get_mean_err(player_scores_pdf),
        "mean": _get_mean_err(player_scores_means),
        "median": _get_mean_err(player_scores_medians),
        "std": _get_mean_err(player_scores_stds),
    }
    team_scores_dict = {
        "pdf": _get_mean_err(team_scores_pdf),
        "mean": _get_mean_err(team_scores_means),
        "median": _get_mean_err(team_scores_medians),
        "std": _get_mean_err(team_scores_stds),
    }
    return x_values, player_scores_dict, team_scores_dict


def get_hist(list_, dx, n_bins):
    assert dx >= (1 - dx) / (n_bins - 1)  # to be sure that there is no gap

    x_values = np.linspace(0, 1, n_bins)

    Y = []
    for x in x_values:
        number_of_element = 0
        for element in list_:
            if x - dx / 2 < element <= x + dx / 2:
                number_of_element += 1
        Y.append(number_of_element / len(list_) / dx)

    return x_values, Y


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES, experiments

    bootstrap_reps = 1000

    print("R2_intra")
    main(PATH_DATA, PATH_DATA_FIGURES, "R2_intra", "R2_intra", "hum", bootstrap_reps)

    for session_type, game_type in experiments:
        for player_type in [None, "hum", "bot"]:
            print(session_type, game_type, player_type)

            main(PATH_DATA, PATH_DATA_FIGURES, session_type, game_type, player_type, bootstrap_reps)
