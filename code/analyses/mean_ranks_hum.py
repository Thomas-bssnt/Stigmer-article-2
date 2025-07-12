import numpy as np
from modules.player import get_players_games_sessions


def main(path_data, path_data_figures, session_type, game_type, bootstrap_reps):
    players = get_players_games_sessions(path_data / session_type, game_type)

    mean, err, mean_0_5, err_0_5, mean_6_10, err_6_10 = bootstrap(players, bootstrap_reps)

    np.savetxt(
        path_data_figures / session_type / game_type / "exp" / "observables_hum" / "rank_mean.txt",
        [np.concatenate(([mean], err))],
        fmt="%f",
    )
    np.savetxt(
        path_data_figures / session_type / game_type / "exp" / "observables_hum" / "rank_mean_0_5.txt",
        [np.concatenate(([mean_0_5], err_0_5))],
        fmt="%f",
    )
    np.savetxt(
        path_data_figures / session_type / game_type / "exp" / "observables_hum" / "rank_mean_6_10.txt",
        [np.concatenate(([mean_6_10], err_6_10))],
        fmt="%f",
    )


def bootstrap(players_grouped, bootstrap_reps):
    bs_props_hum = []
    bs_props_hum_0_5 = []
    bs_props_hum_6_10 = []

    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(len(players_grouped), replace=True, size=len(players_grouped))
        bs_players = [player for i in bs_indices for player in players_grouped[i] if player.bot is False]

        ranks = []
        ranks_0_5 = []
        ranks_6_10 = []
        for player in bs_players:
            for i, game in enumerate(player.games.values()):
                ranks.append(game["rank"])
                if i < 5:
                    ranks_0_5.append(game["rank"])
                else:
                    ranks_6_10.append(game["rank"])

        bs_props_hum.append(np.mean(ranks))
        bs_props_hum_0_5.append(np.mean(ranks_0_5))
        bs_props_hum_6_10.append(np.mean(ranks_6_10))

    mean = np.mean(bs_props_hum, axis=0)
    err = np.abs(np.percentile(bs_props_hum, [50 - 34.13, 50 + 34.13], axis=0) - mean)

    mean_0_5 = np.mean(bs_props_hum_0_5, axis=0)
    err_0_5 = np.abs(np.percentile(bs_props_hum_0_5, [50 - 34.13, 50 + 34.13], axis=0) - mean_0_5)

    mean_6_10 = np.mean(bs_props_hum_6_10, axis=0)
    err_6_10 = np.abs(np.percentile(bs_props_hum_6_10, [50 - 34.13, 50 + 34.13], axis=0) - mean_6_10)

    return mean, err, mean_0_5, err_0_5, mean_6_10, err_6_10


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES, experiments

    bootstrap_reps = 1000

    print("R2_intra")
    main(PATH_DATA, PATH_DATA_FIGURES, "R2_intra", "R2_intra", bootstrap_reps)

    for session_type, game_type in experiments:
        print(session_type, game_type)

        main(PATH_DATA, PATH_DATA_FIGURES, session_type, game_type, bootstrap_reps)
