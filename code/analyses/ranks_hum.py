import numpy as np
from modules.player import get_players_games_sessions


def main(path_data, path_data_figures, session_type, game_type, bootstrap_reps):
    players = get_players_games_sessions(path_data / session_type, game_type)

    mean, err = bootstrap(players, bootstrap_reps)

    np.savetxt(
        path_data_figures / session_type / game_type / "exp" / "observables_hum" / "rank.txt",
        np.column_stack((mean, err.T)),
        fmt="%f",
    )


def bootstrap(players_grouped, bootstrap_reps):
    bs_props_hum = []

    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(len(players_grouped), replace=True, size=len(players_grouped))
        bs_players = [player for i in bs_indices for player in players_grouped[i]]

        ranks = [[] for _ in range(5)]
        for player in bs_players:
            for game in player.games.values():
                ranks[game["rank"] - 1].append(player.bot)

        bs_props_hum.append([rank.count(False) / len(rank) for rank in ranks])

    mean = np.mean(bs_props_hum, axis=0)
    err = np.abs(np.percentile(bs_props_hum, [50 - 34.13, 50 + 34.13], axis=0) - mean)

    return mean, err


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES, experiments

    bootstrap_reps = 1000

    print("R2_intra")
    main(PATH_DATA, PATH_DATA_FIGURES, "R2_intra", "R2_intra", bootstrap_reps)

    for session_type, game_type in experiments:
        print(session_type, game_type)

        main(PATH_DATA, PATH_DATA_FIGURES, session_type, game_type, bootstrap_reps)
