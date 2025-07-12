import numpy as np
from modules.player import get_players_games


def main(path_data, path_data_figures, session_type, game_type, bootstrap_reps):
    thresholds = (-0.5, 0.5)
    players = get_players_games(path_data / session_type, game_type, classify=thresholds)
    players_hum = [player for player in players if not player.bot]

    players_type = [[game["profile"] for game in player.games.values()] for player in players_hum]
    max_games = max(map(len, players_type))
    players_type = np.array([sublist + [None] * (max_games - len(sublist)) for sublist in players_type])

    mean, err = bootstrap(players_type, bootstrap_reps)

    np.savetxt(
        path_data_figures / session_type / game_type / "exp/observables_hum/profiles.txt",
        np.column_stack((mean, err.T)),
        fmt="%f",
    )
    np.savetxt(
        path_data_figures / session_type / game_type / "model/parameters/players_profiles.txt",
        [mean],
        fmt="%f",
    )


def bootstrap(players_type, bootstrap_reps):
    bs_counts = []

    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(len(players_type), replace=True, size=len(players_type))
        bs_players_type = players_type[bs_indices]

        count_non_None = np.count_nonzero(bs_players_type != None, axis=0)
        types = ["col", "neu", "def"]
        counts = np.array([np.count_nonzero(bs_players_type == type_, axis=0) for type_ in types])
        bs_counts.append(np.sum(counts, axis=1) / np.sum(count_non_None))

    mean = np.mean(bs_counts, axis=0)
    err = np.abs(np.percentile(bs_counts, [50 - 34.13, 50 + 34.13], axis=0) - mean)

    return mean, err


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES, experiments

    bootstrap_reps = 1000

    print("R2_intra")
    main(PATH_DATA, PATH_DATA_FIGURES, "R2_intra", "R2_intra", bootstrap_reps)

    for session_type, game_type in experiments:
        print(session_type, game_type)

        main(PATH_DATA, PATH_DATA_FIGURES, session_type, game_type, bootstrap_reps)
