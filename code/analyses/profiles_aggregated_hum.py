import numpy as np
from modules.player import get_players_games


def main(path_data, path_data_figures, session_type, game_type, bootstrap_reps):
    thresholds = (-0.5, 0.5)
    players = get_players_games(path_data / session_type / game_type, game_type, aggregate=True, classify=thresholds)
    players_hum = [player for player in players if not player.bot]

    players_type = [player.aggregated_game["type"] for player in players_hum]

    mean, err = bootstrap(players_type, bootstrap_reps)

    print(game_type)
    for i, type_ in enumerate(["col", "neu", "def"]):
        print(f"{type_}: {round(mean[i]):2}")
        # print(f"{type_}: {mean[i] * 100:3.0f} ± {np.mean(err[:, i]) * 100:2.0f} %")
    print()


def bootstrap(players_type, bootstrap_reps):
    bs_counts = []

    for _ in range(bootstrap_reps):
        bs_players_type = np.random.choice(players_type, replace=True, size=len(players_type))

        types = ["col", "neu", "def"]
        counts = np.array([np.count_nonzero(bs_players_type == type_) for type_ in types])
        bs_counts.append(counts)
        # bs_counts.append(counts / len(bs_players_type))

    mean = np.mean(bs_counts, axis=0)
    err = np.abs(np.percentile(bs_counts, [50 - 34.13, 50 + 34.13], axis=0) - mean)

    return mean, err


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES

    bootstrap_reps = 1000

    main(PATH_DATA, PATH_DATA_FIGURES, "vs_col_def", "0_col_4_def", bootstrap_reps)
    main(PATH_DATA, PATH_DATA_FIGURES, "vs_col_def", "1_col_3_def", bootstrap_reps)
    main(PATH_DATA, PATH_DATA_FIGURES, "vs_col_def", "2_col_2_def", bootstrap_reps)
    main(PATH_DATA, PATH_DATA_FIGURES, "vs_col_def", "3_col_1_def", bootstrap_reps)
    main(PATH_DATA, PATH_DATA_FIGURES, "vs_col_def", "4_col_0_def", bootstrap_reps)

    main(PATH_DATA, PATH_DATA_FIGURES, "vs_const", "4_const1", bootstrap_reps)
    main(PATH_DATA, PATH_DATA_FIGURES, "vs_const", "4_const3", bootstrap_reps)
    main(PATH_DATA, PATH_DATA_FIGURES, "vs_const", "4_const5", bootstrap_reps)

    main(PATH_DATA, PATH_DATA_FIGURES, "vs_opt", "4_opt", bootstrap_reps)
