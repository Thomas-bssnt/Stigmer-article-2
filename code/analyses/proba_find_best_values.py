import numpy as np
from modules.games import Games


def main(path_data, path_data_figures, session_type, game_type, player_type, bootstrap_reps):
    for values in [[99], [86, 85, 84], [72, 71], [99, 86, 85, 84, 72, 71]]:
        games = Games(path_data / session_type, game_type, player_type=player_type)

        probability_finding_value = []
        for games_session in games.session:
            probability_finding_value_session = []
            for game in games_session:
                for value in values:
                    probability_finding_value_session += get_probability_finding_value(game, value, player_type)
            probability_finding_value.append(np.mean(probability_finding_value_session, axis=0))

        mean, err = bootstrap(probability_finding_value, bootstrap_reps)

        obs_folder = "observables" if player_type is None else f"observables_{player_type}"
        path_folder = path_data_figures / session_type / game_type / "exp" / obs_folder

        np.savetxt(
            path_folder / f"proba_find_{'_'.join([str(value) for value in values])}.txt",
            np.column_stack(
                (
                    np.arange(1, len(mean) + 1),
                    mean,
                    err.T,
                )
            ),
            fmt=("%d", "%f", "%f", "%f"),
        )


def get_probability_finding_value(game, value, player_type):
    findings = {index: {} for index, v in enumerate(game.V) if v == value}

    for player_id in game.index_cells_played_players.keys():
        for round_ in range(game.data["numberRounds"]):
            for index, v in zip(
                game.index_cells_played_players[player_id][round_],
                game.value_cells_played_players[player_id][round_],
            ):
                if v == value:
                    if player_id not in findings[index]:
                        findings[index][player_id] = round_

    probas = []
    for findings_cell in findings.values():
        probability_finding_value = np.zeros(game.data["numberRounds"])
        for round_ in findings_cell.values():
            probability_finding_value[round_ - 1 :] += 1
        n_player = len(game.index_cells_played_players.keys())
        probability_finding_value /= n_player
        probas.append(probability_finding_value)
    return probas


def bootstrap(observable_list, bootstrap_reps):
    bs_means = []
    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(range(len(observable_list)), replace=True, size=len(observable_list))
        bs_sample = [observable_list[i] for i in bs_indices]
        bs_means.append(np.mean(bs_sample, axis=0))
    mean = np.mean(bs_means, axis=0)
    err = np.abs(np.percentile(bs_means, [50 - 34.13, 50 + 34.13], axis=0) - mean)
    return mean, err


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES, experiments

    bootstrap_reps = 1000

    print("R2_intra")
    main(PATH_DATA, PATH_DATA_FIGURES, "R2_intra", "R2_intra", "hum", bootstrap_reps)

    for session_type, game_type in experiments:
        for player_type in [None, "hum", "bot"]:
            print(session_type, game_type, player_type)

            main(PATH_DATA, PATH_DATA_FIGURES, session_type, game_type, player_type, bootstrap_reps)
