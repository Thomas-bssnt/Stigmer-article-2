from collections import defaultdict

import numpy as np
from modules.games import Games


def main(path_data, path_data_figures, session_type, game_type, player_type, bootstrap_reps):
    games = Games(path_data / session_type, game_type, player_type=player_type)

    obs_folder = "observables" if player_type is None else f"observables_{player_type}"
    path_folder = path_data_figures / session_type / game_type / "exp" / obs_folder

    observables = bootstrap(games, bootstrap_reps)
    for name, (mean, err) in observables.items():
        np.savetxt(
            path_folder / f"{name}.txt",
            np.column_stack(
                (
                    np.arange(1, len(mean) + 1),
                    mean,
                    err.T,
                )
            ),
            fmt=("%d", "%f", "%f", "%f"),
        )


def bootstrap(games, bootstrap_reps):
    bs_observables = defaultdict(list)
    for _ in range(bootstrap_reps):
        bs_games_groups = np.random.choice(list(games.session), replace=True, size=len(games.session))
        bs_games = [game for bs_games_group in bs_games_groups for game in bs_games_group]
        for observable in bs_games[0].observables.keys():
            bs_observables[observable].append(np.mean([game.observables[observable] for game in bs_games], axis=0))
    return {
        observable: (
            np.mean(bs_observables[observable], axis=0),
            np.abs(
                np.percentile(bs_observables[observable], [50 - 34.13, 50 + 34.13], axis=0)
                - np.mean(bs_observables[observable], axis=0)
            ),
        )
        for observable in bs_observables
    }


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES, experiments

    bootstrap_reps = 1000

    print("R2_intra")
    main(PATH_DATA, PATH_DATA_FIGURES, "R2_intra", "R2_intra", "hum", bootstrap_reps)

    for session_type, game_type in experiments:
        for player_type in [None, "hum", "bot"]:
            print(session_type, game_type, player_type)

            main(PATH_DATA, PATH_DATA_FIGURES, session_type, game_type, player_type, bootstrap_reps)
