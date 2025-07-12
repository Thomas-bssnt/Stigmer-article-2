import numpy as np
from modules.player import get_players_games


def main(path_data, path_data_figures, session_type, game_type, player_type):
    thresholds = (-0.5, 0.5)
    players = get_players_games(path_data / session_type, game_type, classify=thresholds)

    if player_type == "hum":
        filtered_players = [player for player in players if not player.bot]
    elif player_type == "bot":
        filtered_players = [player for player in players if player.bot]
    else:
        filtered_players = players

    params = sorted(
        [(game["u0"], game["u1"]) for player in filtered_players for game in player.games.values()],
        key=lambda x: x[1],
    )

    obs_folder = "observables" if player_type is None else f"observables_{player_type}"
    path_folder = path_data_figures / session_type / game_type / "exp" / obs_folder

    np.savetxt(
        path_folder / "u0_u1.txt",
        params,
        fmt=("%f", "%f"),
    )


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES, experiments

    print("R2_intra")
    main(PATH_DATA, PATH_DATA_FIGURES, "R2_intra", "R2_intra", "hum")

    for session_type, game_type in experiments:
        for player_type in [None, "hum", "bot"]:
            print(session_type, game_type, player_type)

            main(PATH_DATA, PATH_DATA_FIGURES, session_type, game_type, player_type)
