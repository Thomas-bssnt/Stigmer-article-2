import numpy as np
from modules.player import get_players_games


def main(path_data, experiments):

    genders = []
    ages = []

    for session_type, game_type in experiments:

        players = get_players_games(path_data / session_type, game_type, classify=(-0.5, 0.5))

        genders += [player.gender for player in players if not player.bot]
        ages += [player.age for player in players if not player.bot]

    print(f"Male: {genders.count('male')}")
    print(f"Female: {genders.count('female')}")
    print(f"Mean age: {np.mean(ages)}")
    print()


if __name__ == "__main__":
    from modules.constants import PATH_DATA

    main(
        PATH_DATA,
        [
            # vs col-def
            ("R2_vs_col_def/0_col_4_def/", "0_col_4_def"),
            ("R2_vs_col_def/1_col_3_def/", "1_col_3_def"),
            ("R2_vs_col_def/2_col_2_def/", "2_col_2_def"),
            ("R2_vs_col_def/3_col_1_def/", "3_col_1_def"),
            ("R2_vs_col_def/4_col_0_def/", "4_col_0_def"),
            # vs const
            ("R2_vs_const/4_const1/", "4_const1"),
            ("R2_vs_const/4_const3/", "4_const3"),
            ("R2_vs_const/4_const5/", "4_const5"),
            # vs opt
            ("R2_vs_opt/4_opt/", "4_opt"),
        ],
    )

    main(
        PATH_DATA,
        [
            ("R2_intra", "R2_intra"),
        ],
    )
