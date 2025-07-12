from collections import defaultdict

from modules.player import get_players_games


def main(path_data, session_type, game_type):
    players = get_players_games(path_data / session_type, game_type, classify=(-0.5, 0.5))

    players = [player for player in players if not player.bot]

    genders = defaultdict(list)
    genders_all = []
    for player in players:
        for game in player.games.values():
            genders[game["profile"]].append(player.gender)
        genders_all.append(player.gender)

    print(f"{session_type}:")
    print(
        f" - all: "
        f"H = {round(genders_all.count('male') / len(genders_all) * 100):2}% "
        f"F = {round(genders_all.count('female') / len(genders_all) * 100):2}%"
    )
    for profile in ["col", "neu", "def"]:
        print(
            f" - {profile}: "
            f"H = {round(genders[profile].count('male') / len(genders[profile]) * 100):2}% "
            f"F = {round(genders[profile].count('female') / len(genders[profile]) * 100):2}%"
        )


if __name__ == "__main__":
    from modules.constants import PATH_DATA, experiments

    main(PATH_DATA, "R2_intra", "R2_intra")
    for session_type, game_type in experiments:
        main(PATH_DATA, session_type, game_type)
