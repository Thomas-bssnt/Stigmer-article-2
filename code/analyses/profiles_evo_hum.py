from typing import DefaultDict

import matplotlib.pyplot as plt
import numpy as np
from modules.player import get_players_games


def main(path_data, session_type, game_type):
    players = get_players_games(path_data / session_type, game_type, classify=(-0.5, 0.5))
    # players = get_players_games(path_data / session_type, game_type, classify=(-0.5, 0.5), aggregate=True)
    players = [player for player in players if not player.bot]

    # agg_profiles = [player.aggregated_game["profile"] for player in players]
    # print(
    #     f"Aggregated profiles: "
    #     f"col: {agg_profiles.count('col')}/{len(players)}, "
    #     f"neu: {agg_profiles.count('neu')}/{len(players)}, "
    #     f"def: {agg_profiles.count('def')}/{len(players)}"
    # )

    profiles_ranks = []
    for player in players:
        # if player.aggregated_game["profile"] == "neu":
        p_r = []
        for game in player.games.values():
            p_r.append([game["profile"], game["rank"]])
        profiles_ranks.append(p_r)

    n = [0] * 5
    d = [0] * 5
    for p_r in profiles_ranks:
        for i in range(len(p_r) - 1):
            if p_r[i - 1][0] == "def":
                if p_r[i][0] == "col":
                    n[p_r[i][1] - 1] += 1
                d[p_r[i][1] - 1] += 1
    # p = [round(ni / di * 100) for ni, di in zip(n, d)]
    p = [n[0] / d[0] * 100, n[4] / d[4] * 100]

    print(n)
    print(d)
    print(p)

    # profiles = {profile: [] for profile in ["col", "neu", "def"]}
    # for p_r in profiles_ranks:
    #     for p, r in p_r:
    #         profiles[p].append(r)

    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), sharey=True)

    # for ax, profile in zip(axs.flat, ["col", "neu", "def"]):
    #     ax.hist(profiles[profile], bins=5, range=(0.5, 5.5), rwidth=0.8, density=True)
    #     ax.set_title(profile)
    #     ax.set_xticks(np.arange(1, 6))
    #     ax.set_xticklabels(np.arange(1, 6))

    # plt.show()

    # Matrice de transition
    transition_profiles = DefaultDict(list)
    for player in players:
        games = list(player.games.values())
        for i in range(len(games) - 1):
            transition_profiles[games[i]["profile"]].append(games[i + 1]["profile"])

    for profile in ["col", "neu", "def"]:
        print(
            f"{profile}: "
            f"{round(transition_profiles[profile].count('col') / len(transition_profiles[profile])*100)}%, "
            f"{round(transition_profiles[profile].count('neu') / len(transition_profiles[profile])*100)}%, "
            f"{round(transition_profiles[profile].count('def') / len(transition_profiles[profile])*100)}%"
        )


if __name__ == "__main__":
    from modules.constants import PATH_DATA

    main(PATH_DATA, "R2_vs_opt/4_opt/", "4_opt")
