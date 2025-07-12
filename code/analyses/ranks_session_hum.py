import numpy as np
from modules.player import get_players_games_sessions


def main(path_data, path_data_figures, session_type, game_type, bootstrap_reps):
    players = get_players_games_sessions(path_data / session_type, game_type, aggregate=True)

    mean, err = bootstrap(players, bootstrap_reps)

    print(mean, err)

    # np.savetxt(
    #     path_data_figures / session_type / game_type / "exp" / "observables_hum" / "rank.txt",
    #     np.column_stack((mean, err.T)),
    #     fmt="%f",
    # )


def bootstrap(players_grouped, bootstrap_reps):
    bs_mean_rank = []

    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(len(players_grouped), replace=True, size=len(players_grouped))
        bs_players = [players_grouped[i] for i in bs_indices]

        ranks = []
        for players_sessions in bs_players:
            score_hum = 0
            scores_bot = []
            for player in players_sessions:
                if player.bot:
                    scores_bot.append(player.aggregated_game["score"])
                else:
                    score_hum = player.aggregated_game["score"]

            rank = 1 + sum(score_hum < score_bot for score_bot in scores_bot)
            ranks.append(rank)

        bs_mean_rank.append(np.mean(ranks))

    mean = np.mean(bs_mean_rank, axis=0)
    err = np.abs(np.percentile(bs_mean_rank, [50 - 34.13, 50 + 34.13], axis=0) - mean)

    return mean, err


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES, experiments

    bootstrap_reps = 1000

    for session_type, game_type in experiments:
        print(session_type, game_type)

        main(PATH_DATA, PATH_DATA_FIGURES, session_type, game_type, bootstrap_reps)
