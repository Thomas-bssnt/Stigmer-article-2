import numpy as np
from modules.player import get_players_games_sessions


def main(path_data, path_data_figures, session_type, game_type, bootstrap_reps):
    players = get_players_games_sessions(path_data / session_type, game_type, classify=(-0.5, 0.5))

    mean, err, mean_col, err_col, mean_neu, err_neu, mean_def, err_def = bootstrap(players, bootstrap_reps)

    np.savetxt(
        path_data_figures / session_type / game_type / "exp" / "observables_hum" / "rank.txt",
        np.column_stack((mean, err.T)),
        fmt="%f",
    )

    np.savetxt(
        path_data_figures / session_type / game_type / "exp" / "observables_hum" / "rank_col.txt",
        np.column_stack((mean_col, err_col.T)),
        fmt="%f",
    )

    np.savetxt(
        path_data_figures / session_type / game_type / "exp" / "observables_hum" / "rank_neu.txt",
        np.column_stack((mean_neu, err_neu.T)),
        fmt="%f",
    )

    np.savetxt(
        path_data_figures / session_type / game_type / "exp" / "observables_hum" / "rank_def.txt",
        np.column_stack((mean_def, err_def.T)),
        fmt="%f",
    )


def bootstrap(players_grouped, bootstrap_reps):
    bs_props_hum = []
    bs_props_hum_col = []
    bs_props_hum_neu = []
    bs_props_hum_def = []

    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(len(players_grouped), replace=True, size=len(players_grouped))
        bs_players = [player for i in bs_indices for player in players_grouped[i]]

        ranks = [0] * 5
        ranks_col = [0] * 5
        ranks_neu = [0] * 5
        ranks_def = [0] * 5
        for player in bs_players:
            if not player.bot:
                for i, game in enumerate(player.games.values()):
                    ranks[game["rank"] - 1] += 1
                    if game["profile"] == "col":
                        ranks_col[game["rank"] - 1] += 1
                    elif game["profile"] == "neu":
                        ranks_neu[game["rank"] - 1] += 1
                    else:
                        ranks_def[game["rank"] - 1] += 1

        bs_props_hum.append([rank / sum(ranks) for rank in ranks])
        bs_props_hum_col.append(
            [
                rank_c / rank / (sum(ranks_col) / sum(ranks)) if rank > 0 and sum(ranks_col) > 0 else np.nan
                for rank_c, rank in zip(ranks_col, ranks)
            ]
        )
        bs_props_hum_neu.append(
            [
                rank_n / rank / (sum(ranks_neu) / sum(ranks)) if rank > 0 and sum(ranks_neu) > 0 else np.nan
                for rank_n, rank in zip(ranks_neu, ranks)
            ]
        )
        bs_props_hum_def.append(
            [
                rank_d / rank / (sum(ranks_def) / sum(ranks)) if rank > 0 and sum(ranks_def) > 0 else np.nan
                for rank_d, rank in zip(ranks_def, ranks)
            ]
        )

    mean = np.nanmean(bs_props_hum, axis=0)
    err = np.abs(np.nanpercentile(bs_props_hum, [50 - 34.13, 50 + 34.13], axis=0) - mean)

    mean_col = np.nanmean(bs_props_hum_col, axis=0)
    err_col = np.abs(np.nanpercentile(bs_props_hum_col, [50 - 34.13, 50 + 34.13], axis=0) - mean_col)

    mean_neu = np.nanmean(bs_props_hum_neu, axis=0)
    err_neu = np.abs(np.nanpercentile(bs_props_hum_neu, [50 - 34.13, 50 + 34.13], axis=0) - mean_neu)

    mean_def = np.nanmean(bs_props_hum_def, axis=0)
    err_def = np.abs(np.nanpercentile(bs_props_hum_def, [50 - 34.13, 50 + 34.13], axis=0) - mean_def)

    return mean, err, mean_col, err_col, mean_neu, err_neu, mean_def, err_def


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_DATA_FIGURES, experiments

    bootstrap_reps = 1000

    for session_type, game_type in experiments:
        print(session_type, game_type)

        main(PATH_DATA, PATH_DATA_FIGURES, session_type, game_type, bootstrap_reps)
