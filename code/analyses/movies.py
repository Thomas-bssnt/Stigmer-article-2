from json import load

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def create_movies(path_data, session_type, game_type, path_figures):
    path_in_files = list((path_data / session_type / game_type).glob("*/in/*.json"))
    for path_in_file in np.random.choice(path_in_files, 20, replace=False):
        path_out_file = path_in_file.parent.parent / "out" / f"{path_in_file.stem}.csv"
        create_movie(path_in_file, path_out_file, path_figures)


def create_movie(path_in_file, path_out_file, path_figures):
    with open(path_in_file) as in_file:
        in_data = load(in_file)["game"]
    in_data["mapSize"] = len(in_data["map"]["map"])

    maps_visits_all, maps_stars_all = import_game(in_data, path_out_file, "all")
    maps_visits_hum, maps_stars_hum = import_game(in_data, path_out_file, "hum")
    maps_visits_bot, maps_stars_bot = import_game(in_data, path_out_file, "bot")

    def update(
        i_round,
        maps_visits_all,
        maps_visits_hum,
        maps_visits_bot,
        maps_stars_all,
        maps_stars_hum,
        maps_stars_bot,
    ):
        fig.suptitle(f"Round {i_round}", size=14)
        ims[0].set_data(maps_stars_all[i_round])
        ims[1].set_data(maps_stars_hum[i_round])
        ims[2].set_data(maps_stars_bot[i_round])
        ims[3].set_data(maps_visits_all[i_round])
        ims[4].set_data(maps_visits_hum[i_round])
        ims[5].set_data(maps_visits_bot[i_round])

    # Figure

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(12, 8),
    )

    ims = []
    for ax in zip(axs.flat):
        ax = ax[0]
        # the lines
        ax.set_axis_off()
        ax.hlines(
            y=np.arange(-1, in_data["mapSize"]) + 0.5,
            xmin=np.full(in_data["mapSize"] + 1, 0) - 0.5,
            xmax=np.full(in_data["mapSize"] + 1, in_data["mapSize"]) - 0.5,
            color="black",
        )
        ax.vlines(
            x=np.arange(-1, in_data["mapSize"]) + 0.5,
            ymin=np.full(in_data["mapSize"] + 1, 0) - 0.5,
            ymax=np.full(in_data["mapSize"] + 1, in_data["mapSize"]) - 0.5,
            color="black",
        )

        # the text
        for i in range(in_data["mapSize"]):
            for j in range(in_data["mapSize"]):
                ax.text(j, i, in_data["map"]["map"][i][j], ha="center", va="center")

        # the image
        ims.append(ax.imshow(np.zeros((in_data["mapSize"], in_data["mapSize"], 3))))

    axs[0, 0].set_title("All", size=12)
    axs[0, 1].set_title("Hum", size=12)
    axs[0, 2].set_title("Bot", size=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(
        fig,
        update,
        fargs=(
            maps_visits_all,
            maps_visits_hum,
            maps_visits_bot,
            maps_stars_all,
            maps_stars_hum,
            maps_stars_bot,
        ),
        frames=list(np.arange(20 + 1)) + [20] * 3,
        interval=1500,
        repeat=False,
    )

    ani.save(path_figures / f"movies/{path_in_file.stem}.mp4")
    plt.close(fig)


def import_game(in_data, path_out_file, player_type):
    out_data = np.genfromtxt(path_out_file, dtype=None, delimiter=",", names=True, encoding=None)
    visits = np.zeros((in_data["numberRounds"], in_data["mapSize"], in_data["mapSize"]), dtype=int)
    stars = np.zeros((in_data["numberRounds"], in_data["mapSize"], in_data["mapSize"]), dtype=int)
    scores = np.zeros(in_data["numberRounds"], dtype=int)
    for round_, player_id, mapX, mapY, _, numberStars, score in out_data:
        if (
            player_type == "all"
            or (player_type == "hum" and len(player_id) == 3)
            or (player_type == "bot" and len(player_id) > 3)
        ):
            stars[round_ - 1, mapY, mapX] += numberStars
            visits[round_ - 1, mapY, mapX] += 1
            scores[round_ - 1] += score

    map_visits_normalized = safe_normalization(np.cumsum(visits, axis=0))
    map_stars_normalized = safe_normalization(np.cumsum(stars, axis=0))
    scores = np.cumsum([0] + list(scores))

    map_visits_colors = np.full(
        (in_data["numberRounds"] + 1, in_data["mapSize"], in_data["mapSize"], 3), 255, dtype=int
    )
    map_stars_colors = np.full((in_data["numberRounds"] + 1, in_data["mapSize"], in_data["mapSize"], 3), 255, dtype=int)
    for i_round in range(in_data["numberRounds"]):
        for i in range(in_data["mapSize"]):
            for j in range(in_data["mapSize"]):
                map_visits_colors[i_round + 1][i][j] = get_color_visits(np.sqrt(map_visits_normalized[i_round][i][j]))
                map_stars_colors[i_round + 1][i][j] = get_color_stars(np.sqrt(map_stars_normalized[i_round][i][j]))
    return map_visits_colors, map_stars_colors


def safe_normalization(array):
    normalized_array = np.zeros(array.shape)
    for round_ in range(len(array)):
        if (sum_ := np.sum(array[round_])) != 0:
            normalized_array[round_] = array[round_] / sum_

    return normalized_array


def get_color_stars(fraction):
    if fraction <= 0.5:
        value = 255 - round(fraction * 255 / 0.5)
        return 255, value, value
    else:
        value = round((1 - fraction) * 255 / 0.5)
        return value, 0, 0


def get_color_visits(fraction):
    if fraction <= 0.5:
        value = 255 - round(fraction * 255 / 0.5)
        return value, value, 255
    else:
        value = round((1 - fraction) * 255 / 0.5)
        return 0, 0, value


if __name__ == "__main__":
    from modules.constants import PATH_DATA, PATH_FIGURES

    create_movies(PATH_DATA, "R2_vs_opt", "4_opt", PATH_FIGURES)
