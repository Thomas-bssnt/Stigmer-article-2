from collections import defaultdict
from json import load
from warnings import filterwarnings

import numpy as np
from modules.files import get_files, get_files_sessions
from scipy.optimize import OptimizeWarning, curve_fit


class Player:
    def __init__(self, session, player_data, bot=False):
        self.session = session
        self.bot = bot

        if self.bot:
            self.id = player_data["botName"]
        else:
            self.id = player_data["id"]
            self.age = player_data["age"]
            self.gender = player_data["gender"]

        self.games = defaultdict(dict)
        self.aggregated_game = {}

    def __repr__(self):
        return f"Player({self.session['phaseId']}, S{self.session['sessionId']}, {self.id})"

    def __getitem__(self, item):
        return self.games[item]

    def __eq__(self, other):
        return (self.session, self.id) == other

    def aggregate(self):
        self.aggregated_game["ratings"] = defaultdict(list)
        self.aggregated_game["score"] = 0
        self.aggregated_game["score_R2"] = 0
        for game in self.games.values():
            for value, ratings in game["ratings"].items():
                self.aggregated_game["ratings"][value] += ratings
            self.aggregated_game["score"] += game["score"]
            self.aggregated_game["score_R2"] += game["score_R2"]
        self.aggregated_game["score"] /= len(self.games)
        self.aggregated_game["score_R2"] /= len(self.games)

    def classify(self, game, u1_def_neu, u1_neu_col):
        values = [value for value, ratings in game["ratings"].items() if ratings]
        mean_ratings = [np.mean(ratings) for ratings in game["ratings"].values() if ratings]

        filterwarnings("ignore", category=OptimizeWarning)
        try:
            (game["u0"], game["u1"]), _ = curve_fit(
                self._ratings_fit_function,
                values,
                mean_ratings,
                p0=[2.5, 0.0],
            )
        except (TypeError, RuntimeError):
            game["u0"] = mean_ratings[0]
            game["u1"] = 0.0

        if game["u1"] < u1_def_neu:
            game["profile"] = "def"
        elif game["u1"] > u1_neu_col:
            game["profile"] = "col"
        else:
            game["profile"] = "neu"

    @staticmethod
    def _ratings_fit_function(x, u0, u1):
        return u0 + 5 * u1 * x / 99


def get_players_games(path_data, game_type, **kwargs):
    return get_players_list_of_files(get_files(path_data, game_type), **kwargs)


def get_players_games_sessions(path_data, game_type, **kwargs):
    return [
        get_players_list_of_files(files_session, **kwargs) for files_session in get_files_sessions(path_data, game_type)
    ]


def get_players_list_of_files(files, aggregate=False, classify=None):
    players = []
    for in_file, out_file in files:
        # In data
        with in_file.open() as f:
            in_data = load(f)
        session = in_data["session"]

        # Out data
        values = np.unique(in_data["game"]["map"]["map"])
        ratings = defaultdict(lambda: {value: [] for value in values})
        scores = defaultdict(int)
        scores_R2 = defaultdict(int)
        out_data = np.genfromtxt(out_file, dtype=None, delimiter=",", names=True, encoding=None)
        for _, player_id, _, _, value, number_stars, score in out_data:
            ratings[player_id][value].append(number_stars)
            scores[player_id] += score
            scores_R2[player_id] += value
        ranks = {
            player_id: rk
            for rk, (player_id, _) in enumerate(sorted(scores_R2.items(), key=lambda x: x[1], reverse=True), 1)
        }

        bots_id = defaultdict(set)
        for id_ in ratings.keys():
            split_id = id_.split("_")
            if len(split_id) > 1:
                bot_number = split_id[0]
                bot_name = "_".join(split_id[1:])
                bots_id[bot_name].add(bot_number)

        # Add player and data
        for player_data in in_data["players"]:
            if (session, player_data["id"]) not in players:
                players.append(Player(session, player_data))
            player = players[players.index((session, player_data["id"]))]
            player[in_data["id"]]["ratings"] = ratings[player.id]
            player[in_data["id"]]["score"] = scores[player.id]
            player[in_data["id"]]["score_R2"] = scores_R2[player.id]
            player[in_data["id"]]["rank"] = ranks[player.id]

        for player_data in in_data["game"]["bots"]:
            bot_number = bots_id[player_data["botName"]].pop()
            player_data["botName"] = bot_number + "_" + player_data["botName"]
            if (session, player_data["botName"]) not in players:
                players.append(Player(session, player_data, bot=True))
            player = players[players.index((session, player_data["botName"]))]
            player[in_data["id"]]["ratings"] = ratings[player.id]
            player[in_data["id"]]["score"] = scores[player.id]
            player[in_data["id"]]["score_R2"] = scores_R2[player.id]
            player[in_data["id"]]["rank"] = ranks[player.id]

    # Aggregate the games
    if aggregate:
        for player in players:
            player.aggregate()

    # Classify the players
    if classify is not None:
        for player in players:
            for game in player.games.values():
                player.classify(game, *classify)
            if aggregate:
                player.classify(player.aggregated_game, *classify)

    # Order the players games dictionary
    for player in players:
        player.games = {game_id: game for game_id, game in sorted(player.games.items())}

    return players
