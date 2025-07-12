from collections import defaultdict
from json import load
from pathlib import Path


def get_files(path_data: Path, game_type: str):
    for in_file in path_data.glob("S*/in/*.json"):
        with in_file.open() as f:
            in_data = load(f)
        if in_data["game"]["game_type"] == game_type:
            out_file = in_file.parents[1] / "out" / f"{in_file.stem}.csv"
            yield in_file, out_file


def get_files_sessions(path_data: Path, game_type: str):
    files = []
    for path_session in sorted(path_data.glob("S*/")):
        files_session = defaultdict(list)
        for in_file in sorted(path_session.glob("in/*.json")):
            with in_file.open() as f:
                in_data = load(f)
            if in_data["game"]["game_type"] == game_type:
                out_file = in_file.parents[1] / "out" / f"{in_file.stem}.csv"
                players = {player["id"] for player in in_data["players"]}
                files_session[frozenset(players)].append((in_file, out_file))
        for f in files_session.values():  # Separate independent groups within the same session folder
            files.append(f)
    return files


if __name__ == "__main__":
    from constants import PATH_DATA

    for in_file, out_file in get_files(PATH_DATA / "rule_1"):
        print(in_file, out_file)

    for files in get_files_sessions(PATH_DATA / "rule_1"):
        print(files)
