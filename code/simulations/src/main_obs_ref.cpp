#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "agent/Agent.h"
#include "game/Game.h"
#include "game_analyzer/GameAnalyzer.h"
#include "helpers/helper_all.h"

int main()
{
    // Parameters of the program
    const int numberOfGames{100000};
    const std::string gameType{"R2_intra/R2_intra/"};

    // Parameters of the game
    const int numberOfRounds{20};
    const int numberOfPlayers{5};

    // Path of the in and out files
    const std::string pathDataFigures{"../../data_figures/"};
    const std::string pathParameters{pathDataFigures + gameType + "model/parameters/"};

    // Import the optimized parameters
    const std::vector<double> fractionPlayersProfiles{readParameters(pathParameters + "players_profiles.txt")};
    const std::vector<double> parametersOpenings{readParameters(pathParameters + "cells.txt")};
    const nlohmann::json parametersRatings{nlohmann::json::parse(std::ifstream(pathParameters + "stars.json"))};

    //
    GameAnalyzer analyzer(numberOfGames, numberOfPlayers);

#pragma omp parallel for
    for (int iGame = 0; iGame < numberOfGames; ++iGame)
    {
        Game game(numberOfRounds, numberOfPlayers);

        std::vector<Agent> agents{initializePlayers(5, fractionPlayersProfiles, game, parametersOpenings, parametersRatings)};

        for (int iRound{0}; iRound < numberOfRounds; ++iRound)
        {
            for (auto &agent : agents)
            {
                agent.playARound();
            }
        }

        analyzer.analyzeGame(iGame, game, agents);
    }

    analyzer.saveObservables(pathDataFigures + gameType + "model/observables_hum/");

    return 0;
}
