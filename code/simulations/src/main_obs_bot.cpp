#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "agent/Agent.h"
#include "game/Game.h"
#include "game/Rule.h"
#include "game_analyzer/GameAnalyzer.h"
#include "helpers/helper_all.h"
#include "random/myRandom.h"

int main()
{
    // Parameters of the program
    const int numberOfGames{100000};

    std::vector<std::tuple<std::string, std::string, std::vector<std::string>>> experiments;
    experiments.push_back(std::make_tuple("R2_vs_col_def/0_col_4_def/0_col_4_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"def", "def", "def", "def"}));
    experiments.push_back(std::make_tuple("R2_vs_col_def/1_col_3_def/1_col_3_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"col", "def", "def", "def"}));
    experiments.push_back(std::make_tuple("R2_vs_col_def/2_col_2_def/2_col_2_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"col", "col", "def", "def"}));
    experiments.push_back(std::make_tuple("R2_vs_col_def/3_col_1_def/3_col_1_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"col", "col", "col", "def"}));
    experiments.push_back(std::make_tuple("R2_vs_col_def/4_col_0_def/4_col_0_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"col", "col", "col", "col"}));
    experiments.push_back(std::make_tuple("R2_vs_const/4_const1/4_const1/", "R2_vs_const/const/", std::vector<std::string>{"const1", "const1", "const1", "const1"}));
    experiments.push_back(std::make_tuple("R2_vs_const/4_const3/4_const3/", "R2_vs_const/const/", std::vector<std::string>{"const3", "const3", "const3", "const3"}));
    experiments.push_back(std::make_tuple("R2_vs_const/4_const5/4_const5/", "R2_vs_const/const/", std::vector<std::string>{"const5", "const5", "const5", "const5"}));
    experiments.push_back(std::make_tuple("R2_vs_opt/4_opt/4_opt/", "R2_vs_opt/4_opt/4_opt/", std::vector<std::string>{"opt1", "opt1", "opt1", "opt1"}));

    for (const auto &experiment : experiments)
    {
        const std::string gameType{std::get<0>(experiment)};
        const std::string ratingFolder{std::get<1>(experiment)};
        const std::vector<std::string> botTypes{std::get<2>(experiment)};

        std::cerr << "gameType: " << gameType << "\n";

        // Parameters of the game
        const int numberOfRounds{20};
        const int numberOfPlayers{5};

        // Path of the in and out files
        const std::string pathDataFigures{"../../data_figures/"};

        // Import the optimized parameters
        const std::vector<double> fractionPlayersProfiles{readParameters(pathDataFigures + gameType + "model/parameters/players_profiles.txt")};
        const std::vector<double> parametersOpenings{readParameters(pathDataFigures + gameType + "model/parameters/cells.txt")};
        const nlohmann::json parametersRatings{nlohmann::json::parse(std::ifstream(pathDataFigures + ratingFolder + "model/parameters/stars.json"))};

        //
        const std::string pathParametersBots{pathDataFigures + "bots/"};

        //
        GameAnalyzer analyzer_all(numberOfGames, numberOfPlayers);
        GameAnalyzer analyzer_hum(numberOfGames, std::vector<int>{0});
        GameAnalyzer analyzer_bot(numberOfGames, std::vector<int>{1, 2, 3, 4});

#pragma omp parallel for
        for (int iGame = 0; iGame < numberOfGames; ++iGame)
        {
            Game game(numberOfRounds, numberOfPlayers);

            // Creation of the agents
            std::vector<Agent> agents{initializePlayers(1, fractionPlayersProfiles, game, parametersOpenings, parametersRatings)};
            for (const auto &botType : botTypes)
            {
                const std::vector<double> parametersBotsOpenings{readParameters(pathParametersBots + botType + "/cells.txt")};
                const nlohmann::json parametersBotsRatings{nlohmann::json::parse(std::ifstream(pathParametersBots + botType + "/stars.json"))};
                agents.emplace_back(game.getAddress(), parametersBotsOpenings, parametersBotsRatings);
            }

            for (int iRound{0}; iRound < numberOfRounds; ++iRound)
            {
                for (auto &agent : agents)
                {
                    agent.playARound();
                }
            }

            analyzer_all.analyzeGame(iGame, game, agents);
            analyzer_hum.analyzeGame(iGame, game, agents);
            analyzer_bot.analyzeGame(iGame, game, agents);
        }

        analyzer_all.saveObservables(pathDataFigures + gameType + "model/observables/");
        analyzer_hum.saveObservables(pathDataFigures + gameType + "model/observables_hum/");
        analyzer_bot.saveObservables(pathDataFigures + gameType + "model/observables_bot/");
    }

    return 0;
}
