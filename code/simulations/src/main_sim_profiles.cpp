#include <algorithm> // std::max
#include <fstream>   // std::ifstream, std::ofstream
#include <iostream>
#include <stdexcept> // std::runtime_error
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "agent/Agent.h"
#include "game/Game.h"
#include "game_analyzer/GameAnalyzer.h"
#include "helpers/helper_all.h"
#include "random/myRandom.h"

int main()
{
    // Parameters of the program
    const int numberOfGames{20};
    const int numberOfRepetitions{10000};

    const std::string model{"PI"};

    std::vector<std::tuple<std::string, std::vector<std::string>>> experiments;
    experiments.push_back(std::make_tuple("R2_vs_col_def/0_col_4_def/0_col_4_def/", std::vector<std::string>{"def", "def", "def", "def"}));
    experiments.push_back(std::make_tuple("R2_vs_col_def/1_col_3_def/1_col_3_def/", std::vector<std::string>{"col", "def", "def", "def"}));
    experiments.push_back(std::make_tuple("R2_vs_col_def/2_col_2_def/2_col_2_def/", std::vector<std::string>{"col", "col", "def", "def"}));
    experiments.push_back(std::make_tuple("R2_vs_col_def/3_col_1_def/3_col_1_def/", std::vector<std::string>{"col", "col", "col", "def"}));
    experiments.push_back(std::make_tuple("R2_vs_col_def/4_col_0_def/4_col_0_def/", std::vector<std::string>{"col", "col", "col", "col"}));
    experiments.push_back(std::make_tuple("R2_vs_const/4_const1/4_const1/", std::vector<std::string>{"const1", "const1", "const1", "const1"}));
    experiments.push_back(std::make_tuple("R2_vs_const/4_const3/4_const3/", std::vector<std::string>{"const3", "const3", "const3", "const3"}));
    experiments.push_back(std::make_tuple("R2_vs_const/4_const5/4_const5/", std::vector<std::string>{"const5", "const5", "const5", "const5"}));
    experiments.push_back(std::make_tuple("R2_vs_opt/4_opt/4_opt/", std::vector<std::string>{"opt1", "opt1", "opt1", "opt1"}));

    for (const auto &experiment : experiments)
    {
        const std::string gameType{std::get<0>(experiment)};
        const std::vector<std::string> botTypes{std::get<1>(experiment)};

        std::cerr << "gameType: " << gameType << "\n";

        // Parameters of the game
        const int numberOfRounds{20};
        const int numberOfPlayers{5};

        // Path of the in and out files
        const std::string pathDataFigures{"../../data_figures/"};
        const std::string pathObservables{pathDataFigures + gameType + "model/pred_profile_model_" + model + "/"};
        const std::string pathParameters{pathDataFigures + gameType + "model/parameters/"};
        const std::string pathParametersBots{pathDataFigures + "bots/"};

        // Get parameters
        const std::vector<std::string> playersProfiles{"col", "neu", "def"};
        const nlohmann::json parametersRatings{nlohmann::json::parse(std::ifstream(pathDataFigures + "all/model/parameters/stars.json"))};
        const std::vector<double> parametersOpenings{readParameters(pathParameters + "cells.txt")};

        std::vector<std::vector<int>> agentProfiles(numberOfGames, std::vector<int>(3, 0));

        GameAnalyzer analyzerObs_all(numberOfRepetitions * 10, numberOfPlayers);
        GameAnalyzer analyzerObs_hum(numberOfRepetitions * 10, std::vector<int>{0});
        GameAnalyzer analyzerObs_bot(numberOfRepetitions * 10, std::vector<int>{1, 2, 3, 4});

        // Loop over the repetitions
#pragma omp parallel for
        for (int iRepetition = 0; iRepetition < numberOfRepetitions; ++iRepetition)
        {
            std::vector<double> fractionPlayersProfiles(3, 0.);
            std::vector<double> P(numberOfGames, 0.);
            std::vector<double> I(numberOfGames, 0.);
            std::vector<int> R(numberOfGames, 1);

            for (int iGame{0}; iGame < numberOfGames; ++iGame)
            {
                Game game(numberOfRounds, numberOfPlayers);

                // Select profile of Agent
                if (iGame == 0)
                {
                    fractionPlayersProfiles = {0.33, 0.33, 0.34};
                }
                else
                {
                    double p_average{0.};
                    double i_average{0.};
                    double r_average{0.};

                    const int n{3};
                    const int iStart{std::max(0, iGame - n)};
                    for (int i{iStart}; i < iGame; ++i)
                    {
                        p_average += P[i];
                        i_average += I[i];
                        r_average += R[i];
                    }
                    p_average /= iGame - iStart;
                    i_average /= iGame - iStart;
                    r_average /= iGame - iStart;

                    if (model == "PI")
                    {
                        fractionPlayersProfiles[0] = -0.293 + 0.904 * p_average + 0.00917 * i_average;
                        fractionPlayersProfiles[2] = 0.750 + -0.595 * p_average + -0.00942 * i_average;
                    }
                    else if (model == "PIR")
                    {
                        fractionPlayersProfiles[0] = -1.312 + 1.910 * p_average + 0.01779 * i_average + 0.225 * r_average;
                        fractionPlayersProfiles[2] = 1.289 + -1.129 * p_average + -0.01399 * i_average + -0.119 * r_average;
                    }
                    fractionPlayersProfiles[1] = 1 - fractionPlayersProfiles[0] - fractionPlayersProfiles[2];
                }

                const std::string agentProfile{myRandom::choice(playersProfiles, fractionPlayersProfiles)};

#pragma omp critical
                {
                    if (agentProfile == "col")
                    {
                        agentProfiles[iGame][0]++;
                    }
                    else if (agentProfile == "neu")
                    {
                        agentProfiles[iGame][1]++;
                    }
                    else if (agentProfile == "def")
                    {
                        agentProfiles[iGame][2]++;
                    }
                }

                // Initialize agents
                std::vector<Agent> agents;
                agents.reserve(numberOfPlayers);
                agents.emplace_back(game.getAddress(), parametersOpenings, parametersRatings[agentProfile]);
                for (const auto &botType : botTypes)
                {
                    const std::vector<double> parametersBotsOpenings{readParameters(pathParametersBots + botType + "/cells.txt")};
                    const nlohmann::json parametersBotsRatings{nlohmann::json::parse(std::ifstream(pathParametersBots + botType + "/stars.json"))};
                    agents.emplace_back(game.getAddress(), parametersBotsOpenings, parametersBotsRatings);
                }

                // Play the game
                for (int iRound{0}; iRound < numberOfRounds; ++iRound)
                {
                    for (auto &agent : agents)
                    {
                        agent.playARound();
                    }
                }

                // Analyze the game
                GameAnalyzer analyzer(1, numberOfPlayers);
                analyzer.analyzeGame(0, game, agents);
                P[iGame] = analyzer.get_P()[numberOfRounds - 1];
                I[iGame] = analyzer.get_IPR_P()[numberOfRounds - 1];

                for (int iOtherAgent{1}; iOtherAgent < numberOfPlayers; ++iOtherAgent)
                {
                    if (game.getScoreOfPlayer(iOtherAgent) > game.getScoreOfPlayer(0))
                    {
                        ++R[iGame];
                    }
                }

                // Analyze the game for the observables in the 10 last games
                if (iGame >= numberOfGames - 10)
                {
                    analyzerObs_all.analyzeGame(iGame - (numberOfGames - 10) + iRepetition * 10, game, agents);
                    analyzerObs_hum.analyzeGame(iGame - (numberOfGames - 10) + iRepetition * 10, game, agents);
                    analyzerObs_bot.analyzeGame(iGame - (numberOfGames - 10) + iRepetition * 10, game, agents);
                }
            }
        }

        analyzerObs_all.saveObservables(pathObservables + "observables/");
        analyzerObs_hum.saveObservables(pathObservables + "observables_hum/");
        analyzerObs_bot.saveObservables(pathObservables + "observables_bot/");

        std::ofstream file(pathObservables + "fractions.txt");
        if (!file.is_open())
        {
            throw std::runtime_error("The file " + pathObservables + "fractions.txt could not be opened.");
        }
        for (int iGame{0}; iGame < numberOfGames; ++iGame)
        {
            file << static_cast<double>(agentProfiles[iGame][0]) / (numberOfRepetitions) << " "
                 << static_cast<double>(agentProfiles[iGame][1]) / (numberOfRepetitions) << " "
                 << static_cast<double>(agentProfiles[iGame][2]) / (numberOfRepetitions) << "\n";
        }
    }

    return 0;
}
