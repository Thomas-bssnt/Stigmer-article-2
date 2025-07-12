#include <ctime>   // std::time_t, std::difftime, std::time
#include <fstream> // std::ifstream, std::ofstream
#include <iostream>
#include <stdexcept> // std::runtime_error
#include <string>
#include <vector>

#include "agent/Agent.h"
#include "agent/RatingStrategy.h"
#include "game/Game.h"
#include "helpers/helper_all.h"
#include "random/myRandom.h"

std::vector<double> readValuesObservable(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("The file " + filePath + " could not be opened.");
    }

    std::vector<double> observable;
    std::string line;
    while (std::getline(file, line))
    {
        bool round{true};
        std::string number;
        for (const auto x : line)
        {
            if (round)
            {
                if (x == ' ')
                {
                    round = false;
                }
            }
            else
            {
                if (x == ' ')
                {
                    observable.push_back(std::stod(number));
                    break;
                }
                else
                {
                    number += x;
                }
            }
        }
    }
    return observable;
}

void writeBestParameters(const std::string &filePath, const std::vector<double> &bestParameters)
{
    std::ofstream file(filePath, std::ios::app);
    if (!file.is_open())
    {
        throw std::runtime_error("The file " + filePath + " could not be opened.");
    }

    file << bestParameters[0];
    for (int iParameter{1}; iParameter < bestParameters.size(); ++iParameter)
    {
        file << " " << bestParameters[iParameter];
    }
    file << "\n";
}

double getAverageScore(
    const int numberOfGames,
    const int numberOfRounds,
    const int numberOfPlayers,
    const std::vector<double> &parametersOpenings,
    const std::vector<double> &parametersRatings,
    const std::vector<std::string> &botTypes,
    const std::string &pathParametersBots)
{
    int S{0};
#pragma omp parallel for reduction(+ : S)
    for (int iGame = 0; iGame < numberOfGames; ++iGame)
    {
        Game game(numberOfRounds, numberOfPlayers);

        std::vector<Agent> agents;
        agents.emplace_back(game.getAddress(), parametersOpenings, RatingStrategy(parametersRatings, "tanh"));
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

        S += game.getScoreOfPlayer(0);
    }

    return S / static_cast<double>(numberOfGames);
}

void printCurrentState(const std::vector<double> &parametersOpenings, const std::vector<double> &parametersRatings, double averageScore)
{
    std::cerr << "ParametersOpenings: ";
    for (const auto &parameter : parametersOpenings)
    {
        std::cerr << parameter << " ";
    }
    std::cerr << "\n";
    std::cerr << "ParametersRatings: ";
    for (const auto &parameter : parametersRatings)
    {
        std::cerr << parameter << " ";
    }
    std::cerr << "\n";
    std::cerr << "<S> =  " << averageScore << "\n\n";
}

double randomSmallChange(const std::vector<double> &parameters, const int iParameterToChange)
{
    double epsilon{0.};
    switch (iParameterToChange)
    {
    case 0:
        do
        {
            epsilon = myRandom::rand(-0.1, 0.1);
        } while (parameters[0] - epsilon < 0 ||
                 parameters[0] + epsilon < 0 ||
                 parameters[0] - epsilon > 1 ||
                 parameters[0] + epsilon > 1);
        break;
    case 2:
    case 4:
    case 6:
    case 10:
    case 14:
        epsilon = myRandom::rand(-10., 10.);
        break;
    case 1:
    case 3:
    case 5:
    case 7:
    case 8:
    case 9:
    case 11:
    case 12:
    case 13:
    case 15:
        epsilon = myRandom::rand(-0.1, 0.1);
        break;
    default:
        std::cerr << "The parameter " << iParameterToChange << " is not implemented.\n";
        break;
    }
    return epsilon;
}

void doMonteCarloStep(
    const std::vector<bool> &parametersToChange,
    std::vector<double> &bestParametersOpenings,
    std::vector<double> &bestParametersRatings,
    double &bestAverageScore,
    const int numberOfGames,
    const int numberOfRounds,
    const int numberOfPlayers,
    const std::string &pathParameters,
    const std::vector<std::string> &botTypes,
    const std::string &pathParametersBots)
{
    std::vector<double> parameters{bestParametersOpenings};
    parameters.insert(parameters.end(), bestParametersRatings.begin(), bestParametersRatings.end());

    int iParameterToChange;
    do
    {
        iParameterToChange = myRandom::randIndex(parameters.size());
    } while (!parametersToChange[iParameterToChange]);
    parameters[iParameterToChange] += randomSmallChange(parameters, iParameterToChange);

    std::vector<double> parametersOpenings(parameters.begin(), parameters.begin() + 8);
    std::vector<double> parametersRatings(parameters.begin() + 8, parameters.end());

    double averageScore{bestAverageScore};
    averageScore = getAverageScore(numberOfGames, numberOfRounds, numberOfPlayers, parametersOpenings,
                                   parametersRatings, botTypes, pathParametersBots);

    if (averageScore > bestAverageScore)
    {
        bestAverageScore = averageScore;
        bestParametersOpenings = parametersOpenings;
        bestParametersRatings = parametersRatings;
        writeBestParameters(pathParameters + "cells.txt", bestParametersOpenings);
        writeBestParameters(pathParameters + "stars.txt", bestParametersRatings);
    }

    printCurrentState(parametersOpenings, bestParametersRatings, averageScore);
}

int main()
{
    // Parameters of the program
    const int numberOfGamesInEachStep{10000};
    const std::vector<bool> parametersToChange{true, true, true, true, true, true, true, true,
                                               true, true, true, true, true, true, true, true};

    std::tuple<std::string, std::string, std::vector<std::string>> experiment;
    // experiment = std::make_tuple("R2_vs_col_def/0_col_4_def/0_col_4_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"def", "def", "def", "def"});
    // experiment = std::make_tuple("R2_vs_col_def/1_col_3_def/1_col_3_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"col", "def", "def", "def"});
    // experiment = std::make_tuple("R2_vs_col_def/2_col_2_def/2_col_2_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"col", "col", "def", "def"});
    // experiment = std::make_tuple("R2_vs_col_def/3_col_1_def/3_col_1_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"col", "col", "col", "def"});
    // experiment = std::make_tuple("R2_vs_col_def/4_col_0_def/4_col_0_def/", "R2_vs_col_def/col_def/", std::vector<std::string>{"col", "col", "col", "col"});
    // experiment = std::make_tuple("R2_vs_const/4_const1/4_const1/", "R2_vs_const/const/", std::vector<std::string>{"const1", "const1", "const1", "const1"});
    // experiment = std::make_tuple("R2_vs_const/4_const3/4_const3/", "R2_vs_const/const/", std::vector<std::string>{"const3", "const3", "const3", "const3"});
    experiment = std::make_tuple("R2_vs_const/4_const5/4_const5/", "R2_vs_const/const/", std::vector<std::string>{"const5", "const5", "const5", "const5"});
    // experiment = std::make_tuple("R2_vs_opt/4_opt/4_opt/", "R2_vs_opt/4_opt/4_opt/", std::vector<std::string>{"opt1", "opt1", "opt1", "opt1"});

    const std::string gameType{std::get<0>(experiment)};
    const std::string ratingFolder{std::get<1>(experiment)};
    const std::vector<std::string> botTypes{std::get<2>(experiment)};

    std::cerr << "gameType: " << gameType << "\n\n";

    // Parameters of the game
    const int numberOfRounds{20};
    const int numberOfPlayers{5};

    // Path of the in and out files
    const std::string pathDataFigures{"../../data_figures/"};

    //
    const std::string pathParameters{pathDataFigures + gameType + "opt/parameters/"};
    const std::string pathParametersBots{pathDataFigures + "bots/"};

    // Initialization
    std::vector<double> bestParametersOpenings{readParameters(pathParameters + "cells.txt")};
    std::vector<double> bestParametersRatings{readParameters(pathParameters + "stars.txt")};
    double bestAverageScore{getAverageScore(numberOfGamesInEachStep, numberOfRounds, numberOfPlayers,
                                            bestParametersOpenings, bestParametersRatings, botTypes, pathParametersBots)};
    printCurrentState(bestParametersOpenings, bestParametersRatings, bestAverageScore);

    // MC simulation
    while (true)
    {
        // std::time_t t0, t1;
        // time(&t0);

        doMonteCarloStep(parametersToChange, bestParametersOpenings, bestParametersRatings, bestAverageScore,
                         numberOfGamesInEachStep, numberOfRounds, numberOfPlayers,
                         pathParameters, botTypes, pathParametersBots);

        // time(&t1);
        // std::cerr << "t = " << std::difftime(t1, t0) << "s\n\n";
    }

    return 0;
}
