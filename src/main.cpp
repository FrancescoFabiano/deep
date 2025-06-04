/*
 * \brief The main file.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 12, 2025
 */


#include <iostream>
#include <boost/make_shared.hpp>

#include "Configuration.h"
#include "PortfolioSearch.h"
#include "argparse/ArgumentParser.h"
#include "domain/Domain.h"
#include "neuralnets/TrainingDataset.h"
#include "utilities/ExitHandler.h"
#include "states/representations/kripke/KripkeState.h"


int main(int argc, char** argv)
{
    ArgumentParser::create_instance(argc, argv);
    Configuration::create_instance();
    //Domain::create_instance();

    //Dataset Generation for ML Heuristics
    if (ArgumentParser::get_instance().get_dataset_mode())
    {
        TrainingDataset<KripkeState>::create_instance();
        if (!TrainingDataset<KripkeState>::get_instance().generate_dataset())
        {
            std::exit(static_cast<int>(ExitHandler::ExitCode::SuccessNotPlanningMode));
        }
        std::exit(static_cast<int>(ExitHandler::ExitCode::SuccessNotPlanningModeWarning));
    }

    //Standard Search
    const PortfolioSearch searcher;
    const bool goal_found = searcher.run_portfolio_search();
    if (goal_found)
        std::exit(static_cast<int>(ExitHandler::ExitCode::SuccessFoundGoal));
    std::exit(static_cast<int>(ExitHandler::ExitCode::SuccessNotFoundGoal));
}
