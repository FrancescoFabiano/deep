#include "GraphNN.h"
#include "TrainingDataset.h"
#include "ExitHandler.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

// --- Singleton instance initialization ---
GraphNN* GraphNN::instance = nullptr;

GraphNN& GraphNN::get_instance() {
    if (!instance) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNInstanceError,
            "GraphNN instance not created. Call create_instance() first."
        );
        std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
    }
    return *instance;
}

void GraphNN::create_instance() {
    if (!instance) {
        instance = new GraphNN();
    }
}

GraphNN::GraphNN()
{
    // Use default StateRepresentation for dataset folder creation
    TrainingDataset<DefaultStateRepresentation>::create_instance();
    m_checking_file_path = TrainingDataset<DefaultStateRepresentation>::get_instance().get_folder() + "to_predict.dot";
}

/**
 * \brief Get the score for a given state using the neural network heuristic.
 * \tparam StateRepr The state representation type.
 * \param state The state to evaluate.
 * \return The heuristic score for the state.
 */
template <StateRepresentation StateRepr>
[[nodiscard]] unsigned short GraphNN::get_score(const State<StateRepr>& state)
{
    // Print the state in the required dataset format to the checking file
    {
        std::ofstream ofs(m_checking_file_path);
        if (!ofs) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::GNNFileError,
                "Failed to open file for NN state checking: " + m_checking_file_path
            );
        }
        state.print_dataset_format(ofs);
    }

    // Run the external Python script for NN inference
    std::string command = "./lib/RL/run_python_script.sh " + m_checking_file_path + " " + std::to_string(state.get_plan_length());
    int ret = std::system(command.c_str()); // blocks until script (and Python) finishes

    if (ret != 0) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNScriptError,
            "Using GNN for heuristics failed with exit code: " + std::to_string(ret)
        );
    }

    std::ifstream infile("prediction.tmp");
    if (!infile) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNFileError,
            "Failed to open prediction.tmp"
        );
        return 1;
    }

    std::string line;
    unsigned short valueFromFile = 0;

    while (std::getline(infile, line)) {
        if (line.rfind("VALUE:", 0) == 0) { // line starts with "VALUE:"
            std::istringstream iss(line.substr(6)); // Skip "VALUE:"
            iss >> valueFromFile;
            break;
        }
    }

    infile.close();

    return valueFromFile;
}

