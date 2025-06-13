#include "GraphNN.h"
#include "TrainingDataset.h"
#include "ExitHandler.h"
#include <fstream>
#include <sstream>

// --- Singleton instance initialization ---
template <StateRepresentation StateRepr>
GraphNN<StateRepr>* GraphNN<StateRepr>::instance = nullptr;

template <StateRepresentation StateRepr>
GraphNN<StateRepr>& GraphNN<StateRepr>::get_instance()
{
    if (!instance)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNInstanceError,
            "GraphNN instance not created. Call create_instance() first."
        );
        std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
    }
    return *instance;
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::create_instance()
{
    if (!instance)
    {
        instance = new GraphNN();
    }
}

template <StateRepresentation StateRepr>
GraphNN<StateRepr>::GraphNN()
{
    // Use default StateRepresentation for dataset folder creation
    TrainingDataset<StateRepr>::create_instance();
    m_checking_file_path = TrainingDataset<StateRepr>::get_instance().get_folder() + "to_predict.dot";
    m_goal_file_path = TrainingDataset<StateRepr>::get_instance().get_goal_file_path();
}

/**
 * \brief Get the score for a given state using the neural network heuristic.
 * \tparam StateRepr The state representation type.
 * \param state The state to evaluate.
 * \return The heuristic score for the state.
 */
template <StateRepresentation StateRepr>
[[nodiscard]] short GraphNN<StateRepr>::get_score(const State<StateRepr>& state)
{
    // Print the state in the required dataset format to the checking file
    std::ofstream ofs(m_checking_file_path);
    if (!ofs)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNFileError,
            "Failed to open file for NN state checking: " + m_checking_file_path
        );
    }
    state.print_dataset_format(ofs, ArgumentParser::get_instance().get_dataset_mapped(),
                               ArgumentParser::get_instance().get_dataset_merged());


    // Run the external Python script for NN inference
    /// todo Replace this with C++ call that opens the model and runs the inference
    std::string command = "./lib/RL/run_prediction.sh " + m_checking_file_path + " " +
        std::to_string(state.get_plan_length()) + " " + m_goal_file_path + " " + m_model_path;

    std::array<char, 128> buffer{};
    std::string result;

    // Open a pipe to the shell
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNScriptError,
            "Failed to run Python script");
    }

    // Read the output from the Python script
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }

    try
    {
        return static_cast<short>(std::stod(result));
    }
    catch (const std::exception& e)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNScriptError,
            "Error converting prediction to double: " + std::string(e.what()));
    }

    // Just to please the compiler
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
}
