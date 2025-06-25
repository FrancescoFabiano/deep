#include "ExitHandler.h"
#include "GraphNN.h"
#include "TrainingDataset.h"
#include <fstream>
#include <regex>

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
            "GraphNN instance not created. Call create_instance() first.");
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
    m_checking_file_path =
        TrainingDataset<StateRepr>::get_instance().get_folder() +
        "to_predict.dot";

    /// \todo remove this, not needed if inference is done directly
    m_goal_file_path =
        TrainingDataset<StateRepr>::get_instance().get_goal_file_path();


    populate_with_goal();
}


template <StateRepresentation StateRepr>
[[nodiscard]] short
GraphNN<StateRepr>::get_score(const State<StateRepr>& state)
{

 try {
        torch::Tensor tensor = torch::rand({2, 3});
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Libtorch C++ test failed: " << e.what() << std::endl;
        return 1;
    }


    if (ArgumentParser::get_instance().get_dataset_merged()){


        auto& os = ArgumentParser::get_instance().get_output_stream();
        os << "[DEBUG] The dot can be found at: " << m_checking_file_path << std::endl;


        std::ofstream ofs(m_checking_file_path);
        if (!ofs)
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::GNNFileError,
                "Failed to open file for NN state checking: " + m_checking_file_path);
        }
        state.print_dataset_format(
                ofs, !ArgumentParser::get_instance().get_dataset_mapped(),
                ArgumentParser::get_instance().get_dataset_merged());

        const auto state_tensor = state_to_tensor_minimal(state.get_representation());







        os << "[DEBUG] processing the state with this information: ";
        os << "\tedge_ids: " << state_tensor.edge_ids << std::endl;
        os << "\tedge_attrs: " << state_tensor.edge_attrs << std::endl;
        os << "\treal_node_ids: " << state_tensor.real_node_ids << std::endl;


    }

    return 1;
}

template <StateRepresentation StateRepr>
[[nodiscard]] short
GraphNN<StateRepr>::get_score_python(const State<StateRepr>& state)
{
    // Print the state in the required dataset format to the checking file
    std::ofstream ofs(m_checking_file_path);
    if (!ofs)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNFileError,
            "Failed to open file for NN state checking: " + m_checking_file_path);
    }
    state.print_dataset_format(
        ofs, !ArgumentParser::get_instance().get_dataset_mapped(),
        ArgumentParser::get_instance().get_dataset_merged());


    // Run the external Python script for NN inference
    /// todo Replace this with C++ call that opens the model and runs the
    /// inference
    std::string command = "./lib/RL/run_prediction.sh " + m_checking_file_path +
        " " + std::to_string(state.get_plan_length()) + " " +
        m_goal_file_path + " " + m_model_path;

    std::array<char, 128> buffer{};
    std::string result;

    // Open a pipe to the shell
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"),
                                                  pclose);
    if (!pipe)
    {
        ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNScriptError,
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
        ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNScriptError,
                                       "Error converting prediction to double: " +
                                       std::string(e.what()));
    }

    // Just to please the compiler
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
}

template <StateRepresentation StateRepr>
size_t
GraphNN<StateRepr>::get_symbolic_id(const size_t node)
{
    if (!m_node_to_symbolic.contains(node))
    {
        m_node_to_symbolic[node] = m_symbolic_id++;
        m_real_node_ids.push_back(node);
    }
    return m_node_to_symbolic[node];
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::add_edge(const int64_t src, const int64_t dst, const int64_t label)
{
    m_edge_src.push_back(get_symbolic_id(src));
    m_edge_dst.push_back(get_symbolic_id(dst));
    m_edge_labels.push_back(label);
}

template <StateRepresentation StateRepr>
void
GraphNN<StateRepr>::populate_with_goal()
{
    auto const goal_graph_string = TrainingDataset<StateRepr>::get_instance().get_goal_string();
    const std::regex pattern(R"((\d+)\s*->\s*(\d+)\s*\[label=\"(\d+)\"\];)");
    const std::sregex_iterator begin(goal_graph_string.begin(), goal_graph_string.end(), pattern);
    const std::sregex_iterator end;

    if (ArgumentParser::get_instance().get_dataset_merged())
    {
        const auto epsilon_id = TrainingDataset<KripkeState>::get_epsilon_node_id_int();
        const auto goal_parent_id = TrainingDataset<KripkeState>::get_goal_parent_id_int();

        add_edge(epsilon_id, goal_parent_id, TrainingDataset<KripkeState>::get_to_goal_edge_id_int());
    }

    for (auto it = begin; it != end; ++it)
    {
        const size_t src = std::stoul((*it)[1]);
        const size_t dst = std::stoul((*it)[2]);
        const size_t label = std::stoul((*it)[3]);

        add_edge(src, dst, label);
    }

    if (!ArgumentParser::get_instance().get_dataset_merged())
    {
        const auto options = torch::TensorOptions().dtype(torch::kInt64);
        const torch::Tensor edge_ids = torch::stack({
            torch::from_blob(m_edge_src.data(), {static_cast<int64_t>(m_edge_src.size())}, options),
            torch::from_blob(m_edge_dst.data(), {static_cast<int64_t>(m_edge_dst.size())}, options)
        });

        const torch::Tensor edge_attrs = torch::from_blob(m_edge_labels.data(),
                                                          {static_cast<int64_t>(m_edge_labels.size()), 1}, options);
        const torch::Tensor real_node_ids = torch::from_blob(m_real_node_ids.data(),
                                                             {
                                                                 static_cast<int64_t>(m_real_node_ids.size()), 1
                                                             }, options);

        m_goal_graph_tensor.edge_ids = edge_ids.clone();
        m_goal_graph_tensor.edge_attrs = edge_attrs.clone();
        m_goal_graph_tensor.real_node_ids = real_node_ids.clone();

        m_edge_dst.clear();
        m_edge_src.clear();
        m_edge_labels.clear();
        m_real_node_ids.clear();
        m_symbolic_id = 0;
    }

    m_edges_initial_size = m_edge_labels.size();
    m_node_ids_initial_size = m_real_node_ids.size();
    m_starting_symbolic_id = m_symbolic_id;
}

template <StateRepresentation StateRepr>
GraphTensor
GraphNN<StateRepr>::state_to_tensor_minimal(const KripkeState& kstate)
{
#ifdef DEBUG
    if (ArgumentParser::get_instance().get_dataset_mapped())
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNMappedNotSupportedError,
            "We do not support the mapped version with the inference in C++ (useless, hashing should be better).");
    }
#endif


    if (ArgumentParser::get_instance().get_dataset_merged())
    {
        const auto state_parent_id = kstate.get_pointed().get_id();

        add_edge(TrainingDataset<KripkeState>::get_epsilon_node_id_int(), state_parent_id,
                 TrainingDataset<KripkeState>::get_to_state_edge_id_int());
    }

    const auto& training_dataset = TrainingDataset<KripkeState>::get_instance();

    for (const auto& [from_pw, from_map] : kstate.get_beliefs())
    {
        const size_t src = from_pw.get_id();

        for (const auto& [agent, to_set] : from_map)
        {
            const int64_t label = static_cast<int64_t>(
                training_dataset.get_unique_a_id_from_map(agent));

            for (const auto& to_pw : to_set)
            {
                const size_t dst = to_pw.get_id();

                add_edge(src, dst, label);
            }
        }
    }

    const auto options = torch::TensorOptions().dtype(torch::kInt64);
    const torch::Tensor edge_ids = torch::stack({
        torch::from_blob(m_edge_src.data(), {static_cast<int64_t>(m_edge_src.size())}, options),
        torch::from_blob(m_edge_dst.data(), {static_cast<int64_t>(m_edge_dst.size())}, options)
    });

    const torch::Tensor edge_attrs = torch::from_blob(m_edge_labels.data(),
                                                      {static_cast<int64_t>(m_edge_labels.size()), 1},
                                                      options);
    const torch::Tensor real_node_ids = torch::from_blob(m_real_node_ids.data(),
                                                         {static_cast<int64_t>(m_real_node_ids.size()), 1}, options);

    GraphTensor ret;

    ret.edge_ids = edge_ids.clone();
    ret.edge_attrs = edge_attrs.clone();
    ret.real_node_ids = real_node_ids.clone();

    // Erase only the newly inserted elements
    m_edge_src.erase(m_edge_src.begin() + m_edges_initial_size, m_edge_src.end());
    m_edge_dst.erase(m_edge_dst.begin() + m_edges_initial_size, m_edge_dst.end());
    m_edge_labels.erase(m_edge_labels.begin() + m_edges_initial_size, m_edge_labels.end());
    m_real_node_ids.erase(m_real_node_ids.begin() + m_node_ids_initial_size, m_real_node_ids.end());
    m_symbolic_id = m_starting_symbolic_id;
}
