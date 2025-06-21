#pragma once

#include <string>
#include <string_view>
#include <iostream>

/**
 * \class ExitHandler
 * \brief Utility class for handling program exits with error codes and messages.
 *
 * Provides a type-safe way to exit the program with a specific code and message.
 * Useful for consistent error handling and messaging throughout the application.
 *
 * \author Francesco Fabiano
 * \date May 2025
 */
class ExitHandler
{
public:
    /**
     * \enum ExitCode
     * \brief Enumerates exit codes for program termination.
     *
     * Codes are grouped by subsystem for clarity.
     */
    enum class ExitCode : int
    {
        // --- General ---
        ExitForCompiler = -1, ///< Used to satisfy the compiler when it does not recognize that the branch will exit.
        SuccessFoundGoal = 0, ///< Program completed successfully finding a goal.
        SuccessNotFoundGoal = 1, ///< Program completed successfully without finding a goal.
        SuccessNotPlanningMode = 2, ///< Program completed successfully without planning mode.
        SuccessNotPlanningModeWarning = 3, ///< Program completed successfully but something is not as it should be.

        // --- ArgumentParser Related (100-119) ---
        ArgParseError = 100, ///< Error occurred during argument parsing in ArgumentParser.
        ArgParseInstanceError = 101, ///< Error occurred during ArgumentParser singleton instance.

        // --- Parsing Related (150-169) ---
        ParsingError = 150, ///< Error occurred during parsing in Reader.

        // --- Domain Related (200-219) ---
        DomainFileOpenError = 200, ///< Failed to open domain input file.
        DomainInstanceError = 201, ///< Domain singleton instance error.
        DomainBuildError = 202, ///< Error during domain build process.
        DomainUndeclaredFluent = 203, ///< Undeclared fluent error.
        DomainUndeclaredAgent = 204, ///< Undeclared agent error.
        DomainUndeclaredAction = 205, ///< Undeclared action error.
        DomainInitialStateRestrictionError = 206, ///< Initial State restriction error.
        DomainInitialStateTypeError = 207, ///< Initial State type error.

        // --- Action Related (300-319) ---
        ActionTypeConflict = 300, ///< Conflicting action types detected.
        ActionInvalidExecutor = 301, ///< Invalid executor for action.
        ActionEffectError = 302, ///< Error adding or processing action effect.

        // --- Formula/Helper Related (400-419) ---
        FormulaNonDeterminismError = 400, ///< Non-determinism in formula not supported.
        FormulaBadDeclaration = 401, ///< Bad formula declaration.
        FormulaEmptyEffect = 402, ///< Empty action effect.
        FormulaConsistencyError = 403, ///< Consistency check failed in formula helper.

        // --- HelperPrint Related (500-519) ---
        PrintUnsetGrounderError = 500, ///< Attempted to print with unset grounder.
        PrintNullPointerError = 501, ///< Null pointer encountered during print.

        // --- BeliefFormula Related (600-619) ---
        BeliefFormulaTypeUnset = 600, ///< BeliefFormula type not set properly.
        BeliefFormulaEmptyFluent = 601, ///< BeliefFormula has empty fluent formula.
        BeliefFormulaNotGrounded = 602, ///< BeliefFormula not grounded when required.
        BeliefFormulaMissingNested = 603, ///< BeliefFormula missing nested formula.
        BeliefFormulaOperatorUnset = 604, ///< BeliefFormula operator not set properly.
        BeliefFormulaEmptyAgentGroup = 605, ///< BeliefFormula has empty agent group.

        // --- Heuristics Related (650-669) ---
        HeuristicsBadDeclaration = 650, ///< Heuristic type not declared properly in HeuristicsManager.

        // --- Bisimulation Related (670-679) ---
        SearchBisimulationError = 670, ///< Bisimulation reduction failed. Some formulae are entailed differently from bisimilar states.

        // --- KripkeWorldPointer/Storage Related (700-719) ---
        KripkeWorldPointerNullError = 700, ///< Null pointer dereference in KripkeWorldPointer.
        KripkeWorldPointerIdError = 701, ///< KripkeWorldPointer id overflow or underflow.
        KripkeStorageInsertError = 702, ///< Failed to insert or find KripkeWorld in KripkeStorage.
        KripkeWorldEntailmentError = 703, ///< Failed to check for entailment in KripkeWorld.

        // --- Bisimulation Related (800-819) ---
        BisimulationFailed = 800, ///< Bisimulation minimization failed.
        BisimulationWrapperOutOfBounds = 801, ///< Accessed out of bounds in Bisimulation wrapper.

        // --- BreadthFirst Search Related (850-869) ---
        SearchNoActions = 850, ///< No actions available in the domain for the search.

        // --- PlanningGraph Related (851-859) ---
        PlanningGraphErrorInitialState = 851,
        ///< Error in initial state of PlanningGraph, it is goal so we should not generate it.

        // --- PortfolioSearch Related (860-879) ---
        PortfolioConfigFileError = 860, ///< Could not open portfolio configuration file.
        PortfolioConfigError = 861, ///< Internal error in portfolio search.
        PortfolioConfigFieldError = 862, ///< Error in reading a field in portfolio configuration file.
        SearchParallelNotImplemented = 863, ///< Parallel search is not implemented yet.

        // --- NN Related (880-889) ---
        NNTrainingFileError = 880, ///< Error opening NN training files.
        NNMappingError = 881, ///< Error in accessing NN mapped data.
        NNInstanceError = 882, ///< NN singleton instance error.
        NNDirectoryCreationError = 883, ///< Error creating NN output directories.

        // --- GNN Related (890-899) ---
        GNNInstanceError = 890, ///< Error in creating GNN training files.
        GNNFileError = 891, ///< Error in accessing a file.
        GNNScriptError = 892, ///< Error in running the GNN script file.

        // --- State/Action Related (900-919) ---
        StateActionNotExecutableError = 900, ///< Action not executable in state where it is supposed to be.
    };

    // ArgumentParser Related
    /**
     * \brief Suggestion message for argument parsing errors.
     * \details Shown to the user when argument parsing fails in ArgumentParser.
     */
    static constexpr std::string_view arg_parse_suggestion = "\n  Tip: Use -h or --help for usage information.";

    // Domain Related
    /**
     * \brief Suggestion message for domain creation errors.
     * \details Shown to the user when opening a domain file fails in Domain.
     */
    static constexpr std::string_view domain_file_error = "\n  Tip: Check if the domain file exists and is accessible.";

    /**
     * \brief Exits the program with a message and exit code.
     * \param code The exit code to use (see ExitHandler::ExitCode).
     * \param message The message to display before exiting.
     */
    static void exit_with_message(ExitCode code, const std::string_view message)
    {
        std::cerr << "\n" << message << std::endl;
        std::cerr << "\nProcess finished with exit code: " << static_cast<int>(code) <<
            " (Mostly useful for development)\n" << std::endl;
        std::exit(static_cast<int>(code));
    }
};
