#pragma once

#include <string>
#include <string_view>
#include <iostream>
#include <cstdlib>

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
class ExitHandler {
public:
    /**
     * \enum ExitCode
     * \brief Enumerates exit codes for program termination.
     *
     * Includes codes related to \ref ArgumentParser and BeliefFormula.
     */
    enum class ExitCode : int {
        Success = 0,         ///< Program completed successfully.
        ExitForCompiler = -1,         ///< Used after calling this class to satisfy the compiler when it does not recognize that the branch will exit.

        //ArgumentParser Related
        ArgParseError = 100,    ///< Error occurred during argument parsing in \ref ArgumentParser.
        ArgParseInstanceError = 101,    ///< Error occurred during argument parsing in \ref ArgumentParser.

        // Domain Related
        DomainFileOpenError = 200,   ///< Failed to open domain input file.
        DomainInstanceError = 201,   ///< Domain singleton instance error.
        DomainBuildError = 202,      ///< Error during domain build process.
        DomainUndeclaredFluent = 203,   ///< Undeclared fluent error.
        DomainUndeclaredAgent = 204,    ///< Undeclared agent error.
        DomainUndeclaredAction = 205,   ///< Undeclared action error.
        DomainInitialStateRestrictionError = 206, ///< Initial state restriction error.
        DomainInitialStateTypeError = 207,        ///< Initial state type error.

        // Action Related
        ActionTypeConflict = 300,    ///< Conflicting action types detected.
        ActionInvalidExecutor = 301, ///< Invalid executor for action.
        ActionEffectError = 302,      ///< Error adding or processing action effect.

        // Formula/Helper Related
        FormulaNonDeterminismError = 400, ///< Non-determinism in formula not supported.
        FormulaBadDeclaration = 401,      ///< Bad formula declaration.
        FormulaEmptyEffect = 402,         ///< Empty action effect.
        FormulaConsistencyError = 403,    ///< Consistency check failed in formula helper.

        // HelperPrint Related
        PrintUnsetGrounderError = 500, ///< Attempted to print with unset grounder.
        PrintNullPointerError = 501,   ///< Null pointer encountered during print.

        // BeliefFormula Related
        BeliefFormulaTypeUnset = 600,         ///< BeliefFormula type not set properly.
        BeliefFormulaEmptyFluent = 601,       ///< BeliefFormula has empty fluent formula.
        BeliefFormulaNotGrounded = 602,       ///< BeliefFormula not grounded when required.
        BeliefFormulaMissingNested = 603,     ///< BeliefFormula missing nested formula.
        BeliefFormulaOperatorUnset = 604,     ///< BeliefFormula operator not set properly.
        BeliefFormulaEmptyAgentGroup = 605,    ///< BeliefFormula has empty agent group.

        // KripkeWorldPointer Related
        KripkeWorldPointerNullError = 700,      ///< Null pointer dereference in KripkeWorldPointer.
        KripkeStorageInsertError = 701,         ///< Failed to insert or find KripkeWorld in KripkeStorage.
        KripkeWorldEntailmentError = 702,       ///< Failed to check for entailment of something in KripkeWorld.
    };

    // ArgumentParser Related
    /**
     * \brief Suggestion message for argument parsing errors.
     * \details Shown to the user when argument parsing fails in \ref ArgumentParser.
     */
    static constexpr std::string_view arg_parse_suggestion = "\n  Tip: Use -h or --help for usage information.";


    // Domain Related
    /**
    * \brief Suggestion message for domain creation errors.
    * \details Shown to the user when opening a domain file fails in \ref Domain.
    */
    static constexpr std::string_view domain_file_error = "\n  Tip: Check if the domain file exists and is accessible.";

    /**
     * \brief Exits the program with a message and exit code.
     * \param code The exit code to use (see \ref ExitHandler::ExitCode, e.g., for \ref ArgumentParser errors).
     * \param message The message to display before exiting.
     */
    static void exit_with_message(ExitCode code, const std::string_view message) {
        std::cerr << "\n" << message << std::endl;
        std::cerr << "\nProcess finished with exit code: " << static_cast<int>(code) << " (Mostly useful for development)\n" << std::endl;
        std::exit(static_cast<int>(code));
    }
};
