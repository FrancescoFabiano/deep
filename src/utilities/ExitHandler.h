#pragma once

#include <string>
#include <string_view>
#include <iostream>
#include <cstdlib>

#include "../argparse/ArgumentParser.h"

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
     * Includes codes related to \ref ArgumentParser.
     */
    enum class ExitCode : int {
        Success = 0,         ///< Program completed successfully.

        // ArgumentParser Related
        ArgParseError = 100,    ///< Error occurred during argument parsing in \ref ArgumentParser.
        ArgParseInstanceError = 101,    ///< Error occurred during argument parsing in \ref ArgumentParser.

        // Domain Related
        DomainFileOpenError = 200,   ///< Failed to open domain input file.
        DomainInstanceError = 201,   ///< Domain singleton instance error.
        DomainBuildError = 202,      ///< Error during domain build process.

        // Action Related
        ActionTypeConflict = 300,    ///< Conflicting action types detected.
        ActionInvalidExecutor = 301, ///< Invalid executor for action.
        ActionEffectError = 302,      ///< Error adding or processing action effect.


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