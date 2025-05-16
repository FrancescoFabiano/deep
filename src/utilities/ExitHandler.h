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
        ArgParseError = 1    ///< Error occurred during argument parsing in \ref ArgumentParser.
    };

    // ArgumentParser Related
    /**
     * \brief Suggestion message for argument parsing errors.
     * \details Shown to the user when argument parsing fails in \ref ArgumentParser.
     */
    static constexpr std::string_view arg_parse_suggestion = "\n  Tip: Use -h or --help for usage information.";

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