#pragma once

#include <iostream>
#include <string>
#include <string_view>

/**
 * \class ExitHandler
 * \brief Utility class for handling program exits with error codes and
 * messages.
 *
 * Provides a type-safe way to exit the program with a specific code and
 * message. Useful for consistent error handling and messaging throughout the
 * application.
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
   * Codes are grouped by subsystem for clarity.
   */
enum class ExitCode : int {

    NotImplementedError = -100,

  // --- General (0-99) ---
  ExitForCompiler = -1, ///< Used to satisfy the compiler when it does not
                         ///< recognize that the branch will exit.
  SuccessFoundGoal = 0, ///< Program completed successfully finding a goal.
  SuccessNotFoundGoal =
      1, ///< Program completed successfully without finding a goal.
  SuccessNotPlanningMode =
      2, ///< Program completed successfully without planning mode.
  SuccessNotPlanningModeWarning =
      3, ///< Program completed successfully but something is not as it
         ///< should be.

  // ========================================================================
  // Argument / Input
  // ========================================================================

  // --- ArgumentParser Related (100-119) ---
  ArgParseError = 100,
  ArgParseInstanceError = 101,

  // Reserved: 102-119

  // --- Parsing Related (150-169) ---
  ParsingError = 150,

  // Reserved: 151-169

  // ========================================================================
  // Domain / Action / Event Model
  // ========================================================================

  // --- Domain Related (200-219) ---
  DomainFileOpenError = 200,
  DomainInstanceError = 201,
  DomainBuildError = 202,
  DomainUndeclaredFluent = 203,
  DomainUndeclaredAgent = 204,
  DomainUndeclaredAction = 205,
  DomainInitialStateRestrictionError = 206,
  DomainInitialStateTypeError = 207,

  // Reserved: 208-219

  // --- Action / Event Model Related (300-319) ---
  ActionTypeConflict = 300,
  ActionInvalidExecutor = 301,
  ActionEffectError = 302,
  ActionDuplicatePostcondition = 303,

  // Reserved for future Event / DEL errors:
  // 304-309

  // Reserved for future Action errors:
  // 310-319

  // ========================================================================
  // Formulae
  // ========================================================================

  // --- Formula/Helper Related (400-419) ---
  FormulaNonDeterminismError = 400,
  FormulaBadDeclaration = 401,
  FormulaEmptyEffect = 402,
  FormulaConsistencyError = 403,

  // Reserved: 404-419

  // --- HelperPrint Related (500-519) ---
  PrintUnsetGrounderError = 500,
  PrintNullPointerError = 501,

  // Reserved: 502-519

  // --- BeliefFormula Related (600-619) ---
  BeliefFormulaTypeUnset = 600,
  BeliefFormulaEmptyFluent = 601,
  BeliefFormulaNotGrounded = 602,
  BeliefFormulaMissingNested = 603,
  BeliefFormulaOperatorUnset = 604,
  BeliefFormulaEmptyAgentGroup = 605,

  // Reserved: 606-619

  // ========================================================================
  // Search / Heuristics / Bisimulation
  // ========================================================================

  // --- Heuristics Related (650-669) ---
  HeuristicsBadDeclaration = 650,

  // Reserved: 651-669

  // --- Bisimulation Related (670-679) ---
  SearchBisimulationError = 670,

  // Reserved: 671-679

  // --- KripkeWorldPointer / Storage Related (700-719) ---
  KripkeWorldPointerNullError = 700,
  KripkeWorldPointerIdError = 701,
  KripkeStorageInsertError = 702,
  KripkeWorldEntailmentError = 703,

  // Reserved: 704-719

  // --- Bisimulation Related (800-819) ---
  BisimulationFailed = 800,
  BisimulationWrapperOutOfBounds = 801,

  // Reserved: 802-819

  // --- BreadthFirst Related (850-869) ---
  SearchNoActions = 850,

  // --- PlanningGraph Related (851-859) ---
  PlanningGraphErrorInitialState = 851,

  // Reserved: 852-859

  // --- PortfolioSearch Related (860-879) ---
  PortfolioConfigFileError = 860,
  PortfolioConfigError = 861,
  PortfolioConfigFieldError = 862,
  SearchParallelNotImplemented = 863,
  SearchMethodNotImplemented = 864,
  SearchMethodError = 865,

  // Reserved: 866-879

  // ========================================================================
  // Neural Networks
  // ========================================================================

  // --- NN Related (880-889) ---
  NNTrainingFileError = 880,
  NNMappingError = 881,
  NNInstanceError = 882,
  NNDirectoryCreationError = 883,
  DatasetGenerationTypeWrong = 884,

  // Reserved: 885-889

  // --- GNN Related (890-899) ---
  GNNInstanceError = 890,
  GNNFileError = 891,
  GNNScriptError = 892,
  GNNMappedNotSupportedError = 893,
  GNNTensorTranslationError = 894,
  GNNModelLoadError = 895,
  GNNBitmaskLengthError = 896,
  GNNBitmaskRepetitionError = 897,
  GNNBitmaskGOALError = 898,

  // Reserved: 899

  // --- FringeEvalRL Related (900-919) ---
  FringeEvalInstanceError = 900,
  FringeEvalFileError = 901,
  FringeEvalScriptError = 902,
  FringeEvalMappedNotSupportedError = 903,
  FringeEvalTensorTranslationError = 904,
  FringeEvalModelLoadError = 905,
  FringeEvalBitmaskLengthError = 906,
  FringeEvalBitmaskRepetitionError = 907,
  FringeEvalBitmaskGOALError = 908,
  FringeNotImplementedError = 919,

  // Reserved: 909-918

  // ========================================================================
  // State / Action Execution
  // ========================================================================

  // --- State / Action Related (1000-1019) ---
  StateActionNotExecutableError = 1000,

  // Reserved for future state-transition / DEL errors:
  // 1001-1009

  // Reserved for future state-model errors:
  // 1010-1019
};

  // ArgumentParser Related
  /**
   * \brief Suggestion message for argument parsing errors.
   * \details Shown to the user when argument parsing fails in ArgumentParser.
   */
  static constexpr std::string_view arg_parse_suggestion =
      "\n  Tip: Use -h or --help for usage information.";

  // Domain Related
  /**
   * \brief Suggestion message for domain creation errors.
   * \details Shown to the user when opening a domain file fails in Domain.
   */
  static constexpr std::string_view domain_file_error =
      "\n  Tip: Check if the domain file exists and is accessible.";

  /**
   * \brief Exits the program with a message and exit code.
   * \param code The exit code to use (see ExitHandler::ExitCode).
   * \param message The message to display before exiting.
   */
  static void exit_with_message(ExitCode code, const std::string_view message) {
    std::cerr << "\n[ERROR] " << message << std::endl;
    std::cerr << "\nError code: " << static_cast<int>(code)
              << " (Mostly useful for development)\n"
              << std::endl;
    std::exit(static_cast<int>(code));
  }
};
