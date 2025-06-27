#pragma once

/// \file
/// \brief Precompiled headers for commonly used standard and third-party
/// libraries.

#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_NEURALNETS
  #include <onnxruntime_cxx_api.h>
#endif
