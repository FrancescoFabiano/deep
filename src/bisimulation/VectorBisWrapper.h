#pragma once

#include <vector>
#include <string>
#include "ExitHandler.h"

/**
 * @brief A wrapper around std::vector with bounds-checked operator[] and custom error handling.
 *
 * Everything else is a direct pass-through to std::vector.
 *
 * @tparam T Element type.
 */
template<typename T>
class VectorBisWrapper {
private:
    std::vector<T> data;

public:
    // Constructors
    VectorBisWrapper() = default;

    VectorBisWrapper(std::initializer_list<T> init)
        : data(init) {}

    explicit VectorBisWrapper(size_t n)
        : data(n) {}

    explicit VectorBisWrapper(size_t n, const T& value)
        : data(n, value) {}

    // Bounds-checked element access
    // Maybe modify bounds to allow for bisimulation
    T& operator[](size_t index) {
        if (index >= data.size()) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BisimulationWrapperOutOfBounds,
                "Index " + std::to_string(index) + " is out of bounds in one of the bisimulation vectors"
            );
        }
        return data[index];
    }

    const T& operator[](size_t index) const {
        if (index >= data.size()) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BisimulationWrapperOutOfBounds,
                "Index " + std::to_string(index) + " is out of bounds in one of the bisimulation vectors"
            );
        }
        return data[index];
    }

    // Size and capacity
    [[nodiscard]] size_t size() const { return data.size(); }
    [[nodiscard]] size_t capacity() const { return data.capacity(); }
    [[nodiscard]] bool empty() const { return data.empty(); }

    // Modifiers
    void push_back(const T& value) { data.push_back(value); }
    void pop_back() { data.pop_back(); }
    void clear() { data.clear(); }
    void resize(size_t new_size) { data.resize(new_size); }
    void resize(size_t new_size, const T& value) { data.resize(new_size, value); }
    void reserve(size_t new_capacity) { data.reserve(new_capacity); }

    // Element access
    T& front() { return data.front(); }
    const T& front() const { return data.front(); }
    T& back() { return data.back(); }
    const T& back() const { return data.back(); }

    // Iterators
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    auto cbegin() const { return data.cbegin(); }
    auto cend() const { return data.cend(); }

    // Underlying vector access (if needed)
    std::vector<T>& raw() { return data; }
    const std::vector<T>& raw() const { return data; }
};
