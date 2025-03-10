#include <iostream>
#include <tuple>

// Mux function template
template <std::size_t N, typename... Args>
auto mux(const std::tuple<Args...>& inputs) -> decltype(std::get<N>(inputs)) {
  return std::get<N>(inputs);
}

int main() {
  // Example usage of the mux function template
  int a = 10;
  float b = 20.5;
  char c = 'Z';

  // Using the mux function to select inputs based on the index
  std::cout << "Input at index 0: " << mux<0>(a, b, c) << std::endl;
  std::cout << "Input at index 1: " << mux<1>(a, b, c) << std::endl;
  std::cout << "Input at index 2: " << mux<2>(a, b, c) << std::endl;

  return 0;
}
