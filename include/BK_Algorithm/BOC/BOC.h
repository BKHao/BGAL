#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
namespace BKHao {
class _BOC {
 public:
  enum class _Sign {
    NegativE, ZerO, PositivE, FaileD
  };
  static inline double rand_() {
    return double(rand() / (RAND_MAX + 1.0));
  }
  static inline _Sign sign_(const double _real, const double &_precision = 1e-32) {
    if (abs(_real) < _precision)
      return _Sign::ZerO;
    else if (_real > 0)
      return _Sign::PositivE;
    else
      return _Sign::NegativE;
  }
  static inline double PI() {
    return 3.1415926535897932384626433832795;
  }
//		static int search_files_(const std::string& path, const std::string& ext, std::vector<std::string>& filenames);
};
}


