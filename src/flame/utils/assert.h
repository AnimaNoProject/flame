/**
 * This file is part of FLaME.
 * Copyright (C) 2017 W. Nicholas Greene (wng@csail.mit.edu)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 * @file assert.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 18:22:29 (Fri)
 */

#pragma once

#ifdef FLAME_NO_ASSERT
#define FLAME_ASSERT(x)
#else
#include <cstdlib>   // for abort

#define FLAME_COLOR_RESET   "\033[0m"
#define FLAME_COLOR_RED     "\033[31m"
#define FLAME_COLOR_GREEN   "\033[32m"

namespace flame {

namespace utils {

inline void assert_fail(const char *condition, const char *function,
                        const char *file, int line) {
  fprintf(stderr, FLAME_COLOR_RED "FLAME_ASSERT failed: %s in function %s at %s: %i\n" FLAME_COLOR_RESET,
          condition, function, file, line);

  fprintf(stderr, FLAME_COLOR_RED "Stacktrace:\n" FLAME_COLOR_RESET);

  exit(1);
  // abort();
}

}  // namespace utils

}  // namespace flame

#define FLAME_ASSERT(condition) \
  do { \
    if (!(condition)) \
      flame::utils::assert_fail(#condition, __FUNCTION__, __FILE__, __LINE__); \
  } while (false)

#endif
