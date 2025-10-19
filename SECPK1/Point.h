/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef POINTH
#define POINTH

#include "Int.h"

#ifdef __APPLE__
// macOS system headers (e.g. <MacTypes.h>) define a struct Point in the global
// namespace. Undefine it so the Secp256k1 Point class keeps its original name
// without colliding with the system definition when compiling Objective-C++
// sources.
#undef Point
#endif

class Point {

public:

  Point();
  Point(Int *cx,Int *cy,Int *cz);
  Point(Int *cx, Int *cz);
  Point(const Point &p);
  ~Point();
  bool isZero();
  bool equals(Point &p);
  void Set(Point &p);
  void Set(Int *cx, Int *cy,Int *cz);
  void Clear();
  void Reduce();
  std::string toString();

  Int x;
  Int y;
  Int z;

};

#endif // POINTH
