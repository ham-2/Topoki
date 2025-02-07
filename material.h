#ifndef MATERIAL_INCLUDED
#define MATERIAL_INCLUDED

#include <cstdint>

#include "pieces.h"
#include "board.h"

enum Score : int {};

constexpr Score S(int mg, int eg) {
    return Score(mg | ((unsigned int)eg << 16));
}

inline int get_eg(Score s) {
    union { uint16_t u; int16_t s; } eg = { uint16_t(unsigned(s + 0x8000) >> 16) };
    return int(eg.s);
}

inline int get_mg(Score s) {
    union { uint16_t u; int16_t s; } mg = { uint16_t(unsigned(s)) };
    return int(mg.s);
}

constexpr Score operator+(Score s1, Score s2) { return Score(int(s1) + s2); }
constexpr Score operator-(Score s1, Score s2) { return Score(int(s1) - s2); }
inline Score& operator+=(Score& s1, Score s2) { return s1 = s1 + s2; }
inline Score& operator-=(Score& s1, Score s2) { return s1 = s1 - s2; }
constexpr Score operator-(Score s) { return Score(-int(s)); }
constexpr Score operator*(Score s, int i) { return Score(int(s) * i); }

constexpr int pawn_eg = 204; // score - cp conversion
constexpr int Tempo = 20;

constexpr int Material_MG[7] = {
    0, // EMPTY
    142, // PAWN
    400, // KNIGHT
    378, // BISHOP
    704, // ROOK
    1496, // QUEEN
    0, // KING
};

constexpr Score Material[7] = {
    S(Material_MG[0], 0), // EMPTY
    S(Material_MG[1], pawn_eg ), // PAWN
    S(Material_MG[2], 618 ), // KNIGHT
    S(Material_MG[3], 639 ), // BISHOP
    S(Material_MG[4], 1174), // ROOK
    S(Material_MG[5], 2217), // QUEEN
    S(Material_MG[6], 0   ), // KING
};

#endif