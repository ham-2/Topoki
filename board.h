#ifndef BOARD_INCLUDED
#define BOARD_INCLUDED

#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifdef _WIN64
#include <nmmintrin.h>
#include <immintrin.h>
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "pieces.h"

enum Square : int {
	A1, B1, C1, D1, E1, F1, G1, H1,
	A2, B2, C2, D2, E2, F2, G2, H2,
	A3, B3, C3, D3, E3, F3, G3, H3,
	A4, B4, C4, D4, E4, F4, G4, H4,
	A5, B5, C5, D5, E5, F5, G5, H5,
	A6, B6, C6, D6, E6, F6, G6, H6,
	A7, B7, C7, D7, E7, F7, G7, H7,
	A8, B8, C8, D8, E8, F8, G8, H8,
	SQ_END
};

inline Square operator+(Square s, int i) { return Square(int(s) + i); }
inline Square operator-(Square s, int i) { return Square(int(s) - i); }
inline Square& operator++(Square& s) { return s = Square(int(s) + 1); }
inline Square& operator+=(Square& s, int i) { return s = Square(int(s) + i); }

Square parse_square(char file, char rank);

inline int get_file(Square s) { return s & 7; }
inline int get_rank(Square s) { return s >> 3; }
inline Square mirror(Square s) { return Square(s ^ 56); }

std::ostream& operator<<(std::ostream& os, Square s);

// Use 16bit integer for moves - 0bXXAABBBBBBCCCCCC
// C: 6 bits indicating from square
// B: 6 bits indicating to square
// A: 00: knight 01: bishop 10: rook 11: queen promotion
// X: 00: normal 01: promotion 10: castling 11: en passant
enum Move : uint16_t {
	NULL_MOVE = 0
};

std::ostream& operator<<(std::ostream& os, Move m);

inline Move make_move(Square from, Square to, int type = 0, int promotion = 0) {
	return Move(from | (to << 6) | (promotion << 12) | (type << 14));
}

inline constexpr Square get_from(Move m) { return Square(m & 0x3F); }
inline constexpr Square get_to(Move m)   { return Square((m >> 6) & 0x3F); }
inline constexpr UPiece get_promotion(Move m) { return UPiece(((m >> 12) & 3) + KNIGHT); }
inline constexpr int get_movetype(Move m) { return (m >> 14); }

typedef uint64_t Bitboard;
constexpr Bitboard EmptyBoard = 0;
constexpr Bitboard FullBoard  = ~EmptyBoard;
	
extern Bitboard SquareBoard[64];
extern Bitboard PseudoAttacks[7][64];
extern Bitboard Rays[64][64];
extern int Distance[64][64];

constexpr Bitboard FileBoard[8] = {
	0x0101010101010101ULL,
	0x0101010101010101ULL << 1,
	0x0101010101010101ULL << 2,
	0x0101010101010101ULL << 3,
	0x0101010101010101ULL << 4,
	0x0101010101010101ULL << 5,
	0x0101010101010101ULL << 6,
	0x0101010101010101ULL << 7
};

constexpr Bitboard RankBoard[8] = {
	0xFFULL      , 0xFFULL << 8 , 0xFFULL << 16, 0xFFULL << 24,
	0xFFULL << 32, 0xFFULL << 40, 0xFFULL << 48, 0xFFULL << 56
};

constexpr Bitboard LightSquares = 0x55AA55AA55AA55AAULL;
constexpr Bitboard DarkSquares = ~LightSquares;

constexpr Bitboard CenterFiles = FileBoard[3] | FileBoard[4];
constexpr Bitboard BoardEdges = FileBoard[0] | FileBoard[7] | RankBoard[0] | RankBoard[7];
constexpr Bitboard White6L = FullBoard >> 16;
constexpr Bitboard Black6L = FullBoard << 16;

inline Bitboard get_fileboard(Square s) { return FileBoard[get_file(s)]; }
inline Bitboard get_rankboard(Square s) { return RankBoard[get_rank(s)]; }
inline Bitboard get_forward(Color c, Square s) {
	return c ? ~RankBoard[7] >> (8 * (7 ^ get_rank(s))) :
				~RankBoard[0] << (8 * get_rank(s));
}

void print(std::ostream& os, Bitboard b);

inline int popcount(Bitboard b) {
	return (int)_mm_popcnt_u64(b);
}

inline Bitboard pext(Bitboard b, Bitboard r) {
	return (Bitboard)_pext_u64(b, r);
}

inline Square lsb_square(Bitboard b) {
#ifdef _WIN64
	unsigned long _index = 0;
	_BitScanForward64(&_index, b);
	return Square(_index);
#else
	return Square(__builtin_ctzll(b));
#endif
}

inline Square pop_lsb(Bitboard* b) {
	Square s = lsb_square(*b);
	*b &= *b - 1;
	return s;
}

template <int T>
constexpr Bitboard shift(Bitboard b) {
	return T == -9 ? (b & ~FileBoard[0]) >> 9
		: T == -8 ? b >> 8
		: T == -7 ? (b & ~FileBoard[7]) >> 7
		: T == -1 ? (b & ~FileBoard[0]) >> 1
		: T == 1 ? (b & ~FileBoard[7]) << 1
		: T == 7 ? (b & ~FileBoard[0]) << 7
		: T == 8 ? b << 8
		: T == 9 ? (b & ~FileBoard[7]) << 9
		: 0;
}

inline Bitboard adj_fileboard(Square s) {
	return shift<1>(get_fileboard(s)) | shift<-1>(get_fileboard(s));
}

struct Attacks {
	Bitboard relevant;
	Bitboard* attacks;
};

extern Attacks BishopAttacks[64];
extern Attacks RookAttacks[64];

inline Bitboard index_attack(Attacks a, Bitboard occupied) {
	return a.attacks[pext(occupied, a.relevant)];
}

template<Color side>
Bitboard attacks_pawn(Bitboard b) {
	switch (side) {
	case WHITE: {
		return shift<7>(b) | shift<9>(b);
	}
	case BLACK: {
		return shift<-7>(b) | shift<-9>(b);
	}
	default:
		return 0;
	}
}

template<Color side>
Bitboard attacks_pawn(Square s) {
	return PseudoAttacks[int(side)][s];
}

template<UPiece attacker>
Bitboard attacks(Square s, Bitboard occupied) {
	switch (attacker) {
	case KNIGHT: {
		return PseudoAttacks[KNIGHT][s];
	}
	case BISHOP: {
		return index_attack(BishopAttacks[s], occupied);
	}
	case ROOK: {
		return index_attack(RookAttacks[s], occupied);
	}
	case QUEEN: {
		return index_attack(BishopAttacks[s], occupied) 
				| index_attack(RookAttacks[s], occupied);
	}
	case KING: {
		return PseudoAttacks[KING][s];
	}
	default:
		return 0;
	}
}

namespace Board {
	void init();
}

#endif
