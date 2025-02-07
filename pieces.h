#ifndef PIECES_INCLUDED
#define PIECES_INCLUDED

#include <string>
	
enum Color : bool { WHITE = false, BLACK = true };

inline constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum UPiece : int {
	UEMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
};

enum Piece : int {
	EMPTY,
	W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, W_P1, W_P2,
	B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING, B_P1, B_P2
};

inline Color to_color(Piece p) {
	return Color(p > 8);
}

const std::string FEN_Pieces = " PNBRQK  pnbrqk";

bool parse_piece(char c, Piece& p);

char print_piece(Piece p);

inline UPiece to_upiece(Piece p) {
	return UPiece(int(p & 7));
}

inline Piece to_piece(UPiece u, Color c) {
	return c ? Piece(u + 8) : Piece(u);
}

#endif