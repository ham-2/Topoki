#include "pieces.h"

using namespace std;

bool parse_piece(char c, Piece& p) { 
	size_t idx = FEN_Pieces.find(c);
	if (idx == string::npos) { return false; }
	p = Piece(idx);
	return true;
}

char print_piece(Piece p) { return FEN_Pieces[p]; }