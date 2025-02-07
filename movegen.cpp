#include "movegen.h"

using namespace std;

void MoveList::generate(Position& board) {
	Move* list_ptr = list;
	Color c = board.get_side();
	Bitboard occupied = board.get_occupied();
	Square king_square = board.get_king_square(c);
	Bitboard pinned = board.get_pinned(c, king_square);
	Bitboard to_board;

	if (board.get_checkers()) {
		// i) Block / Capture
		Bitboard checkers = board.get_checkers();
		to_board = ~board.get_pieces(c);
		Square checker;
		while (checkers) {
			checker = pop_lsb(&checkers);
			switch (to_upiece(board.get_piece(checker))) {
			case PAWN: {
				to_board &= SquareBoard[checker];
				break;
			}
			case KNIGHT: {
				to_board &= SquareBoard[checker];
				break;
			}
			default: { // sliding pieces
				to_board &= Rays[checker][king_square]
				& ((FullBoard << checker) ^ (FullBoard << king_square))
				| SquareBoard[checker];
				break;
			}
			}
		}
			
		list_ptr = add_pawn_moves(list_ptr, board, pinned, to_board);
		list_ptr = add_moves<KNIGHT>(list_ptr, board, pinned, to_board);
		list_ptr = add_moves<BISHOP>(list_ptr, board, pinned, to_board);
		list_ptr = add_moves<ROOK>(list_ptr, board, pinned, to_board);
		list_ptr = add_moves<QUEEN>(list_ptr, board, pinned, to_board);
	}
	else {
		// i) Normal moves
		to_board = ~board.get_pieces(c);
		list_ptr = add_pawn_moves(list_ptr, board, pinned, to_board);
		list_ptr = add_moves<KNIGHT>(list_ptr, board, pinned, to_board);
		list_ptr = add_moves<BISHOP>(list_ptr, board, pinned, to_board);
		list_ptr = add_moves<ROOK>(list_ptr, board, pinned, to_board);
		list_ptr = add_moves<QUEEN>(list_ptr, board, pinned, to_board);

		// ii) Castling
		if (c) {
			if (board.get_castling_right(2)) {
				if (check_castling(F8, G8, board)) {
					*(list_ptr++) = make_move(E8, G8, 2, 0);
				}
			}
			if (board.get_castling_right(3)) {
				if (check_castling(C8, D8, board)
					&& (board.get_piece(B8) == EMPTY)) {
					*(list_ptr++) = make_move(E8, C8, 2, 0);
				}
			}
		}
		else {
			if (board.get_castling_right(0)) {
				if (check_castling(F1, G1, board)) {
					*(list_ptr++) = make_move(E1, G1, 2, 0);
				}
			}
			if (board.get_castling_right(1)) {
				if (check_castling(C1, D1, board)
					&& (board.get_piece(B1) == EMPTY)) {
					*(list_ptr++) = make_move(E1, C1, 2, 0);
				}
			}
		}

	}

	// King Moves
	to_board = PseudoAttacks[KING][king_square];
	to_board &= ~board.get_pieces(c);
	occupied ^= SquareBoard[king_square];
	Bitboard tb = to_board;
	Square s;
	while (tb) {
		s = pop_lsb(&tb);
		if (board.get_attackers(s, occupied) &
			board.get_pieces(~c)) {
			to_board ^= SquareBoard[s];
		}
	}
	list_ptr = add_moves<KING>(list_ptr, board, pinned, to_board);

	end = list_ptr;
}

void MoveList::show() {
	for (Move* m = list; m < end; m++) {
		cout << *m << " ";
	}
	cout << endl;
}

template<UPiece u>
Move* MoveList::add_moves(Move* list_ptr, Position& board, Bitboard pinned, Bitboard _to_board) {
	// u != PAWN - use add_pawn_moves
	Color c = board.get_side();
	Bitboard from_board = board.get_pieces(c, u);
	Bitboard to_board;
	Square from, to;
	while (from_board) {
		from = pop_lsb(&from_board);
		to_board = _to_board & attacks<u>(from, board.get_occupied());
		if (SquareBoard[from] & pinned) {
			to_board &= Rays[from][board.get_king_square(c)];
		}
		while (to_board) {
			to = pop_lsb(&to_board);
			*(list_ptr++) = make_move(from, to);
		}
	}
	return list_ptr;
}

inline bool check_pawn_move(Square from, Square to, Square k, Bitboard pinned) {
	// helper - have to check if pinned
	if (SquareBoard[from] & pinned) { 
		return SquareBoard[to] & Rays[from][k];
	}
	return true;
}

inline bool check_enpassant(Square from, Square to, Color c, Position& board) {
	Square cp = c ? Square(to + 8) : Square(to - 8);
	return !(board.get_attackers(board.get_king_square(c),
		board.get_occupied() ^ SquareBoard[from] ^ SquareBoard[to] ^ SquareBoard[cp]) &
		(board.get_pieces(~c) ^ SquareBoard[cp]));
}

Move* MoveList::add_pawn_moves(Move* list_ptr, Position& board, Bitboard pinned, Bitboard _to_board) {
	Color c = board.get_side();
	Square k = board.get_king_square(c);
	Bitboard pawns = board.get_pieces(c, PAWN);
	Bitboard promotable = c ? pawns & RankBoard[1] : pawns & RankBoard[6];
	Bitboard rank3 = c ? RankBoard[5] : RankBoard[2];
	int advance = c ? -8 : 8;
	int p_left = c ? -9 : 7;
	int p_right = c ? -7 : 9;
	Square from, to;

	if (to = board.get_enpassant()) {
		Bitboard ep = SquareBoard[to];
		Bitboard ep_to = c ? shift<8>(ep) : shift<-8>(ep);
		Bitboard cp_left = c ? shift<-9>(pawns) : shift<7>(pawns);
		Bitboard cp_right = c ? shift<-7>(pawns) : shift<9>(pawns);
		if (ep & _to_board ||
			_to_board == ep_to) { // get out of check by en passant
			// first handle en passant and remove from _to_board
			if (cp_left & ep) {
				from = Square(int(to) - p_left);
				if (check_pawn_move(from, to, k, pinned) &&
					check_enpassant(from, to, c, board)) {
					*(list_ptr++) = make_move(from, to, 3, 0);
				}
			}
			if (cp_right & ep) {
				from = Square(int(to) - p_right);
				if (check_pawn_move(from, to, k, pinned) &&
					check_enpassant(from, to, c, board)) {
					*(list_ptr++) = make_move(from, to, 3, 0);
				}
			}
			_to_board ^= ep;
		}
	}
		
	if (promotable) { // promotions
		Bitboard pa = (c ? shift<-8>(promotable) : shift<8>(promotable))
			& ~board.get_occupied() & _to_board;
		Bitboard pl = (c ? shift<-9>(promotable) : shift<7>(promotable))
			& board.get_pieces(~c) & _to_board;
		Bitboard pr = (c ? shift<-7>(promotable) : shift<9>(promotable))
			& board.get_pieces(~c) & _to_board;

		while (pa) {
			to = pop_lsb(&pa);
			from = Square(int(to) - advance);
			if (check_pawn_move(from, to, k, pinned)) {
				*(list_ptr++) = make_move(from, to, 1, 3);
				*(list_ptr++) = make_move(from, to, 1, 0);
				*(list_ptr++) = make_move(from, to, 1, 2);
				*(list_ptr++) = make_move(from, to, 1, 1);
			}
		}

		while (pl) {
			to = pop_lsb(&pl);
			from = Square(int(to) - p_left);
			if (check_pawn_move(from, to, k, pinned)) {
				*(list_ptr++) = make_move(from, to, 1, 3);
				*(list_ptr++) = make_move(from, to, 1, 0);
				*(list_ptr++) = make_move(from, to, 1, 2);
				*(list_ptr++) = make_move(from, to, 1, 1);
			}
		}

		while (pr) {
			to = pop_lsb(&pr);
			from = Square(int(to) - p_right);
			if (check_pawn_move(from, to, k, pinned)) {
				*(list_ptr++) = make_move(from, to, 1, 3);
				*(list_ptr++) = make_move(from, to, 1, 0);
				*(list_ptr++) = make_move(from, to, 1, 2);
				*(list_ptr++) = make_move(from, to, 1, 1);
			}
		}

		pawns ^= promotable;
	}

	Bitboard pa = (c ? shift<-8>(pawns) : shift<8>(pawns)) & ~board.get_occupied();
	Bitboard paa = (c ? shift<-8>(pa & rank3) : shift<8>(pa & rank3))
		& ~board.get_occupied() & _to_board;
	pa &= _to_board; // need to do this later : might block check with double advance
	Bitboard pl = (c ? shift<-9>(pawns) : shift<7>(pawns)) & board.get_pieces(~c) & _to_board;
	Bitboard pr = (c ? shift<-7>(pawns) : shift<9>(pawns)) & board.get_pieces(~c) & _to_board;

	while (pa) {
		to = pop_lsb(&pa);
		from = Square(int(to) - advance);
		if (check_pawn_move(from, to, k, pinned)) {
			*(list_ptr++) = make_move(from, to);
		}
	}

	while (paa) {
		to = pop_lsb(&paa);
		from = Square(int(to) - 2 * advance);
		if (check_pawn_move(from, to, k, pinned)) {
			*(list_ptr++) = make_move(from, to);
		}
	}

	while (pl) {
		to = pop_lsb(&pl);
		from = Square(int(to) - p_left);
		if (check_pawn_move(from, to, k, pinned)) {
			*(list_ptr++) = make_move(from, to);
		}
	}

	while (pr) {
		to = pop_lsb(&pr);
		from = Square(int(to) - p_right);
		if (check_pawn_move(from, to, k, pinned)) {
			*(list_ptr++) = make_move(from, to);
		}
	}

	return list_ptr;
}

bool MoveList::check_castling(Square low, Square high, Position& board)
{
	for (low; low <= high; ++low) {
		if (board.get_piece(low) != EMPTY) { return false; }
		if (board.get_attackers(low, board.get_occupied()) 
			& board.get_pieces(~board.get_side())) { return false; }
	}
	return true;
}