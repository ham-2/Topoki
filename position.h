#ifndef POSITION_INCLUDED
#define POSITION_INCLUDED

#include <string>
#include <sstream>
#include <bitset>
#include <memory.h>

#include "material.h"
#include "network.h"
#include "misc.h"
#include "pieces.h"

typedef uint64_t Key;

struct Undo {
	// To detect repetition, we store key in this struct
	// repetition values
	// positive : repeated once, distance to last occurance
	// negative : repeated twice, distance to last occurance
	// negative, <1024 : repeated > 2, distance to last occurance
	Key key;
	int repetition;
	int fifty_move;

	// Informations needed to undo a move. Stored in a stack
	Undo* prev;
	Piece captured;
	Square enpassant; // if en passant is possible
	int castling_rights; // WK, WQ, BK, BQ

	// Other useful informations
	Bitboard checkers;
	Square king_square[2];
};

class Position {
private:
	Undo* undo_stack;
	Piece squares[64];
	Bitboard pieces[7];
	Bitboard colors[2];
	int piece_count;
	Color side_to_move;

	int material_total;

	int m[2];

	// For eval
	alignas(32)
	int16_t accumulator[2 * SIZE_O0];
	Net* net;

	void pop_stack();
	void clear_stack();

	void rebuild();

	void place(Piece p, Square s);
	void remove(Piece p, Square s);
	void move_piece(Piece p, Square from, Square to);


public:
	Position(Net* n);

	static void init();

	void verify();
	Key get_key();
	inline Color get_side() { return side_to_move; }
	inline Piece get_piece(Square s) { return squares[s]; }
	inline Bitboard get_pieces(Color c) { return colors[c]; }
	inline Bitboard get_pieces(UPiece u) { return pieces[u]; }
	inline Bitboard get_pieces(Color c, UPiece u) { return colors[c] & pieces[u]; };
	inline Bitboard get_pieces(Color c, UPiece u1, UPiece u2) { return colors[c] & (pieces[u1] | pieces[u2]); }
	Bitboard get_attackers(Square s, Bitboard occupied);
	Bitboard see_least_piece(Color c, Bitboard attackers, UPiece& u);
	inline Bitboard get_checkers() { return undo_stack->checkers;  }
	inline Bitboard get_occupied() { return ~pieces[EMPTY]; }
	Bitboard get_pinned(Color c, Square k);
	inline Square get_king_square(Color c) { return undo_stack->king_square[c]; }
	inline Square get_enpassant() { return undo_stack->enpassant; }
	inline int get_piece_count() { return piece_count; }
	inline int get_fiftymove() { return undo_stack->fifty_move; }
	inline int get_repetition() { return undo_stack->repetition; }
	inline bool get_repetition(int root_dist) { 
		return undo_stack->repetition == 0 ? false :
			undo_stack->repetition > 0 ? (undo_stack->repetition <= root_dist) :
			true;
	}
	inline bool get_threefold() { return undo_stack->repetition < 0; }
	inline bool get_castling_tb() { return undo_stack->castling_rights; }
	inline bool get_castling_right(int i) {
		return bool(undo_stack->castling_rights & (1 << i));
	}
	inline bool has_castling_rights(Color c) {
		return c ? undo_stack->castling_rights & 12 : undo_stack->castling_rights & 3;
	}
	inline bool get_nh_condition() {
		return !(undo_stack->checkers || (piece_count < 8));
	};

	// functions for eval
	inline bool semiopen_file(Square s, Color c) { 
		return (popcount(FileBoard[get_file(s)] & pieces[PAWN] & colors[~c]) == 1);
	}
	inline bool is_passed_pawn(Square s, Color c) {
		return !bool(get_forward(c, s) & (get_fileboard(s) | adj_fileboard(s)) & pieces[PAWN] & colors[~c]);
	}
	inline bool is_passed_pawn_push(Move m, Color c) {
		return bool(get_from(m) & pieces[PAWN]) && is_passed_pawn(get_from(m), c);
	}
	inline bool material_capture(Move m) {
		return to_upiece(squares[get_to(m)]) > to_upiece(squares[get_from(m)]);
	}
	int see(Move m);
	bool is_check(Move m);
	inline bool is_capture(Move m)
	{
		return squares[get_to(m)] != EMPTY;
	}
	inline bool is_non_quiesce(Move m)
	{
		return get_movetype(m) == 0
			? (squares[get_to(m)] != EMPTY && see(m) > 0)
			: get_movetype(m) != 2;
	}
	inline Net* get_net() { return net; }
	inline void get_eval(int* dst) {
		compute(dst, accumulator, net, side_to_move);
	}
	inline int get_mg() {
		return material_total < 2000 ? 0 :
			material_total > 6000 ? 100 : 
			(material_total - 2000) / 40;
	}
	inline int get_material() {
		return side_to_move ? m[BLACK] - m[WHITE] : m[WHITE] - m[BLACK];
	}
	inline int16_t* get_accumulator() { return accumulator; }
	inline void set_accumulator() {
		compute_L0(accumulator, squares, net);
	}
	void show();
	void set(std::string fen);
	Move parse_move(std::string string_move);
	void do_move(Move m, Undo* new_undo);
	void undo_move(Move m);
	void do_null_move(Undo* new_undo);
	void undo_null_move();

	void set_random(int pieces, PRNG* r);

	Position& operator=(const Position board);

	friend std::ostream& operator<<(std::ostream& os, Position& pos);
};

std::ostream& operator<<(std::ostream& os, Position& pos);


#endif
