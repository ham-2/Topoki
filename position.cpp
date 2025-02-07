#include "position.h"

using namespace std;

// For Zobrist Hashing
Key piece_keys[15][64];
Key castle_keys[16];
Key enpassant_keys[8];
Key side_to_move_key;
Key fifty_move_key[8];
Key threefold_key;

constexpr Piece piece_list[] = { W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
								 B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING };

int castling_helper[64];
	
void Position::init() {
	//set Zobrist keys at startup
	PRNG generator = PRNG(9235123129483259312ULL);

	for (Piece p : piece_list) {
		for (int j = 0; j < 64; j++) {
			piece_keys[p][j] = Key(generator.get());
		}
	}

	for (int i = 0; i < 16; i++) {
		castle_keys[i] = Key(generator.get());
	}

	for (int i = 0; i < 8; i++) {
		enpassant_keys[i] = Key(generator.get());
	}

	side_to_move_key = Key(generator.get());

	for (int i = 0; i < 8; i++) {
		fifty_move_key[i] = Key(generator.get());
	}

	//set castling_helper
	for (int i = A1; i < SQ_END; i++) {
		castling_helper[i] = ~(0);
	}
	castling_helper[A1] = ~(2);
	castling_helper[E1] = ~(1 | 2);
	castling_helper[H1] = ~(1);
	castling_helper[A8] = ~(8);
	castling_helper[E8] = ~(4 | 8);
	castling_helper[H8] = ~(4);

	threefold_key = Key(generator.get());
}

void Position::pop_stack() {
	Undo* t = undo_stack;
	undo_stack = undo_stack->prev;
}

void Position::clear_stack() {
	while (undo_stack->prev != nullptr) { pop_stack(); }
}

void Position::rebuild() { // computes others from squares data.
	for (Square i = A1; i < SQ_END; ++i) {
		Piece p = squares[i];
		pieces[to_upiece(p)] ^= SquareBoard[i];
		if (p != EMPTY) {
			colors[to_color(p)] ^= SquareBoard[i];
			material_total += Material_MG[to_upiece(p)];
			m[to_color(p)] += Material_MG[to_upiece(p)];
			piece_count++;
			undo_stack->key ^= piece_keys[p][i];
		}
	}

	undo_stack->checkers =
		get_attackers(lsb_square(get_pieces(side_to_move, KING)), ~pieces[EMPTY])
		& colors[~side_to_move];

	compute_L0(accumulator, squares, net);
}

// place/remove/move functions - dont handle key xor
inline void Position::place(Piece p, Square s) {
	squares[s] = p;
	pieces[UEMPTY] ^= SquareBoard[s];
	colors[to_color(p)] ^= SquareBoard[s];
	pieces[to_upiece(p)] ^= SquareBoard[s];
	piece_count++;
	material_total += Material_MG[to_upiece(p)];
	m[to_color(p)] += Material_MG[to_upiece(p)];

	update_L0_place(accumulator, p, s, net);
}

inline void Position::remove(Piece p, Square s) {
	squares[s] = EMPTY;
	pieces[UEMPTY] ^= SquareBoard[s];
	colors[to_color(p)] ^= SquareBoard[s];
	pieces[to_upiece(p)] ^= SquareBoard[s];
	piece_count--;
	material_total -= Material_MG[to_upiece(p)];
	m[to_color(p)] -= Material_MG[to_upiece(p)];

	update_L0_remove(accumulator, p, s, net);
}

inline void Position::move_piece(Piece p, Square from, Square to) {
	squares[from] = EMPTY;
	squares[to] = p;
	pieces[UEMPTY] ^= SquareBoard[from] ^ SquareBoard[to];
	colors[to_color(p)] ^= SquareBoard[from] ^ SquareBoard[to];
	pieces[to_upiece(p)] ^= SquareBoard[from] ^ SquareBoard[to];

	update_L0_remove(accumulator, p, from, net);
	update_L0_place(accumulator, p, to, net);
}

Position::Position(Net* n) {
	memset(this, 0, sizeof(Position));
	undo_stack = new Undo;
	memset(undo_stack, 0, sizeof(Undo));
	undo_stack->prev = nullptr;

	net = n;
}

void Position::verify() {
	// Verify all data are consistant.
	if (colors[0] & colors[1]) {
		cout << "Colors overlap detected" << endl;
	}

	Position testpos(net);
	for (int i = 0; i < 64; i++) { testpos.squares[i] = squares[i]; }
	testpos.rebuild();

	Bitboard test = EmptyBoard;
	for (int i = 0; i < 7; i++) {
		if (test & pieces[i]) {
			cout << "Pieces overlap detected at " << i << endl;
		}
		test |= pieces[i];
		if (testpos.pieces[i] != pieces[i]) {
			cout << "Pieces inconsistent at " << i << endl;
		}
		if (i != 0 && testpos.piece_count != piece_count) {
			cout << "piece count inconsistent" << endl;
		}
	}

	if (testpos.colors[WHITE] != colors[WHITE]) { cout << "White inconsistent" << endl; }
	if (testpos.colors[BLACK] != colors[BLACK]) { cout << "Black inconsistent" << endl; }

	for (int i = 0; i < 2 * SIZE_O0; i++) {
		if (testpos.accumulator[i] != accumulator[i]) {
			cout << "Accumulator inconsistent at" << i << endl;
		}
	}
}

Key Position::get_key() {
	if (undo_stack->repetition < 0) {
		return undo_stack->key ^ threefold_key;
	}
	// we need to modify key if 50move count is high
	if (undo_stack->fifty_move > 20) {
		return undo_stack->key ^ fifty_move_key[((undo_stack->fifty_move - 20) >> 2) & 8];
	}
	return undo_stack->key;
}

Bitboard Position::get_attackers(Square s, Bitboard occupied) {
	return attacks_pawn<BLACK>(s) & get_pieces(WHITE, PAWN) |
		attacks_pawn<WHITE>(s) & get_pieces(BLACK, PAWN) |
		attacks<KNIGHT>(s, occupied) & pieces[KNIGHT] |
		attacks<BISHOP>(s, occupied) & (pieces[BISHOP] | pieces[QUEEN]) |
		attacks<ROOK>(s, occupied) & (pieces[ROOK] | pieces[QUEEN]) |
		attacks<KING>(s, occupied) & pieces[KING];
}

Bitboard Position::get_pinned(Color c, Square k) {
	Bitboard pinned = EmptyBoard;
	Bitboard occupied = ~pieces[EMPTY];
	Bitboard ab = PseudoAttacks[BISHOP][k] & colors[~c] & (pieces[BISHOP] | pieces[QUEEN])
		| PseudoAttacks[ROOK][k] & colors[~c] & (pieces[ROOK] | pieces[QUEEN]);

	Bitboard ray;
	Square b;

	while (ab) {
		b = pop_lsb(&ab);
		ray = (Rays[b][k] & ((FullBoard << b) ^ (FullBoard << k)));
		ray = (ray & (ray - 1)) & occupied;
		if (popcount(ray) == 1) { pinned |= ray; }
	}

	return pinned;
}

inline Bitboard see_attackers(Square s, Bitboard occupied, Bitboard bq, Bitboard rq) {
	return attacks<BISHOP>(s, occupied) & bq | attacks<ROOK>(s, occupied) & rq;
}

Bitboard Position::see_least_piece(Color c, Bitboard attackers, UPiece& u) {
	attackers &= colors[c];
	for (UPiece v : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
		if (Bitboard b = attackers & pieces[v]) {
			u = v;
			return b & (b ^ (b - 1));
		}
	}
	return 0;
}

int Position::see(Move m) {
	Color c = side_to_move;
	Square s = get_to(m);
	Bitboard lp = SquareBoard[get_from(m)];
	Bitboard occupied = ~pieces[EMPTY];
	Bitboard bq = (pieces[BISHOP] | pieces[QUEEN]);
	Bitboard rq = (pieces[ROOK] | pieces[QUEEN]);
	Bitboard attackers = get_attackers(s, occupied);
	UPiece u = to_upiece(squares[get_from(m)]);
	int value[8];
	int depth = 0;
	value[0] = Material_MG[to_upiece(squares[s])];
	do {
		depth++;
		value[depth] = Material_MG[u] - value[depth - 1];
		if ((value[depth] < 0) && (value[depth - 1] > 0)) { break; }
		attackers ^= lp;
		occupied ^= lp;
		bq &= ~lp;
		rq &= ~lp;
		attackers |= see_attackers(s, occupied, bq, rq);
		c = ~c;
		if (depth > 6) { return 1024; }
	} 
	while (lp = see_least_piece(c, attackers, u));

	while (--depth) {
		value[depth - 1] = -max(-value[depth - 1], value[depth]);
	}
	return value[0];
}

bool Position::is_check(Move m)
{
	Square from = get_from(m);
	Square to = get_to(m);
	Color c = side_to_move;
	Square king = undo_stack->king_square[~c];
	UPiece u = to_upiece(squares[from]);
	Bitboard occupied = ~pieces[EMPTY];

	if (get_movetype(m) == 1) { u = get_promotion(m); }
	if (get_movetype(m) == 2) { 
		u = ROOK;
		to = get_file(to) > 4 ? (c ? F8 : F1) : (c ? D8 : D1);
	}
	switch (u) {
	case PAWN: {
		if (c) {
			if (attacks_pawn<WHITE>(king) & SquareBoard[to]) { return true; }
		}
		else {
			if (attacks_pawn<BLACK>(king) & SquareBoard[to]) { return true; }
		}
		break;
	}
	case KNIGHT: {
		if (attacks<KNIGHT>(king, occupied) & SquareBoard[to]) {
			return true;
		}
		break;
	}
	case BISHOP: {
		if (attacks<BISHOP>(king, occupied) & SquareBoard[to]) {
			return true;
		}
		break;
	}
	case ROOK: {
		if (attacks<ROOK>(king, occupied) & SquareBoard[to]) {
			return true;
		}
		break;
	}
	case QUEEN: {
		if (attacks<QUEEN>(king, occupied) & SquareBoard[to]) {
			return true;
		}
		break;
	}
	}
	// Discovered check
	Bitboard ba = (pieces[BISHOP] | pieces[QUEEN]) & colors[c];
	Bitboard ra = (pieces[ROOK] | pieces[QUEEN]) & colors[c];
	if (get_movetype(m) == 3) {
		Square cp = c ? to + 8 : to - 8;
		occupied ^= cp;
	}
	if ((PseudoAttacks[BISHOP][king] & ba & SquareBoard[from])
		&& !(Rays[from][king] & SquareBoard[to])) {
		if (attacks<BISHOP>(king, occupied ^ SquareBoard[from]) & ba) { return true; }
	}
	if ((PseudoAttacks[ROOK][king] & ra & SquareBoard[from])
		&& !(Rays[from][king] & SquareBoard[to])) {
		if (attacks<ROOK>(king, occupied ^ SquareBoard[from]) & ra) { return true; }
	}

	return false;
}

void Position::show() {
	cout << *this << endl;
	verify();
}

ostream& operator<<(ostream& os, Position& pos) {
	// Show board
	Square sq = A8;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			os << print_piece(pos.squares[sq]) << " ";
			++sq;
		}
		sq += (-16);
		os << "\n";
	}
	// Side to move, 50move count
	if (pos.side_to_move) { os << "Black to move, "; }
	else { os << "White to move, "; }
	os << "50-move count " << pos.undo_stack->fifty_move << "\n";

	// Castling Rights
	os << "Castling rights: "
		<< bitset<8>(pos.undo_stack->castling_rights) << "\n";

	// En Passant SQUARE
	os << "En passant square: " << pos.undo_stack->enpassant << "\n";

	// Key
	os << "Key: " << pos.get_key() << "\n";

	//
	os << "Checkers : ";
	Bitboard checkers = pos.undo_stack->checkers;
	Square s;
	while (checkers) {
		s = pop_lsb(&checkers);
		os << s << " ";
	}
	os << "\n";

	// repetition
	os << "Repetition: " << pos.undo_stack->repetition << endl;

	return os;
}

void Position::set(string fen) {
	char c;
	istringstream ss(fen);

	// first clear stack
	clear_stack();
	Undo* temp = undo_stack;
	Net* temp2 = net;
	memset(this, 0, sizeof(Position));
	undo_stack = temp;
	net = temp2;
	memset(undo_stack, 0, sizeof(Undo));

	// set pieces from rank 8.
	ss >> noskipws >> c;
	Square sq = A8;
	Piece p;
	while (c != ' ') {
		if (isdigit(c)) {
			sq += (c - '0');
		}
		else if (c == '/') {
			sq += (-16);
		}
		else if (parse_piece(c, p)) {
			squares[sq] = p;
			if (to_upiece(p) == KING) {
				undo_stack->king_square[to_color(p)] = sq;
			}
			++sq;
		}
		ss >> c;
	}
		
	// white / black to move
	ss >> c;
	side_to_move = c == 'w' ? WHITE : BLACK;
	if (side_to_move == BLACK) { undo_stack->key ^= side_to_move_key; }
	ss >> c;

	// castling rights
	undo_stack->castling_rights = 0;

	ss >> c;
	while (c != ' ') {
		switch (c) {
		case 'K':
			undo_stack->castling_rights ^= 1;
			break;
		case 'Q':
			undo_stack->castling_rights ^= 2;
			break;
		case 'k':
			undo_stack->castling_rights ^= 4;
			break;
		case 'q':
			undo_stack->castling_rights ^= 8;
			break;
		}
		ss >> c;
	}
	undo_stack->key ^= castle_keys[undo_stack->castling_rights];

	// en passant square
	char file, rank;
	ss >> file >> rank;
	if ((file >= 'a' && file <= 'h') && (rank == '3' || rank == '6')) {
		// otherwise file = '-' && rank == ' ' probably
		sq = parse_square(file, rank);
		undo_stack->enpassant = sq;
		undo_stack->key ^= enpassant_keys[get_file(sq)];
	}

	// 50-move rule
	int count;
	ss >> skipws >> count;
	undo_stack->fifty_move = count;
		
	// move count - doesn't matter

	// write other data
	undo_stack->captured = EMPTY;
	rebuild();
}

Move Position::parse_move(string string_move) {
	uint16_t move_int = 0;

	Square from = parse_square(string_move[0], string_move[1]);
	Square to   = parse_square(string_move[2], string_move[3]);
	int promotion = 0;
	int type = 0;
		
	if (string_move.size() == 5) { // Promotion?
		type = 1;
		string piece_finder = "nbrq";
		promotion = int(piece_finder.find(tolower(string_move[4])));
	}

	if (squares[from] == W_KING || squares[from] == B_KING) { // Castling?
		if (get_file(from) == 4 &&
			(get_file(to) == 2 || get_file(to) == 6)) {
			type = 2;
		}
	}

	if (to == undo_stack->enpassant && to != A1) { // En passant?
		if (squares[from] == W_PAWN || squares[from] == B_PAWN) {
			type = 3;
		}
	}

	return make_move(from, to, type, promotion);
}

void Position::do_move(Move m, Undo* new_undo) {
	memcpy(new_undo, undo_stack, sizeof(Undo));
	new_undo->prev = undo_stack;
	new_undo->fifty_move++;
	new_undo->enpassant = Square(0);
		
	if (undo_stack->enpassant != Square(0)) {
		new_undo->key ^= enpassant_keys[get_file(undo_stack->enpassant)];
	} // unset en passant key
	undo_stack = new_undo;

	Square from = get_from(m);
	Square to = get_to(m);

	Piece moved = squares[from];
	Piece captured = squares[to];
		

	if (captured != EMPTY) {
		remove(captured, to);
		new_undo->key ^= piece_keys[captured][to];
		new_undo->fifty_move = 0;
	}

	// set castling rights
	new_undo->key ^= castle_keys[new_undo->castling_rights];
	new_undo->castling_rights &= castling_helper[from] & castling_helper[to];
	new_undo->key ^= castle_keys[new_undo->castling_rights];
		
	switch (get_movetype(m)) {
	case 0: { // Normal
		move_piece(moved, from, to);
		new_undo->key ^= piece_keys[moved][from] ^ piece_keys[moved][to];

		if (to_upiece(moved) == PAWN) {
			// other stack variables
			new_undo->fifty_move = 0;

			// set en passant square
			Square epp;
			if (side_to_move) {
				if (from - to == 16) {
					epp = Square(to + 8);
					if (attacks_pawn<WHITE>(get_pieces(WHITE, PAWN)) & SquareBoard[epp]) {
						new_undo->enpassant = epp;
						new_undo->key ^= enpassant_keys[get_file(to)];
					}
				}
			}
			else {
				if (to - from == 16) {
					epp = Square(to - 8);
					if (attacks_pawn<BLACK>(get_pieces(BLACK, PAWN)) & SquareBoard[epp]) {
						new_undo->enpassant = epp;
						new_undo->key ^= enpassant_keys[get_file(to)];
					}
				}
			}
		}

		break;
	}

	case 1: { // Promotion
		remove(moved, from);
		new_undo->key ^= piece_keys[moved][from];
		// change moved to promoted piece
		moved = to_piece(get_promotion(m), side_to_move);
		place(moved, to);
		new_undo->key ^= piece_keys[moved][to];
		new_undo->fifty_move = 0;
		break;
	}

	case 2: { // Castle
		// do castling
		Square rook_from, rook_to;
		Piece rook = to_piece(ROOK, side_to_move);
		if (get_file(to) > 4) {
			// Kingside
			rook_from = Square(to + 1);
			rook_to = Square(to - 1);
		}
		else {
			// Queenside
			rook_from = Square(to - 2);
			rook_to = Square(to + 1);
		}
		move_piece(moved, from, to);
		move_piece(rook, rook_from, rook_to);
		undo_stack->key ^= piece_keys[moved][from] ^ piece_keys[moved][to]
			^ piece_keys[rook][rook_from] ^ piece_keys[rook][rook_to];
		break;
	}

	case 3: { // En Passant
		Square captured_square = side_to_move ? Square(to + 8) : Square(to - 8);
		Piece captured_pawn = to_piece(PAWN, ~side_to_move);
		remove(captured_pawn, captured_square);
		move_piece(moved, from, to);
		Key k = piece_keys[captured_pawn][captured_square]
			^ piece_keys[moved][from] ^ piece_keys[moved][to];
		new_undo->key ^= k;
		break;
	}
	}

	if (to_upiece(moved) == KING) {
		new_undo->king_square[side_to_move] = to;
	}

	side_to_move = ~side_to_move;
	new_undo->key ^= side_to_move_key;

	new_undo->captured = captured;
	new_undo->checkers = 
		get_attackers(lsb_square(get_pieces(side_to_move, KING)), ~pieces[EMPTY]) 
		& colors[~side_to_move];

	// set repetition
	Undo* p = new_undo;
	new_undo->repetition = 0;
	int dist = 0;
	while ((p->prev != nullptr) && p->prev->fifty_move == p->fifty_move - 1) {
		dist++;
		if (new_undo->key == p->prev->key) {
			new_undo->repetition = (p->prev->repetition == 0) ? dist : -dist;
			break;
		}
		p = p->prev;
	}
}

void Position::undo_move(Move m) {
	Square from = get_from(m);
	Square to = get_to(m);

	Piece moved = squares[to];
	Piece captured = undo_stack->captured;

	side_to_move = ~side_to_move;
		

	switch (get_movetype(m)) {
	case 0: { // Normal
		move_piece(moved, to, from);
		break;
	}

	case 1: { // Promotion
		remove(moved, to);
		moved = to_piece(PAWN, side_to_move);
		place(moved, from);
		break;
	}

	case 2: { // Castle
		// do castling
		Square rook_from, rook_to;
		Piece rook = to_piece(ROOK, side_to_move);
		if (get_file(to) > 4) {
			// Kingside
			rook_from = Square(to + 1);
			rook_to = Square(to - 1);
		}
		else {
			// Queenside
			rook_from = Square(to - 2);
			rook_to = Square(to + 1);
		}
		move_piece(moved, to, from);
		move_piece(rook, rook_to, rook_from);
		break;
	}

	case 3: { // En Passant
		Square captured_square = side_to_move ? Square(to + 8) : Square(to - 8);
		place(to_piece(PAWN, ~side_to_move), captured_square);
		move_piece(moved, to, from);
		break;
	}
	}

	if (captured != EMPTY) {
		place(captured, to);
	}

	pop_stack();
}

void Position::do_null_move(Undo* new_undo) {
	memcpy(new_undo, undo_stack, sizeof(Undo));
	new_undo->prev = undo_stack;
	new_undo->enpassant = Square(0);

	if (undo_stack->enpassant != Square(0)) {
		new_undo->key ^= enpassant_keys[get_file(undo_stack->enpassant)];
	} // unset en passant key
	undo_stack = new_undo;

	side_to_move = ~side_to_move;
	new_undo->key ^= side_to_move_key;

	new_undo->captured = EMPTY;
	new_undo->checkers = EmptyBoard;

	// set repetition
	new_undo->repetition = 0;
}

void Position::undo_null_move() {
	side_to_move = ~side_to_move;
	pop_stack();
}

void Position::set_random(int pieces, PRNG* r) {
	if (pieces < 2) { pieces = 2; }
	uint64_t seed = r->get_seed();

	// Zero
	clear_stack();
	Undo* temp = undo_stack;
	Net* temp2 = net;
	memset(this, 0, sizeof(Position));
	undo_stack = temp;
	net = temp2;
	memset(undo_stack, 0, sizeof(Undo));

	// Pick Side
	side_to_move = Color(r->get() % 2);
	if (side_to_move == BLACK) { undo_stack->key ^= side_to_move_key; }

	Square sq[32];
	Bitboard occ = EmptyBoard;

	// Place Kings
	do {
		sq[0] = Square(r->get() % SQ_END);
		sq[1] = Square(r->get() % SQ_END);
	} while (Distance[sq[0]][sq[1]] < 2);

	squares[sq[0]] = W_KING;
	squares[sq[1]] = B_KING;
	undo_stack->king_square[WHITE] = sq[0];
	undo_stack->king_square[BLACK] = sq[1];
	Bitboard ksq[2];
	ksq[WHITE] = SquareBoard[sq[0]];
	ksq[BLACK] = SquareBoard[sq[1]];
	occ ^= ksq[WHITE] ^ ksq[BLACK];

	int i = 2;
	while (i < pieces) {
		sq[i] = Square(r->get() % SQ_END);
		bool skip = false;
		for (int j = 0; j < i; j++) {
			if (sq[i] == sq[j]) { 
				skip = true;
				break;
			}
		}
		if (skip) { continue; }

		Color c = Color(r->get() % 2);
		int u_r = r->get() % 15;
		UPiece u = u_r < 8 ? PAWN :
			u_r < 10 ? KNIGHT :
			u_r < 12 ? BISHOP :
			u_r < 14 ? ROOK :
			QUEEN;
		Piece p = to_piece(u, c);

		if (u == PAWN &&
			(get_rank(sq[i]) == 0 || get_rank(sq[i]) == 7)) {
			continue;
		}

		occ ^= SquareBoard[sq[i]];

		switch (u) {
		case PAWN: {
			if (c == WHITE) {
				if (attacks_pawn<WHITE>(sq[i]) & ksq[BLACK]) { 
					skip = true;
				}
			}
			else {
				if (attacks_pawn<BLACK>(sq[i]) & ksq[WHITE]) { 
					skip = true;
				}
			}
			break;
		}
		case KNIGHT: {
			if (attacks<KNIGHT>(sq[i], occ) & ksq[~c]) {
				skip = true;
			}
			break;
		}
		case BISHOP: {
			if (attacks<BISHOP>(sq[i], occ) & ksq[~c]) {
				skip = true;
			}
			break;
		}
		case ROOK: {
			if (attacks<ROOK>(sq[i], occ) & ksq[~c]) {
				skip = true;
			}
			break;
		}
		case QUEEN: {
			if (attacks<QUEEN>(sq[i], occ) & ksq[~c]) {
				skip = true;
			}
			break;
		}
		}

		if (skip) {
			occ ^= SquareBoard[sq[i]];
			continue;
		}

		squares[sq[i]] = p;
		i++;
	}

	rebuild();

	if (undo_stack->checkers) { throw seed; }
}

Position& Position::operator=(const Position board) {
	clear_stack();
	memcpy(this, &board, sizeof(Position));
	undo_stack = new Undo;
	memcpy(undo_stack, board.undo_stack, sizeof(Undo));
	undo_stack->prev = nullptr;
	return *this;
}