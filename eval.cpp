#include "eval.h"

using namespace std;

atomic<long> node_count(0);
bool limit_strength = false;
int max_noise = 0;
int contempt = 0;

int end_eval(Position* board) {
	node_count++;

	int v[SIZE_O3];
	board->get_eval(v);
	int score = v[0] * board->get_mg()
		+ v[1] * (100 - board->get_mg());

	// 50-move, shuffling
	score *= (100 - board->get_fiftymove());
	score /= 10000;

	score = score > EVAL_HIGH ? EVAL_HIGH :
		score < EVAL_LOW ? EVAL_LOW : score;

	return score;
}

int eval(Position* board, int depth, int alpha, int beta)
{
	// Move Generation
	MoveList legal_moves;
	legal_moves.generate(*board);

	// Mates and Stalemates
	if (legal_moves.length() == 0) {
		if (board->get_checkers()) { return EVAL_MIN; }
		else { return 0; }
	}

	// repetition
	if (board->get_repetition(depth)) { return 0; }

	// 50-move
	if (board->get_fiftymove() > 99) { return 0; }

	int new_eval = EVAL_INIT;
	int comp_eval;

	if (board->get_checkers()) { // Checked

		for (auto move = legal_moves.list; move != legal_moves.end; move++) {
			Undo u;
			board->do_move(*move, &u);
			comp_eval = -eval(board, depth + 1, -beta, -alpha);
			board->undo_move(*move);
			dec_mate(comp_eval);
			if (comp_eval > new_eval) {
				new_eval = comp_eval;
				if (comp_eval > alpha) { 
					alpha = comp_eval;
					if (alpha > beta) { break; }
				}
			}
		}

		return new_eval;
	}
		
	else {
		new_eval = end_eval(board);

		if (depth > 1) { return new_eval; }

		// ab
		if (new_eval > alpha) { alpha = new_eval; }
		if (alpha > beta) { return new_eval; }

		for (auto move = legal_moves.list; move != legal_moves.end; move++) {
			if (board->is_non_quiesce(*move))
			{
				Undo u;
				board->do_move(*move, &u);
				comp_eval = -eval(board, depth + 1, -beta, -alpha);
				board->undo_move(*move);
				dec_mate(comp_eval);
				if (comp_eval > new_eval) { 
					new_eval = comp_eval;
					if (comp_eval > alpha) { 
						alpha = comp_eval;
						if (alpha > beta) { break; }
					}
				}
			}
		}

		return new_eval;
	}
}

string eval_print(int eval) {
	if (eval <= EVAL_FAIL || eval >= -EVAL_FAIL) {
		return "cp 0";
	}
	else if (eval > EVAL_END) {
		int ply_to_mate = EVAL_MAX - eval;
		return "mate " + to_string((ply_to_mate + 1) / 2);
	}
	else if (eval < -EVAL_END) {
		int ply_to_mate = eval + EVAL_MAX;
		return "mate -" + to_string((ply_to_mate + 1) / 2);
	}
	else if (eval >= 0) {
		return "cp " + to_string(int(200.0 / (2.01 - 2.0 * eval / EVAL_HIGH) - 100));
	}
	else {
		return "cp -" + to_string(int(200.0 / (2.01 - 2.0 * (-eval) / EVAL_HIGH) - 100));
	}
}

int add_noise(int& eval) {
	eval += (int(rng.get()) % max_noise
		+ int(rng.get()) % max_noise
		+ int(rng.get()) % max_noise
		+ int(rng.get()) % max_noise) / 4;
	return eval;
}