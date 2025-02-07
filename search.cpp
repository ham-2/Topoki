#include "search.h"

using namespace std;

bool ATM = false;
bool lichess_timing = false;
bool stop_if_mate = false;

bool ponder = false;
atomic<bool> ponder_continue(false);

int multipv = 1;

void get_time(istringstream& ss, Color c, float& time, float& max_time, bool& force_time, int& max_ply) {
	string word;
	time = DEFAULT_TIME;
	max_time = DEFAULT_TIME_MAX;
	max_ply = MAX_PLY;
	force_time = false;
	while (ss >> word) {
		if (word == "wtime") {
			ss >> word;
			if (!c) {
				time = stof(word) / 28000;
				if (lichess_timing) { time -= 1.05f; }
				max_time = stof(word) / 5000;
				if (lichess_timing) { max_time -= 1.05f; }
			}
		}
		else if (word == "btime") {
			ss >> word;
			if (c) {
				time = stof(word) / 28000;
				if (lichess_timing) { time -= 1.05f; }
				max_time = stof(word) / 5000;
				if (lichess_timing) { max_time -= 1.05f; }
			}
		}
		else if (word == "winc") {
			ss >> word;
			if (!c) {
				time += stof(word) / 1000;
				max_time += stof(word) / 1000;
			}
		}
		else if (word == "binc") {
			ss >> word;
			if (c) {
				time += stof(word) / 1000;
				max_time += stof(word) / 1000;
			}
		}
		else if (word == "movetime") {
			ss >> word;
			time = stof(word) / 1000;
			force_time = true;
		}
		else if (word == "infinite") {
			ss >> word;
			time = -1;
			max_time = -1;
			force_time = true;
			break;
		}
		else if (word == "depth") {
			ss >> word;
			max_ply = stoi(word);
		}
	}
	if (!force_time) {
		if (time < 0) { time = 0.05f; }
	}
}

void clear1ply(Position& board) {
	MoveList legal_moves;
	legal_moves.generate(board);
	for (auto dmove = legal_moves.list; dmove != legal_moves.end; dmove++) {
		Undo u;
		board.do_move(*dmove, &u);
		Main_TT.clear_entry(board.get_key());
		board.undo_move(*dmove);
	}
	Main_TT.clear_entry(board.get_key());
}

void search_start(Thread* t, float time, float max_time, bool force_time, int max_ply)
{
	Threads.stop = false;
		
	int multipv_max = multipv;
	Move bmove = NULL_MOVE;
	atomic<bool>* complete = new atomic<bool>;
	*complete = false;
	condition_variable* print_cond = new condition_variable;
	force_time = !ATM || force_time;
	Position* board = t->board;

	node_count.exchange(0);
	Main_TT.increment();

	thread print_t = thread(printer, time, complete, print_cond, max_time, force_time);

	Main_TT.clear_entry(board->get_key());
	Threads.depth.exchange(DEFAULT_PLY);

	// Start parallel search
	Threads.t_wait.notify_all();
	Threads.threads[0]->m.lock();

	TTEntry probe = {};
	Main_TT.probe(board->get_key(), &probe);

	// Move Generation
	MoveList legal_moves;
	legal_moves.generate(*board);
	if (legal_moves.length() < multipv_max) { multipv_max = legal_moves.length(); }
	vector<pair<Move, int>> pvmoves(legal_moves.length());
	if (multipv_max > 1) {		
		for (int i = 0; i < legal_moves.length(); i++) {
			pvmoves[i] = make_pair(*(legal_moves.list + i), -EVAL_FAIL);
		}
	}

	int window_a = 2 << (EVAL_BITS - 6);
	int window_b = 2 << (EVAL_BITS - 6);
	int window_c = probe.eval;

	SearchParams sp = {
		board, &(Threads.stop), &Main_TT, t->step
	};

	while (Threads.depth <= max_ply) {
		window_c = alpha_beta(&sp, &probe,
			Threads.depth, t->board->get_side(), 0,
			window_c - window_a,
			window_c + window_b);

		// MultiPV search
		if (multipv_max > 1) {
			// Update scores
			for (int i = 0; i < legal_moves.length(); i++) {
				Undo u;
				board->do_move(pvmoves[i].first, &u);
				TTEntry probe = {};
				if (Main_TT.probe(board->get_key(), &probe) != -1) {
					pvmoves[i].second = -probe.eval;
				}
				board->undo_move(pvmoves[i].first);
			}

			// Find top moves
			sort(pvmoves.begin(), pvmoves.end(), compare);

			// Search Top Moves
			for (int i = 1; i < multipv_max; i++) {
				Undo u;
				board->do_move(pvmoves[i].first, &u);

				Main_TT.probe(board->get_key(), &probe);

				alpha_beta(&sp, &probe,
					Threads.depth - i, ~(t->board->get_side()), 0,
					window_c - window_a,
					window_c + window_b);

				board->undo_move(pvmoves[i].first);
			}
		}
			
		// Forced Stop
		if (Threads.stop) { break; }

		if (probe.type == 1) {
			window_a *= 4;
		}
		if (probe.type == -1) {
			window_b *= 4;
		}
		if (probe.type == 0) {
			// Stop if mate
			if (stop_if_mate && is_mate(window_c)) {
				break;
			}

			// Print and search again
			Threads.depth++;
			print_cond->notify_all();

			window_a = 2 << (EVAL_BITS - 6);
			window_b = 2 << (EVAL_BITS - 6);
		}
	}

	// terminate print thread
	*complete = true;
	print_cond->notify_all();
	print_t.join();
	delete complete;
	Threads.stop = true;
	Threads.threads[0]->m.unlock();
	Threads.sync();

	// Print Bestmove
	Main_TT.probe(board->get_key(), &probe);
	Threads.acquire_cout();
	cout << "bestmove " << move(probe.nmove) << endl;		
	Threads.release_cout();

	// Ponder
	if (ponder_continue && !is_mate(probe.eval)) {
		Threads.do_move(bmove);
		Threads.depth.exchange(DEFAULT_PLY);
		Threads.stop = false;
		ponder_continue = false;

		// Start parallel search
		Threads.t_wait.notify_all();

		Threads.threads[0]->m.lock();

		TTEntry probe_ponder = {};
		Main_TT.probe(board->get_key(), &probe_ponder);
		while (Threads.depth <= max_ply && !Threads.stop) {
			window_c = alpha_beta(&sp, &probe,
				Threads.depth, t->board->get_side(), 0,
				window_c - window_a,
				window_c + window_b);

			// Forced Stop
			if (Threads.stop) { break; }

			if (probe.type == 1) {
				window_a *= 4;
			}
			if (probe.type == -1) {
				window_b *= 4;
			}
			if (probe.type == 0) {
				// Stop if mate
				if (stop_if_mate && is_mate(window_c)) {
					break;
				}

				Threads.depth++;

				window_a = 2 << (EVAL_BITS - 6);;
				window_b = 2 << (EVAL_BITS - 6);;
			}
		}

		Threads.stop = true;
		Threads.threads[0]->m.unlock();
	}

	return;
}