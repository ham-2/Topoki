#include "printer.h"

using namespace std;

void getpv(ostream& os, Position* board, int& depth) {
	string pv;
	TTEntry probe = {};
	Main_TT.probe(board->get_key(), &probe);
	if (probe.nmove != NULL_MOVE) {
		Undo u;
		board->do_move(probe.nmove, &u);
		os << probe.nmove << " ";
		depth++;
		getpv(os, board, depth);
		board->undo_move(probe.nmove);
	}
}

bool compare(pair<Move, int> a, pair<Move, int> b) { return a.second > b.second; }

void printer(float time, atomic<bool>* stop, condition_variable* cv, float max_time, bool force_time)
{
	using namespace std::chrono;
	milliseconds limit_time{ static_cast<long int>(1000 * time) };
	milliseconds limit_time_max{ static_cast<long int>(1000 * max_time) };
	system_clock::time_point time_start = system_clock::now();
	milliseconds search_time = milliseconds(0);
	system_clock::time_point time_now;
	Move bmove = NULL_MOVE;
	int score = 0;
	int max_depth = 0;
	int currdepth;
	int nodes;
	mutex mu;
	int multipv_max = multipv;

	// Move Generation
	MoveList legal_moves;
	legal_moves.generate(*Threads.board);
	if (legal_moves.length() < multipv_max) { multipv_max = legal_moves.length(); }
	vector<pair<Move, int>> pvmoves(legal_moves.length());
	if (multipv_max > 1) {
		for (int i = 0; i < legal_moves.length(); i++) {
			pvmoves[i] = make_pair(*(legal_moves.list + i), -EVAL_FAIL);
		}
	}

	while (true) {
		time_now = system_clock::now();
		search_time = duration_cast<milliseconds>(time_now - time_start);
		unique_lock<mutex> m(mu);
		if (force_time) { // Normal Search
			if (time == -1 || limit_time - search_time > milliseconds(PRINT_MIN_MS)) {
				cv->wait_for(m, milliseconds(PRINT_MIN_MS));
			}
			else {
				cv->wait_for(m, limit_time - search_time);
			}
		}

		// Limit Time
		time_now = system_clock::now();
		search_time = duration_cast<milliseconds>(time_now - time_start);
		nodes = node_count;

		if (force_time) { // Normal Search
			if (time != -1 && search_time >= limit_time) {
				if (ponder) { ponder_continue = true; }
				Threads.stop = true;
				*stop = true;
			}
		}

		Threads.acquire_cout();
		if (multipv_max > 1) {
			// Update pvs
			for (int i = 0; i < legal_moves.length(); i++) {
				Undo u;
				Move m = pvmoves[i].first;
				Threads.board->do_move(m, &u);
				TTEntry probe = {};
				if (Main_TT.probe(Threads.board->get_key(), &probe) != -1) {
					pvmoves[i].second = -probe.eval;
				}
				Threads.board->undo_move(m);
			}

			// Find top moves
			sort(pvmoves.begin(), pvmoves.end(), compare);

			// Print PVs
			for (int i = 0; i < multipv_max; i++) {			
				Move m = pvmoves[i].first;
				score  = pvmoves[i].second;
				Undo u;
				Threads.board->do_move(m, &u);
				stringstream buf;
				currdepth = 0;
				buf << m << " ";
				getpv(buf, Threads.board, currdepth);
				Threads.board->undo_move(m);
					
				if (i == 0) { max_depth = Threads.depth; }
				if (score != -EVAL_FAIL) { dec_mate(score); }
					
				cout << "info time " << search_time.count()
					<< " depth " << max_depth
					<< " seldepth " << currdepth + 1
					<< " nodes " << nodes << " multipv " << i + 1
					<< " score " << eval_print(score)
					<< " pv " << buf.str() << "\n";
			}
			cout << endl;

		}
		else {
			// Get candidate move
			TTEntry probe = {};
			Main_TT.probe(Threads.board->get_key(), &probe);
			stringstream buf;
			currdepth = 0;
			getpv(buf, Threads.board, currdepth);
			max_depth = Threads.depth;

			// Print
			cout << "info time " << search_time.count() << " depth " << max_depth
				<< " currmove " << move(probe.nmove) << " score " << eval_print(probe.eval)
				<< " nodes " << nodes 
				<< " nps " << int(double(nodes) * 1000 / (search_time.count() + 1))
				<< " pv " << buf.str() << endl;
		}
		Threads.release_cout();

		if (*stop) { break; }
	}

}