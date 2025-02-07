#include "threads.h"

using namespace std;

Threadmgr Threads;
const char* startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const char* default_weight = "weights.bin";

int SEARCH_THREADS_MAX = 32;
	
void lazy_smp(Thread* t) {
	int currdepth;
	TTEntry probe = {};

	SearchParams sp = {
		t->board, &Threads.stop, &Main_TT, t->step
	};

	while (!(t->kill)) {
		unique_lock<mutex> m(t->m);
		Threads.t_wait.wait(m);

		Main_TT.probe(t->board->get_key(), &probe);
		int window_a = 2 << (EVAL_BITS - 6);
		int window_b = 2 << (EVAL_BITS - 6);
		int window_c = probe.eval;

		while (!Threads.stop) {
			currdepth = Threads.depth + 1;

			alpha_beta(&sp, &probe,
				currdepth, t->board->get_side(), 0,
				window_c - window_a,
				window_c + window_b);

			if (probe.type == 1) {
				window_a *= 4;
			}
			if (probe.type == -1) {
				window_b *= 4;
			}
			if (probe.type == 0) {
				window_a = 2 << (EVAL_BITS - 6);
				window_b = 2 << (EVAL_BITS - 6);
			}
		}
	}

	(t->kill) = false;
	return;
}

Thread::Thread(int id, Net* n) {
	board = new Position(n);
	this->id = id;
	step = Primes[id];
}

Thread::~Thread() {
	delete board;
	delete t;
}

void Threadmgr::init() {
	stop = new atomic<bool>;
	n = new Net;
	load_weights<Net>(n, default_weight);
	board = new Position(n);
	Threads.threads.push_back(new Thread(0, n));
	set_all(startpos_fen);
}

void Threadmgr::add_thread() {
	Thread* new_thread = new Thread(num_threads, n);
	*(new_thread->board) = *board;
	threads.push_back(new_thread);
	new_thread->t = new thread(lazy_smp, new_thread);
	num_threads++;
}

void Threadmgr::del_thread() {
	Thread* t = threads.back();
	t->kill = true;
	t_wait.notify_all();
	while (t->kill) {
		this_thread::sleep_for(chrono::milliseconds(5));
	}
	threads.pop_back();
	delete t;
	num_threads--;
}

void Threadmgr::set_threads(int new_threads) {
	if (new_threads < 1) { new_threads = 1; }
	else if (new_threads > SEARCH_THREADS_MAX) { new_threads = SEARCH_THREADS_MAX; }

	if (new_threads > num_threads) {
		while (new_threads != num_threads) {
			add_thread();
		}
	}

	if (new_threads < num_threads) {
		while (new_threads != num_threads) {
			del_thread();
		}
	}
}

void Threadmgr::acquire_cout() { cout_lock.lock(); }

void Threadmgr::release_cout() { cout_lock.unlock(); }

void Threadmgr::acquire_lock() {
	for (int i = 0; i < threads.size(); i++) {
		threads[i]->m.lock();
	}
}

void Threadmgr::release_lock() {
	for (int i = 0; i < threads.size(); i++) {
		threads[i]->m.unlock();
	}
}

void Threadmgr::sync() {
	acquire_lock();
	release_lock();
}

void Threadmgr::set_all(string fen) {
	board->set(fen);
	for (int i = 0; i < threads.size(); i++) {
		threads[i]->board->set(fen);
	}
}

void Threadmgr::setr(int pieces, uint64_t seed)
{
	PRNG p(seed);
	board->set_random(pieces, &p);
	for (int i = 0; i < threads.size(); i++) {
		*threads[i]->board = *board;
	}
}

void Threadmgr::show(int i) {
	if (i < 0) { board->show(); }
	if (i < threads.size()) {
		threads[i]->board->show();
	}
}

void Threadmgr::do_move(Move m) {
	Undo* u = new Undo;
	board->do_move(m, u);
	for (int i = 0; i < threads.size(); i++) {
		Undo* u = new Undo;
		threads[i]->board->do_move(m, u);
	}
}

void Threadmgr::do_move(string ms) {
	Move m = threads[0]->board->parse_move(ms);
	do_move(m);
}

void Threadmgr::undo_move(Move m) {
	board->undo_move(m);
	for (int i = 0; i < threads.size(); i++) {
		threads[i]->board->undo_move(m);
	}
}

void Threadmgr::test_eval() {
	int endeval = end_eval(board);
	cout << endeval << " " << eval_print(endeval) << "\n";
	cout << board->get_material() << ' ' << board->get_material() * 4 << '\n';
}

void Threadmgr::gen()
{
	MoveList ml;
	ml.generate(*board);
	ml.show();
	cout << "\n" << ml.length() << " moves" << endl;
}

void Threadmgr::test_see(string ms) {
	Move m = board->parse_move(ms);
	cout << board->see(m) << endl;
	//cout << board->is_check(m) << endl;
}

void Threadmgr::set_weights() {
	board->set_accumulator();
	for (int i = 0; i < threads.size(); i++) {
		memcpy(threads[i]->board->get_net(), n, sizeof(Net));
		threads[i]->board->set_accumulator();
	}
}