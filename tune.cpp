#include "tune.h"

using namespace std;

constexpr int LOSS_SMOOTH = (1 << 12);
constexpr int THREAD_LOOP = 8;
constexpr int PICK_THRESHOLD = (4 << (EVAL_BITS - 6));
constexpr int PICK_DIV = PICK_THRESHOLD / 64;

int tb_probe(Position* board) {
	return tb_probe_wdl(
		board->get_pieces(WHITE),
		board->get_pieces(BLACK),
		board->get_pieces(KING),
		board->get_pieces(QUEEN),
		board->get_pieces(ROOK),
		board->get_pieces(BISHOP),
		board->get_pieces(KNIGHT),
		board->get_pieces(PAWN),
		board->get_fiftymove(),
		board->get_castling_tb(),
		board->get_enpassant(),
		!board->get_side());
}

template <typename T1, typename T2, int size> 
inline void convert_(T1* dst, T2* src) {
	for (int i = 0; i < size; i++) { dst[i] = src[i]; }
}

void convert_to_float(Net_train* dst, Net* src) 
{
	dst->m.lock();

	convert_<float, int8_t, SIZE_F0 * SIZE_O0>(dst->L0_a, src->L0_a);
	convert_<float, int16_t, SIZE_O0>(dst->L0_b, src->L0_b);
	convert_<float, int8_t, SIZE_F1 * SIZE_O1>(dst->L1_a, src->L1_a);
	convert_<float, int16_t, SIZE_O1>(dst->L1_b, src->L1_b);
	convert_<float, int8_t, SIZE_F2 * SIZE_O2>(dst->L2_a, src->L2_a);
	convert_<float, int16_t, SIZE_O2>(dst->L2_b, src->L2_b);
	convert_<float, int16_t, SIZE_F3 * SIZE_O3>(dst->L3_a, src->L3_a);
	convert_<float, int32_t, SIZE_O3>(dst->L3_b, src->L3_b);

	dst->m.unlock();
}

void convert_to_int(Net* dst, Net_train* src) 
{
	src->m.lock();

	convert_<int8_t, float, SIZE_F0 * SIZE_O0>(dst->L0_a, src->L0_a);
	convert_<int16_t, float, SIZE_O0>(dst->L0_b, src->L0_b);
	convert_<int8_t, float, SIZE_F1 * SIZE_O1>(dst->L1_a, src->L1_a);
	convert_<int16_t, float, SIZE_O1>(dst->L1_b, src->L1_b);
	convert_<int8_t, float, SIZE_F2 * SIZE_O2>(dst->L2_a, src->L2_a);
	convert_<int16_t, float, SIZE_O2>(dst->L2_b, src->L2_b);
	convert_<int16_t, float, SIZE_F3 * SIZE_O3>(dst->L3_a, src->L3_a);
	convert_<int32_t, float, SIZE_O3>(dst->L3_b, src->L3_b);

	src->m.unlock();
}

void copy_float(Net_train* dst, Net_train* src) {
	src->m.lock();
	constexpr size_t C = offsetof(Net_train, L0_a);

	memcpy((char*)(dst) + C, 
		(const char*)(src) + C,
		sizeof(Net_train) - C);

	src->m.unlock();
}

template<int size, int clip>
inline void add_float_(float* dst, float* src) {
	for (int i = 0; i < size; i += 8) {
		__m256 dst_ = _mm256_load_ps(dst + i);
		__m256 src_ = _mm256_load_ps(src + i);
		dst_ = _mm256_add_ps(dst_, src_);
		dst_ = _mm256_max_ps(dst_, _mm256_set1_ps(-clip));
		dst_ = _mm256_min_ps(dst_, _mm256_set1_ps(clip));
		_mm256_store_ps(dst + i, dst_);
	}
}

void add_float(Net_train* dst, Net_train* src) {

	src->m.lock();
	dst->m.lock();

	add_float_<SIZE_F0 * SIZE_O0, 127>(dst->L0_a, src->L0_a);
	add_float_<SIZE_O0, 16383>(dst->L0_b, src->L0_b);
	add_float_<SIZE_F1 * SIZE_O1, 127>(dst->L1_a, src->L1_a);
	add_float_<SIZE_F2, 16383>(dst->L1_b, src->L1_b);
	add_float_<SIZE_F2 * SIZE_O2, 127>(dst->L2_a, src->L2_a);
	add_float_<SIZE_F3, 16383>(dst->L2_b, src->L2_b);
	add_float_<SIZE_F3 * SIZE_O3, 32767>(dst->L3_a, src->L3_a);
	for (int i = 0; i < SIZE_O3; i++) {
		dst->L3_b[i] += src->L3_b[i];
	}

	dst->m.unlock();
	src->m.unlock();
}

template <int size_3, int size_2, int max_2, int shift_2>
void back_b_(__m256 c, float* d_3, float* d_2,
	int16_t* p2r, int8_t* a_2, float* b_1) {

	for (int i = 0; i < size_2; i++) {
		if (p2r[i] > (max_2 << shift_2) || p2r[i] < 0) {
			d_2[i] = 0;
			continue;
		}

		float temp[8];
		__m256 d_2_ = _mm256_setzero_ps();
		for (int j = 0; j < size_3; j += 8) {
			__m256 d_3_ = _mm256_load_ps(d_3 + j);
			__m256 a_2_ = _mm256_cvtepi32_ps(
				_mm256_cvtepi8_epi32(
					_mm_load_si128((__m128i*)(a_2 + j + i * size_3))));
			d_2_ = _mm256_add_ps(d_2_, _mm256_mul_ps(d_3_, a_2_));
		}
		d_2_ = _mm256_hadd_ps(d_2_, _mm256_setzero_ps());
		d_2_ = _mm256_hadd_ps(d_2_, _mm256_setzero_ps());
		d_2_ = _mm256_hadd_ps(d_2_, _mm256_setzero_ps());
		_mm256_store_ps(temp, d_2_);
		d_2[i] = temp[0] / (1 << shift_2);
	}

	//c = _mm256_mul_ps(c, _mm256_set1_ps(128.0));
	for (int i = 0; i < size_2; i += 8) {
		__m256 d_2_ = _mm256_load_ps(d_2 + i);
		d_2_ = _mm256_mul_ps(d_2_, c);
		__m256 b_1_ = _mm256_load_ps(b_1 + i);
		b_1_ = _mm256_add_ps(b_1_, d_2_);
		_mm256_store_ps(b_1 + i, b_1_);
	}
}

template <int size_3, int size_2>
void back_a_(__m256 c, float* d_3, float* a_2, int16_t* p2) {
	for (int i = 0; i < size_2; i++) {
		__m256 p2_ = _mm256_set1_ps(float(p2[i]));
		for (int j = 0; j < size_3; j += 8) {
			__m256 d_3_ = _mm256_load_ps(d_3 + j);
			d_3_ = _mm256_mul_ps(d_3_, p2_);
			d_3_ = _mm256_mul_ps(d_3_, c);
			__m256 a2_ = _mm256_load_ps(a_2 + j + i * size_3);
			a2_ = _mm256_add_ps(a2_, d_3_);
			_mm256_store_ps(a_2 + j + i * size_3, a2_);
		}
	}
}

void backpropagate(Net_train* dst, Position* board,
	int score_true, atomic<double>* loss, double learning_rate)
{
	alignas(32)
	int16_t P1[SIZE_F1];
	int16_t P2_RAW[SIZE_O1];
	int16_t P2[SIZE_F2];
	int16_t P3_RAW[SIZE_O2];
	int16_t P3[SIZE_F3];
	int P[SIZE_O3];
	int PS;

	float mg = float(board->get_mg()) / 100;
	float eg = 1 - mg;

	float dPdP3R[SIZE_F3];
	float dPdP2R[SIZE_F2];
	float dPdP1R[SIZE_F1];

	//board->set_state();
	int16_t* acc = board->get_accumulator() + (board->get_side() ? SIZE_O0 : 0);
	Net* n = board->get_net();

	ReLUClip<SIZE_F1, SHIFT_L1, MAX_L1>(P1, acc);
	compute_layer<SIZE_F2, SIZE_F1>(P2_RAW, P1, n->L1_a, n->L1_b);
	ReLUClip<SIZE_F2, SHIFT_L2, MAX_L2>(P2, P2_RAW);
	compute_layer<SIZE_F3, SIZE_F2>(P3_RAW, P2, n->L2_a, n->L2_b);
	ReLUClip<SIZE_F3, SHIFT_L3, MAX_L3>(P3, P3_RAW);
	compute_L3(P, P3, n);

	PS = P[0] * mg + P[1] * eg;

	// -dE/dP = true - curr
	float _coeff = learning_rate * (score_true - PS);

	// clip: only for loss
	PS = PS > EVAL_HIGH ? EVAL_HIGH : PS < EVAL_LOW ? EVAL_LOW : PS;
	double loss_ = double(score_true - PS) * (score_true - PS);
	*loss = loss_ + (*loss) * (LOSS_SMOOTH - 1) / LOSS_SMOOTH;

	dst->L3_b[0] += _coeff * mg * (1 << 8);
	dst->L3_b[1] += _coeff * eg * (1 << 8);

	__m256 _coeff256 = _mm256_set1_ps(_coeff);

	for (int i = 0; i < SIZE_F3; i++) {
		dst->L3_a[0 + i * SIZE_O3] += _coeff * mg * P3[i];
		dst->L3_a[1 + i * SIZE_O3] += _coeff * eg * P3[i];

		dPdP3R[i] = P3_RAW[i] > (MAX_L3 << SHIFT_L3) ? 0.0 :
			P3_RAW[i] < 0 ? 0.0 :
			mg * n->L3_a[i] / (1 << SHIFT_L3) +
			eg * n->L3_a[i + SIZE_F3] / (1 << SHIFT_L3);

		dst->L2_b[i] += _coeff * dPdP3R[i] * (1 << 8);
	}

	back_a_<SIZE_F3, SIZE_F2>(_coeff256, dPdP3R, dst->L2_a, P2);

	back_b_<SIZE_F3, SIZE_F2, MAX_L2, SHIFT_L2>
		(_coeff256, dPdP3R, dPdP2R, P2_RAW, n->L2_a, dst->L1_b);

	back_a_<SIZE_F2, SIZE_F1>(_coeff256, dPdP2R, dst->L1_a, P1);

	back_b_<SIZE_F2, SIZE_F1, MAX_L1, SHIFT_L1>
		(_coeff256, dPdP2R, dPdP1R, acc, n->L1_a, dst->L0_b);

	for (Square s = A1; s < SQ_END; ++s) {
		Piece p = board->get_piece(s);
		if (p == EMPTY) { continue; }

		int u = (to_upiece(p) - 1) * 64;
		int addr = board->get_side() == to_color(p) ?
			(u + s) * SIZE_O0 : (u + s + L0_OFFSET) * SIZE_O0;

		for (int i = 0; i < SIZE_O0; i += 8) {
			__m256 _src = _mm256_load_ps(dPdP1R + i);
			__m256 _dst = _mm256_load_ps(dst->L0_a + addr + i);

			_src = _mm256_mul_ps(_coeff256, _src);
			_dst = _mm256_add_ps(_dst, _src);
			_mm256_store_ps(dst->L0_a + addr + i, _dst);
		}
	}
}

int _play_rand(Position* board, PRNG* rng_, Undo* u) 
{	
	MoveList legal_moves;
	legal_moves.generate(*board);

	if (legal_moves.list == legal_moves.end) {
		return 0;
	}

	else {
		Move m = legal_moves.list[rng.get() % legal_moves.length()];
		board->do_move(m, u);
		return 1;
	}
}

struct PBS {
	Net_train* dst_;
	atomic<double>* loss_curr;
	double lr;
	PRNG* r;
	TT* table;
	Move* game;
	bool* quiet;
	//int* decay;
	int score_true;
};

int _find_best(Position* board, int find_depth, PBS* p, int pos)
{
	MoveList legal_moves;
	legal_moves.generate(*board);
	bool quiet = !bool(board->get_checkers());

	if (legal_moves.list == legal_moves.end) {
		p->score_true = board->get_checkers() ? EVAL_LOW : 0;
		p->quiet[pos] = quiet;
		//p->decay[pos] = 100;
		return 0;
	}
	else if (board->get_piece_count() <= 5) { 
		int tb = tb_probe(board);
		p->score_true = tb == 4 ? EVAL_HIGH :
			tb == 0 ? EVAL_LOW : 0;
		//p->quiet[pos] = false;
		p->quiet[pos] = quiet;
		for (int i = 0; i < legal_moves.length(); i++) {
			Move m = legal_moves.list[i];
			if (board->is_non_quiesce(m)) {
				p->quiet[pos] = false;
			}
		}
		//p->decay[pos] = 100;
		return 0;
	}
	else {
		Move m;
		Move nmove = NULL_MOVE;
		int comp_eval;
		int new_eval = EVAL_INIT;
		int eval_list[256] = {};
		int eval_pick[256] = {};

		SearchParams sp = { board, &(Threads.stop), p->table, 1 };
		p->table->increment();

		for (int i = 0; i < legal_moves.length(); i++) {
			m = legal_moves.list[i];

			Undo u;
			board->do_move(m, &u);

			// Repetition + 50-move
			if (board->get_repetition(100) ||
				board->get_fiftymove() > 99)
			{
				eval_list[i] = EVAL_FAIL;
				if (nmove == NULL_MOVE) { 
					nmove = m;
					new_eval = EVAL_FAIL;
				}
			}
			else 
			{
				TTEntry entry = {};
				Key root_key = board->get_key();
				p->table->probe(root_key, &entry);
				int window_c = entry.key != root_key ? 0 : entry.eval;
				int window_a = 2 << (EVAL_BITS - 6);
				int window_b = 2 << (EVAL_BITS - 6);
				int depth = 1;
				while (!Threads.stop) {
					int alpha = new_eval - (2 << (EVAL_BITS - 6));
					bool fail = false;
					int alpha_s = max(alpha, (window_c - window_a));
					if (alpha > window_c - window_a) {
						fail = true;
					}
					window_c = alpha_beta(&sp,
						&entry, depth, board->get_side(), 0,
						alpha_s,
						window_c + window_b);
					if (entry.type == 1) {
						if (fail) {
							break;
						}
						else {
							window_a += 2 << (EVAL_BITS - 6);
						}
					}
					else if (entry.type == -1) {
						window_b += 2 << (EVAL_BITS - 6);
					}
					else if (entry.type == 0) {
						if (depth++ >= find_depth) {
							break;
						}
					}
				}

				comp_eval = -window_c;
				eval_list[i] = comp_eval;

				if (comp_eval > new_eval) {
					new_eval = comp_eval;
					nmove = m;
				}
			}

			board->undo_move(m);
		}

		int cumul = 0;
		for (int i = 0; i < legal_moves.length(); i++) {
			eval_list[i] = eval_list[i] - new_eval + PICK_THRESHOLD;
			if (eval_list[i] < 0) { eval_list[i] = 0; }
			eval_pick[i] = eval_list[i] / PICK_DIV;
			eval_pick[i] = eval_pick[i] * eval_pick[i] * eval_pick[i];
			cumul += eval_pick[i];
			eval_pick[i] = cumul;
		}
		int movepick = p->r->get() % cumul;
		int nmove_i = 0;
		while (eval_pick[nmove_i] < movepick &&
			nmove_i < legal_moves.length() - 1) { nmove_i++; }
		nmove = legal_moves.list[nmove_i];

		if (board->is_non_quiesce(nmove)) {
			quiet = false;
		}

		p->game[pos] = nmove;
		p->quiet[pos] = quiet;
		//p->decay[pos] = eval_list[nmove_i] * 100 / PICK_THRESHOLD;
		return 1;
	}
}

void _do_learning_thread(Net_train* src,
	atomic<bool>* stop, atomic<double>* loss, 
	atomic<uint64_t>* games, atomic<uint64_t>* poss,
	int find_depth, int rand_depth, double lr, PRNG* p) 
{
	Net* tmp = new Net;
	Net_train* src_ = new Net_train;
	Net_train* dst_ = new Net_train;

	Position* board = new Position(tmp);

	TT* table = new TT;

	Undo* rands = (Undo*)malloc(sizeof(Undo) * 1024);
	Move* game = (Move*)malloc(sizeof(Move) * 8192);
	bool* quiet = (bool*)malloc(sizeof(bool) * 8192);
	Undo* stack = (Undo*)malloc(sizeof(Undo) * 8192);
	//int* decay = (int*)malloc(sizeof(int) * 8192);

	while (!(*stop)) {
		copy_float(src_, src);
		convert_to_int(tmp, src_);
		memset(dst_, 0, sizeof(Net_train));
		
		for (int rep = 0; rep < THREAD_LOOP; rep++) {
			//board->set(startpos_fen);
			//int depth = 0;
			//while (depth < rand_depth) 
			//{
			//	if (_play_rand(board, p, rands + depth) == 0) { break; }
			//	depth++;
			//}
			//board->set_accumulator();

			board->set_random(rand_depth, p);
			
			table->clear();
			int score_true = 0;
			int pos = 0;
			PBS pb = { dst_, loss, lr, p, table, game, quiet, 0 };
			
			while (pos < 8192 &&
				_find_best(board, find_depth, &pb, pos) > 0 &&
				(!Threads.stop)) {
				board->do_move(game[pos], stack + pos);
				pos++;
			}

			score_true = pb.score_true;
			if (quiet[pos]) {
				backpropagate(pb.dst_, board,
					score_true,
					pb.loss_curr, pb.lr);
				(*poss)++;
			}

			while (!Threads.stop && pos > 0) {
				pos--;
				board->undo_move(game[pos]);
				score_true = -score_true;
				if (quiet[pos]) {
					backpropagate(pb.dst_, board,
						score_true,
						pb.loss_curr, pb.lr);
					(*poss)++;
				}
			}

			//while (pos < 8192 &&
			//	(_play_rand(board, p, stack + pos) > 0) &&
			//	(!Threads.stop)) {
			//	pos++;
			//	int sc = board->get_material() * 16;
			//	if (sc != 0) {
			//		sc = sc > EVAL_HIGH ? EVAL_HIGH :
			//			sc < EVAL_LOW ? EVAL_LOW :
			//			sc;
			//		backpropagate(pb.dst_, board,
			//			sc,
			//			pb.loss_curr, pb.lr);
			//		(*poss)++;
			//	}
			//}

			(*games)++;
		}

		add_float(src, dst_);
	}

	delete board;
	delete tmp;
	delete src_;
	delete dst_;
	delete table;
	free(game);
	free(quiet);
	free(stack);
}

void do_learning(Net_train* src,
	uint64_t* time_curr, uint64_t* game_curr, uint64_t games,
	int threads, int find_depth, int rand_depth, double lr, bool* c) {

	PRNG rng_0(3245356235923498ULL);

	atomic<double> loss_[32];
	atomic<uint64_t> games_(0);
	atomic<uint64_t> poss_[32];
	thread thread_[32];

	double loss;
	uint64_t poss;

	using namespace std::chrono;

	system_clock::time_point time_start = system_clock::now();
	milliseconds time = milliseconds(0);

	Threads.stop = false;

	for (int i = 0; i < threads; i++) {
		PRNG* n = new PRNG(rng_0.get());

		loss_[i].exchange(0);
		poss_[i].exchange(0);
		
		thread_[i] = thread(
			_do_learning_thread,
			src, &(Threads.stop),
			loss_ + i, 
			&games_, poss_ + i, 
			find_depth, rand_depth,
			lr, n
		);
	}

	std::cout << std::setprecision(1);

	std::cout << '\n'
		<< "Learning rate: " << std::scientific << lr << '\n'
		<< "NGames: " << games / 1000 << 'K' << '\n'
		<< "From: " << rand_depth << '\n'
		<< "Depth: " << find_depth << '\n' << std::endl;

	std::cout << std::setprecision(2);

	while (!(Threads.stop)) {
		std::this_thread::sleep_for(milliseconds(3000));
		loss = 0.0;
		poss = 0;
		for (int i = 0; i < threads; i++) {
			loss += loss_[i];
			poss += poss_[i];
		}
		loss /= threads;
		poss;

		system_clock::time_point time_now = system_clock::now();
		time = duration_cast<milliseconds>(time_now - time_start);

		double div_by = (1.0 - pow(double(LOSS_SMOOTH - 1) / LOSS_SMOOTH, poss / threads)) * LOSS_SMOOTH;
		double accuracy = 100.0 - sqrt(loss / div_by) * 50 / EVAL_HIGH;
		std::cout
			<< "Time: " << std::setw(8) << *time_curr + time.count() / 1000 << "s // "
			<< "Pos: " << std::setw(10) << std::fixed << float(poss) / 1000000 << "M // "
			<< "Games: " << std::setw(8) << std::fixed << float(games_ + *game_curr) / 1000 << "K // "
			<< "Accuracy: " << std::setw(5) << accuracy << "% // "
			<< "Loss: " << std::scientific << loss
			<< std::endl;

		if (games_ > games) { 
			Threads.stop = true;
			break;
		}
	}

	for (int i = 0; i < threads; i++) {
		thread_[i].join();
	}

	if (games_ < games) { *c = false; }

	*time_curr += time.count() / 1000;
	*game_curr += games_;
}

void do_learning_cycle(Net* src, uint64_t* game_switch,
	int threads, int* find_depth, int* rand_depth, double* lr, int cycles)
{
	int count = 0;
	uint64_t game_curr = 0;
	uint64_t time_curr = 0;
	bool c = true;

	Net_train* train = new Net_train;
	convert_to_float(train, src);

	while (c) {
		cout << "Playing Self... ";

		for (int i = 0; i < cycles; i++) {
			do_learning(train,
				&time_curr, &game_curr, game_switch[i],
				threads, find_depth[i], rand_depth[i], lr[i], &c);
		}

		string leading_zero = count < 10 ? "00" : count < 100 ? "0" : "";

		convert_to_int(src, train);
		save_weights<Net>(src,
			"ep" + leading_zero + to_string(count)
			+ "-" + to_string(game_curr / 1000) + "K.bin");

		count++;
	}

	delete train;
}