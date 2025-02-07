#ifndef LEARNING_INCLUDED
#define LEARNING_INCLUDED

#include <atomic>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <vector>

#include "benchmark.h"
#include "eval.h"
#include "position.h"
#include "movegen.h"
#include "network.h"
#include "threads.h"

#include "tablebase/tbprobe.h"

struct Net_train {
	std::mutex m;

	alignas(32)
	float L0_a[SIZE_F0 * SIZE_O0];
	float L0_b[SIZE_O0];

	float L1_a[SIZE_F1 * SIZE_O1];
	float L1_b[SIZE_O1];

	float L2_a[SIZE_F2 * SIZE_O2];
	float L2_b[SIZE_O2];

	float L3_a[SIZE_F3 * SIZE_O3];
	float L3_b[SIZE_O3];
};

void do_learning(Net_train* dst,
	uint64_t* time_curr, uint64_t* game_curr, uint64_t games,
	int threads, int find_depth, int rand_depth, double lr, bool* c);

void do_learning_cycle(Net* src, uint64_t* game_switch,
	int threads, int* find_depth, int* rand_depth, double* lr, int cycles);

int tb_probe(Position* board);

#endif
