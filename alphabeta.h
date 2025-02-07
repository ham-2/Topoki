#ifndef ALPHABETA_INCLUDED
#define ALPHABETA_INCLUDED

#include <atomic>
#include <iostream>
#include <mutex>

#include "eval.h"
#include "position.h"
#include "movegen.h"
#include "table.h"

constexpr int NULLMOVE_MAX_PLY = 8;

struct SearchParams {
	Position* board;
	std::atomic<bool>* stop;
	TT* table;
	int step;
};

int alpha_beta(SearchParams* sp, TTEntry* probe,
	int ply, Color root_color, int root_dist,
	int alpha, int beta);

#endif