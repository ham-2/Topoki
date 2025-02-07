#ifndef EVAL_INCLUDED
#define EVAL_INCLUDED

#include <iostream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <atomic>

#include "misc.h"
#include "material.h"
#include "movegen.h"
#include "position.h"
#include "table.h"

extern std::atomic<long> node_count;

extern bool limit_strength;
extern int max_noise;
extern int contempt;

constexpr int EVAL_END = (1 << (EVAL_BITS + 1)); // Not real value, set if certain win

constexpr int EVAL_HIGH = (1 << EVAL_BITS); // +64 disks
constexpr int EVAL_LOW = -EVAL_HIGH; // -64 disks
constexpr int EVAL_FAIL = EVAL_LOW - 1;
constexpr int EVAL_MAX = EVAL_HIGH ^ EVAL_END;
constexpr int EVAL_MIN = EVAL_LOW ^ EVAL_END;
constexpr int EVAL_INIT = EVAL_MIN - 1;

int end_eval(Position* board);

int eval(Position* board, int depth, int alpha = EVAL_MIN, int beta = EVAL_MAX);

std::string eval_print(int eval);

inline bool is_mate(int eval) { return eval > EVAL_END || eval < -EVAL_END; }
inline void inc_mate(int& eval) {
	if (eval > EVAL_END) { eval++; }
	else if (eval < -EVAL_END) { eval--; }
}
inline void dec_mate(int& eval) {
	if (eval > EVAL_END) { eval--; }
	else if (eval < -EVAL_END) { eval++; }
}

int add_noise(int& eval);

#endif