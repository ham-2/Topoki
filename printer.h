#ifndef PRINTER_INCLUDED
#define PRINTER_INCLUDED

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sstream>

#include "board.h"
#include "eval.h"
#include "search.h"
#include "table.h"
#include "threads.h"

constexpr int PRINT_MIN_MS = 500;
constexpr int PRINT_ATM_MS = 100;

bool compare(std::pair<Move, int> a, std::pair<Move, int> b);

void printer(float time, std::atomic<bool>* stop,
	std::condition_variable* cv, float max_time, bool force_time);


#endif