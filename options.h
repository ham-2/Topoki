#ifndef OPTIONS_INCLUDED
#define OPTIONS_INCLUDED

#include <sstream>

#include "search.h"
#include "eval.h"
#include "table.h"
#include "threads.h"

void print_option();

void set_option(std::istringstream& ss);

#endif