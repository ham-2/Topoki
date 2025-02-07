#ifndef TABLE_INCLUDED
#define TABLE_INCLUDED

#include <cstring>
#include <thread>
#include <mutex>

#include "position.h"

#ifdef _WIN64
#include <windows.h>
#include <memoryapi.h>
#else
#include <stdlib.h>
#endif

extern int num_threads;

constexpr int TABLE_MB_DEFAULT = 32;
constexpr int TABLE_MB_MAX = 4096;

class spinlock {
	std::atomic_flag locked = ATOMIC_FLAG_INIT;
public:
	void lock() {
		while (locked.test_and_set(std::memory_order_acquire)) { ; }
	}
	void unlock() {
		locked.clear(std::memory_order_release);
	}
};

struct TTEntry {
	Key key;
	int eval;
	Move nmove;
	uint16_t table_sn;
	uint8_t depth;
	int8_t type;
	spinlock m;
	Key pad;
};

struct TT {
	TTEntry* table;
		
	static size_t TT_MB_SIZE;
	static uint64_t TT_LENGTH;
	static int tt_sn;

	TT();
	~TT();

	int probe(Key key, TTEntry* probe);
	void register_entry(Key key, int eval, Move move, uint8_t depth, int8_t type);
	TTEntry* backup_entry(Key key);
	void write_entry(TTEntry* ptr);
	void clear_entry(Key key);
	void clear();
	void change_size(size_t new_size);
	void increment() { tt_sn++; }
};
	

extern TT Main_TT;

void init_table();

#endif