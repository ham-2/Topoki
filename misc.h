#ifndef MISC_INCLUDED
#define MISC_INCLUDED

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

class PRNG {
private:
	uint64_t state;

public:
	PRNG(uint64_t seed) { state = seed; }
		
	uint64_t get() {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		return state;
	}

	inline uint64_t get_seed() { return state; }
};

int load_file(char* dst, std::string filename, size_t size);
void save_file(char* src, std::string filename, size_t size);

extern PRNG rng;

#endif
