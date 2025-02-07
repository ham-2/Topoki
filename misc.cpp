#include "misc.h"

PRNG rng = PRNG(3245356235923498ULL);

int load_file(char* dst, std::string filename, size_t size)
{
	std::cout << "Loading \"" << filename << "\"\n";
	std::ifstream input(filename, std::ios::binary);

	input.read(dst, size);

	if (input.fail() || (input.peek() != EOF)) {
		std::cout << "Failed to load" << std::endl;
		return -1;
	}
	else {
		std::cout << "Loaded \"" << filename << "\"" << std::endl;
	}

	input.close();
	return 0;
}

void save_file(char* src, std::string filename, size_t size)
{
	std::cout << "Saving to \"" << filename << "\"\n";
	std::ofstream output(filename, std::ios::binary);

	output.write(src, size);

	if (output.fail()) {
		std::cout << "Failed to save" << std::endl;
	}
	else {
		std::cout << "Saved to \"" << filename << "\"" << std::endl;
	}

	output.close();
}