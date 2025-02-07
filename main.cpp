#include <iostream>
#include <sstream>
#include <string>
#include <filesystem>

#include "benchmark.h"
#include "board.h"
#include "material.h"
#include "options.h"
#include "position.h"
#include "search.h"
#include "threads.h"
#include "tune.h"

using namespace std;

int main() {
	
	cout << "Topoki 4 2025-02-05\n";

	string input, word;

	Board::init();
	Position::init();
	Threads.init();
	
	if (tb_init("tablebase/5")) {
		cout << "Loaded Tablebase for " << TB_LARGEST << " Pieces\n";
	}

	cout << endl;

	while (true) {
		getline(cin, input);

		istringstream ss(input);
		word.clear();
		ss >> skipws >> word;

		if (word == "quit") { break; }

		else if (word == "uci") {
			cout << "id name Topoki\n"
				<< "id author Seungrae Kim" << endl;
			// Options
			print_option();
			cout << "uciok" << endl;
		}

		else if (word == "isready") {
			cout << "readyok" << endl;
		}
		
		else if (word == "position") {
			ss >> word;
			string fen;
			if (word == "fen") {
				while (ss >> word && word != "moves") {
					fen += word + " ";
				}
			}
			else if (word == "startpos") {
				fen = startpos_fen;
				ss >> word; // "moves"
			}

			Threads.stop = true;
			Threads.acquire_lock();
			Threads.set_all(fen);
			if (word == "moves") {
				while (ss >> word) {
					Threads.do_move(word);
				}
			}
			Threads.release_lock();
		}

		else if (word == "go") {
			Color c = Threads.get_color();
			float time;
			float max_time;
			bool force_time;
			int max_ply;
			get_time(ss, c, time, max_time, force_time, max_ply);
			thread t = thread(search_start, Threads.threads[0], time, max_time, force_time, max_ply);
			t.detach();
		}

		else if (word == "stop") {
			Threads.stop = true;
		}

		else if (word == "setoption") {
			set_option(ss);
		}

		else if (word == "perft") {
			int depth;
			ss >> depth;
			perft(Threads.threads[0]->board, depth);
		}

		else if (word == "showboard") {
			int threadidx;
			threadidx = ss >> threadidx ? threadidx : 0;
			Threads.show(threadidx);
		}

		else if (word == "moves") {
			Threads.acquire_lock();
			while (ss >> word) {
				Threads.do_move(word);
			}
			Threads.release_lock();
		}

		else if (word == "generate") {
			Threads.gen();
		}

		else if (word == "test") {
			ss >> word;
			Threads.board->verify();
			//Threads.see(word);
			Threads.test_eval();
		}

		else if (word == "tb") {
			cout << tb_probe(Threads.board) << endl;
		}

		// Weight

		else if (word == "load") {
			ss >> word;
			word = std::filesystem::current_path().string() + "/" + word;
			Threads.acquire_lock();
			load_weights<Net>(Threads.n, word);
			Threads.set_weights();
			Threads.release_lock();
		}

		else if (word == "save") {
			ss >> word;
			word = std::filesystem::current_path().string() + "/" + word;
			save_weights<Net>(Threads.n, word);
		}

		else if (word == "zero") {
			zero_weights(Threads.n);
			Threads.set_weights();
		}

		else if (word == "rand") {
			ss >> word;
			rand_weights_all(Threads.n, stoi(word));
			Threads.set_weights();
		}

		else if (word == "tune") {
			int cycle = 0;
			int find_depth[32] = { };
			int rand_depth[32] = { };
			uint64_t games_[32] = { };
			double lr[32] = { };

			ss >> word;
			int threads = stoi(word);

			cout << "\nLearning with: " << threads << " Threads\n" << endl;

			while (ss >> word) {
				find_depth[cycle] = stoi(word);

				ss >> word;
				rand_depth[cycle] = stoi(word);

				ss >> word;
				games_[cycle] = int64_t(stoi(word)) * 1000;

				ss >> word;
				lr[cycle] = stod(word);

				cycle++;
			}

			thread t = thread(do_learning_cycle, Threads.n,
				games_, threads, find_depth, rand_depth, lr, cycle);
			t.detach();

		}

		else if (word == "setacc") {
			Threads.set_weights();
		}

		else if (word == "stats") {
			get_stats(Threads.n);
		}

		else if (word == "setmat") {
			set_material(Threads.n);
			Threads.set_weights();
		}

		else if (word == "setr") {
			ss >> word;
			int pieces = stoi(word);
			ss >> word;
			uint64_t seed = stoull(word);
			Threads.setr(pieces, seed);
			Threads.show(-1);
		}

	}

	return 44;
}
