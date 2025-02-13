#ifndef NETWORK_INCLUDED
#define NETWORK_INCLUDED

#include "board.h"
#include "material.h"
#include "misc.h"

constexpr int SIZE_F0 = 6 * 2 * 64;
constexpr int SIZE_O0 = 192;

constexpr int SIZE_F1 = SIZE_O0;
constexpr int SIZE_O1 = 64;

constexpr int SIZE_F2 = SIZE_O1;
constexpr int SIZE_O2 = 64;

constexpr int SIZE_F3 = SIZE_O2;
constexpr int SIZE_O3 = 2;

constexpr int SHIFT_L1 = 2;
constexpr int SHIFT_L2 = 5;
constexpr int SHIFT_L3 = 5;

constexpr int MAX_L1 = 255;
constexpr int MAX_L2 = 255;
constexpr int MAX_L3 = (1 << 15) - 1;

constexpr int EVAL_BITS = 16;

constexpr int L0_OFFSET = 384;

struct Net {

	alignas(32) 
	int8_t L0_a[SIZE_F0 * SIZE_O0];
	int16_t L0_b[SIZE_O0];

	int8_t L1_a[SIZE_F1 * SIZE_O1];
	int16_t L1_b[SIZE_O1];

	int8_t L2_a[SIZE_F2 * SIZE_O2];
	int16_t L2_b[SIZE_O2];

	int16_t L3_a[SIZE_F3 * SIZE_O3];
	int32_t L3_b[SIZE_O3];
};

inline void zero_weights(Net* net) { memset(net, 0, sizeof(Net)); }
void rand_weights_all(Net* net, int bits);
//void rand_weights_1(Net* net, int mm);
void set_material(Net* net);

template <typename T>
inline int load_weights(T* net, std::string filename) {
	return load_file((char*)net, filename, sizeof(T));
}

template <typename T>
inline void save_weights(T* net, std::string filename) {
	return save_file((char*)net, filename, sizeof(T));
}

void get_stats(Net* net);

void compute_L0(int16_t* dst_w, Piece* squares, Net* n);
void update_L0_place(int16_t* dst_w, Piece p, Square s, Net* n);
void update_L0_remove(int16_t* dst_w, Piece p, Square s, Net* n);
template <int S, int shift, int max>
void ReLUClip(int16_t* dst, int16_t* src);
template <int size_dst, int size_src>
void compute_layer(int16_t* dst, int16_t* src,
	               int8_t* a, int16_t* b);
void compute_L3(int* dst, int16_t* src, Net* n);

void compute(int* dst, int16_t* src, Net* n, Color side_to_move);

#endif