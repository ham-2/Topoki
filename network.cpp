#include "network.h"

// Read / Write / Modify

template <typename T, int size>
void get_minmax(T* addr) {
	int max = -65536; int min = 65536;
	int64_t sum = 0; uint64_t suma = 0;
	int cnt = 0; 
	for (int i = 0; i < size; i++) {
		if (addr[i] > max) { max = addr[i]; }
		if (addr[i] < min) { min = addr[i]; }
		sum += addr[i];
		suma += addr[i] > 0 ? addr[i] : -addr[i];
		if (addr[i] > 124 || addr[i] < -124) {
			cnt++;
		}
	}
	std::cout << "max " << max << " min " << min <<
		"\navg " << double(sum) / size << 
		"\nabs " << double(suma) / size <<
		"\n124cnt " << cnt << " / " << size << '\n';
}

void get_stats(Net* net) {
	std::cout << "L0_a: ";
	get_minmax< int8_t, SIZE_F0 * SIZE_O0>(net->L0_a);
	std::cout << "L0_b: ";
	get_minmax<int16_t, SIZE_O0>(net->L0_b);
	std::cout << "L1_a: ";
	get_minmax< int8_t, SIZE_F1 * SIZE_O1>(net->L1_a);
	std::cout << "L1_b: ";
	get_minmax<int16_t, SIZE_O1>(net->L1_b);
	std::cout << "L2_a: ";
	get_minmax< int8_t, SIZE_F2 * SIZE_O2>(net->L2_a);
	std::cout << "L2_b: ";
	get_minmax<int16_t, SIZE_O2>(net->L2_b);
	std::cout << "L3_a: ";
	get_minmax<int16_t, SIZE_F3 * SIZE_O3>(net->L3_a);
	std::cout << "L3_b: ";
	for (int i = 0; i < SIZE_O3; i++) {
		std::cout << net->L3_b[i] << ' ';
	}
	std::cout << std::endl;
}

void rand_weights_all(Net* net, int bits) {
	if (bits < 0) { return; }
	bits = bits > 7 ? 7 : bits;

	uint8_t mask = uint8_t(~0) >> (8 - bits);

	for (int i = 0; i < SIZE_F0 * SIZE_O0; i++) {
		net->L0_a[i] += (rng.get() & rng.get() & rng.get() & mask) 
			- (rng.get() & rng.get() & rng.get() & mask);
	}
	for (int i = 0; i < SIZE_F1 * SIZE_O1; i++) {
		net->L1_a[i] += (rng.get() & rng.get() & rng.get() & mask)
			- (rng.get() & rng.get() & rng.get() & mask);
	}
	for (int i = 0; i < SIZE_F2 * SIZE_O2; i++) {
		net->L2_a[i] += (rng.get() & rng.get() & rng.get() & mask)
			- (rng.get() & rng.get() & rng.get() & mask);
	}
	for (int i = 0; i < SIZE_F3 * SIZE_O3; i++) {
		net->L3_a[i] += (rng.get() & rng.get() & rng.get() & mask)
			- (rng.get() & rng.get() & rng.get() & mask);
	}
}

//void rand_weights_1(Net* net, int mm) {
//	if (mm & 1) {
//		net->L0_a[rng.get() % (SIZE_F0 * SIZE_F1)] += 1;
//	}
//	mm >>= 1;
//	if (mm & 1) {
//		net->L1_a[rng.get() % (SIZE_F1 * SIZE_F2)] += 1;
//	}
//	mm >>= 1;
//	if (mm & 1) {
//		net->L2_a[rng.get() % (SIZE_F2 * SIZE_F3)] += 1;
//	}
//	mm >>= 1;
//	if (mm & 1) {
//		net->L3_a[rng.get() % (SIZE_F3 * SIZE_OUT)] += 1;
//	}
//}

void set_material(Net* net) {
	zero_weights(net);
	
	for (UPiece u : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN }) {
		for (Square s = A1; s < SQ_END; ++s) {
			int a = (u - 1) * 64 + s;
			int b = (u - 1) * 64 + s + L0_OFFSET;

			net->L0_a[a * SIZE_O0 + u] = 4;
			net->L0_a[b * SIZE_O0 + 16 + u] = 4;
		}

		for (int j = 0; j < 2; j++) {
			net->L3_a[u * 2 + j] = Material_MG[u];
			net->L3_a[(u + 16) * 2 + j] = -Material_MG[u];
		}
	}
}

void _sub_L0(int16_t* dst, int addr, Net* n) {
#ifdef _AVX256_
	for (int i = 0; i < SIZE_O0; i += 16) {
		__m256i dst_ = _mm256_load_si256((__m256i*)(dst + i));
		__m256i src_ = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i*)(n->L0_a + addr + i)));
		dst_ = _mm256_subs_epi16(dst_, src_);
		_mm256_store_si256((__m256i*)(dst + i), dst_);
	}
#else
	for (int i = 0; i < SIZE_O0; i++) {
		dst[i] -= n->L0_a[addr + i];
	}
#endif
}

void _add_L0(int16_t* dst, int addr, Net* n) {
#ifdef _AVX256_
	for (int i = 0; i < SIZE_O0; i += 16) {
		__m256i dst_ = _mm256_load_si256((__m256i*)(dst + i));
		__m256i src_ = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i*)(n->L0_a + addr + i)));
		dst_ = _mm256_adds_epi16(dst_, src_);
		_mm256_store_si256((__m256i*)(dst + i), dst_);
	}
#else
	for (int i = 0; i < SIZE_O0; i++) {
		dst[i] += n->L0_a[addr + i];
	}
#endif
}

// Evaluation
// L0 idx
// US   : ((UPiece - 1) * 64 + sq      ) * SIZE_O0
// THEM : ((UPiece - 1) * 64 + sq + 384) * SIZE_O0

void compute_L0(int16_t* dst_w, Piece* squares, Net* n) {
	
	int16_t* dst_b = dst_w + SIZE_O0;
	
	for (int i = 0; i < SIZE_O0; i++) {
		// Bias
		dst_w[i] = n->L0_b[i];
		dst_b[i] = n->L0_b[i];
	}

	for (Square s = A1; s < SQ_END; ++s) {
		Piece p = squares[s];
		if (p != EMPTY) {
			update_L0_place(dst_w, p, s, n);
		}
	}
}

void update_L0_place(int16_t* dst_w, Piece p, Square s, Net* n) 
{
	int16_t* dst_b = dst_w + SIZE_O0;
	int u = (to_upiece(p) - 1) * 64;

	if (to_color(p) == WHITE) {
		_add_L0(dst_w, (u + s            ) * SIZE_O0, n);
		_add_L0(dst_b, (u + s + L0_OFFSET) * SIZE_O0, n);
	}
	else {
		_add_L0(dst_w, (u + s + L0_OFFSET) * SIZE_O0, n);
		_add_L0(dst_b, (u + s            ) * SIZE_O0, n);
	}
}

void update_L0_remove(int16_t* dst_w, Piece p, Square s, Net* n) 
{
	int16_t* dst_b = dst_w + SIZE_O0;
	int u = (to_upiece(p) - 1) * 64;

	if (to_color(p) == WHITE) {
		_sub_L0(dst_w, (u + s            ) * SIZE_O0, n);
		_sub_L0(dst_b, (u + s + L0_OFFSET) * SIZE_O0, n);
	}
	else {
		_sub_L0(dst_w, (u + s + L0_OFFSET) * SIZE_O0, n);
		_sub_L0(dst_b, (u + s            ) * SIZE_O0, n);
	}
}

template <int S, int shift, int max>
void ReLUClip(int16_t* dst, int16_t* src) {
#ifdef _AVX256_
	for (int i = 0; i < S; i += 16) {
		__m256i src_ = _mm256_load_si256((__m256i*)(src + i));
		src_ = _mm256_srai_epi16(src_, shift);
		src_ = _mm256_max_epi16(src_, _mm256_setzero_si256());
		src_ = _mm256_min_epi16(src_, _mm256_set1_epi16(max));
		_mm256_store_si256((__m256i*)(dst + i), src_);
	}
#else
	for (int i = 0; i < S; i++) {
		dst[i] = src[i] >> shift;
		dst[i] = dst[i] < 0 ? 0 :
			dst[i] > max ? max : dst[i];
	}
#endif
}

template <int S, int shift, int max>
void ReLUClip_fallback(int16_t* dst, int16_t* src) {
	for (int i = 0; i < S; i++) {
		dst[i] = src[i] >> shift;
		dst[i] = dst[i] < 0 ? 0 :
			dst[i] > max ? max : dst[i];
	}
}

template <int size_dst, int size_src>
void compute_layer_fallback(int16_t* dst, int16_t* src,
	int8_t* a, int16_t* b)
{
	for (int i = 0; i < size_dst; i += 32) {
		int tmp[32] = { };

		for (int j = 0; j < size_src; j++) {
			for (int k = 0; k < 32; k++) {
				tmp[k] += src[j] * a[j * size_dst + i + k];
			}
		}

		for (int k = 0; k < 32; k++) {
			tmp[k] += b[i + k];
			dst[i + k] = tmp[k] > 32767 ? 32767 :
				tmp[k] < -32768 ? 32768 : tmp[k];
		}
	}
}

template <int size_dst, int size_src>
void compute_layer(int16_t* dst,
	int16_t* src,
	int8_t* a, int16_t* b)
{
#ifdef _AVX256_

	for (int i = 0; i < size_dst; i += 32) {
		__m256i dst1_ = _mm256_setzero_si256();
		__m256i dst2_ = _mm256_setzero_si256();

	for (int j = 0; j < size_src; j += 2) {
		__m256i src1_ = _mm256_set1_epi8(uint8_t(*(src + j)));
		__m256i src2_ = _mm256_set1_epi8(uint8_t(*(src + j + 1)));
		__m256i src_ = _mm256_unpackhi_epi8(src1_, src2_);
		__m256i l1w1_ = _mm256_load_si256((__m256i*)(a +  j      * size_dst + i));
		__m256i l1w2_ = _mm256_load_si256((__m256i*)(a + (j + 1) * size_dst + i));
		__m256i s1 = _mm256_maddubs_epi16(src_, _mm256_unpackhi_epi8(l1w1_, l1w2_));
		__m256i s2 = _mm256_maddubs_epi16(src_, _mm256_unpacklo_epi8(l1w1_, l1w2_));
		dst1_ = _mm256_adds_epi16(dst1_, s1);
		dst2_ = _mm256_adds_epi16(dst2_, s2);
	}

	_mm256_store_si256((__m256i*)(dst + i), _mm256_adds_epi16(
		_mm256_permute2f128_si256(dst2_, dst1_, 0b00100000),
		_mm256_load_si256((__m256i*)(b + i))
	));
	_mm256_store_si256((__m256i*)(dst + i + 16), _mm256_adds_epi16(
		_mm256_permute2f128_si256(dst2_, dst1_, 0b00110001),
		_mm256_load_si256((__m256i*)(b + i + 16))
	));
	}

#else

	compute_layer_fallback<size_dst, size_src>(dst, src, a, b);

#endif
}

void compute_L3(int* dst, int16_t* src, Net* n) {
	for (int j = 0; j < SIZE_O3; j++) {
		dst[j] = n->L3_b[j];
	}
	for (int i = 0; i < SIZE_F3; i++) {
	for (int j = 0; j < SIZE_O3; j++) {
		dst[j] += int64_t(src[i]) * n->L3_a[j + SIZE_O3 * i];
	}
	}
}

void compute(int* dst, int16_t* src, Net* n, Color side_to_move) {

	alignas(32)
	int16_t P1[SIZE_F1];
	int16_t P2[SIZE_F2];
	int16_t P3[SIZE_F3];
	
	ReLUClip<SIZE_F1, SHIFT_L1, MAX_L1>(P1, src + (side_to_move ? SIZE_O0 : 0));
	compute_layer<SIZE_F2, SIZE_F1>(P2, P1, n->L1_a, n->L1_b);
	ReLUClip<SIZE_F2, SHIFT_L2, MAX_L2>(P2, P2);
	compute_layer<SIZE_F3, SIZE_F2>(P3, P2, n->L2_a, n->L2_b);
	ReLUClip<SIZE_F3, SHIFT_L3, MAX_L3>(P3, P3);
	compute_L3(dst, P3, n);
}