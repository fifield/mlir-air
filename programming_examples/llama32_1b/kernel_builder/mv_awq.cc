//===- mv_awq.cc --------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Fused AWQ int4 dequant + BF16 GEMV external kernel for AIE.
//
// Computes C[M] = dequant(Q[M,K], params[M,2*K/G]) @ B[K].
//
// qweights layout:
//   row-major (M, K/2) uint8, low nibble for even K, high nibble for odd K.
// params layout:
//   row-major (M, 2*K/GROUP_SIZE) bf16, interleaved [scale_g, zero_g].
//
// The row_offset argument offsets only the output pointer. qweights and params
// are expected to point at the current row tile; AIR DMA controls tile offsets.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <aie_api/aie.hpp>

#ifndef GROUP_SIZE
#define GROUP_SIZE 128
#endif

#ifndef DIM_M_OUTPUT
#define DIM_M_OUTPUT 2048
#endif

static inline uint8_t unpack_u4(const uint8_t byte, const uint32_t k) {
  return (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
}

static void matvec_awq_scalar(uint32_t m, uint32_t k,
                              const uint8_t *__restrict qweights,
                              const bfloat16 *__restrict params,
                              const bfloat16 *__restrict b,
                              bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const uint32_t packed_k = k / 2;
  const uint32_t groups = (k + GROUP_SIZE - 1) / GROUP_SIZE;

  for (uint32_t row = 0; row < m; ++row) {
    const uint8_t *__restrict row_q = qweights + row * packed_k;
    const bfloat16 *__restrict row_params = params + row * groups * 2;
    float acc = 0.0f;

    for (uint32_t kk = 0; kk < k; ++kk) {
      const uint32_t group = kk / GROUP_SIZE;
      const float scale = static_cast<float>(row_params[2 * group]);
      const float zero = static_cast<float>(row_params[2 * group + 1]);
      const uint8_t q = unpack_u4(row_q[kk >> 1], kk);
      const float w = (static_cast<float>(q) - zero) * scale;
      acc += w * static_cast<float>(b[kk]);
    }

    c[row] = static_cast<bfloat16>(acc);
  }
}

extern "C" {

void matvec_awq_bf16(uint32_t m, uint32_t k, uint32_t row_offset,
                     const uint8_t *__restrict qweights,
                     const bfloat16 *__restrict params,
                     const bfloat16 *__restrict b_in,
                     bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_awq_scalar(m, k, qweights, params, b_in, c_out);
}

void linalg_fill_bf16(bfloat16 *c_out) {
  for (uint32_t i = 0; i < DIM_M_OUTPUT; ++i) {
    c_out[i] = static_cast<bfloat16>(0.0f);
  }
}

} // extern "C"
