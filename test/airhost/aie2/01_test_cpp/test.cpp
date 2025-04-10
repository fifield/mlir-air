//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <air.hpp>

#include <dlfcn.h>
#include <iostream>

int main(int argc, char *argv[]) {

  hsa_status_t init_status = air_init();
  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed with code " << init_status << ", Exiting"
              << "\n";
    return -1;
  }

  air_module_handle_t handle = air_module_load_from_file(nullptr);
  if (!handle) {
    std::cout << "Failed to load air module"
              << "\n";
    return -1;
  }

  // get the mlir generated c-interface to the air mul() function:
  // void _mlir_ciface_mul(tensor_t<uint32_t, 1> *input0,
  //                       tensor_t<uint32_t, 1> *input1,
  //                       tensor_t<uint32_t, 1> *output0);
  auto mul_fn = (void (*)(tensor_t<uint32_t, 1> *, tensor_t<uint32_t, 1> *,
                          tensor_t<uint32_t, 1> *))dlsym((void *)handle,
                                                         "_mlir_ciface_mul");
  if (!mul_fn) {
    std::cout << "Failed to locate _mlir_ciface_mul in .so"
              << "\n";
    return -1;
  }

  // create input and output tensors
  const int N = 1024;
  tensor_t<uint32_t, 1> input0;
  tensor_t<uint32_t, 1> input1;
  tensor_t<uint32_t, 1> output0;
  input0.shape[0] = N;
  input1.shape[0] = N;
  output0.shape[0] = N;

  input0.alloc = input0.data =
      (uint32_t *)air_malloc(input0.shape[0] * sizeof(uint32_t));
  input1.alloc = input1.data =
      (uint32_t *)air_malloc(input1.shape[0] * sizeof(uint32_t));
  output0.alloc = output0.data =
      (uint32_t *)air_malloc(output0.shape[0] * sizeof(uint32_t));

  for (int i = 0; i < N; i++) {
    input0.data[i] = i + 1;
    input1.data[i] = i + 1;
    output0.data[i] = 0xffcafeff;
  }
  mul_fn(&input0, &input1, &output0);

  std::vector<void *> args{input0.data, input1.data, output0.data};
  dispatch_segment(args);

  // check output tensor
  int errs = 0;
  for (size_t i = 0; i < output0.shape[0]; i++) {
    uint32_t ref = input0.data[i] * input1.data[i];
    if (output0.data[i] != ref) {
      std::cout << "Mismatch at index " << i << ": expected " << std::hex << ref
                << ", but got " << std::hex << ((uint32_t *)output0.data)[i]
                << "\n";
      errs++;
    }
  }

  air_free(input0.data);
  air_free(input1.data);
  air_free(output0.data);

  if (errs) {
    std::cout << "failed."
              << "\n";
    return -1;
  }
  std::cout << "PASS!"
            << "\n";
  return 0;
}