// SPDX-License-Identifier: MIT
// Copyright (C) 2023, Advanced Micro Devices, Inc.

#ifndef RUNTIME_H_
#define RUNTIME_H_

#include <vector>
#include <string>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

namespace air {
namespace rocm {

class Runtime {
public:
  Runtime() = default;
  static void Init();
  static void ShutDown();

  void *AllocateMemory(size_t size);
  void FreeMemory(void *ptr);

  hsa_status_t RunKernel(const std::string &pdi_file, const std::string &insts_file, std::vector<void *> &args);
  static Runtime *runtime_;

private:
  static hsa_status_t IterateAgents(hsa_agent_t agent, void *data);
  static hsa_status_t IterateMemPool(hsa_amd_memory_pool_t pool, void *data);
  static hsa_status_t IterateDevPool(hsa_amd_memory_pool_t pool, void *data);
  void FindAieAgents();
  void InitMemSegments();
  void InitAieQueue(uint32_t size = 64);

  hsa_amd_memory_pool_t global_mem_pool_;
  hsa_amd_memory_pool_t global_dev_pool_;
  std::vector<hsa_agent_t> aie_agents_;
  hsa_queue_t *aie_queue_;
};

} // namespace rocm
} // namespace air

#endif // RUNTIME_H_
