// SPDX-License-Identifier: MIT
// Copyright (C) 2023, Advanced Micro Devices, Inc.

#ifndef RUNTIME_H_
#define RUNTIME_H_

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace air {
namespace rocm {

class Runtime {
public:
  Runtime()
      : global_mem_pool_({0}), global_dev_pool_({0}), aie_queue_(nullptr) {}
  ~Runtime() {
    while (_alloc_map.size() > 0)
      FreeMemory(_alloc_map.begin()->first);
  }
  static void Init();
  static void ShutDown();
  static Runtime *getRuntime() { return runtime_; }

  void *AllocateMemory(size_t size);
  hsa_status_t FreeMemory(void *ptr);

  hsa_status_t RunKernel(const std::string &pdi_file,
                         const std::string &insts_file,
                         std::vector<void *> &args);
  hsa_status_t loadSegmentPdi(const std::string &pdi_file);
  hsa_status_t dispatchRutimeSequence(std::vector<uint32_t> &sequence_vector,
                                      std::vector<void *> &args);

  hsa_queue_t *getAieQueue() { return aie_queue_; }
  std::vector<hsa_agent_t> getAieAgents() { return aie_agents_; }

private:
  void FindAieAgents();
  void InitMemSegments();
  void InitAieQueue(uint32_t size = 64);

  std::optional<std::size_t> getAllocationSize(void *ptr);


  hsa_amd_memory_pool_t global_mem_pool_;
  hsa_amd_memory_pool_t global_dev_pool_;
  hsa_queue_t *aie_queue_;
  std::vector<hsa_agent_t> aie_agents_;
  std::unordered_map<void *, size_t> _alloc_map;

  static Runtime *runtime_;
};

} // namespace rocm
} // namespace air

#endif // RUNTIME_H_
