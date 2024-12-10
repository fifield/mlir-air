// SPDX-License-Identifier: MIT
// Copyright (C) 2023, Advanced Micro Devices, Inc.

#include "runtime.h"

#include "debug.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>

air::rocm::Runtime *air::rocm::Runtime::runtime_ = nullptr;

#define LOW_ADDR(addr) (reinterpret_cast<uint64_t>(addr) & 0xFFFFFFFF)
#define HIGH_ADDR(addr) (reinterpret_cast<uint64_t>(addr) >> 32)

static std::optional<size_t> load_pdi_file(hsa_amd_memory_pool_t mem_pool,
                                           const std::string &file_name,
                                           void **buf) {
  std::ifstream bin_file(file_name,
                         std::ios::binary | std::ios::ate | std::ios::in);
  if (bin_file.fail())
    return std::nullopt;

  auto size(bin_file.tellg());

  bin_file.seekg(0, std::ios::beg);
  hsa_status_t r = hsa_amd_memory_pool_allocate(mem_pool, size, 0, buf);
  if (r != HSA_STATUS_SUCCESS)
    return std::nullopt;

  bin_file.read(reinterpret_cast<char *>(*buf), size);
  return size;
}

static std::vector<uint32_t> load_instr_file(const std::string &file_name) {

  std::vector<uint32_t> instr_vec;

  std::ifstream bin_file(file_name,
                         std::ios::binary | std::ios::ate | std::ios::in);
  if (bin_file.fail())
    return instr_vec;

  auto size(bin_file.tellg());
  bin_file.seekg(0, std::ios::beg);

  std::string val;
  while (bin_file >> val)
    instr_vec.push_back(std::stoul(val, nullptr, 16));
  return instr_vec;
}

void air::rocm::Runtime::Init() {
  if (!runtime_) {
    runtime_ = new Runtime();
  }

  runtime_->FindAieAgents();
  runtime_->InitMemSegments();
  runtime_->InitAieQueue();
}

void air::rocm::Runtime::ShutDown() {
  delete runtime_;
  runtime_ = nullptr;
}

void *air::rocm::Runtime::AllocateMemory(size_t size) {
  void *mem = nullptr;
  hsa_status_t r =
      hsa_amd_memory_pool_allocate(global_mem_pool_, size, 0, &mem);
  if (r != HSA_STATUS_SUCCESS)
    return nullptr;
  _alloc_map.insert({mem, size});
  return mem;
}

hsa_status_t air::rocm::Runtime::FreeMemory(void *ptr) {
  auto it = _alloc_map.find(ptr);
  if (it == _alloc_map.end())
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  hsa_status_t r = hsa_amd_memory_pool_free(ptr);
  _alloc_map.erase(it);
  return r;
}

std::optional<size_t> air::rocm::Runtime::getAllocationSize(void *ptr) {
  for (auto it = _alloc_map.begin(); it != _alloc_map.end(); it++) {
    size_t p = reinterpret_cast<size_t>(ptr);
    if (p >= reinterpret_cast<size_t>(it->first) &&
        p < reinterpret_cast<size_t>(it->first) + it->second)
      return it->second;
  }
  return std::nullopt;
}

hsa_status_t air::rocm::Runtime::dispatchRutimeSequence(
    void *pdi_buf, std::vector<uint32_t> &sequence_vector,
    std::vector<void *> &args) {

  void *sequence = nullptr;
  hsa_status_t r = hsa_amd_memory_pool_allocate(
      global_dev_pool_, sequence_vector.size() * sizeof(uint32_t), 0,
      &sequence);
  if (r != HSA_STATUS_SUCCESS)
    return r;

  std::memcpy(sequence, sequence_vector.data(),
              sequence_vector.size() * sizeof(uint32_t));

  uint64_t wr_idx = 0;
  uint64_t packet_id = 0;

  // get a packet in the queue
  wr_idx = hsa_queue_add_write_index_relaxed(aie_queue_, 1);
  packet_id = wr_idx % aie_queue_->size;

  // create a packet to store the command
  hsa_amd_aie_ert_packet_t *cmd_pkt =
      static_cast<hsa_amd_aie_ert_packet_t *>(aie_queue_->base_address) +
      packet_id;
  cmd_pkt->state = HSA_AMD_AIE_ERT_STATE_NEW;
  cmd_pkt->count = args.size() * 2 + 6; // # of arguments to put in command
  cmd_pkt->opcode = HSA_AMD_AIE_ERT_START_CU;
  cmd_pkt->header.AmdFormat = HSA_AMD_PACKET_TYPE_AIE_ERT;
  cmd_pkt->header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC
                           << HSA_PACKET_HEADER_TYPE;

  // Creating the payload for the packet
  hsa_amd_aie_ert_start_kernel_data_t *cmd_payload = nullptr;
  hsa_status_t status = hsa_amd_memory_pool_allocate(
      global_mem_pool_, 64, 0, reinterpret_cast<void **>(&cmd_payload));
  if (status != HSA_STATUS_SUCCESS)
    return status;

  // Selecting the PDI to use with this command
  cmd_payload->pdi_addr = pdi_buf;
  // Transaction opcode
  cmd_payload->data[0] = 0x3;
  cmd_payload->data[1] = 0x0;
  cmd_payload->data[2] = LOW_ADDR(sequence);
  cmd_payload->data[3] = HIGH_ADDR(sequence);
  cmd_payload->data[4] = sequence_vector.size();
  int idx = 5;
  for (auto &a : args) {
    cmd_payload->data[idx++] = LOW_ADDR(a);
    cmd_payload->data[idx++] = HIGH_ADDR(a);
  }
  for (auto &a : args) {
    std::optional<size_t> size = getAllocationSize(a);
    if (!size) {
      std::cerr << "Error: could not find allocation " << std::hex << a
                << std::dec << "\n";
      return HSA_STATUS_ERROR;
    }
    cmd_payload->data[idx++] = size.value();
  }
  cmd_pkt->payload_data = reinterpret_cast<uint64_t>(cmd_payload);

  hsa_signal_store_screlease(aie_queue_->doorbell_signal, wr_idx);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air::rocm::Runtime::loadSegmentPdi(void **pdi_buf,
                                                const void *pdi_data,
                                                size_t pdi_size) {

  hsa_status_t r = hsa_amd_memory_pool_allocate(global_dev_pool_, pdi_size, 0,
                                                const_cast<void **>(pdi_buf));
  if (r != HSA_STATUS_SUCCESS)
    return r;
  std::memcpy(*pdi_buf, pdi_data, pdi_size);
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t IterateAgents(hsa_agent_t agent, void *data) {
  hsa_status_t status(HSA_STATUS_SUCCESS);
  hsa_device_type_t device_type;
  std::vector<hsa_agent_t> *aie_agents(nullptr);

  if (!data) {
    status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
    return status;
  }

  aie_agents = static_cast<std::vector<hsa_agent_t> *>(data);
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (device_type == HSA_DEVICE_TYPE_AIE) {
    aie_agents->push_back(agent);
  }

  return status;
}

static hsa_status_t IterateMemPool(hsa_amd_memory_pool_t pool, void *data) {
  hsa_region_segment_t segment_type;
  hsa_status_t status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (status != HSA_STATUS_SUCCESS || segment_type != HSA_REGION_SEGMENT_GLOBAL)
    return status;

  hsa_amd_memory_pool_global_flag_t global_pool_flags;
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_pool_flags);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  if ((global_pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
      (global_pool_flags & HSA_REGION_GLOBAL_FLAG_KERNARG)) {
    debug_print("Runtime: found global segment");
    reinterpret_cast<std::vector<hsa_amd_memory_pool_t> *>(data)->push_back(
        pool);
  }
  return status;
}

static hsa_status_t IterateDevPool(hsa_amd_memory_pool_t pool, void *data) {
  hsa_region_segment_t segment_type;
  hsa_status_t status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (status != HSA_STATUS_SUCCESS || segment_type != HSA_REGION_SEGMENT_GLOBAL)
    return status;

  hsa_amd_memory_pool_global_flag_t global_pool_flags;
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_pool_flags);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  if ((global_pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
      !(global_pool_flags & HSA_REGION_GLOBAL_FLAG_KERNARG)) {
    debug_print("Runtime: found dev segment");
    reinterpret_cast<std::vector<hsa_amd_memory_pool_t> *>(data)->push_back(
        pool);
  }

  return status;
}

void air::rocm::Runtime::FindAieAgents() {
  hsa_iterate_agents(&IterateAgents, reinterpret_cast<void *>(&aie_agents_));
  debug_print("Runtime: found ", aie_agents_.size(), " AIE agents");
}

void air::rocm::Runtime::InitMemSegments() {
  debug_print("Runtime: initializing memory pools");
  std::vector<hsa_amd_memory_pool_t> mem_pools;
  std::vector<hsa_amd_memory_pool_t> dev_pools;
  hsa_amd_agent_iterate_memory_pools(aie_agents_.front(), &IterateMemPool,
                                     reinterpret_cast<void *>(&mem_pools));
  hsa_amd_agent_iterate_memory_pools(aie_agents_.front(), &IterateDevPool,
                                     reinterpret_cast<void *>(&dev_pools));
  assert(mem_pools.size() >= 1);
  assert(dev_pools.size() >= 1);
  global_mem_pool_ = mem_pools.front();
  global_dev_pool_ = dev_pools.front();
}

void air::rocm::Runtime::InitAieQueue(uint32_t size) {
  hsa_status_t r =
      hsa_queue_create(aie_agents_.front(), size, HSA_QUEUE_TYPE_SINGLE,
                       nullptr, nullptr, 0, 0, &aie_queue_);
  assert(r == HSA_STATUS_SUCCESS);
  assert(aie_queue_);
  assert(aie_queue_->base_address);
}