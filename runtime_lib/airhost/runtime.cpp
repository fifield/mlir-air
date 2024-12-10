// SPDX-License-Identifier: MIT
// Copyright (C) 2023, Advanced Micro Devices, Inc.

#include "runtime.h"

#include "debug.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>

using namespace air::rocm;

Runtime *Runtime::runtime_ = nullptr;

#define LOW_ADDR(addr) (reinterpret_cast<uint64_t>(addr) & 0xFFFFFFFF)
#define HIGH_ADDR(addr) (reinterpret_cast<uint64_t>(addr) >> 32)

static std::optional<uint32_t> load_pdi_file(hsa_amd_memory_pool_t mem_pool,
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
  uint32_t pdi_size = size;
  return pdi_size;
}

static std::optional<uint32_t> load_instr_file(hsa_amd_memory_pool_t mem_pool,
                                               const std::string &file_name,
                                               void **buf) {
  std::ifstream bin_file(file_name,
                         std::ios::binary | std::ios::ate | std::ios::in);
  if (bin_file.fail())
    return std::nullopt;

  auto size(bin_file.tellg());
  bin_file.seekg(0, std::ios::beg);
  std::vector<uint32_t> pdi_vec;
  std::string val;

  while (bin_file >> val) {
    pdi_vec.push_back(std::stoul(val, nullptr, 16));
  }
  hsa_status_t r = hsa_amd_memory_pool_allocate(mem_pool, size, 0, buf);
  if (r != HSA_STATUS_SUCCESS)
    return std::nullopt;
  std::memcpy(*buf, pdi_vec.data(), pdi_vec.size() * sizeof(uint32_t));
  uint32_t num_instr = pdi_vec.size();
  return num_instr;
}

void air::rocm::Runtime::Init() {
  if (!runtime_) {
    runtime_ = new Runtime();
  }

  runtime_->FindAieAgents();
  runtime_->InitMemSegments();
  runtime_->InitAieQueue();
}

void air::rocm::Runtime::ShutDown() { delete runtime_; }

void *air::rocm::Runtime::AllocateMemory(size_t size) {
  void *mem(nullptr);

  hsa_amd_memory_pool_allocate(global_mem_pool_, size, 0, &mem);

  return mem;
}

void air::rocm::Runtime::FreeMemory(void *ptr) {
  hsa_amd_memory_pool_free(ptr);
}

hsa_status_t air::rocm::Runtime::RunKernel(const std::string &pdi_file,
                                           const std::string &insts_file,
                                           std::vector<void *> &args) {
  void *pdi_buf = nullptr;
  void *insts_buf = nullptr;
  hsa_status_t status = HSA_STATUS_SUCCESS;

  std::optional<uint32_t> pdi_size = load_pdi_file(
      global_dev_pool_, pdi_file, reinterpret_cast<void **>(&pdi_buf));
  std::optional<uint32_t> insts_size = load_instr_file(
      global_dev_pool_, insts_file, reinterpret_cast<void **>(&insts_buf));

  if (!pdi_size || !insts_size)
    return HSA_STATUS_ERROR;

  std::cout << "loaded pdi file: " << pdi_file << " size: " << pdi_size.value()
            << std::endl;
  std::cout << "loaded insts file: " << insts_file
            << " size: " << insts_size.value() << std::endl;

  hsa_amd_aie_ert_hw_ctx_cu_config_addr_t cu_config{
      .cu_config_addr = reinterpret_cast<uint64_t>(pdi_buf),
      .cu_func = 0,
      .cu_size = pdi_size.value()};

  hsa_amd_aie_ert_hw_ctx_config_cu_param_addr_t config_cu_args{
      .num_cus = 1, .cu_configs = &cu_config};

  // Configure the queue's hardware context.
  status = hsa_amd_queue_hw_ctx_config(
      aie_queue_, HSA_AMD_QUEUE_AIE_ERT_HW_CXT_CONFIG_CU, &config_cu_args);
  if (status != HSA_STATUS_SUCCESS)
    return status;

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
  cmd_pkt->count = 0xA; // # of arguments to put in command
  cmd_pkt->opcode = HSA_AMD_AIE_ERT_START_CU;
  cmd_pkt->header.AmdFormat = HSA_AMD_PACKET_TYPE_AIE_ERT;
  cmd_pkt->header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC
                           << HSA_PACKET_HEADER_TYPE;

  // Creating the payload for the packet
  hsa_amd_aie_ert_start_kernel_data_t *cmd_payload = nullptr;
  status = hsa_amd_memory_pool_allocate(
      global_dev_pool_, 64, 0, reinterpret_cast<void **>(&cmd_payload));
  if (status != HSA_STATUS_SUCCESS)
    return status;

  std::cout << "args[0]: " << args[0] << std::endl;
  std::cout << "args[1]: " << args[1] << std::endl;

  // Selecting the PDI to use with this command
  cmd_payload->cu_mask = 0x1;
  // Transaction opcode
  cmd_payload->data[0] = 0x3;
  cmd_payload->data[1] = 0x0;
  cmd_payload->data[2] = LOW_ADDR(insts_buf);
  cmd_payload->data[3] = HIGH_ADDR(insts_buf);
  cmd_payload->data[4] = insts_size.value();
  cmd_payload->data[5] = LOW_ADDR(args[0]);
  cmd_payload->data[6] = HIGH_ADDR(args[0]);
  cmd_payload->data[7] = LOW_ADDR(args[1]);
  cmd_payload->data[8] = HIGH_ADDR(args[1]);
  cmd_payload->data[9] = 256 * sizeof(uint32_t);
  cmd_payload->data[10] = 256 * sizeof(uint32_t);
  cmd_pkt->payload_data = reinterpret_cast<uint64_t>(cmd_payload);

  hsa_signal_store_screlease(aie_queue_->doorbell_signal, wr_idx);

  return status;
}

hsa_status_t Runtime::IterateAgents(hsa_agent_t agent, void *data) {
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

hsa_status_t air::rocm::Runtime::IterateMemPool(hsa_amd_memory_pool_t pool,
                                                void *data) {
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
    *static_cast<hsa_amd_memory_pool_t *>(data) = pool;
  }
  return status;
}

hsa_status_t air::rocm::Runtime::IterateDevPool(hsa_amd_memory_pool_t pool,
                                                void *data) {
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
    *static_cast<hsa_amd_memory_pool_t *>(data) = pool;
  }

  return status;
}

void air::rocm::Runtime::FindAieAgents() {
  hsa_iterate_agents(&Runtime::IterateAgents,
                     reinterpret_cast<void *>(&aie_agents_));
  debug_print("Runtime: found ", aie_agents_.size(), " AIE agents");
}

void air::rocm::Runtime::InitMemSegments() {
  debug_print("Runtime: initializing memory pools");
  hsa_amd_agent_iterate_memory_pools(
      aie_agents_.front(), &Runtime::IterateMemPool,
      reinterpret_cast<void *>(&global_mem_pool_));
  hsa_amd_agent_iterate_memory_pools(
      aie_agents_.front(), &Runtime::IterateDevPool,
      reinterpret_cast<void *>(&global_dev_pool_));
}

void air::rocm::Runtime::InitAieQueue(uint32_t size) {
  hsa_status_t r =
      hsa_queue_create(aie_agents_.front(), size, HSA_QUEUE_TYPE_SINGLE,
                       nullptr, nullptr, 0, 0, &aie_queue_);
  assert(r == HSA_STATUS_SUCCESS);
  assert(aie_queue_);
  assert(aie_queue_->base_address);
}