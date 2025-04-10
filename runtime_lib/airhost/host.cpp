//===- host.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air.hpp"
#include "air_host.h"
#include "air_host_impl.h"
#include "runtime.h"
#include "test_library.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include <assert.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream> // ifstream
#include <iomanip> // setbase()
#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define BOUNCE_BUFFER_SIZE 0x8000

#define ROCR_XDNA 1

#define VERBOSE true

// temporary solution to stash some state
extern "C" {

air_rt_herd_desc_t _air_host_active_herd = {nullptr, nullptr, nullptr};
air_rt_segment_desc_t _air_host_active_segment = {nullptr, nullptr, nullptr};
air_module_handle_t _air_host_active_module = (air_module_handle_t) nullptr;
}

// Determining if an hsa agent is an AIE agent or not
hsa_status_t find_aie(hsa_agent_t agent, void *data) {
  hsa_status_t status(HSA_STATUS_SUCCESS);
  hsa_device_type_t device_type;
  std::vector<hsa_agent_t> *aie_agents = nullptr;

  if (!data) {
    status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
    printf("find_aie: INVALID ARGUMENT\n");
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

hsa_status_t air_init() {
  printf("%s\n", __func__);

  hsa_status_t hsa_ret = hsa_init();
  air::rocm::Runtime::Init();

  if (hsa_ret != HSA_STATUS_SUCCESS) {
    std::cerr << "hsa_init failed" << std::endl;
    return hsa_ret;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_shut_down() {

  if (_air_host_active_module)
    air_module_unload(_air_host_active_module);

  hsa_status_t hsa_ret = hsa_shut_down();
  if (hsa_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] hsa_shut_down() failed\n");
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t run_kernel(const std::string &pdi_file,
                        const std::string &insts_file,
                        std::vector<void *> &args) {
  return air::rocm::Runtime::getRuntime()->RunKernel(pdi_file, insts_file,
                                                     args);
}

hsa_status_t dispatch_segment(std::vector<void *> &args) {
  auto segment_desc = _air_host_active_segment.segment_desc;
  if (!segment_desc)
    return HSA_STATUS_ERROR;

  std::string segment_name(segment_desc->name, segment_desc->name_length);

  std::string func_name = "__airrt_" + segment_name + "_aie_functions";
  air_rt_aie_functions_t *mlir = (air_rt_aie_functions_t *)dlsym(
      (void *)_air_host_active_module, func_name.c_str());

  void *pdi_buf = nullptr;
  size_t pdi_size = mlir->get_pdi(&pdi_buf);
  if (!pdi_size || !pdi_buf) {
    printf("Failed to get PDI\n");
    return HSA_STATUS_ERROR;
  }

  hsa_status_t r =
      air::rocm::Runtime::getRuntime()->loadSegmentPdi(&pdi_buf, pdi_size);
  if (r != HSA_STATUS_SUCCESS) {
    printf("Failed to load segment PDI!\n");
    return HSA_STATUS_ERROR;
  }

  uint32_t *insts_buf = nullptr;
  size_t insts_size = mlir->get_insts(reinterpret_cast<void **>(&insts_buf));
  if (!insts_size || !insts_buf) {
    printf("Failed to get instructions\n");
    return HSA_STATUS_ERROR;
  }
  std::vector<uint32_t> sequence_vector(
      insts_buf, insts_buf + (insts_size / sizeof(uint32_t)));
  return air::rocm::Runtime::getRuntime()->dispatchRutimeSequence(
      pdi_buf, sequence_vector, args);
}

air_module_handle_t air_module_load_from_file(const char *filename) {
  if (_air_host_active_module)
    air_module_unload(_air_host_active_module);

  air_module_handle_t handle;
  void *_handle = dlopen(filename, RTLD_NOW);
  if (!_handle) {
    printf("%s\n", dlerror());
    return 0;
  }
  auto q = air::rocm::Runtime::getRuntime()->getAieQueue();
  auto agent = &air::rocm::Runtime::getRuntime()->getAieAgents()[0];

  _air_host_active_module = (air_module_handle_t)_handle;
  _air_host_active_herd = {q, agent, nullptr};
  _air_host_active_segment = {q, agent, nullptr};

  return (air_module_handle_t)_handle;
}

int32_t air_module_unload(air_module_handle_t handle) {
  if (!handle)
    return -1;

  if (auto module_desc = air_module_get_desc(handle)) {
    for (int i = 0; i < module_desc->segment_length; i++) {
      for (int j = 0; j < module_desc->segment_descs[i]->herd_length; j++) {
        auto herd_desc = module_desc->segment_descs[i]->herd_descs[j];
        if (herd_desc == _air_host_active_herd.herd_desc) {
          if (_air_host_active_segment.q) {
            hsa_queue_destroy(_air_host_active_segment.q);
          }
          _air_host_active_herd = {nullptr, nullptr};
          _air_host_active_segment = {nullptr, nullptr, nullptr};
        }
      }
    }
  }
  if (_air_host_active_module == handle) {
    _air_host_active_module = (air_module_handle_t) nullptr;
  }

  return dlclose((void *)handle);
}

air_herd_desc_t *air_herd_get_desc(air_module_handle_t handle,
                                   air_segment_desc_t *segment_desc,
                                   const char *herd_name) {
  if (!handle)
    return nullptr;
  if (!segment_desc)
    return nullptr;

  auto module_desc = air_module_get_desc(handle);
  if (!module_desc)
    return nullptr;

  if (!air_segment_get_desc(handle, segment_desc->name))
    return nullptr;

  for (int i = 0; i < segment_desc->herd_length; i++) {
    auto herd_desc = segment_desc->herd_descs[i];
    if (!strncmp(herd_name, herd_desc->name, herd_desc->name_length))
      return herd_desc;
  }
  return nullptr;
}

air_segment_desc_t *air_segment_get_desc(air_module_handle_t handle,
                                         const char *segment_name) {
  if (!handle)
    return nullptr;

  auto module_desc = air_module_get_desc(handle);
  if (!module_desc)
    return nullptr;

  for (int i = 0; i < module_desc->segment_length; i++) {
    auto segment_desc = module_desc->segment_descs[i];
    if (!strncmp(segment_name, segment_desc->name, segment_desc->name_length)) {
      return segment_desc;
    }
  }
  return nullptr;
}

air_module_desc_t *air_module_get_desc(air_module_handle_t handle) {
  if (!handle)
    return nullptr;
  return (air_module_desc_t *)dlsym((void *)handle,
                                    "__airrt_module_descriptor");
}

uint64_t air_segment_load(const char *name) {

  auto segment_desc = air_segment_get_desc(_air_host_active_module, name);
  if (!segment_desc) {
    printf("Failed to locate segment descriptor '%s'!\n", name);
    assert(0);
  }
  std::string segment_name(segment_desc->name, segment_desc->name_length);

  if (VERBOSE)
    std::cout << "load segment: " << segment_name << "\n";

  _air_host_active_segment.segment_desc = segment_desc;
  return reinterpret_cast<uint64_t>(segment_desc);
}

uint64_t air_herd_load(const char *name) {

  // If no segment is loaded, load the segment associated with this herd
  if (!_air_host_active_segment.segment_desc) {
    bool loaded = false;
    if (auto module_desc = air_module_get_desc(_air_host_active_module)) {
      for (int i = 0; !loaded && i < module_desc->segment_length; i++) {
        for (int j = 0;
             !loaded && j < module_desc->segment_descs[i]->herd_length; j++) {
          auto herd_desc = module_desc->segment_descs[i]->herd_descs[j];
          // use the segment of the first herd with a matching name
          if (!strncmp(name, herd_desc->name, herd_desc->name_length)) {
            air_segment_load(module_desc->segment_descs[i]->name);
            loaded = true; // break
          }
        }
      }
    }
  }
  auto herd_desc = air_herd_get_desc(
      _air_host_active_module, _air_host_active_segment.segment_desc, name);
  // In some scenarios load_segment is not called. This is a temporary hack
  // to support that case.
  if (!herd_desc) {
    if (_air_host_active_segment.segment_desc) {
      _air_host_active_segment.segment_desc = 0;
      return air_herd_load(name);
    }
    printf("Failed to locate herd descriptor '%s'!\n", name);
    assert(0);
  }
  _air_host_active_herd.herd_desc = herd_desc;

  return 0;
}

uint64_t air_wait_all(std::vector<uint64_t> &signals) {
  return 0;
  hsa_queue_t *q = _air_host_active_segment.q;
  if (!q) {
    printf("WARNING: no queue provided, air_wait_all will return without "
           "waiting\n");
    return 0;
  }

  // Containing all of the barrier packets. This is needed because
  // we might have to wait on more than 5 signals.
  std::vector<hsa_barrier_and_packet_t> packets;

  // iterate over the signals in chunks of 5
  while (signals.size()) {
    if (signals.size() < 5)
      signals.resize(5, 0);

    // Vector which contains the handles of the signals that we are going to
    // wait on
    std::vector<hsa_signal_t> signals_in_pkt;
    bool non_zero = false;
    for (auto s : signals) {
      if (s) {
        // Push back the proper signal
        signals_in_pkt.push_back(*reinterpret_cast<hsa_signal_t *>(s));
        non_zero = true;
      } else {
        // Create a dummy signal that will have a handle of 0
        hsa_signal_t dummy_signal;
        // hsa_amd_signal_create_on_agent(
        //     0, 0, nullptr, _air_host_active_segment.agent, 0, &dummy_signal);
        dummy_signal.handle =
            0; // The barrier and packet will ignore a signal with handle of 0
        signals_in_pkt.push_back(dummy_signal);
      }
    }
    if (non_zero) {

      // Submit a barrier packet for 5 signals that we are waiting on
      uint64_t wr_idx =
          hsa_queue_add_write_index_relaxed(_air_host_active_segment.q, 1);
      uint64_t packet_id = wr_idx % _air_host_active_segment.q->size;
      hsa_barrier_and_packet_t barrier_pkt;
      air_packet_barrier_and(&barrier_pkt, signals_in_pkt[0], signals_in_pkt[1],
                             signals_in_pkt[2], signals_in_pkt[3],
                             signals_in_pkt[4]);
      // hsa_amd_signal_create_on_agent(1, 0, nullptr,
      //                                _air_host_active_segment.agent, 0,
      //                                &barrier_pkt.completion_signal);
      air_queue_dispatch(_air_host_active_segment.q, packet_id, wr_idx,
                         &barrier_pkt);

      // Put it in a vector of barrier packets so we can wait on all of them
      // after they are submitted
      packets.push_back(barrier_pkt);
    }

    // Remove the 5 signals from our vector and keep going if we have more
    signals.resize(signals.size() - 5);
  }

  // Submit each packet and delete the completion signal
  for (auto p : packets) {
    air_queue_wait(q, &p);
    hsa_signal_destroy(p.completion_signal);
  }

  return 0;
}

extern "C" {

uint64_t _mlir_ciface___airrt_herd_load(const char *name) {
  return air_herd_load(name);
}

uint64_t _mlir_ciface___airrt_segment_load(const char *name) {
  return air_segment_load(name);
}

void _mlir_ciface___airrt_wait_all_0_0() { return; }
void _mlir_ciface___airrt_wait_all_0_1(uint64_t e0) {
  std::vector<uint64_t> events{e0, 0, 0, 0, 0};
  air_wait_all(events);
  return;
}
void _mlir_ciface___airrt_wait_all_0_2(uint64_t e0, uint64_t e1) {
  std::vector<uint64_t> events{e0, e1, 0, 0, 0};
  air_wait_all(events);
  return;
}
void _mlir_ciface___airrt_wait_all_0_3(uint64_t e0, uint64_t e1, uint64_t e2) {
  std::vector<uint64_t> events{e0, e1, e2, 0, 0};
  air_wait_all(events);
  return;
}

uint64_t _mlir_ciface___airrt_wait_all_1_0() {
  std::vector<uint64_t> events{};
  return air_wait_all(events);
}
uint64_t _mlir_ciface___airrt_wait_all_1_1(uint64_t e0) {
  std::vector<uint64_t> events{e0, 0, 0, 0, 0};
  return air_wait_all(events);
}
uint64_t _mlir_ciface___airrt_wait_all_1_2(uint64_t e0, uint64_t e1) {
  std::vector<uint64_t> events{e0, e1, 0, 0, 0};
  return air_wait_all(events);
}
uint64_t _mlir_ciface___airrt_wait_all_1_3(uint64_t e0, uint64_t e1,
                                           uint64_t e2) {
  std::vector<uint64_t> events{e0, e1, e2, 0, 0};
  return air_wait_all(events);
}

} // extern C
