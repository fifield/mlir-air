//===- AirHostModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air.hpp"

#include "hsa/hsa.h"
#include <hsa/hsa_ext_amd.h>

#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <sstream>
#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;

namespace {
void defineAIRHostModule(nb::module_ &m) {

  // HSA helper classes

  // hsa_agent_t
  nb::class_<hsa_agent_t> Agent(m, "Agent");
  Agent.def("__str__", [](const hsa_agent_t &self) -> std::string {
    std::stringstream ss;
    ss << "<hsa_agent_t at 0x" << std::hex << reinterpret_cast<uint64_t>(&self)
       << ">";
    return ss.str();
  });
  Agent.def("type", [](const hsa_agent_t &self) -> hsa_device_type_t {
    hsa_device_type_t type;
    hsa_agent_get_info(self, HSA_AGENT_INFO_DEVICE, &type);
    return type;
  });

  // hsa_amd_memory_pool_t
  nb::class_<hsa_amd_memory_pool_t> MemoryPool(m, "AmdMemoryPool");

  // hsa_device_type_t
  nb::enum_<hsa_device_type_t> HsaDeviceType(m, "DeviceType");
  HsaDeviceType.value("HSA_DEVICE_TYPE_CPU", HSA_DEVICE_TYPE_CPU)
      .value("HSA_DEVICE_TYPE_GPU", HSA_DEVICE_TYPE_GPU)
      .value("HSA_DEVICE_TYPE_DSP", HSA_DEVICE_TYPE_DSP)
      .value("HSA_DEVICE_TYPE_AIE", HSA_DEVICE_TYPE_AIE);

  nb::class_<hsa_queue_t> Queue(m, "Queue");

  // AIR host functions
  m.def("allocate_memory", [](size_t size) -> nb::ndarray<nb::numpy, uint8_t> {
    void *mem = air_malloc(size);
    std::cout << "Allocated memory at " << mem << "\n";
    nb::capsule capsule(mem, [](void *mem) noexcept {
      std::cout << "Deallocating memory at " << mem << "\n";
      air_free(mem);
    });
    return nb::ndarray<nb::numpy, uint8_t>(mem, {size}, capsule);
  });

  m.def("run",
        [](const std::string &pdi_file, const std::string &insts_file,
           std::vector<nb::ndarray<uint8_t>> &args) -> uint64_t {
          std::cout << "Running " << insts_file << " with " << pdi_file << "\n"; 
          for (auto &arg : args) {
            std::cout << "arg size: " << arg.size() << "\n";
          }
          std::vector<void *> arg_ptrs;
          for (auto &arg : args) {
            arg_ptrs.push_back(arg.data());
          }
          return run_kernel(pdi_file, insts_file, arg_ptrs);
        });
  m.def("dispatch",
        [](const std::string &insts_file,
           std::vector<pybind11::array_t<uint8_t>> &args) -> uint64_t {
          std::cout << "Running " << insts_file << "\n";
          for (auto &arg : args) {
            std::cout << "arg size: " << arg.size() << "\n";
          }
          std::vector<void *> arg_ptrs;
          for (auto &arg : args) {
            arg_ptrs.push_back(arg.request().ptr);
          }
          return dispatch_sequence(insts_file, arg_ptrs);
        });

  m.def(
      "init_libxaie", []() -> uint64_t { return (uint64_t)air_init_libxaie(); },
      nb::rv_policy::reference);

  m.def("deinit_libxaie", [](uint64_t ctx) -> void {
    air_deinit_libxaie((air_libxaie_ctx_t)ctx);
  });

  m.def("init", []() -> uint64_t { return (uint64_t)air_init(); });

  m.def("shut_down", []() -> uint64_t { return (uint64_t)air_shut_down(); });

  nb::class_<air_module_desc_t>(m, "ModuleDescriptor")
      .def(
          "getSegments",
          [](const air_module_desc_t &d) -> std::vector<air_segment_desc_t *> {
            std::vector<air_segment_desc_t *> segments;
            for (uint64_t i = 0; i < d.segment_length; i++)
              segments.push_back(d.segment_descs[i]);
            return segments;
          },
          nb::rv_policy::reference);

  nb::class_<air_segment_desc_t>(m, "SegmentDescriptor")
      .def(
          "getHerds",
          [](const air_segment_desc_t &d) -> std::vector<air_herd_desc_t *> {
            std::vector<air_herd_desc_t *> herds;
            for (uint64_t i = 0; i < d.herd_length; i++)
              herds.push_back(d.herd_descs[i]);
            return herds;
          },
          nb::rv_policy::reference)
      .def("getName", [](const air_segment_desc_t &d) -> std::string {
        return {d.name, static_cast<size_t>(d.name_length)};
      });

  nb::class_<air_herd_desc_t>(m, "HerdDescriptor")
      .def("getName", [](const air_herd_desc_t &d) -> std::string {
        return {d.name, static_cast<size_t>(d.name_length)};
      });

  m.def("module_load_from_file",
        [](const std::string &filename) -> air_module_handle_t {
          return air_module_load_from_file(filename.c_str());
        });

  m.def("module_unload", &air_module_unload);

  m.def("get_module_descriptor", &air_module_get_desc,
        nb::rv_policy::reference);

  // nb::class_<hsa_agent_t> Agent(m, "Agent");

  m.def(
      "get_agents",
      []() -> std::vector<hsa_agent_t> {
        std::vector<hsa_agent_t> agents;
        hsa_iterate_agents(
            [](hsa_agent_t agent, void *data) {
              static_cast<std::vector<hsa_agent_t> *>(data)->push_back(agent);
              return HSA_STATUS_SUCCESS;
            },
            (void *)&agents);
        return agents;
      },
      nb::rv_policy::reference);

  // nb::class_<hsa_queue_t> Queue(m, "Queue");

  m.def(
      "queue_create",
      [](const hsa_agent_t &a) -> hsa_queue_t * {
        hsa_queue_t *q = nullptr;
        uint32_t aie_max_queue_size(0);

        // Query the queue size the agent supports
        auto queue_size_ret = hsa_agent_get_info(
            a, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &aie_max_queue_size);
        if (queue_size_ret != HSA_STATUS_SUCCESS)
          return nullptr;

        // Creating the queue
        auto queue_create_ret =
            hsa_queue_create(a, aie_max_queue_size, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, 0, 0, &q);

        if (queue_create_ret != 0)
          return nullptr;
        return q;
      },
      nb::rv_policy::reference);

  m.def(
      "read32", [](uint64_t addr) -> uint32_t { return air_read32(addr); },
      nb::rv_policy::copy);

  m.def("write32", [](uint64_t addr, uint32_t val) -> void {
    return air_write32(addr, val);
  });

}

} // namespace

NB_MODULE(_airRt, m) {
  m.doc() = R"pbdoc(
        AIR Runtime Python bindings
        --------------------------

        .. currentmodule:: _airRt

        .. autosummary::
           :toctree: _generate

    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  auto airhost = m.def_submodule("host", "libairhost bindings");
  defineAIRHostModule(airhost);
}
