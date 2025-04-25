# ./python/air/backend/rocr.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import air.ir

from .abc import AirBackend, AirBackendError

import air.compiler.aircc.main as aircc
from air.compiler.aircc.configure import rocm_path, install_path

from aie.extras.runtime.refbackend import LLVMJITBackend
import aie.ir as aieir

import ctypes

# load runtime into python
ctypes.CDLL(f"{rocm_path}/../../libhsa-runtime64.so.1", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(
    f"{install_path()}/runtime_lib/x86_64/airhost/libairhost_shared.so",
    mode=ctypes.RTLD_GLOBAL,
)
import air._mlir_libs._airRt as airrt


class HSACompileArtifact:
    """A class encompassing information on the artifacts produced by compilation for the NPU/HSA"""

    def __init__(
        self,
        so,
        function_name,
        compiled_module,
    ):
        """
        Constructor for an HSACompileArtifact

        Args:
            pdi: pdi file name/path
            so: shared object file name/path
            function_name: function name
            compiled_module: compiled module
        """
        self.so_file = so
        self.function_name = function_name
        self.compiled_module = compiled_module


class HSABackend(AirBackend):
    """Main entry-point for the hsa based AIR backend."""

    def __init__(
        self,
        verbose=False,
        experimental_passes=False,
        omit_while_true_loop=False,
        omit_pingpong=False,
    ):
        """Constructor for HSABackend

        Args:
            verbose: verbose output
            experimental_passes: configure aircc to run additional experimental passes
            omit_while_true_loop: configure aircc to omit the while true loop it traditionally emits.
            omit_pingpong: configure aircc to omit the generation of ping-pong buffering.
        """
        super().__init__()
        self.verbose = verbose
        self.experimental_passes = experimental_passes
        self.omit_while_true_loop = omit_while_true_loop
        self.omit_pingpong = omit_pingpong
        self.currently_loaded = False
        self.backend = LLVMJITBackend()

    def __del__(self):
        self.unload()

    def compile(
        self,
        air_module: air.ir.Module,
        function_name,
    ):
        """Compiles an AIR module for the NPU / HSA Runtime with aircc.

        The module is expected to be AIR dialect IR. The input IR is passed directly to aircc.

        Args:
            air_module: The MLIR module consisting of funcs in the AIR dialect.
        Returns:
            An HSACompileArtifact object
        """
        if self.currently_loaded:
            raise AirBackendError(
                "Cannot use HSABackend to compile while the artifact is currently loaded. Call unload() first."
            )

        so_file = "./air.mlir.so"

        with air.ir.Context():

            if self.verbose:
                print("AIR Module:")
                print(air_module)

            aircc_options = [
                "--device",
                "npu1_4col",
                "--shared",
                "-o",
                "air.mlir.so",
                "air.mlir",
            ]

            if self.verbose:
                aircc_options = aircc_options + ["-v"]

            if self.experimental_passes:
                aircc_options += ["--experimental-passes"]

            if self.omit_while_true_loop:
                aircc_options += ["--omit-while-true-loop"]

            if self.omit_pingpong:
                aircc_options += ["--omit-ping-pong-transform"]

            aircc.run(air_module, aircc_options)

            REF_BACKEND_LOWERING_PIPELINE = (
                "builtin.module("
                + ",".join(
                    [
                        "func.func(convert-math-to-llvm)",
                        "convert-math-to-libm",
                        "expand-strided-metadata",
                        "finalize-memref-to-llvm",
                        "lower-affine",
                        "func.func(convert-arith-to-llvm)",
                        "convert-func-to-llvm",
                        "convert-cf-to-llvm",
                        "reconcile-unrealized-casts",
                    ]
                )
                + ")"
            )
            # open mlir file in new context
            with open(
                "air_project/aie_ctrl.air.mlir", "r"
            ) as f, aieir.Context(), aieir.Location.unknown():
                # parse mlir file
                module = aieir.Module.parse(f.read())
                # compile the module
                compiled_module = self.backend.compile(
                    module,
                    pipeline=REF_BACKEND_LOWERING_PIPELINE,
                    kernel_name=function_name,
                )

        return HSACompileArtifact(so_file, function_name, compiled_module)

    def load(self, artifact: HSACompileArtifact):
        """Load a compiled artifact into the air runtime.

        Args:
            artifact: The result of calling compile with HSABackend on an MLIR-AIR module.

        Returns: A callable that can be used to invoke the loaded module.
            The callable takes a list of numpy arrays. Each numpy array is
            assumed to be an input/output tensor. The callable also returns a
            list of numpy arrays, one for each tensor.
        """
        if self.currently_loaded:
            raise AirBackendError(
                "Cannot use HSABackend to compile while the artifact is currently loaded. Call unload() first."
            )

        # init runtime
        airrt.host.init()

        # register the aircc generated shared library with the runtime
        self.airrt_handle = airrt.host.module_load_from_file(artifact.so_file)

        loaded = self.backend.load(artifact.compiled_module)

        # return the function 'function_name()' of the loaded module
        f = getattr(loaded, artifact.function_name)

        def wrapped_function(*args):
            """Wrap the function"""
            try:
                with aieir.Context():
                    f(*args)
                    airrt.host.dispatch_segment(args)
            except Exception as e:
                print(f"Error in wrapped function: {e}")
                pass
            return None

        return wrapped_function

    def compile_and_load(
        self,
        air_module: air.ir.Module,
        function_name,
    ):
        """
        Compile and load a module in one step.

        Args:
            air_module: The MLIR module consisting of funcs in the AIR dialect.

        Returns: A callable that can be used to invoke the loaded module.
            The callable takes a list of numpy arrays. Each numpy array is
            assumed to be an input/output tensor. The callable also returns a
            list of numpy arrays, one for each tensor.
        """
        c = self.compile(air_module, function_name)
        return self.load(c)

    def unload(self):
        """Unload any loaded module and shutdown the air runtime."""
        # self.kernel = None
        # self.context = None
        # self.xclbin = None
        # self.device = None
        # self.bo_instr = None
        # self.instr_v = None
        self.currently_loaded = False
