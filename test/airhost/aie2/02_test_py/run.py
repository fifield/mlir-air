from air.compiler.aircc.configure import *
import ctypes
import numpy as np

# load runtime into python
ctypes.CDLL(f"{rocm_path}/../../libhsa-runtime64.so.1", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(
    f"{install_path()}/runtime_lib/x86_64/airhost/libairhost_shared.so",
    mode=ctypes.RTLD_GLOBAL,
)
import air._mlir_libs._airRt as airrt

# init runtime
airrt.host.init()

# register the aircc generated shared library with the runtime
h = airrt.host.module_load_from_file("./air.mlir.so")

input_buffer0 = airrt.host.allocate_memory(1024)
input_buffer1 = airrt.host.allocate_memory(1024)
output_buffer = airrt.host.allocate_memory(1024)

input_buffer0.fill(2)
input_buffer1.fill(21)
output_buffer.fill(123)

from aie.extras.runtime.refbackend import LLVMJITBackend
import aie.ir as aieir

backend = LLVMJITBackend()

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
    "./air_project/aie_ctrl.air.mlir", "r"
) as f, aieir.Context(), aieir.Location.unknown():
    # parse mlir file
    module = aieir.Module.parse(f.read())
    # compile the module
    compiled = backend.compile(
        module, pipeline=REF_BACKEND_LOWERING_PIPELINE, kernel_name="mul"
    )
    # load and run the compiled function
    mul = backend.load(compiled).mul
    mul(input_buffer0, input_buffer1, output_buffer)

# temporary hack: dispatch the instruction sequence of current segment,
# which was loaded during mul() above.
r = airrt.host.dispatch_segment(
    [input_buffer0, input_buffer1, output_buffer],
)

# compare the results
print(r, input_buffer0, input_buffer1, output_buffer)
ref = input_buffer0.view(np.int32) + input_buffer1.view(np.int32)
output = output_buffer.view(np.int32)

errs = 0
for i in range(64):
    if output[i] != ref[i]:
        print(f"Error at {i}: {hex(output[i])} != {hex(ref[i])}")
        errs = errs + 1
        break

if errs == 0:
    print("PASS!")
    exit(0)
else:
    print(f"Found {errs} errors")
    exit(-1)
