# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.dialects.air import module_builder, herd, launch, segment
from air.dialects.air import ChannelOp, ChannelPut, ChannelGet, WaitAllOp
from air.dialects.air import MemorySpace, AsyncTokenType
from air.dialects.air import arith, T, F32Type, BF16Type
from air.dialects.func import FuncOp
from air.dialects.linalg import elemwise_binary
from air.dialects.linalg.opdsl.lang import BinaryFn, TypeFn
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
from air.ir import MemRefType, IntegerAttr

import numpy as np
from ml_dtypes import bfloat16


def to_type(dtype):
    if dtype == np.int32:
        return T.i32()
    if dtype == np.int16:
        return T.i16()
    if dtype == np.float32:
        return F32Type.get()
    if dtype == bfloat16:
        return BF16Type.get()
    return None


@module_builder
def build_module(shape, dtype, tile_size):
    memrefTyIn = MemRefType.get(shape, to_type(dtype))
    memrefTyOut = MemRefType.get(shape, to_type(dtype))
    tokenTy = AsyncTokenType.get()

    ChannelOp("ChanA")
    ChannelOp("ChanB")
    ChannelOp("ChanC")

    # The air_eltwise_add() function is defined here
    @FuncOp.from_py_func(memrefTyIn, memrefTyIn, memrefTyOut)
    def air_eltwise_add(arg0, arg1, arg2):

        @launch(operands=[arg0, arg1, arg2])
        def launch_body(a, b, c):
            ChannelPut("ChanA", a, async_token=tokenTy)
            ChannelPut("ChanB", b, async_token=tokenTy)

            @segment(name="segment_0", async_token=tokenTy)
            def segment_body():

                @herd(name="herd_0", sizes=[1, 1], async_token=tokenTy)
                def herd_body(x, y, sx, sy):
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                    itile_type = MemRefType.get(
                        shape=tile_size,
                        element_type=to_type(dtype),
                        memory_space=mem_space,
                    )
                    otile_type = MemRefType.get(
                        shape=tile_size,
                        element_type=to_type(dtype),
                        memory_space=mem_space,
                    )
                    count = arith.ConstantOp.create_index(shape[0] // tile_size[0])
                    for _ in for_(count):
                        tile_a = AllocOp(itile_type, [], [])
                        tile_b = AllocOp(itile_type, [], [])
                        tile_c = AllocOp(otile_type, [], [])
                        token_a = ChannelGet("ChanA", tile_a, async_token=tokenTy)
                        token_b = ChannelGet("ChanB", tile_b, async_token=tokenTy)
                        WaitAllOp(None, [token_a, token_b])
                        elemwise_binary(
                            tile_a,
                            tile_b,
                            outs=[tile_c],
                            fun=BinaryFn.add,
                            cast=TypeFn.cast_unsigned,
                        )
                        ChannelPut("ChanC", tile_c)
                        DeallocOp(tile_a)
                        DeallocOp(tile_b)
                        DeallocOp(tile_c)
                        yield_([])

                WaitAllOp(None, [herd_body])

            token_c = ChannelGet(
                "ChanC", c, async_dependencies=[segment_body], async_token=tokenTy
            )
            WaitAllOp(None, [token_c])


mlir_module = build_module(shape=[1024], dtype=np.int32, tile_size=[32])

print(mlir_module)
# # write to file
# with open("air.mlir", "w") as f:
#     f.write(str(mlir_module))

from air.backend import rocr
import air._mlir_libs._airRt as airrt

backend = rocr.HSABackend()
air_eltwise_add = backend.compile_and_load(mlir_module, "air_eltwise_add")

input_buffer0 = airrt.host.allocate_memory(1024)
input_buffer1 = airrt.host.allocate_memory(1024)
output_buffer = airrt.host.allocate_memory(1024)

input_buffer0.fill(2)
input_buffer1.fill(21)
output_buffer.fill(123)

air_eltwise_add(input_buffer0, input_buffer1, output_buffer)

print(input_buffer0, input_buffer1, output_buffer)

ref = input_buffer0.view(np.int32) + input_buffer1.view(np.int32)
output = output_buffer.view(np.int32)

errs = 0
for i in range(1024 // 4):
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
