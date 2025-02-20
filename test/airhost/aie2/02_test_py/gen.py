# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import air.compiler.aircc.main as aircc
from air.compiler.aircc.configure import install_path
from air.dialects.air import *
from air.dialects.func import FuncOp
from air.dialects.linalg import elemwise_binary
from air.dialects.linalg.opdsl.lang import BinaryFn, TypeFn
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
from air.ir import *

import ctypes
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

    # The mul() function defined here is the function called by test.cpp
    @FuncOp.from_py_func(memrefTyIn, memrefTyIn, memrefTyOut)
    def mul(arg0, arg1, arg2):

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
# print(mlir_module)

# # write to file
# with open("air.mlir", "w") as f:
#     f.write(str(mlir_module))

aircc.run(
    mlir_module,
    [
        # "-v",
        "-xchesscc",
        "-xbridge",
        "--device",
        "npu1_4col",
        "--shared",
        "-o",
        "air.mlir.so",
        "air.mlir",  # options parser requires something here?
    ],
)
