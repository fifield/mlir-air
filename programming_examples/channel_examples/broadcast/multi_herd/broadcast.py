# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

INOUT_DATATYPE = np.int32

OUTPUT_HERD_NAMES = ["ChanOutB", "ChanOutC", "ChanOutD"]


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, xrt_dtype)

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    image_type_l1 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    Channel("ChanIn", size=[1, 1], broadcast_shape=[3, 1])
    for name in OUTPUT_HERD_NAMES:
        Channel(name)

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut, memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1, arg2, arg3):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1, arg2, arg3])
        def launch_body(a, b, c, d):

            ChannelPut("ChanIn", a)
            ChannelGet(OUTPUT_HERD_NAMES[0], b)
            ChannelGet(OUTPUT_HERD_NAMES[1], c)
            ChannelGet(OUTPUT_HERD_NAMES[2], d)

            @segment(name="seg")
            def segment_body():

                for herd_num in range(3):

                    @herd(name="broadcastherd" + str(herd_num), sizes=[1, 1])
                    def herd_body(_tx, _ty, _sx, _sy):

                        # We must allocate a buffer of image size for the input/output
                        image_in = AllocOp(image_type_l1, [], [])
                        image_out = AllocOp(image_type_l1, [], [])

                        ChannelGet("ChanIn", image_in, indices=[herd_num, 0])

                        # Access every value in the image
                        for i in range_(IMAGE_HEIGHT):
                            for j in range_(IMAGE_WIDTH):
                                # Load the input value
                                val_in = load(image_in, [i, j])

                                # Calculate the output value
                                val_out = arith.addi(
                                    val_in, arith.ConstantOp(T.i32(), herd_num + 1)
                                )

                                # Store the output value
                                store(val_out, image_out, [i, j])
                                yield_([])
                            yield_([])

                        ChannelPut(OUTPUT_HERD_NAMES[herd_num], image_out)

                        DeallocOp(image_in)
                        DeallocOp(image_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the channel broadcast multi herd example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(np.prod(IMAGE_SIZE), dtype=INOUT_DATATYPE).reshape(IMAGE_SIZE)
    output_b = np.arange(1, np.prod(IMAGE_SIZE) + 1, dtype=INOUT_DATATYPE).reshape(
        IMAGE_SIZE
    )
    output_c = np.arange(2, np.prod(IMAGE_SIZE) + 2, dtype=INOUT_DATATYPE).reshape(
        IMAGE_SIZE
    )
    output_d = np.arange(3, np.prod(IMAGE_SIZE) + 3, dtype=INOUT_DATATYPE).reshape(
        IMAGE_SIZE
    )

    runner = XRTRunner(verbose=args.verbose)
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_a],
            expected_outputs=[output_b, output_c, output_d],
        )
    )
