
// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie

module {
  %t2_0 = AIE.tile(2, 0)
  %t2_1 = AIE.tile(2, 1)
  %t3_0 = AIE.tile(3, 0)
  %t3_1 = AIE.tile(3, 1)
  %t6_0 = AIE.tile(6, 0)
  %t6_1 = AIE.tile(6, 1)
  %t7_0 = AIE.tile(7, 0)
  %t7_1 = AIE.tile(7, 1)
  %t10_0 = AIE.tile(10, 0)
  %t10_1 = AIE.tile(10, 1)
  %t11_0 = AIE.tile(11, 0)
  %t11_1 = AIE.tile(11, 1)
  %t18_0 = AIE.tile(18, 0)
  %t18_1 = AIE.tile(18, 1)
  %t19_0 = AIE.tile(19, 0)
  %t19_1 = AIE.tile(19, 1)

  %t8_3 = AIE.tile(8,3)
  %t8_4 = AIE.tile(8,4)
  %t8_5 = AIE.tile(8,5)
  %t8_6 = AIE.tile(8,6)
  %t9_3 = AIE.tile(9,3)
  %t9_4 = AIE.tile(9,4)
  %t9_5 = AIE.tile(9,5)
  %t9_6 = AIE.tile(9,6)
  %t10_3 = AIE.tile(10,3)
  %t10_4 = AIE.tile(10,4)
  %t10_5 = AIE.tile(10,5)
  %t10_6 = AIE.tile(10,6)
  %t11_3 = AIE.tile(11,3)
  %t11_4 = AIE.tile(11,4)
  %t11_5 = AIE.tile(11,5)
  %t11_6 = AIE.tile(11,6)

  // Per shim DMA routing
  %sw2_0 = AIE.switchbox(%t2_0) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux2_0 = AIE.shimmux(%t2_0) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }
  AIE.flow(%t2_1, "South" : 0, %t8_3, "DMA" : 0)
  AIE.flow(%t2_1, "South" : 1, %t8_4, "DMA" : 0)
  AIE.flow(%t8_3, "DMA" : 0, %t2_1, "South" : 0)
  AIE.flow(%t8_4, "DMA" : 0, %t2_1, "South" : 1)


  %sw3_0 = AIE.switchbox(%t3_0) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux3_0 = AIE.shimmux(%t3_0) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }
  AIE.flow(%t3_1, "South" : 0, %t8_5, "DMA" : 0)
  AIE.flow(%t3_1, "South" : 1, %t8_6, "DMA" : 0)
  AIE.flow(%t8_5, "DMA" : 0, %t3_1, "South" : 0)
  AIE.flow(%t8_6, "DMA" : 0, %t3_1, "South" : 1)


  %sw6_0 = AIE.switchbox(%t6_0) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux6_0 = AIE.shimmux(%t6_0) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }
  AIE.flow(%t6_1, "South" : 0, %t9_3, "DMA" : 0)
  AIE.flow(%t6_1, "South" : 1, %t9_4, "DMA" : 0)
  AIE.flow(%t9_3, "DMA" : 0, %t6_1, "South" : 0)
  AIE.flow(%t9_4, "DMA" : 0, %t6_1, "South" : 1)


  %sw7_0 = AIE.switchbox(%t7_0) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux7_0 = AIE.shimmux(%t7_0) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }
  AIE.flow(%t7_1, "South" : 0, %t9_5, "DMA" : 0)
  AIE.flow(%t7_1, "South" : 1, %t9_6, "DMA" : 0)
  AIE.flow(%t9_5, "DMA" : 0, %t7_1, "South" : 0)
  AIE.flow(%t9_6, "DMA" : 0, %t7_1, "South" : 1)

  %sw10_0 = AIE.switchbox(%t10_0) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux10_0 = AIE.shimmux(%t10_0) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }
  AIE.flow(%t10_1, "South" : 0, %t10_3, "DMA" : 0)
  AIE.flow(%t10_1, "South" : 1, %t10_4, "DMA" : 0)
  AIE.flow(%t10_3, "DMA" : 0, %t10_1, "South" : 0)
  AIE.flow(%t10_4, "DMA" : 0, %t10_1, "South" : 1)

  %sw11_0 = AIE.switchbox(%t11_0) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux11_0 = AIE.shimmux(%t11_0) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }
  AIE.flow(%t11_1, "South" : 0, %t10_5, "DMA" : 0)
  AIE.flow(%t11_1, "South" : 1, %t10_6, "DMA" : 0)
  AIE.flow(%t10_5, "DMA" : 0, %t11_1, "South" : 0)
  AIE.flow(%t10_6, "DMA" : 0, %t11_1, "South" : 1)

  %sw18_0 = AIE.switchbox(%t18_0) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux18_0 = AIE.shimmux(%t18_0) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }
  AIE.flow(%t18_1, "South" : 0, %t11_3, "DMA" : 0)
  AIE.flow(%t18_1, "South" : 1, %t11_4, "DMA" : 0)
  AIE.flow(%t11_3, "DMA" : 0, %t18_1, "South" : 0)
  AIE.flow(%t11_4, "DMA" : 0, %t18_1, "South" : 1)

  %sw19_0 = AIE.switchbox(%t19_0) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux19_0 = AIE.shimmux(%t19_0) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }
  AIE.flow(%t19_1, "South" : 0, %t11_5, "DMA" : 0)
  AIE.flow(%t19_1, "South" : 1, %t11_6, "DMA" : 0)
  AIE.flow(%t11_5, "DMA" : 0, %t19_1, "South" : 0)
  AIE.flow(%t11_6, "DMA" : 0, %t19_1, "South" : 1)

  // etc ...


  %buf8_3_0 = AIE.buffer(%t8_3) : memref<16xi32>
  %buf8_3_1 = AIE.buffer(%t8_3) : memref<16xi32>
  %buf8_4_0 = AIE.buffer(%t8_4) : memref<16xi32>
  %buf8_4_1 = AIE.buffer(%t8_4) : memref<16xi32>

  %buf8_5_0 = AIE.buffer(%t8_5) : memref<16xi32>
  %buf8_5_1 = AIE.buffer(%t8_5) : memref<16xi32>
  %buf8_6_0 = AIE.buffer(%t8_6) : memref<16xi32>
  %buf8_6_1 = AIE.buffer(%t8_6) : memref<16xi32>

  %buf9_3_0 = AIE.buffer(%t9_3) : memref<16xi32>
  %buf9_3_1 = AIE.buffer(%t9_3) : memref<16xi32>
  %buf9_4_0 = AIE.buffer(%t9_4) : memref<16xi32>
  %buf9_4_1 = AIE.buffer(%t9_4) : memref<16xi32>

  %buf9_5_0 = AIE.buffer(%t9_5) : memref<16xi32>
  %buf9_5_1 = AIE.buffer(%t9_5) : memref<16xi32>
  %buf9_6_0 = AIE.buffer(%t9_6) : memref<16xi32>
  %buf9_6_1 = AIE.buffer(%t9_6) : memref<16xi32>

  %buf10_3_0 = AIE.buffer(%t10_3) : memref<16xi32>
  %buf10_3_1 = AIE.buffer(%t10_3) : memref<16xi32>
  %buf10_4_0 = AIE.buffer(%t10_4) : memref<16xi32>
  %buf10_4_1 = AIE.buffer(%t10_4) : memref<16xi32>

  %buf10_5_0 = AIE.buffer(%t10_5) : memref<16xi32>
  %buf10_5_1 = AIE.buffer(%t10_5) : memref<16xi32>
  %buf10_6_0 = AIE.buffer(%t10_6) : memref<16xi32>
  %buf10_6_1 = AIE.buffer(%t10_6) : memref<16xi32>

  %buf11_3_0 = AIE.buffer(%t11_3) : memref<16xi32>
  %buf11_3_1 = AIE.buffer(%t11_3) : memref<16xi32>
  %buf11_4_0 = AIE.buffer(%t11_4) : memref<16xi32>
  %buf11_4_1 = AIE.buffer(%t11_4) : memref<16xi32>

  %buf11_5_0 = AIE.buffer(%t11_5) : memref<16xi32>
  %buf11_5_1 = AIE.buffer(%t11_5) : memref<16xi32>
  %buf11_6_0 = AIE.buffer(%t11_6) : memref<16xi32>
  %buf11_6_1 = AIE.buffer(%t11_6) : memref<16xi32>

  // ...
  %l8_3_0 = AIE.lock(%t8_3, 0)
  %l8_3_1 = AIE.lock(%t8_3, 1)
  %l8_4_0 = AIE.lock(%t8_4, 0)
  %l8_4_1 = AIE.lock(%t8_4, 1)
  %l8_5_0 = AIE.lock(%t8_5, 0)
  %l8_5_1 = AIE.lock(%t8_5, 1)
  %l8_6_0 = AIE.lock(%t8_6, 0)
  %l8_6_1 = AIE.lock(%t8_6, 1)
  %l9_3_0 = AIE.lock(%t9_3, 0)
  %l9_3_1 = AIE.lock(%t9_3, 1)
  %l9_4_0 = AIE.lock(%t9_4, 0)
  %l9_4_1 = AIE.lock(%t9_4, 1)
  %l9_5_0 = AIE.lock(%t9_5, 0)
  %l9_5_1 = AIE.lock(%t9_5, 1)
  %l9_6_0 = AIE.lock(%t9_6, 0)
  %l9_6_1 = AIE.lock(%t9_6, 1)
  %l10_3_0 = AIE.lock(%t10_3, 0)
  %l10_3_1 = AIE.lock(%t10_3, 1)
  %l10_4_0 = AIE.lock(%t10_4, 0)
  %l10_4_1 = AIE.lock(%t10_4, 1)
  %l10_5_0 = AIE.lock(%t10_5, 0)
  %l10_5_1 = AIE.lock(%t10_5, 1)
  %l10_6_0 = AIE.lock(%t10_6, 0)
  %l10_6_1 = AIE.lock(%t10_6, 1)
  %l11_3_0 = AIE.lock(%t11_3, 0)
  %l11_3_1 = AIE.lock(%t11_3, 1)
  %l11_4_0 = AIE.lock(%t11_4, 0)
  %l11_4_1 = AIE.lock(%t11_4, 1)
  %l11_5_0 = AIE.lock(%t11_5, 0)
  %l11_5_1 = AIE.lock(%t11_5, 1)
  %l11_6_0 = AIE.lock(%t11_6, 0)
  %l11_6_1 = AIE.lock(%t11_6, 1)

  // ...



  %m8_3 = AIE.mem(%t8_3) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l8_3_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf8_3_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_3_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l8_3_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf8_3_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_3_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l8_3_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf8_3_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_3_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l8_3_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf8_3_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_3_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m8_4 = AIE.mem(%t8_4) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l8_4_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf8_4_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_4_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l8_4_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf8_4_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_4_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l8_4_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf8_4_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_4_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l8_4_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf8_4_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_4_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m8_5 = AIE.mem(%t8_5) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l8_5_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf8_5_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_5_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l8_5_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf8_5_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_5_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l8_5_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf8_5_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_5_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l8_5_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf8_5_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_5_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m8_6 = AIE.mem(%t8_6) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l8_6_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf8_6_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_6_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l8_6_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf8_6_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_6_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l8_6_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf8_6_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_6_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l8_6_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf8_6_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l8_6_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m9_3 = AIE.mem(%t9_3) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l9_3_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf9_3_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_3_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l9_3_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf9_3_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_3_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l9_3_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf9_3_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_3_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l9_3_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf9_3_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_3_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m9_4 = AIE.mem(%t9_4) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l9_4_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf9_4_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_4_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l9_4_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf9_4_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_4_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l9_4_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf9_4_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_4_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l9_4_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf9_4_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_4_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }


  %m9_5 = AIE.mem(%t9_5) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l9_5_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf9_5_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_5_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l9_5_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf9_5_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_5_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l9_5_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf9_5_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_5_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l9_5_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf9_5_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_5_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m9_6 = AIE.mem(%t9_6) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l9_6_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf9_6_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_6_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l9_6_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf9_6_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_6_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l9_6_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf9_6_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_6_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l9_6_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf9_6_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l9_6_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m10_3 = AIE.mem(%t10_3) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l10_3_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf10_3_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_3_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l10_3_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf10_3_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_3_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l10_3_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf10_3_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_3_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l10_3_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf10_3_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_3_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m10_4 = AIE.mem(%t10_4) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l10_4_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf10_4_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_4_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l10_4_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf10_4_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_4_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l10_4_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf10_4_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_4_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l10_4_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf10_4_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_4_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m10_5 = AIE.mem(%t10_5) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l10_5_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf10_5_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_5_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l10_5_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf10_5_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_5_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l10_5_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf10_5_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_5_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l10_5_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf10_5_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_5_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m10_6 = AIE.mem(%t10_6) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l10_6_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf10_6_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_6_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l10_6_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf10_6_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_6_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l10_6_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf10_6_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_6_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l10_6_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf10_6_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l10_6_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m11_3 = AIE.mem(%t11_3) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l11_3_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf11_3_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_3_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l11_3_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf11_3_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_3_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l11_3_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf11_3_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_3_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l11_3_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf11_3_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_3_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m11_4 = AIE.mem(%t11_4) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l11_4_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf11_4_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_4_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l11_4_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf11_4_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_4_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l11_4_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf11_4_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_4_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l11_4_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf11_4_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_4_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m11_5 = AIE.mem(%t11_5) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l11_5_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf11_5_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_5_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l11_5_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf11_5_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_5_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l11_5_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf11_5_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_5_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l11_5_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf11_5_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_5_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m11_6 = AIE.mem(%t11_6) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l11_6_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf11_6_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_6_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l11_6_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf11_6_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_6_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l11_6_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf11_6_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_6_0, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l11_6_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf11_6_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l11_6_1, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }



}