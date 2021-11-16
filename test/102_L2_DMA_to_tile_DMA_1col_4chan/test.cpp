// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cstdio>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "air_host.h"
#include "test_library.h"

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie_inc.cpp"
#undef TileInst
#undef TileDMAInst

}

#define L2_DMA_BASE 0x020240000000LL
#define SHMEM_BASE  0x020100000000LL

struct dma_cmd_t {
  uint8_t select;
  uint16_t length;
  uint16_t uram_addr;
  uint8_t id;
};

struct dma_rsp_t {
	uint8_t id;
};

int main(int argc, char *argv[])
{

  xaie = air_init_libxaie1();
  
  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  for (int i=0; i<16; i++) {
    mlir_write_buffer_buf1(i, 0x9a01);
    mlir_write_buffer_buf2(i, 0xba01);
    mlir_write_buffer_buf4(i, 0x9a02);
    mlir_write_buffer_buf5(i, 0xba02);
    mlir_write_buffer_buf7(i, 0x9401);
    mlir_write_buffer_buf8(i, 0xb401);
    mlir_write_buffer_buf10(i, 0x9402);
    mlir_write_buffer_buf11(i, 0xb402);
  }

  ACDC_print_dma_status(xaie->TileInst[7][1]);
  ACDC_print_dma_status(xaie->TileInst[7][2]);
  ACDC_print_dma_status(xaie->TileInst[7][3]);
  ACDC_print_dma_status(xaie->TileInst[7][4]);

  XAieGbl_Write32(xaie->TileInst[7][0].TileAddr + 0x00033008, 0xFF);

  uint32_t reg = XAieGbl_Read32(xaie->TileInst[7][0].TileAddr + 0x00033004);
  printf("REG %x\n", reg);
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE);
  uint32_t *bank1_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE+0x20000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 2 for the upper memory as it goes in
  for (int i=0;i<32;i++) {
    uint32_t upper_lower = (i%8)/4;
    uint32_t first128_second128 = i%2;
    uint32_t first64_second64 = (i%16)/8;
    uint32_t first32_second32 = (i/2)%2;
    uint32_t offset = (first128_second128)*4;
    offset += (first64_second64)*2;
    offset += first32_second32;
    offset += (i/16)*8;
    uint32_t toWrite = i + (((upper_lower)+1) << 28);

    printf("%d : %d %d %d %d %d %08X\n",i,upper_lower, first128_second128, first64_second64, first32_second32, offset, toWrite);
    if (upper_lower)
      bank1_ptr[offset] = toWrite;
    else
      bank0_ptr[offset] = toWrite;
  }

  // Read back the values from above 
  for (int i=0;i<16;i++) {
    uint32_t word0 = bank0_ptr[i];
    uint32_t word1 = bank1_ptr[i];

    printf("%x %08X %08X\r\n", i, word0, word1);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(pkt);
  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  //
  // Set up a 1x4 herd starting 7,1
  //

  pkt->arg[0]  = AIR_PKT_TYPE_HERD_INITIALIZE;
  pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  pkt->arg[0] |= (1L << 40);
  pkt->arg[0] |= (7L << 32);
  pkt->arg[0] |= (4L << 24);
  pkt->arg[0] |= (1L << 16);
  
  pkt->arg[1] = 0;  // Herd ID 0
  pkt->arg[2] = 0;
  pkt->arg[3] = 0;

  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);

  // globally bypass headers
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(pkt);
  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
  pkt->arg[0] = AIR_PKT_TYPE_PUT_STREAM;

  static dma_cmd_t cmd;
  cmd.select = 7;
  cmd.length = 0;
  cmd.uram_addr = 1;
  cmd.id = 0;

  uint64_t stream = 0;
  pkt->arg[1] = stream;
  pkt->arg[2] = 0;
  pkt->arg[2] |= ((uint64_t)cmd.select) << 32; // 3 bits
  pkt->arg[2] |= cmd.length << 18; // 14 bits
  pkt->arg[2] |= cmd.uram_addr << 5; // 13 bits
  pkt->arg[2] |= cmd.id; // 5 bits

  signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
  signal_store_release((signal_t*)&q->doorbell, wr_idx);

  //
  // send the data
  //

  for (int sel=0; sel<4; sel++) {

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;

    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    initialize_packet(pkt);
    pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
    pkt->arg[0] = AIR_PKT_TYPE_PUT_STREAM;

    cmd.select = sel;
    cmd.length = (sel>1) ? 8 : 4;
    cmd.uram_addr = 0;
    cmd.id = sel+1;

    pkt->arg[1] = stream;
    pkt->arg[2] = 0;
    pkt->arg[2] |= ((uint64_t)cmd.select) << 32;
    pkt->arg[2] |= cmd.length << 18;
    pkt->arg[2] |= cmd.uram_addr << 5;
    pkt->arg[2] |= cmd.id;

    signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
    if (sel == 3) {
      signal_store_release((signal_t*)&q->doorbell, wr_idx);
      while (signal_wait_aquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
        printf("packet completion signal timeout!\n");
        printf("%x\n", pkt->header);
        printf("%x\n", pkt->type);
        printf("%lx\n", pkt->completion_signal);
        break;
      }
    }
  }

  ACDC_print_dma_status(xaie->TileInst[7][1]);
  ACDC_print_dma_status(xaie->TileInst[7][2]);
  ACDC_print_dma_status(xaie->TileInst[7][3]);
  ACDC_print_dma_status(xaie->TileInst[7][4]);
  
  uint32_t errs = 0;
  for (int i=0; i<16; i++) {
    uint32_t check = (i % 4) + ((i / 4) * 8);
    uint32_t d;
    d = mlir_read_buffer_buf1(i);
    if ((d & 0x0fffffff) != check) {
      printf("Part 0 : Word %i : Expect %x, got %08X\n", i, check, d);
      errs++;
    }
  }
  for (int i=0; i<16; i++) {
    uint32_t check = (i % 4) + ((i / 4) * 8) + 4;
    uint32_t d;
    d = mlir_read_buffer_buf4(i);
    if ((d & 0x0fffffff) != check) {
      printf("Part 1 : Word %i : Expect %x, got %08X\n", i, check, d);
      errs++;
    }
  }
  for (int i=0; i<32; i++) {
    uint32_t d;
    if (i<16)
      d = mlir_read_buffer_buf7(i);
    else 
      d = mlir_read_buffer_buf8(i-16);
    if ((d & 0x0fffffff) != (i)) {
      printf("Prim 0 : Word %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }
  for (int i=0; i<32; i++) {
    uint32_t d;
    if (i<16)
      d = mlir_read_buffer_buf10(i);
    else 
      d = mlir_read_buffer_buf11(i-16);
    if ((d & 0x0fffffff) != (i)) {
      printf("Prim 1 : Word %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }

  if (errs) {
    printf("FAIL: %d errors\n", errs);
    return -1;
  }
  else {
    printf("PASS!\n");
    return 0;
  }
}