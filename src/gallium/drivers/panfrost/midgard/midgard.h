/* Author(s):
 *   Connor Abbott
 *   Alyssa Rosenzweig
 *
 * Copyright (c) 2013 Connor Abbott (connor@abbott.cx)
 * Copyright (c) 2018 Alyssa Rosenzweig (alyssa@rosenzweig.io)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* midgard.h - definitions for the Midgard shader architecture */

typedef unsigned midgard_word_type;

typedef enum
{
	midgard_alu_vmul,
	midgard_alu_sadd,
	midgard_alu_smul,
	midgard_alu_vadd,
	midgard_alu_lut
} midgard_alu_e;

/*
 * ALU words
 */

typedef enum
{
	midgard_alu_op_fadd       = 0x10,
	midgard_alu_op_fmul       = 0x14,
	midgard_alu_op_fmin       = 0x28,
	midgard_alu_op_fmax       = 0x2C,
	midgard_alu_op_fmov       = 0x30,
	midgard_alu_op_ffloor     = 0x36,
	midgard_alu_op_fceil      = 0x37,
	midgard_alu_op_fdot3      = 0x3C,
	midgard_alu_op_fdot3r     = 0x3D,
	midgard_alu_op_fdot4      = 0x3E,
	midgard_alu_op_freduce    = 0x3F,
	midgard_alu_op_iadd       = 0x40,
	midgard_alu_op_isub       = 0x46,
	midgard_alu_op_imul       = 0x58,
	midgard_alu_op_imov       = 0x7B,
	midgard_alu_op_feq        = 0x80,
	midgard_alu_op_fne        = 0x81,
	midgard_alu_op_flt        = 0x82,
	midgard_alu_op_fle        = 0x83,
	midgard_alu_op_f2i        = 0x99,
	midgard_alu_op_ieq        = 0xA0,
	midgard_alu_op_ine        = 0xA1,
	midgard_alu_op_ilt        = 0xA4,
	midgard_alu_op_ile        = 0xA5,
	midgard_alu_op_ball       = 0xA9,
	midgard_alu_op_bany       = 0xB1,
	midgard_alu_op_i2f        = 0xB8,
	midgard_alu_op_csel       = 0xC5,
	midgard_alu_op_fatan_pt2  = 0xE8,
	midgard_alu_op_frcp       = 0xF0,
	midgard_alu_op_frsqrt     = 0xF2,
	midgard_alu_op_fsqrt      = 0xF3,
	midgard_alu_op_fexp2      = 0xF4,
	midgard_alu_op_flog2      = 0xF5,
	midgard_alu_op_fsin       = 0xF6,
	midgard_alu_op_fcos       = 0xF7,
	midgard_alu_op_fatan_pt1  = 0xF9,

	/* Not a real op, just used within the compiler to signal that a framebuffer write should be placed later */
	midgard_alu_op_synthwrite = 0xFF,
} midgard_alu_op_e;

typedef enum
{
	midgard_outmod_none = 0,
	midgard_outmod_pos  = 1,
	midgard_outmod_int  = 2,
	midgard_outmod_sat  = 3
} midgard_outmod_e;

typedef enum
{
	midgard_reg_mode_half = 1,
	midgard_reg_mode_full = 2
} midgard_reg_mode_e;

typedef enum
{
	midgard_dest_override_lower = 0,
	midgard_dest_override_upper = 1,
	midgard_dest_override_none = 2
} midgard_dest_override_e;

typedef struct
__attribute__((__packed__))
{
	bool abs         : 1;
	bool negate      : 1;
	
	/* replicate lower half if dest = half, or low/high half selection if
	 * dest = full
	 */
	bool rep_low     : 1;
	bool rep_high    : 1; /* unused if dest = full */
	bool half        : 1; /* only matters if dest = full */
	unsigned swizzle : 8;
} midgard_vector_alu_src_t;

typedef struct
__attribute__((__packed__))
{
	midgard_alu_op_e op               :  8;
	midgard_reg_mode_e reg_mode   :  2;
	unsigned src1 : 13;
	unsigned src2 : 13;
	midgard_dest_override_e dest_override : 2;
	midgard_outmod_e outmod               : 2;
	unsigned mask                           : 8;
} midgard_vector_alu_t;

typedef struct
__attribute__((__packed__))
{
	bool abs           : 1;
	bool negate        : 1;
	bool full          : 1; /* 0 = half, 1 = full */
	unsigned component : 3;
} midgard_scalar_alu_src_t;

typedef struct
__attribute__((__packed__))
{
	midgard_alu_op_e op         :  8;
	unsigned src1             :  6;
	unsigned src2             : 11;
	unsigned unknown          :  1;
	midgard_outmod_e outmod :  2;
	bool output_full          :  1;
	unsigned output_component :  3;
} midgard_scalar_alu_t;

/* ALU control words are single bit fields with a lot of space */

#define ALU_ENAB_VEC_MUL    (1 << 17)
#define ALU_ENAB_SCAL_ADD   (1 << 19)
#define ALU_ENAB_VEC_ADD    (1 << 21)
#define ALU_ENAB_SCAL_MUL   (1 << 23)
#define ALU_ENAB_LUT        (1 << 25)
#define ALU_ENAB_BR_COMPACT (1 << 26)
#define ALU_ENAB_BRANCH     (1 << 27)

/* ALU register fields are weird because of inline constants */

typedef struct
__attribute__((__packed__))
{
	unsigned input1_reg : 5;
	unsigned input2_reg : 5;
	unsigned output_reg : 5;
	unsigned inline_2   : 1;
} alu_register_word;

typedef struct
__attribute__((__packed__))
{
	unsigned src1_reg : 5;
	unsigned src2_reg : 5;
	unsigned out_reg  : 5;
	bool src2_imm     : 1;
} midgard_reg_info_t;

/*
 * Load/store words
 */

#define OP_IS_STORE(op) (\
		op == midgard_op_store_vary_16 || \
		op == midgard_op_store_vary_32 \
	)

typedef enum
{
	midgard_op_ld_st_noop   = 0x03,
	midgard_op_load_attr_16 = 0x95,
	midgard_op_load_attr_32 = 0x94,
	midgard_op_load_vary_16 = 0x99,
	midgard_op_load_vary_32 = 0x98,
	midgard_op_load_uniform_16 = 0xAC,
	midgard_op_load_uniform_32 = 0xB0,
	midgard_op_store_vary_16 = 0xD5,
	midgard_op_store_vary_32 = 0xD4
} midgard_load_store_op_e;

typedef struct
__attribute__((__packed__))
{
	midgard_load_store_op_e op : 8;
	unsigned reg     : 5;
	unsigned mask    : 4;
	unsigned swizzle : 8;
	unsigned unknown : 26;
	unsigned address : 9;
} midgard_load_store_word_t;

typedef struct
__attribute__((__packed__))
{
	uint8_t  tag       : 8;
	uint64_t word1     : 60;
	uint64_t word2     : 60;
} midgard_load_store_t;

/* Some defines not found in the disassembler */

/* 4-bit type tags */

#define TAG_TEXTURE_4 0x3
#define TAG_LOAD_STORE_4 0x5
#define TAG_ALU_4 0x8
#define TAG_ALU_8 0x9
#define TAG_ALU_12 0xA
#define TAG_ALU_16 0xB

/* Special register aliases */

#define REGISTER_UNUSED 24
#define REGISTER_CONSTANT 26
#define REGISTER_OFFSET 27
#define REGISTER_TEXTURE_1 28
#define REGISTER_TEXTURE_2 29
#define REGISTER_SELECT  31

/* Swizzle support */

#define SWIZZLE(A, B, C, D) ((D << 6) | (C << 4) | (B << 2) | (A << 0))
#define SWIZZLE_FROM_ARRAY(r) SWIZZLE(r[0], r[1], r[2], r[3])
#define COMPONENT_X 0x0
#define COMPONENT_Y 0x1
#define COMPONENT_Z 0x2
#define COMPONENT_W 0x3
