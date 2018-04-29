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

#ifndef __midgard_h__
#define __midgard_h__

#include <stdint.h>
#include <stdbool.h>

typedef enum
{
	midgard_word_type_alu,
	midgard_word_type_load_store,
	midgard_word_type_texture,
	midgard_word_type_unknown
} midgard_word_type;

typedef enum
{
	midgard_alu_vmul,
	midgard_alu_sadd,
	midgard_alu_smul,
	midgard_alu_vadd,
	midgard_alu_lut
} midgard_alu;

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
	midgard_alu_op_ishladd    = 0x41,
	midgard_alu_op_isub       = 0x46,
	midgard_alu_op_imul       = 0x58,
	midgard_alu_op_imin       = 0x60,
	midgard_alu_op_imax       = 0x62,
	midgard_alu_op_iasr       = 0x68,
	midgard_alu_op_ilsr       = 0x69,
	midgard_alu_op_ishl       = 0x6E,
	midgard_alu_op_iand       = 0x70,
	midgard_alu_op_ior        = 0x71,
	midgard_alu_op_inot       = 0x72,
	midgard_alu_op_ixor       = 0x76,
	midgard_alu_op_imov       = 0x7B,
	midgard_alu_op_feq        = 0x80,
	midgard_alu_op_fne        = 0x81,
	midgard_alu_op_flt        = 0x82,
	midgard_alu_op_fle        = 0x83,
	midgard_alu_op_f2i        = 0x99,
	midgard_alu_op_f2u        = 0x9D,
	midgard_alu_op_ieq        = 0xA0,
	midgard_alu_op_ine        = 0xA1,
	midgard_alu_op_ilt        = 0xA4,
	midgard_alu_op_ile        = 0xA5,
	midgard_alu_op_ball       = 0xA9,
	midgard_alu_op_bany       = 0xB1,
	midgard_alu_op_i2f        = 0xB8,
	midgard_alu_op_u2f        = 0xBC,
	midgard_alu_op_icsel      = 0xC1,
	midgard_alu_op_fcsel      = 0xC5,
	midgard_alu_op_fatan_pt2  = 0xE8,
	midgard_alu_op_frcp       = 0xF0,
	midgard_alu_op_frsqrt     = 0xF2,
	midgard_alu_op_fsqrt      = 0xF3,
	midgard_alu_op_fexp2      = 0xF4,
	midgard_alu_op_flog2      = 0xF5,
	midgard_alu_op_fsin       = 0xF6,
	midgard_alu_op_fcos       = 0xF7,
	midgard_alu_op_fatan2_pt1 = 0xF9,
} midgard_alu_op;

typedef enum
{
	midgard_outmod_none = 0,
	midgard_outmod_pos  = 1,
	midgard_outmod_int  = 2,
	midgard_outmod_sat  = 3
} midgard_outmod;

typedef enum
{
	midgard_reg_mode_half = 1,
	midgard_reg_mode_full = 2
} midgard_reg_mode;

typedef enum
{
	midgard_dest_override_lower = 0,
	midgard_dest_override_upper = 1,
	midgard_dest_override_none = 2
} midgard_dest_override;

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
} midgard_vector_alu_src;

typedef struct
__attribute__((__packed__))
{
	midgard_alu_op op               :  8;
	midgard_reg_mode reg_mode   :  2;
	unsigned src1 : 13;
	unsigned src2 : 13;
	midgard_dest_override dest_override : 2;
	midgard_outmod outmod               : 2;
	unsigned mask                           : 8;
} midgard_vector_alu;

typedef struct
__attribute__((__packed__))
{
	bool abs           : 1;
	bool negate        : 1;
	bool full          : 1; /* 0 = half, 1 = full */
	unsigned component : 3;
} midgard_scalar_alu_src;

typedef struct
__attribute__((__packed__))
{
	midgard_alu_op op         :  8;
	unsigned src1             :  6;
	unsigned src2             : 11;
	unsigned unknown          :  1;
	midgard_outmod outmod :  2;
	bool output_full          :  1;
	unsigned output_component :  3;
} midgard_scalar_alu;

typedef struct
__attribute__((__packed__))
{
	unsigned src1_reg : 5;
	unsigned src2_reg : 5;
	unsigned out_reg  : 5;
	bool src2_imm     : 1;
} midgard_reg_info;

typedef enum
{
	midgard_jmp_writeout_op_branch_uncond = 1,
	midgard_jmp_writeout_op_branch_cond = 2,
	midgard_jmp_writeout_op_discard = 4,
	midgard_jmp_writeout_op_writeout = 7,
} midgard_jmp_writeout_op;

typedef struct
__attribute__((__packed__))
{
	midgard_jmp_writeout_op op : 3; /* == branch_uncond */
	unsigned dest_tag : 4; /* tag of branch destination */
	unsigned unknown : 2;
	int offset : 7;
} midgard_branch_uncond;

typedef struct
__attribute__((__packed__))
{
	midgard_jmp_writeout_op op : 3; /* == branch_cond */
	unsigned dest_tag : 4; /* tag of branch destination */
	int offset : 7;
	unsigned cond : 2;
} midgard_branch_cond;

typedef struct
__attribute__((__packed__))
{
	midgard_jmp_writeout_op op : 3; /* == writeout */
	unsigned unknown : 13;
} midgard_writeout;

/*
 * Load/store words
 */

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
} midgard_load_store_op;

typedef enum
{
	midgard_interp_centroid = 1,
	midgard_interp_default = 2
} midgard_interpolation;

typedef struct
__attribute__((__packed__))
{
	midgard_load_store_op op : 8;
	unsigned reg     : 5;
	unsigned mask    : 4;
	unsigned swizzle : 8;
	unsigned unknown : 16;

	unsigned unknown0_1 : 4; /* Always zero */

	/* Varying qualifiers, zero if not a varying */
	unsigned flat    : 1;
	unsigned is_varying : 1; /* Always one for varying, but maybe something else? */
	midgard_interpolation interpolation : 2;

	unsigned unknown0_2 : 2; /* Always zero */

	unsigned address : 9;
} midgard_load_store_word;

typedef struct
__attribute__((__packed__))
{
	unsigned type      : 4;
	unsigned next_type : 4;
	uint64_t word1     : 60;
	uint64_t word2     : 60;
} midgard_load_store;

/* Texture pipeline results are in r28-r29 */
#define REG_TEX_BASE 28

/* Texture opcodes... maybe? */
#define TEXTURE_OP_NORMAL 0x11
#define TEXTURE_OP_TEXEL_FETCH 0x14

/* Texture format types, found in format */
#define TEXTURE_2D 0x02
#define TEXTURE_3D 0x03

typedef struct
__attribute__((__packed__))
{
	unsigned type      : 4;
	unsigned next_type : 4;

	unsigned op  : 6;
	unsigned shadow    : 1;
	unsigned unknown3  : 1;

	/* A little obscure, but last is set for the last texture operation in
	 * a shader. cont appears to just be last's opposite (?). Yeah, I know,
	 * kind of funky.. BiOpen thinks it could do with memory hinting, or
	 * tile locking? */

	unsigned cont  : 1;
	unsigned last  : 1;

	unsigned format    : 5;
	unsigned has_offset : 1;

	/* Like in Bifrost */
	unsigned filter  : 1;

	unsigned in_reg_select : 1;
	unsigned in_reg_upper  : 1;
	unsigned unknown1  : 1;
	unsigned in_reg_full : 1;

	unsigned in_reg_swizzle_right : 2;
	unsigned in_reg_swizzle_left : 2;

	unsigned unknown8  : 4;

	unsigned out_full  : 1;

	/* Always 1 afaict... */
	unsigned unknown7  : 2;

	unsigned out_reg_select : 1;
	unsigned out_upper : 1;

	unsigned mask : 4;

	unsigned unknown2  : 2;

	unsigned swizzle  : 8;
	unsigned unknown4  : 8;

	unsigned unknownA  : 4;

	unsigned offset_unknown1  : 1;
	unsigned offset_reg_select : 1;
	unsigned offset_reg_upper : 1;
	unsigned offset_unknown4  : 1;
	unsigned offset_unknown5  : 1;
	unsigned offset_unknown6  : 1;
	unsigned offset_unknown7  : 1;
	unsigned offset_unknown8  : 1;
	unsigned offset_unknown9  : 1;

	unsigned unknownB  : 3;

	/* Texture bias or LOD, depending on whether it is executed in a
	 * fragment/vertex shader respectively. Compute as int(2^8 * biasf).
	 *
	 * For texel fetch, this is the LOD as is. */
	unsigned bias  : 8;

	unsigned unknown9  : 8;

	unsigned texture_handle : 16;
	unsigned sampler_handle : 16;
} midgard_texture_word;

#endif
