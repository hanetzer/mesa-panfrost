/* Author(s):
 *   Alyssa Rosenzweig
 *
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

/* Some constants and macros not found in the disassembler */

#define OP_IS_STORE(op) (\
		op == midgard_op_store_vary_16 || \
		op == midgard_op_store_vary_32 \
	)

/* ALU control words are single bit fields with a lot of space */

#define ALU_ENAB_VEC_MUL    (1 << 17)
#define ALU_ENAB_SCAL_ADD   (1 << 19)
#define ALU_ENAB_VEC_ADD    (1 << 21)
#define ALU_ENAB_SCAL_MUL   (1 << 23)
#define ALU_ENAB_VEC_LUT    (1 << 25)
#define ALU_ENAB_BR_COMPACT (1 << 26)
#define ALU_ENAB_BRANCH     (1 << 27)

/* Vector-independant shorthands for the above; these numbers are arbitrary and
 * not from the ISA. Convert to the above with unit_enum_to_midgard */

#define UNIT_MUL 0
#define UNIT_ADD 1
#define UNIT_LUT 2

/* 4-bit type tags */

#define TAG_TEXTURE_4 0x3
#define TAG_LOAD_STORE_4 0x5
#define TAG_ALU_4 0x8
#define TAG_ALU_8 0x9
#define TAG_ALU_12 0xA
#define TAG_ALU_16 0xB

/* Special register aliases */

#define MAX_WORK_REGISTERS 16

/* Uniforms are begin at (REGISTER_UNIFORMS - uniform_count) */
#define REGISTER_UNIFORMS 24

#define REGISTER_UNUSED 24
#define REGISTER_CONSTANT 26
#define REGISTER_VARYING_BASE 26
#define REGISTER_OFFSET 27
#define REGISTER_TEXTURE_BASE 28
#define REGISTER_SELECT  31

/* SSA helper aliases to mimic the registers. UNUSED_0 encoded as an inline
 * constant. UNUSED_1 encoded as REGISTER_UNUSED */

#define SSA_UNUSED_0 0
#define SSA_UNUSED_1 -2

#define SSA_FIXED_SHIFT 24
#define SSA_FIXED_REGISTER(reg) ((1 + reg) << SSA_FIXED_SHIFT)
#define SSA_REG_FROM_FIXED(reg) ((reg >> SSA_FIXED_SHIFT) - 1)
#define SSA_FIXED_MINIMUM SSA_FIXED_REGISTER(0)

/* Swizzle support */

#define SWIZZLE(A, B, C, D) ((D << 6) | (C << 4) | (B << 2) | (A << 0))
#define SWIZZLE_FROM_ARRAY(r) SWIZZLE(r[0], r[1], r[2], r[3])
#define COMPONENT_X 0x0
#define COMPONENT_Y 0x1
#define COMPONENT_Z 0x2
#define COMPONENT_W 0x3

/* Output writing "condition" for the branch (all one's) */

#define COND_FBWRITE 0x3

/* See ISA notes */

#define LDST_NOP (3)

/* Is this opcode that of an integer? */
static bool
midgard_is_integer_op(int op)
{
	switch (op) {
		case midgard_alu_op_iadd: 
		case midgard_alu_op_ishladd: 
		case midgard_alu_op_isub: 
		case midgard_alu_op_imul: 
		case midgard_alu_op_imin: 
		case midgard_alu_op_imax: 
		case midgard_alu_op_iasr: 
		case midgard_alu_op_ilsr: 
		case midgard_alu_op_ishl: 
		case midgard_alu_op_iand: 
		case midgard_alu_op_ior: 
		case midgard_alu_op_inot: 
		case midgard_alu_op_iandnot: 
		case midgard_alu_op_ixor: 
		case midgard_alu_op_imov: 
		//case midgard_alu_op_f2i: 
		//case midgard_alu_op_f2u: 
		case midgard_alu_op_ieq: 
		case midgard_alu_op_ine: 
		case midgard_alu_op_ilt: 
		case midgard_alu_op_ile: 
		case midgard_alu_op_iball_eq: 
		case midgard_alu_op_ibany_neq: 
		//case midgard_alu_op_i2f: 
		//case midgard_alu_op_u2f: 
		case midgard_alu_op_icsel: 
			return true;
		default:
			return false;
	}
}
