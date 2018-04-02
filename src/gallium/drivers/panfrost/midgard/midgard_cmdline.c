/*
 * Copyright (C) 2018 Alyssa Rosenzweig <alyssa@rosenzweig.io>
 *
 * Copyright (C) 2014 Rob Clark <robclark@freedesktop.org>
 * Copyright (c) 2014 Scott Mansell
 * Copyright © 2014 Broadcom
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <err.h>

#include "compiler/glsl/standalone.h"
#include "compiler/glsl/glsl_to_nir.h"
#include "compiler/nir_types.h"
#include "main/imports.h"
#include "compiler/nir/nir_builder.h"

#include "midgard.h"

/* Instruction arguments represented as block-local SSA indices, rather than
 * registers. Negative values mean unused. */

typedef struct {
	int src0;
	int src1;
	int dest;

	/* The output is -not- SSA -- it's a direct register from I/O -- and
	 * must not be culled/renamed */
	bool literal_out;

	/* src1 is -not- SSA but instead a 16-bit inline constant to be smudged
	 * in. Only valid for ALU ops. */
	bool inline_constant;
} ssa_args;

/* Generic in-memory data type repesenting a single logical instruction, rather
 * than a single instruction group. This is the preferred form for code gen.
 * Multiple midgard_insturctions will later be combined during scheduling,
 * though this is not represented in this structure.  Its format bridges
 * the low-level binary representation with the higher level semantic meaning.
 *
 * Notably, it allows registers to be specified as block local SSA, for code
 * emitted before the register allocation pass.
 */

typedef struct midgard_instruction {
	midgard_word_type type; /* ALU, load/store, texture */

	/* If the register allocator has not run yet... */
	bool uses_ssa;
	ssa_args ssa_args;

	/* Special fields for an ALU instruction */
	bool vector; 
	alu_register_word registers;

	/* I.e. (1 << alu_bit) */
	int unit;

	bool has_constants;
	float constants[4];

	bool compact_branch;

	/* dynarray's are O(n) to delete from, which makes peephole
	 * optimisations a little awkward. Instead, just have an unused flag
	 * which the code gen will skip over */

	bool unused;

	union {
		midgard_load_store_word_t load_store;
		midgard_scalar_alu_t scalar_alu;
		midgard_vector_alu_t vector_alu;
		uint16_t br_compact;
		/* TODO Texture */
	};
} midgard_instruction;

/* Helpers to generate midgard_instruction's using macro magic, since every
 * driver seems to do it that way */

#define EMIT(op, ...) util_dynarray_append(&(ctx->current_block), midgard_instruction, v_##op(__VA_ARGS__));

#define M_LOAD_STORE(name, rname, uname) \
	static midgard_instruction m_##name(unsigned ssa, unsigned address) { \
		midgard_instruction i = { \
			.type = TAG_LOAD_STORE_4, \
			.uses_ssa = true, \
			.ssa_args = { \
				.rname = ssa, \
				.uname = -1, \
				.src1 = -1 \
			}, \
			.unused = false, \
			.load_store = { \
				.op = midgard_op_##name, \
				.mask = 0xF, \
				.swizzle = SWIZZLE(COMPONENT_X, COMPONENT_Y, COMPONENT_Z, COMPONENT_W), \
				.address = address \
			} \
		}; \
		\
		return i; \
	}

#define M_LOAD(name) M_LOAD_STORE(name, dest, src0)
#define M_STORE(name) M_LOAD_STORE(name, src0, dest)

const midgard_vector_alu_src_t blank_alu_src = {
	.swizzle = SWIZZLE(COMPONENT_X, COMPONENT_Y, COMPONENT_Z, COMPONENT_W),
};

const midgard_scalar_alu_src_t blank_scalar_alu_src = {};

/* Used for encoding the unused source of 1-op instructions */
const midgard_vector_alu_src_t zero_alu_src = { 0 };

/* Coerce structs to integer */

static unsigned
vector_alu_src_to_unsigned(midgard_vector_alu_src_t src)
{
	unsigned u;
	memcpy(&u, &src, sizeof(src));
	return u;
}

static unsigned
scalar_alu_src_to_unsigned(midgard_scalar_alu_src_t src)
{
	unsigned u;
	memcpy(&u, &src, sizeof(src));
	return u;
}

/* Inputs a NIR ALU source, with modifiers attached if necessary, and outputs
 * the corresponding Midgard source */

static midgard_vector_alu_src_t
vector_alu_modifiers(nir_alu_src *src)
{
	if (!src) return blank_alu_src;

	midgard_vector_alu_src_t alu_src = {
		.abs = src->abs,
		.negate = src->negate,
		.rep_low = 0,
		.rep_high = 0,
		.half = 0, /* TODO */
		.swizzle = SWIZZLE_FROM_ARRAY(src->swizzle)
	};

	return alu_src;
}

/* Full 1 parameters refer to "non-half-float mode" and "first src in scalar
 * instruction" to account for a weird special case */

static midgard_scalar_alu_src_t
scalar_alu_modifiers(nir_alu_src *src, bool full1)
{
	if (!src) return blank_scalar_alu_src;

	midgard_scalar_alu_src_t alu_src = {
		.abs = src->abs,
		.negate = src->negate,
		.full = 1, /* TODO */
		.component = src->swizzle[0] << (full1 ? 1 : 0) /* TODO: Is this actually correct? */
	};

	return alu_src;
}

static unsigned 
scalar_move_src(int component, bool full) {
	midgard_scalar_alu_src_t src = {
		.full = full,
		.component = component << 1 /* XXX: Ditto */
	};

	return scalar_alu_src_to_unsigned(src);
}

static midgard_outmod_e
n2m_alu_outmod(bool saturate)
{
	return saturate ? midgard_outmod_sat : midgard_outmod_none;
}

static midgard_instruction
m_alu_vector(midgard_alu_op_e op, int unit, unsigned src0, midgard_vector_alu_src_t mod1, unsigned src1, midgard_vector_alu_src_t mod2, unsigned dest, bool literal_out, midgard_outmod_e outmod)
{
	/* TODO: Use literal_out hint during register allocation */
	midgard_instruction ins = {
		.type = TAG_ALU_4,
		.unit = unit,
		.unused = false,
		.uses_ssa = true,
		.ssa_args = {
			.src0 = src0,
			.src1 = src1,
			.dest = dest,
			.literal_out = literal_out
		},
		.vector = true,
		.vector_alu = {
			.op = op,
			.reg_mode = midgard_reg_mode_full,
			.dest_override = midgard_dest_override_none,
			.outmod = outmod,
			.mask = 0xFF,
			.src1 = vector_alu_src_to_unsigned(mod1),
			.src2 = vector_alu_src_to_unsigned(mod2)
		},
	};

	return ins;
}

#define M_ALU_VECTOR_1(unit, name) \
	static midgard_instruction v_##name(unsigned src, midgard_vector_alu_src_t mod1, unsigned dest, bool literal, midgard_outmod_e outmod) { \
		return m_alu_vector(midgard_alu_op_##name, ALU_ENAB_VEC_##unit, SSA_UNUSED_1, zero_alu_src, src, mod1, dest, literal, outmod); \
	}

#define M_ALU_VECTOR_2(unit, name) \
	static midgard_instruction v_##name(unsigned src1, midgard_vector_alu_src_t mod1, unsigned src2, midgard_vector_alu_src_t mod2, unsigned dest, bool literal, midgard_outmod_e outmod) { \
		return m_alu_vector(midgard_alu_op_##name, ALU_ENAB_VEC_##unit, src1, mod1, src2, mod2, dest, literal, outmod); \
	}

/* load/store instructions have both 32-bit and 16-bit variants, depending on
 * whether we are using vectors composed of highp or mediump. At the moment, we
 * don't support half-floats -- this requires changes in other parts of the
 * compiler -- therefore the 16-bit versions are commented out. */

//M_LOAD(ld_st_noop);
//M_LOAD(load_attr_16);
M_LOAD(load_attr_32);
//M_LOAD(load_vary_16);
M_LOAD(load_vary_32);
//M_LOAD(load_uniform_16);
M_LOAD(load_uniform_32);
//M_STORE(store_vary_16);
M_STORE(store_vary_32);

/* Used as a sort of intrinsic outside of the ALU code */
M_ALU_VECTOR_1(MUL, fmov);

/* TODO: Expand into constituent parts since we do understand how this works,
 * no? */

static midgard_instruction
v_alu_br_compact_cond(midgard_jmp_writeout_op_e op, unsigned tag, signed offset, unsigned cond)
{
	midgard_branch_cond_t branch = {
		.op = op,
		.dest_tag = tag,
		.offset = offset,
		.cond = cond
	};

	uint16_t compact;
	memcpy(&compact, &branch, sizeof(branch));

	midgard_instruction ins = {
		.type = TAG_ALU_4,
		.unit = ALU_ENAB_BR_COMPACT ,
		.unused = false,
		.uses_ssa = false,

		.compact_branch = true, 
		.br_compact = compact
	};

	return ins;
}

static void
attach_constants(midgard_instruction *ins, void *constants)
{
	ins->has_constants = true;
	memcpy(&ins->constants, constants, 16); /* TODO: How big? */
}

typedef struct compiler_context {
	gl_shader_stage stage;

	/* List of midgard_instructions emitted for the current block */
	struct util_dynarray current_block;

	/* Constants which have been loaded, for later inlining */
	struct hash_table_u64 *ssa_constants;

	/* SSA values / registers which have been aliased. Naively, these
	 * demand a fmov output; instead, we alias them in a later pass to
	 * avoid the wasted op.
	 *
	 * A note on encoding: to avoid dynamic memory management here, rather
	 * than ampping to a pointer, we map to the source index; the key
	 * itself is just the destination index. */

	struct hash_table_u64 *ssa_to_alias;
	struct set *leftover_ssa_to_alias;
	
	/* Encoded the same as ssa_to_alias, except now it's mapping SSA source indicdes as the keys to fixed destination registers as the values */
	struct hash_table_u64 *register_to_ssa;
} compiler_context;

static int
glsl_type_size(const struct glsl_type *type)
{
	return glsl_count_attribute_slots(type, false);
}

static void
optimise_nir(nir_shader *nir)
{
	bool progress;

	//NIR_PASS_V(nir, nir_lower_io_to_temporaries, nir_shader_get_entrypoint(nir), true, true);
	//NIR_PASS(progress, nir, nir_opt_global_to_local);
	//NIR_PASS(progress, nir, nir_lower_regs_to_ssa);

	//NIR_PASS(progress, nir, nir_lower_global_vars_to_local);
	//NIR_PASS(progress, nir, nir_lower_locals_to_regs);
	//NIR_PASS_V(nir, nir_lower_io_types);

	do {
		progress = false;

		NIR_PASS(progress, nir, nir_lower_io, nir_var_all, glsl_type_size, 0);
		NIR_PASS(progress, nir, nir_lower_var_copies);
		NIR_PASS(progress, nir, nir_lower_vars_to_ssa);

		//NIR_PASS(progress, nir, nir_lower_vec_to_movs);
		NIR_PASS(progress, nir, nir_copy_prop);
		NIR_PASS(progress, nir, nir_opt_remove_phis);
		NIR_PASS(progress, nir, nir_opt_dce);
		NIR_PASS(progress, nir, nir_opt_dead_cf);
		NIR_PASS(progress, nir, nir_opt_cse);
		NIR_PASS(progress, nir, nir_opt_peephole_select, 8);
		NIR_PASS(progress, nir, nir_opt_algebraic);
		NIR_PASS(progress, nir, nir_opt_constant_folding);
		NIR_PASS(progress, nir, nir_opt_undef);
		NIR_PASS(progress, nir, nir_opt_loop_unroll, 
				nir_var_shader_in |
				nir_var_shader_out |
				nir_var_local);
	} while(progress);
	printf("---\n");

	NIR_PASS(progress, nir, nir_lower_to_source_mods);
	NIR_PASS(progress, nir, nir_copy_prop);
	NIR_PASS(progress, nir, nir_opt_dce);
}

/* Front-half of aliasing the SSA slots, merely by inserting the flag in the
 * appropriate hash table. Intentional off-by-one to avoid confusing NULL with
 * r0. See the comments in compiler_context */

static void
alias_ssa(compiler_context *ctx, int dest, int src, bool literal_dest)
{
	if (literal_dest) {
		_mesa_hash_table_u64_insert(ctx->register_to_ssa, src, (void *) (uintptr_t) dest + 1);
	} else {
		_mesa_hash_table_u64_insert(ctx->ssa_to_alias, dest, (void *) (uintptr_t) src + 1);
		_mesa_set_add(ctx->leftover_ssa_to_alias, (void *) (uintptr_t) dest);
	}
}

static void
emit_load_const(compiler_context *ctx, nir_load_const_instr *instr)
{
	nir_ssa_def def = instr->def;

	float *v = ralloc_array(NULL, float, 4);
	memcpy(v, &instr->value.f32, 4 * sizeof(float));
	_mesa_hash_table_u64_insert(ctx->ssa_constants, def.index, v);

	midgard_instruction ins = v_fmov(REGISTER_CONSTANT, blank_alu_src, def.index, false, midgard_outmod_none);
	attach_constants(&ins, &instr->value.f32);
	util_dynarray_append(&ctx->current_block, midgard_instruction, ins);
}

static unsigned
unit_enum_to_midgard(int unit_enum, int is_vector) {
	if (is_vector) {
		switch(unit_enum) {
			case UNIT_MUL: return ALU_ENAB_VEC_MUL;
			case UNIT_ADD: return ALU_ENAB_VEC_ADD;
			case UNIT_LUT: return ALU_ENAB_VEC_LUT;
			default: return 0; /* Should never happen */
		}
	} else {
		switch(unit_enum) {
			case UNIT_MUL: return ALU_ENAB_SCAL_MUL;
			case UNIT_ADD: return ALU_ENAB_SCAL_ADD;
			default: return 0; /* Should never happen */
		}
	}
}

/* Unit: shorthand for the unit used by this instruction (MUL, ADD, LUT).
 * Components: Number/style of arguments:
 * 	2: Standard two argument op (fadd, fmul)
 * 	1: Flipped one-argument op (fmov, imov)
 * 	0: Standard one-argument op (frcp)
 * NIR: NIR instruction op.
 * Op: Midgard instruction op.
 */

#define ALU_CASE(_unit, _components, nir, _op) \
	case nir_op_##nir: \
		unit = UNIT_##_unit; \
		components = _components; \
		op = midgard_alu_op_##_op; \
		break;

static void
emit_alu(compiler_context *ctx, nir_alu_instr *instr)
{
	unsigned dest = instr->dest.dest.ssa.index;

	/* lower_vec_to_moves generates really bad code, so we use the pass in Freedreno instead */
	if ((instr->op == nir_op_vec2) ||
		(instr->op == nir_op_vec3) ||
		(instr->op == nir_op_vec4)) {

		for (int i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
			nir_alu_src *asrc = &instr->src[i];

			int input = asrc->src.ssa->index;
			int component = asrc->swizzle[0];

			midgard_instruction ins = {
				.type = TAG_ALU_4,
				.unit = ALU_ENAB_SCAL_MUL,
				.unused = false,
				.uses_ssa = true,
				.ssa_args = {
					.src0 = SSA_UNUSED_1,
					.src1 = input,
					.dest = dest,
				},
				.scalar_alu = {
					.op = midgard_alu_op_fmov,
					.src1 = 0,
					.src2 = scalar_move_src(component, true),
					.outmod = midgard_outmod_none,
					.output_full = true,
					.output_component = i << 1 /* Skew full */
				}
			};

			util_dynarray_append(&(ctx->current_block), midgard_instruction, ins); 
		}

		return;
	}

	/* ALU ops are unified in NIR between scalar/vector, but partially
	 * split in Midgard. Reconcile that here, to avoid diverging code paths
	 */
	bool is_vector = instr->dest.dest.ssa.num_components != 1;

	/* Most Midgard ALU ops have a 1:1 correspondance to NIR ops; these are
	 * supported. A few do not and are therefore commented and TODO to
	 * figure out what code paths would generate these. Also, there are a
	 * number of NIR ops which Midgard does not support and need to be
	 * lowered, also TODO. This switch block emits the opcode and calling
	 * convention of the Midgard instruction; actual packing is done in emit_alu below
	 * */

	unsigned op, unit, components;

	switch(instr->op) {
		ALU_CASE(ADD, 2, fadd, fadd);
		ALU_CASE(MUL, 2, fmul, fmul);
		ALU_CASE(MUL, 2, fmin, fmin);
		ALU_CASE(MUL, 2, fmax, fmax);
		ALU_CASE(MUL, 1, fmov, fmov);
		ALU_CASE(MUL, 1, ffloor, ffloor);
		ALU_CASE(MUL, 1, fceil, fceil);
		//ALU_CASE(MUL, 2, fdot3);
		//ALU_CASE(MUL, 2, fdot3r);
		//ALU_CASE(MUL, 2, fdot4);
		//ALU_CASE(MUL, 2, freduce);
		ALU_CASE(ADD, 2, iadd, iadd);
		ALU_CASE(ADD, 2, isub, isub);
		ALU_CASE(MUL, 2, imul, imul);

		/* TODO: How does imov work, exactly? */
		ALU_CASE(MUL, 1, imov, fmov);

		ALU_CASE(MUL, 2, feq, feq);
		ALU_CASE(MUL, 2, fne, fne);
		ALU_CASE(MUL, 2, flt, flt);
		//ALU_CASE(MUL, 2, fle);
		ALU_CASE(MUL, 1, f2i32, f2i);
		ALU_CASE(MUL, 1, f2u32, f2i);
		ALU_CASE(MUL, 2, ieq, ieq);
		ALU_CASE(MUL, 2, ine, ine);
		ALU_CASE(MUL, 2, ilt, ilt);
		//ALU_CASE(MUL, 2, ile);
		//ALU_CASE(MUL, 2, csel, csel);
		ALU_CASE(MUL, 1, i2f32, i2f);
		ALU_CASE(MUL, 1, u2f32, i2f);
		//ALU_CASE(LUT, 0, fatan_pt2);
		ALU_CASE(LUT, 0, frcp, frcp);
		ALU_CASE(LUT, 0, frsq, frsqrt);
		ALU_CASE(LUT, 0, fsqrt, fsqrt);
		ALU_CASE(LUT, 0, fexp2, fexp2);
		ALU_CASE(LUT, 0, flog2, flog2);

		// Input needs to be divided by pi due to Midgard weirdness We
		// define special NIR ops, fsinpi and fcospi, that include the
		// division correctly, supplying appropriately lowering passes.
		// That way, the division by pi can take advantage of constant
		// folding, algebraic simplifications, and so forth.

		ALU_CASE(LUT, 0, fsinpi, fsin);
		ALU_CASE(LUT, 0, fcospi, fcos);

		//ALU_CASE(LUT, 0, fatan_pt1);


		default:
			printf("Unhandled ALU op\n");
			break;
	}

	/* slut doesn't exist; lower to vlut which acts as scalar
	 * despite the name */

	if (unit == UNIT_LUT)
		is_vector = true;

	/* Initialise fields common between scalar/vector instructions */
	midgard_outmod_e outmod = n2m_alu_outmod(instr->dest.saturate);

	/* src0 will always exist afaik, but src1 will not for 1-argument
	 * instructions. The latter can only be fetched if the instruction
	 * needs it, or else we may segfault. */

	unsigned src0 = instr->src[0].src.ssa->index;
	unsigned src1 = components > 1 ? instr->src[1].src.ssa->index : 0;

	/* Rather than use the instruction generation helpers, we do it
	 * ourselves here to avoid the mess */

	midgard_instruction ins = {
		.type = TAG_ALU_4,
		.unit = unit_enum_to_midgard(unit, is_vector),
		.unused = false,
		.uses_ssa = true,
		.ssa_args = {
			.src0 = components == 2 || components == 0 ? src0 : SSA_UNUSED_1,
			.src1 = components == 2 ? src1 : components == 1 ? src0 : SSA_UNUSED_0,
			.dest = dest,
			.inline_constant = components == 0
		},
		.vector = is_vector
	};

	nir_alu_src *nirmod0 = NULL;
	nir_alu_src *nirmod1 = NULL;

	if (components == 2) {
		nirmod0 = &instr->src[0];
		nirmod1 = &instr->src[1];
	} else if (components == 1) {
		nirmod1 = &instr->src[0];
	} else if (components == 0) {
		nirmod0 = &instr->src[0];
	}

	if (is_vector) {
		midgard_vector_alu_t alu = {
			.op = op,
			.reg_mode = midgard_reg_mode_full,
			.dest_override = midgard_dest_override_none,
			.outmod = outmod,
			.mask = unit == UNIT_LUT ? 0x3 : 0xFF, /* XXX */
			.src1 = vector_alu_src_to_unsigned(vector_alu_modifiers(nirmod0)),
			.src2 = vector_alu_src_to_unsigned(vector_alu_modifiers(nirmod1)),
		};

		ins.vector_alu = alu;
	} else {
		bool is_full = true; /* TODO */

		midgard_scalar_alu_t alu = {
			.op = op,
			.src1 = scalar_alu_src_to_unsigned(scalar_alu_modifiers(nirmod0, is_full)),
			.src2 = scalar_alu_src_to_unsigned(scalar_alu_modifiers(nirmod1, true)),
			.unknown = 0, /* XXX */
			.outmod = outmod,
			.output_full = true, /* XXX */
			.output_component = 0, /* XXX output_component */
		};

		ins.scalar_alu = alu;
	}

	util_dynarray_append(&ctx->current_block, midgard_instruction, ins);
}

static void
emit_intrinsic(compiler_context *ctx, nir_intrinsic_instr *instr)
{
        nir_const_value *const_offset;
        unsigned offset, reg;

	switch(instr->intrinsic) {
		case nir_intrinsic_load_uniform:
		case nir_intrinsic_load_input:
			const_offset = nir_src_as_const_value(instr->src[0]);
			assert (const_offset && "no indirect inputs");

			offset = nir_intrinsic_base(instr) + const_offset->u32[0];
			assert(offset % 4 == 0);
			offset = offset / 4;

			reg = instr->dest.ssa.index;

			/* What this means depends on the type of instruction */
			/* TODO: Pack? */

			if (instr->intrinsic == nir_intrinsic_load_uniform) {
				/* TODO: half-floats */
				/* TODO: Wrong order, plus how do we know how many? */
				/* TODO: Spill to ld_uniform */

				int reg_slot = 23 - offset;
				
				/* Uniform accesses are 0-cycle, since they're
				 * just a register fetch in the usual case. So,
				 * we alias the registers while we're still in
				 * SSA-space */

				alias_ssa(ctx, reg, SSA_FIXED_REGISTER(reg_slot), false);
			} else if (ctx->stage == MESA_SHADER_FRAGMENT) {
				/* XXX: Half-floats? */
				/* TODO: swizzle, mask, decode unknown */

				midgard_instruction ins = m_load_vary_32(reg, offset);
				ins.load_store.unknown = 0xA01E9E; /* XXX: What is this? */
				util_dynarray_append(&ctx->current_block, midgard_instruction, ins);
			} else if(ctx->stage == MESA_SHADER_VERTEX) {
				midgard_instruction ins = m_load_attr_32(reg, offset);
				ins.load_store.unknown = 0x1E1E; /* XXX: What is this? */
				util_dynarray_append(&ctx->current_block, midgard_instruction, ins);
			} else {
				printf("Unknown load\n");

				/* Worst case, emit a load varying and at least
				 * that'll show up in the disassembly */

				util_dynarray_append(&ctx->current_block, midgard_instruction, m_load_vary_32(reg, 0));
			}

			break;

		case nir_intrinsic_store_output:
			const_offset = nir_src_as_const_value(instr->src[1]);
			assert(const_offset && "no indirect outputs");

			offset = nir_intrinsic_base(instr) + const_offset->u32[0];
			offset = offset * 2 + (nir_intrinsic_component(instr) / 2);

			reg = instr->src[0].ssa->index;

			if (ctx->stage == MESA_SHADER_FRAGMENT) {
				/* gl_FragColor is not emitted with load/store
				 * instructions. Instead, it gets plonked into
				 * r0 at the end of the shader and we do the
				 * framebuffer writeout dance. TODO: Defer
				 * writes */

				alias_ssa(ctx, 0, reg, true);
			} else if (ctx->stage == MESA_SHADER_VERTEX) {
				/* Either this is a write from the perspective
				 * division / viewport scaling code and should
				 * be translated to the special output
				 * register, or otherwise it's just a varying
				 * */

				if (nir_intrinsic_base(instr) == VERTEX_EPILOGUE_BASE) {
					alias_ssa(ctx, REGISTER_VERTEX, reg, true);
				} else {
					reg = 1; /* XXX WTF WTF WTF WHY DOES THIS WORK WTF */

					midgard_instruction ins = m_store_vary_32(reg, offset);
					ins.load_store.unknown = 0x1E9E; /* XXX: What is this? */
					util_dynarray_append(&ctx->current_block, midgard_instruction, ins);
				}
			} else {
				printf("Unknown store\n");
				util_dynarray_append(&ctx->current_block, midgard_instruction, m_store_vary_32(reg, offset));
			}

			break;


		default:
			printf ("Unhandled intrinsic\n");
			break;
	}
}

static void
emit_instr(compiler_context *ctx, struct nir_instr *instr)
{
	nir_print_instr(instr, stdout);
	putchar('\n');

	switch(instr->type) {
		case nir_instr_type_load_const:
			emit_load_const(ctx, nir_instr_as_load_const(instr));
			break;

		case nir_instr_type_intrinsic:
			emit_intrinsic(ctx, nir_instr_as_intrinsic(instr));
			break;

		case nir_instr_type_alu:
			emit_alu(ctx, nir_instr_as_alu(instr));
			break;

		default:
			printf("Unhandled instruction type\n");
			break;
	}
}

/* TODO: Write a register allocator. But for now, just set register = ssa index... */

/* Transform to account for SSA register aliases */

static int
dealias_register(int reg)
{
	if (reg >= SSA_FIXED_MINIMUM)
		return SSA_REG_FROM_FIXED(reg);

	if (reg >= 0)
		return reg;

	switch(reg) {
		/* fmov style unused */
		case SSA_UNUSED_0: return 0;
		
		/* lut style unused */
		case SSA_UNUSED_1: return REGISTER_UNUSED;

		default:
		   printf("Unknown SSA register alias %d\n", reg);
		   return 31;
	}
}


static void
allocate_registers(compiler_context *ctx)
{
	util_dynarray_foreach(&ctx->current_block, midgard_instruction, ins) {
		ssa_args args = ins->ssa_args;

		switch (ins->type) {
			case TAG_ALU_4:
				ins->registers.output_reg = dealias_register(args.dest);
				ins->registers.input1_reg = dealias_register(args.src0);

				ins->registers.inline_2 = args.inline_constant;

				if (args.inline_constant && args.src1 != 0) {
					printf("TODO: Encode inline constant %d\n", args.src1);
				} else {
					ins->registers.input2_reg = dealias_register(args.src1);
				}

				break;
			
			case TAG_LOAD_STORE_4:
				ins->load_store.reg = (args.dest >= 0) ? args.dest : args.src0;
				break;

			default: 
				printf("Unknown tag in register assignment pass\n");
				break;
		}
	}
}

/* Midgard prefetches instruction types, so during emission we need to
 * lookahead too. Unless this is the last instruction, in which we return 1. Or
 * if this is the second to last and the last is an ALU, then it's also 1... */

#define IN_ARRAY(n, arr) (n < (arr.data + arr.size))
#define IS_ALU(tag) (tag == TAG_ALU_4 || tag == TAG_ALU_8 ||  \
		     tag == TAG_ALU_12 || tag == TAG_ALU_16)

#define EMIT_AND_COUNT(type, val) util_dynarray_append(emission, type, val); \
				  bytes_emitted += sizeof(type)

static void
emit_binary_vector_instruction(midgard_instruction *ains,
		uint16_t *register_words, int *register_words_count, 
		uint64_t *body_words, size_t *body_size, int *body_words_count, 
		size_t *bytes_emitted)
{
	memcpy(&register_words[(*register_words_count)++], &ains->registers, sizeof(ains->registers));
	*bytes_emitted += sizeof(alu_register_word);

	body_size[*body_words_count] = sizeof(midgard_vector_alu_t);
	memcpy(&body_words[(*body_words_count)++], &ains->vector_alu, sizeof(ains->vector_alu));
	*bytes_emitted += sizeof(midgard_vector_alu_t);
}

/* Returns the number of instructions emitted (minus one). In trivial cases,
 * this equals one (zero returned), but when instructions are paired (the
 * optimal case) this can be two, or in the best case for ALUs, up to five. */

static int
emit_binary_instruction(compiler_context *ctx, midgard_instruction *ins, struct util_dynarray *emission)
{
	int instructions_emitted = 0;

	uint8_t tag = ins->type;

	switch(ins->type) {
		case TAG_ALU_4:
		case TAG_ALU_8:
		case TAG_ALU_12:
		case TAG_ALU_16: {
			uint32_t control = 0;
			size_t bytes_emitted = sizeof(control);

			uint16_t register_words[8];
			int register_words_count = 0;

			uint64_t body_words[8];
			size_t body_size[8];
			int body_words_count = 0;
			
			/* TODO: Constant combining */
			int index = 0, last_unit = 0;
			bool has_embedded_constants = false;

			while (ins + index) {
				midgard_instruction *ains = ins + index; 

				/* Ensure that the chain can continue */
				if (ains->unused) goto skip_instruction;
				if (ains->type != TAG_ALU_4 || ains->unit <= last_unit) break;

				/* Only one set of embedded constants per
				 * bundle possible; if we duplicate, we must
				 * break the chain early, unfortunately */

				if (ains->has_constants) {
					if (has_embedded_constants) break;

					has_embedded_constants = true;
				}

				control |= ains->unit;
				last_unit = ains->unit;

				if (ains->vector) {
					emit_binary_vector_instruction(ains, register_words,
							&register_words_count, body_words,
							body_size, &body_words_count, &bytes_emitted);
				} else if (ains->compact_branch) {
					/* XXX: Workaround hardware errata where branches cannot standalone in a word by including a dummy move */
					if (index == 0) {
						midgard_instruction ins = v_fmov(0, blank_alu_src, 0, true, midgard_outmod_none);

						control |= ins.unit;

						emit_binary_vector_instruction(&ins, register_words,
								&register_words_count, body_words,
								body_size, &body_words_count, &bytes_emitted);
					}

					body_size[body_words_count] = sizeof(ains->br_compact);
					memcpy(&body_words[body_words_count++], &ains->br_compact, sizeof(ains->br_compact));
					bytes_emitted += sizeof(ains->br_compact);
				} else {
					/* TODO: Vector/scalar stuff operates in parallel. This is probably faulty logic */

					memcpy(&register_words[register_words_count++], &ains->registers, sizeof(ains->registers));
					bytes_emitted += sizeof(alu_register_word);

					body_size[body_words_count] = sizeof(midgard_scalar_alu_t);
					memcpy(&body_words[body_words_count++], &ains->scalar_alu, sizeof(ains->scalar_alu));
					bytes_emitted += sizeof(midgard_scalar_alu_t);

					/* TODO: Emit pipeline registers and batch instructions once we know how XXX */
					++index;	
					break;
				}

skip_instruction:
				++index;
			}

			/* Bubble up the number of instructions for skipping */
			instructions_emitted = index - 1;

			int padding = 0;

			/* Pad ALU op to nearest word */

			if (bytes_emitted & 15) {
				padding = 16 - (bytes_emitted & 15);
				bytes_emitted += padding;
			}

			/* Constants must always be quadwords */
			if (ins->has_constants) {
				bytes_emitted += 16;
			}

			/* Size ALU instruction for tag */
			control |= (TAG_ALU_4) + (bytes_emitted / 16) - 1;
			
			/* Actually emit each component */
			EMIT_AND_COUNT(uint32_t, control);

			for (int i = 0; i < register_words_count; ++i)
				util_dynarray_append(emission, uint16_t, register_words[i]);

			for (int i = 0; i < body_words_count; ++i)
				memcpy(util_dynarray_grow(emission, body_size[i]), &body_words[i], body_size[i]);

			/* Emit padding */
			util_dynarray_grow(emission, padding);

			/* Tack on constants */

			if (ins->has_constants) {
				EMIT_AND_COUNT(float, ins->constants[0]);
				EMIT_AND_COUNT(float, ins->constants[1]);
				EMIT_AND_COUNT(float, ins->constants[2]);
				EMIT_AND_COUNT(float, ins->constants[3]);
			}

			break;
		 }

		case TAG_LOAD_STORE_4: {
			/* Load store instructions have two words at once. If we
			 * only have one queued up, we need to NOP pad.
			 * Otherwise, we store both in succession to save space
			 * (and cycles? Unclear) and skip the next. The
			 * usefulness of this optimisation is greatly dependent
			 * on the quality of the (presently nonexistent)
			 * instruction scheduler.
			 */

			uint64_t current64, next64;
			
			midgard_load_store_word_t current = ins->load_store;
			memcpy(&current64, &current, sizeof(current));

			bool filled_next = false;

			if ((ins + 1)->type == TAG_LOAD_STORE_4) {
				midgard_load_store_word_t next = (ins + 1)->load_store;

				/* As the two operate concurrently (TODO:
				 * verify), make sure they are not dependent */

				if (!(OP_IS_STORE(next.op) && !OP_IS_STORE(current.op) &&
				      next.reg == current.reg)) {
					memcpy(&next64, &next, sizeof(next));

					/* Skip ahead one, since it's redundant with the pair */
					instructions_emitted++;

					filled_next = true;
				}
			}

			if (!filled_next) {
				/* While this is good reason for this number
				 * (see the ISA notes), for our purposes we can
				 * just use it as a magic number until it
				 * breaks ;) */

				next64 = 3;
			}

			midgard_load_store_t instruction = {
				.tag = tag,
				.word1 = current64,
				.word2 = next64
			};

			util_dynarray_append(emission, midgard_load_store_t, instruction);

			break;
		}

		default:
			printf("Unknown midgard instruction type\n");
			break;
	}

	return instructions_emitted;
}


/* ALU instructions can inline constants, which decreases register pressure.
 * This is handled here. It does *not* remove the original move, since this is
 * not safe at this stage. eliminate_constant_mov will handle this */

#define CONDITIONAL_ATTACH(src) { \
	void *entry = _mesa_hash_table_u64_search(ctx->ssa_constants, alu->ssa_args.src); \
\
	if (entry) { \
		attach_constants(alu, entry); \
		alu->ssa_args.src = REGISTER_CONSTANT; \
	} \
}

static void
inline_alu_constants(compiler_context *ctx)
{
	util_dynarray_foreach(&ctx->current_block, midgard_instruction, alu) {
		/* Other instructions cannot inline constants */
		if (alu->type != TAG_ALU_4) continue;

		/* If there is already a constant here, we can do nothing */
		if (alu->has_constants) continue;

		/* Constants should always be SSA... */
		if (!alu->uses_ssa) continue;

		CONDITIONAL_ATTACH(src0);
		CONDITIONAL_ATTACH(src1);
	}
}

/* While NIR handles this most of the time, sometimes we generate unnecessary
 * mov instructions ourselves, in particular from the load_const routine if
 * constants are inlined. As a peephole optimisation, eliminate redundant moves
 * in the current block here.
 */

static void
eliminate_constant_mov(compiler_context *ctx)
{
	util_dynarray_foreach(&ctx->current_block, midgard_instruction, move) {
		/* Only interest ourselves with fmov instructions */
		
		if (move->type != TAG_ALU_4) continue;
		if (move->vector && move->vector_alu.op != midgard_alu_op_fmov) continue;
		if (!move->vector && move->scalar_alu.op != midgard_alu_op_fmov) continue;

		/* If this is a literal move (used in tandem with I/O), it
		 * cannot be removed. Similarly, if it -will- be a literal move
		 * based on register_to_ssa, it cannot be removed. */
		
		if (move->ssa_args.literal_out) continue;
		if (_mesa_hash_table_u64_search(ctx->register_to_ssa, move->ssa_args.dest)) continue;

		unsigned target_reg = move->ssa_args.dest;

		/* Scan the succeeding instructions for usage */

		bool used = false;

		for (midgard_instruction *candidate = (move + 1);
		     IN_ARRAY(candidate, ctx->current_block);
		     candidate += 1) {
			/* Check this candidate for usage */

			if (candidate->ssa_args.src0 == target_reg ||
			    candidate->ssa_args.src1 == target_reg) {
				used = true;
				break;
			}
		}

		/* At this point, we know if the move is used or not. If it's
		 * not, delete it! */

		if (!used)
			move->unused = true;
	}
}

/* Map normal SSA sources to other SSA sources / fixed registers (like
 * uniforms) */

static void
map_ssa_to_alias(compiler_context *ctx, int *ref)
{
	uintptr_t alias = _mesa_hash_table_u64_search(ctx->ssa_to_alias, *ref);
	
	if (alias) {
		/* Remove entry in leftovers to avoid a redunant fmov */

		struct set_entry *leftover = _mesa_set_search(ctx->leftover_ssa_to_alias, (void *) (uintptr_t) *ref);

		if (leftover)
			_mesa_set_remove(ctx->leftover_ssa_to_alias, leftover);

		/* Assign the alias map */
		*ref = alias - 1;
		return;
	}

	alias = _mesa_hash_table_u64_search(ctx->register_to_ssa, *ref);

	if (alias) {
		*ref = alias - 1;
		return;
	}
}

/* If there are leftovers after the below pass, emit actual fmov
 * instructions for the slow-but-correct path */

static void
emit_leftover_move(compiler_context *ctx)
{
	struct set_entry *leftover;

	set_foreach(ctx->leftover_ssa_to_alias, leftover) {
		int base = (uintptr_t) leftover->key;
		int mapped = base;

		map_ssa_to_alias(ctx, &mapped);
		EMIT(fmov, mapped, blank_alu_src, base, false, midgard_outmod_none);
	}
}

static void
actualise_ssa_to_alias(compiler_context *ctx)
{
	util_dynarray_foreach(&ctx->current_block, midgard_instruction, ins) {
		map_ssa_to_alias(ctx, &ins->ssa_args.src0);
		map_ssa_to_alias(ctx, &ins->ssa_args.src1);
	}

	emit_leftover_move(ctx);
}

/* Sort of opposite of the above */

static void
actualise_register_to_ssa(compiler_context *ctx)
{
	util_dynarray_foreach(&ctx->current_block, midgard_instruction, ins) {
		uintptr_t reg = (uintptr_t) _mesa_hash_table_u64_search(ctx->register_to_ssa, ins->ssa_args.dest);

		if (reg) {
			ins->ssa_args.dest = reg - 1;
			ins->ssa_args.literal_out = true;
		}
	}
}

/* At least for storing the end gl_Position, there appears to be a hardware
 * quirk that stores must be at the end. This pass, intended to occur right
 * after Midgard instruction emission, defer them to the end. It must be run
 * while the code is still in SSA form, to avoid conflicts with the RA. */

static void
defer_stores(compiler_context *ctx)
{
	/* As we are appending mid foreach, break out at end */
	int cap = ctx->current_block.size / sizeof(midgard_instruction);

	util_dynarray_foreach(&ctx->current_block, midgard_instruction, store) {
		/* Search for stores */

		if (store->type != TAG_LOAD_STORE_4) goto skip;
		if (!OP_IS_STORE(store->load_store.op)) goto skip;

		/* Splice out the instruction and move to the end of stream*/

		util_dynarray_append(&ctx->current_block, midgard_instruction, *store);
		store->unused = true;

		/* Check for maximum condition */
skip:
		cap--;
		if (!cap) break;
	}
}

/* Shader epilogues */

static void
emit_vertex_epilogue(nir_builder *b, nir_ssa_def *input_point)
{
	/* TODO: Don't assume 400x240 screen, nor 0.5, nor NDC input */
	nir_ssa_def *window = nir_vec4(b, nir_imm_float(b, 200.0f), nir_imm_float(b, 120.0f), nir_imm_float(b, 0.5f), nir_imm_float(b, 0.0));
	nir_ssa_def *persp = nir_vec4(b, nir_imm_float(b, 0), nir_imm_float(b, 0), nir_imm_float(b, 0), nir_imm_float(b, 1.0));
	nir_ssa_def *transformed_point = nir_fadd(b, nir_fadd(b, nir_fmul(b, input_point, window), window), persp);

	/* Finally, write out the transformed values to VERTEX_EPILOGUE_BASE
	 * (which ends up being r27) */

	nir_intrinsic_instr *store;
	store = nir_intrinsic_instr_create(b->shader, nir_intrinsic_store_output);
	store->num_components = 4;
	nir_intrinsic_set_base(store, VERTEX_EPILOGUE_BASE);
	nir_intrinsic_set_write_mask(store, 0xf);
	store->src[0].ssa = transformed_point;
	store->src[0].is_ssa = true;
	store->src[1] = nir_src_for_ssa(nir_imm_int(b, 0));
	nir_builder_instr_insert(b, &store->instr);
}

/* XXX: From nir_lower_clip.c, genericise the code before merging XXX FIXME */
static nir_ssa_def *
find_output_in_block(nir_block *block, unsigned drvloc)
{
   nir_foreach_instr(instr, block) {

      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if ((intr->intrinsic == nir_intrinsic_store_output) &&
             nir_intrinsic_base(intr) == drvloc) {
            assert(intr->src[0].is_ssa);
            assert(nir_src_as_const_value(intr->src[1]));
            return intr->src[0].ssa;
         }
      }
   }

   return NULL;
}

/* TODO: maybe this would be a useful helper?
 * NOTE: assumes each output is written exactly once (and unconditionally)
 * so if needed nir_lower_outputs_to_temporaries()
 */
static nir_ssa_def *
find_output(nir_shader *shader, unsigned drvloc)
{
	printf("Finding %d\n", drvloc);
   nir_ssa_def *def = NULL;
   nir_foreach_function(function, shader) {
      if (function->impl) {
         nir_foreach_block_reverse(block, function->impl) {
            nir_ssa_def *new_def = find_output_in_block(block, drvloc);
            assert(!(new_def && def));
            def = new_def;
#if !defined(DEBUG)
            /* for debug builds, scan entire shader to assert
             * if output is written multiple times.  For release
             * builds just assume all is well and bail when we
             * find first:
             */
            if (def)
               break;
#endif
         }
      }
   }

   return def;
}

static void
append_vertex_epilogue(nir_shader *shader)
{
	nir_ssa_def *gl_Position;

	/* First, find gl_Position for later pass */

	nir_foreach_variable(var, &shader->outputs) {
		if (var->data.location == VARYING_SLOT_POS)
			gl_Position = find_output(shader, var->data.driver_location);
	}

	if (!gl_Position) {
		printf("gl_Position not written in vertex shader\n");
		return;
	}

	nir_foreach_function(func, shader) {
		if (!strcmp(func->name, "main")) {
			nir_builder b;

			nir_builder_init(&b, func->impl);
			b.cursor = nir_after_cf_list(&func->impl->body);
			emit_vertex_epilogue(&b, gl_Position);
		}
	}
}


static void
emit_fragment_epilogue(compiler_context *ctx)
{
	/* See the docs for why this works */

	EMIT(alu_br_compact_cond, midgard_jmp_writeout_op_writeout, TAG_ALU_4, 0, COND_FBWRITE);
	EMIT(alu_br_compact_cond, midgard_jmp_writeout_op_writeout, TAG_ALU_4, -1, COND_FBWRITE);
}

static int
midgard_compile_shader_nir(nir_shader *nir, struct util_dynarray *compiled)
{
	compiler_context ictx = {
		.stage = nir->info.stage
	};

	compiler_context *ctx = &ictx;

	/* Assign var locations early, so the epilogue can use them if necessary */
	nir_assign_var_locations(&nir->outputs, &nir->num_outputs, glsl_type_size);
	nir_assign_var_locations(&nir->inputs, &nir->num_inputs, glsl_type_size);

	/* Lower I/O before the vertex epilogue as well */
	bool progress;
	NIR_PASS(progress, nir, nir_lower_io, nir_var_all, glsl_type_size, 0);
	NIR_PASS(progress, nir, nir_lower_var_copies);
	NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
	NIR_PASS(progress, nir, nir_lower_io, nir_var_all, glsl_type_size, 0);

	/* Append vertex epilogue before optimisation, so the epilogue itself
	 * is optimised */

	if (ctx->stage == MESA_SHADER_VERTEX)
		append_vertex_epilogue(nir);

	/* Optimisation passes */

	nir_print_shader(nir, stdout);
	optimise_nir(nir);
	nir_print_shader(nir, stdout);

	nir_foreach_function(func, nir) {
		if (!func->impl)
			continue;

		nir_foreach_block(block, func->impl) {
			util_dynarray_init(&ctx->current_block, NULL);
			ctx->ssa_constants = _mesa_hash_table_u64_create(NULL); 
			ctx->ssa_to_alias = _mesa_hash_table_u64_create(NULL); 
			ctx->register_to_ssa = _mesa_hash_table_u64_create(NULL); 
			ctx->leftover_ssa_to_alias = _mesa_set_create(NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);

			nir_foreach_instr(instr, block) {
				emit_instr(ctx, instr);
			}

			/* Workaround hardware quirk */
			defer_stores(ctx);

			/* Artefact of load_const, etc in the average case */
			inline_alu_constants(ctx);
			eliminate_constant_mov(ctx);

			/* Perform heavylifting for aliasing */
			actualise_ssa_to_alias(ctx);
			actualise_register_to_ssa(ctx);

			/* Append fragment shader epilogue (value writeout) */
			if (ctx->stage == MESA_SHADER_FRAGMENT)
				emit_fragment_epilogue(ctx);

			/* Finally, register allocation! Must be done after everything else */
			allocate_registers(ctx);

			break; /* TODO: Multi-block shaders */
		}

		break; /* TODO: Multi-function shaders */
	}

	struct util_dynarray tags;

	util_dynarray_init(compiled, NULL);
	util_dynarray_init(&tags, NULL);

	/* XXX: Workaround hardware errata where shaders must start with a
	 * load/store instruction by adding a noop load */

	int first_tag = 0;

	util_dynarray_foreach(&ctx->current_block, midgard_instruction, ins) {
		if (!ins->unused) {
			first_tag = ins->type;
			break;
		}
	}

	if (unlikely(first_tag != TAG_LOAD_STORE_4)) {
		midgard_load_store_t instruction = {
			.tag = TAG_LOAD_STORE_4,
			.word1 = 3,
			.word2 = 3
		};

		util_dynarray_append(&tags, int, compiled->size);
		util_dynarray_append(compiled, midgard_load_store_t, instruction);
	}

	/* Emit flat binary from the instruction array. Save instruction boundaries such that lookahead tags can be assigned easily */
	util_dynarray_foreach(&ctx->current_block, midgard_instruction, ins) {
		if (!ins->unused) {
			util_dynarray_append(&tags, int, compiled->size);
			ins += emit_binary_instruction(ctx, ins, compiled);
		}
	}

	/* Now just perform lookahead */
	util_dynarray_foreach(&tags, int, tag) {
		int lookahead;

		uint8_t *ins = ((uint8_t *) compiled->data) + *tag;

		if (IN_ARRAY(tag + 1, tags)) {
			uint8_t *next = ((uint8_t *) compiled->data) + *(tag + 1);

			if (!IN_ARRAY(tag + 2, tags) && IS_ALU(*next)) {
				lookahead = 1;
			} else {
				lookahead = *next;
			}
		} else {
			lookahead = 1;
		}

		*ins |= lookahead << 4;
	}

	util_dynarray_fini(&ctx->current_block);

	/* TODO: Propagate compiled code up correctly */
	return 0;
}

static void
finalise_to_disk(const char *filename, struct util_dynarray *data)
{
	FILE *fp;
	fp = fopen(filename, "wb");
	fwrite(data->data, 1, data->size, fp);
	fclose(fp);

	util_dynarray_fini(data);
}

static const nir_shader_compiler_options nir_options = {
	.lower_sub = true,
	.lower_fpow = true,
	.lower_scmp = true,
	.lower_flrp32 = true,
	.lower_flrp64 = true,
	.lower_ffract = true,
	.lower_fmod32 = true,
	.lower_fmod64 = true,
	.lower_fdiv = true,
	.lower_fsinpi = true,

	.vertex_id_zero_based = true,
	.lower_extract_byte = true,
	.lower_extract_word = true,

	/* TODO: Reenable when integer ops are understood */
	.native_integers = false
};

int main(int argc, char **argv)
{
	struct gl_shader_program *prog;
	nir_shader *nir;

	struct standalone_options options = {
		.glsl_version = 140,
		.do_link = true,
	};

	if (argc != 3) {
		printf("Must pass exactly two GLSL files\n");
		exit(1);
	}

	prog = standalone_compile_shader(&options, 2, &argv[1]);
	prog->_LinkedShaders[MESA_SHADER_FRAGMENT]->Program->info.stage = MESA_SHADER_FRAGMENT;

	struct util_dynarray compiled;

	nir = glsl_to_nir(prog, MESA_SHADER_VERTEX, &nir_options);
	midgard_compile_shader_nir(nir, &compiled);
	finalise_to_disk("/dev/shm/vertex.bin", &compiled);

	nir = glsl_to_nir(prog, MESA_SHADER_FRAGMENT, &nir_options);
	midgard_compile_shader_nir(nir, &compiled);
	finalise_to_disk("/dev/shm/fragment.bin", &compiled);
}
