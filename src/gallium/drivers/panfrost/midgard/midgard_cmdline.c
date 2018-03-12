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

#define EMIT(op, ...) util_dynarray_append(&(ctx->current_block), midgard_instruction, m_##op(__VA_ARGS__));

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

/* Used for encoding the unused source of 1-op instructions */
const midgard_vector_alu_src_t zero_alu_src = { 0 };

static midgard_vector_alu_src_t
n2m_alu_modifiers(nir_alu_src *src)
{
	midgard_vector_alu_src_t alu_src = {
		.abs = src->abs,
		.negate = src->negate,
		.rep_low = 0,
		.rep_high = 0,
		.half = 0,
		.swizzle = SWIZZLE_FROM_ARRAY(src->swizzle)
	};

	return alu_src;
}

static unsigned
alu_src_to_unsigned(midgard_vector_alu_src_t src)
{
	unsigned u;
	memcpy(&u, &src, sizeof(src));
	return u;
}

static midgard_instruction
m_alu_vector(midgard_alu_op_e op, int unit, unsigned src0, midgard_vector_alu_src_t mod1, unsigned src1, midgard_vector_alu_src_t mod2, unsigned dest, bool literal_out)
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
			.outmod = midgard_outmod_none,
			.mask = 0xFF,
			.src1 = alu_src_to_unsigned(mod1),
			.src2 = alu_src_to_unsigned(mod2)
		},
	};

	return ins;
}

#define M_ALU_VECTOR_1(unit, name) \
	static midgard_instruction m_##name(unsigned src, midgard_vector_alu_src_t mod1, unsigned dest, bool literal) { \
		return m_alu_vector(midgard_alu_op_##name, ALU_ENAB_VEC_##unit, -1, zero_alu_src, src, mod1, dest, literal); \
	}

#define M_ALU_VECTOR_2(unit, name) \
	static midgard_instruction m_##name(unsigned src1, midgard_vector_alu_src_t mod1, unsigned src2, midgard_vector_alu_src_t mod2, unsigned dest, bool literal) { \
		return m_alu_vector(midgard_alu_op_##name, ALU_ENAB_VEC_##unit, src1, mod1, src2, mod2, dest, literal); \
	}

/* load/store instructions have both 32-bit and 16-bit variants, depending on
 * whether we are using vectors composed of highp or mediump. At the moment, we
 * don't support half-floats -- this requires changes in other parts of the
 * compiler -- therefore the 16-bit versions are commented out. */

M_LOAD(ld_st_noop);
//M_LOAD(load_attr_16);
M_LOAD(load_attr_32);
//M_LOAD(load_vary_16);
M_LOAD(load_vary_32);
//M_LOAD(load_uniform_16);
M_LOAD(load_uniform_32);
//M_STORE(store_vary_16);
M_STORE(store_vary_32);

M_ALU_VECTOR_2(MUL, fadd);
M_ALU_VECTOR_2(MUL, fmul);
M_ALU_VECTOR_2(MUL, fmin);
M_ALU_VECTOR_2(MUL, fmax);
M_ALU_VECTOR_1(MUL, fmov);
M_ALU_VECTOR_1(MUL, ffloor);
M_ALU_VECTOR_1(MUL, fceil);
//M_ALU_VECTOR_2(fdot3);
//M_ALU_VECTOR_2(fdot3r);
//M_ALU_VECTOR_2(fdot4);
//M_ALU_VECTOR_2(freduce);
M_ALU_VECTOR_2(MUL, iadd);
M_ALU_VECTOR_2(MUL, isub);
M_ALU_VECTOR_2(MUL, imul);
M_ALU_VECTOR_2(MUL, imov);
M_ALU_VECTOR_2(MUL, feq);
M_ALU_VECTOR_2(MUL, fne);
M_ALU_VECTOR_2(MUL, flt);
//M_ALU_VECTOR_2(fle);
M_ALU_VECTOR_1(MUL, f2i);
M_ALU_VECTOR_2(MUL, ieq);
M_ALU_VECTOR_2(MUL, ine);
M_ALU_VECTOR_2(MUL, ilt);
//M_ALU_VECTOR_2(ile);
//M_ALU_VECTOR_2(csel);
M_ALU_VECTOR_1(MUL, i2f);
//M_ALU_VECTOR_2(fatan_pt2);
M_ALU_VECTOR_1(MUL, frcp);
M_ALU_VECTOR_1(MUL, frsqrt);
M_ALU_VECTOR_1(MUL, fsqrt);
M_ALU_VECTOR_1(MUL, fexp2);
M_ALU_VECTOR_1(MUL, flog2);
M_ALU_VECTOR_1(MUL, fsin);
M_ALU_VECTOR_1(MUL, fcos);
//M_ALU_VECTOR_2(fatan_pt1);

M_ALU_VECTOR_1(MUL, synthwrite);

/* TODO: Expand into constituent parts since we do understand how this works,
 * no? */

static midgard_instruction
m_alu_br_compact_cond(midgard_jmp_writeout_op_e op, unsigned tag, signed offset, unsigned cond)
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

	struct hash_table_u64 *ssa_constants;
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

	do {
		progress = false;

		NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
		NIR_PASS(progress, nir, nir_lower_vec_to_movs);

		/* Midgard does not support I/O->I/O copies; lower these */
		NIR_PASS(progress, nir, nir_lower_var_copies);

		//NIR_PASS(progress, nir, nir_lower_io, nir_var_all, glsl_type_size, 0);
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
}

static void
emit_load_const(compiler_context *ctx, nir_load_const_instr *instr)
{
	nir_ssa_def def = instr->def;

	float *v = ralloc_array(ctx->ssa_constants, float, 4);
	memcpy(v, &instr->value.f32, 4 * sizeof(float));
	_mesa_hash_table_u64_insert(ctx->ssa_constants, def.index, v);

	midgard_instruction ins = m_fmov(REGISTER_CONSTANT, blank_alu_src, def.index, false);
	attach_constants(&ins, &instr->value.f32);
	util_dynarray_append(&ctx->current_block, midgard_instruction, ins);
}

#define EMIT_ALU_CASE_1(op_nir, op_midgard) \
	case nir_op_##op_nir: \
		ins = m_##op_midgard( \
			instr->src[0].src.ssa->index, \
			n2m_alu_modifiers(&instr->src[0]), \
			instr->dest.dest.ssa.index, \
			false); \
		util_dynarray_append(&ctx->current_block, midgard_instruction, ins); \
		break;

#define EMIT_ALU_CASE_2(op_nir, op_midgard) \
	case nir_op_##op_nir: \
		ins = m_##op_midgard( \
			instr->src[0].src.ssa->index, \
			n2m_alu_modifiers(&instr->src[0]), \
			instr->src[1].src.ssa->index, \
			n2m_alu_modifiers(&instr->src[1]), \
			instr->dest.dest.ssa.index, \
			false); \
		util_dynarray_append(&ctx->current_block, midgard_instruction, ins); \
		break;


static void
emit_alu(compiler_context *ctx, nir_alu_instr *instr)
{
	midgard_instruction ins;

	/* Most Midgard ALU ops have a 1:1 correspondance to NIR ops; these are
	 * supported. A few do not and are therefore commented and TODO to
	 * figure out what code paths would generate these. Also, there are a
	 * number of NIR ops which Midgard does not support and need to be
	 * lowered, also TODO */

	switch(instr->op) {
		EMIT_ALU_CASE_2(fadd, fadd);
		EMIT_ALU_CASE_2(fmul, fmul);
		EMIT_ALU_CASE_2(fmin, fmin);
		EMIT_ALU_CASE_2(fmax, fmax);
		EMIT_ALU_CASE_1(fmov, fmov);
		EMIT_ALU_CASE_1(ffloor, ffloor);
		EMIT_ALU_CASE_1(fceil, fceil);
		//EMIT_ALU_CASE_2(fdot3);
		//EMIT_ALU_CASE_2(fdot3r);
		//EMIT_ALU_CASE_2(fdot4);
		//EMIT_ALU_CASE_2(freduce);
		EMIT_ALU_CASE_2(iadd, iadd);
		EMIT_ALU_CASE_2(isub, isub);
		EMIT_ALU_CASE_2(imul, imul);
		EMIT_ALU_CASE_2(imov, imov);
		EMIT_ALU_CASE_2(feq, feq);
		EMIT_ALU_CASE_2(fne, fne);
		EMIT_ALU_CASE_2(flt, flt);
		//EMIT_ALU_CASE_2(fle);
		EMIT_ALU_CASE_1(f2i32, f2i);
		EMIT_ALU_CASE_1(f2u32, f2i);
		EMIT_ALU_CASE_2(ieq, ieq);
		EMIT_ALU_CASE_2(ine, ine);
		EMIT_ALU_CASE_2(ilt, ilt);
		//EMIT_ALU_CASE_2(ile);
		//EMIT_ALU_CASE_2(csel, csel);
		EMIT_ALU_CASE_1(i2f32, i2f);
		EMIT_ALU_CASE_1(u2f32, i2f);
		//EMIT_ALU_CASE_2(fatan_pt2);
		EMIT_ALU_CASE_1(frcp, frcp);
		EMIT_ALU_CASE_1(frsq, frsqrt);
		EMIT_ALU_CASE_1(fsqrt, fsqrt);
		EMIT_ALU_CASE_1(fexp2, fexp2);
		EMIT_ALU_CASE_1(flog2, flog2);

		// Input needs to be divided by pi due to Midgard weirdness We
		// define special NIR ops, fsinpi and fcospi, that include the
		// division correctly, supplying appropriately lowering passes.
		// That way, the division by pi can take advantage of constant
		// folding, algebraic simplifications, and so forth.

		EMIT_ALU_CASE_1(fsinpi, fsin);
		EMIT_ALU_CASE_1(fcospi, fcos);

		//EMIT_ALU_CASE_2(fatan_pt1);

		default:
			printf("Unhandled ALU op\n");
			break;
	}
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

			util_dynarray_append(&ctx->current_block, midgard_instruction, 
				instr->intrinsic == nir_intrinsic_load_uniform ? 
					m_load_uniform_32(reg, offset) :
				ctx->stage == MESA_SHADER_FRAGMENT ?
					m_load_vary_32(reg, offset) :
					m_load_attr_32(reg, offset));

			break;

		case nir_intrinsic_store_output:
			const_offset = nir_src_as_const_value(instr->src[1]);
			assert(const_offset && "no indirect outputs");

			offset = nir_intrinsic_base(instr) + const_offset->u32[0];
			offset = offset * 4 + nir_intrinsic_component(instr);

			reg = instr->src[0].ssa->index;

			util_dynarray_append(&ctx->current_block, midgard_instruction, m_store_vary_32(reg, offset));

			break;

		case nir_intrinsic_store_var: {
			nir_variable *out = instr->variables[0]->var;
			reg = instr->src[0].ssa->index;

			if (out->data.mode == nir_var_shader_out) {
				if (out->data.location == FRAG_RESULT_COLOR) {
					/* gl_FragColor is not emitted with
					 * load/store instructions. Instead, it
					 * gets plonked into r0 at the end of
					 * the shader and we do the framebuffer
					 * writeout dance. TODO: Defer writes */

					EMIT(fmov, reg, blank_alu_src, 0, true);
					break;
				}
			}

			/* Worst case, emit a store varying and at least
			 * that'll show up in the disassembly */

			util_dynarray_append(&ctx->current_block, midgard_instruction, m_store_vary_32(reg, offset));
	      }


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

static void
allocate_registers(compiler_context *ctx)
{
	util_dynarray_foreach(&ctx->current_block, midgard_instruction, ins) {
		ssa_args args = ins->ssa_args;

		switch (ins->type) {
			case TAG_ALU_4:
				ins->registers.output_reg = args.dest;
				ins->registers.input1_reg = (args.src0 >= 0) ? args.src0 : REGISTER_UNUSED;
				ins->registers.input2_reg = args.src1;
				
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

			while ((ins + index) &&
				(ins + index)->type == TAG_ALU_4 &&
			       	(ins + index)->unit > last_unit) {

				midgard_instruction *ains = ins + index;

				control |= ains->unit;
				last_unit = ains->unit;

				if (ains->vector) {
					memcpy(&register_words[register_words_count++], &ains->registers, sizeof(ains->registers));
					bytes_emitted += sizeof(alu_register_word);

					body_size[body_words_count] = sizeof(midgard_vector_alu_t);
					memcpy(&body_words[body_words_count++], &ains->vector_alu, sizeof(ains->vector_alu));
					bytes_emitted += sizeof(midgard_vector_alu_t);

				} else if (ains->compact_branch) {
					body_size[body_words_count] = sizeof(ains->br_compact);
					memcpy(&body_words[body_words_count++], &ains->br_compact, sizeof(ains->br_compact));
					bytes_emitted += sizeof(ains->br_compact);
				} else {
					/* TODO: Scalar ops */
					printf("Scalar the huh?\n");
				}

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
		break; \
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

		/* If this is a literal move (used in tandem with I/O), it cannot be removed */
		
		if (move->ssa_args.literal_out) continue;

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

static int
midgard_compile_shader_nir(nir_shader *nir, struct util_dynarray *compiled)
{
	compiler_context ictx = {
		.stage = nir->info.stage
	};

	compiler_context *ctx = &ictx;

	optimise_nir(nir);
	nir_print_shader(nir, stdout);

	nir_foreach_function(func, nir) {
		if (!func->impl)
			continue;

		nir_foreach_block(block, func->impl) {
			util_dynarray_init(&ctx->current_block, NULL);
			ctx->ssa_constants = _mesa_hash_table_u64_create(NULL); 

			nir_foreach_instr(instr, block) {
				emit_instr(ctx, instr);
			}

			inline_alu_constants(ctx);

			/* Artefact of load_const in the average case */
			eliminate_constant_mov(ctx);

			/* Append fragment shader epilogue (value writeout) */
			EMIT(alu_br_compact_cond, midgard_jmp_writeout_op_writeout, TAG_ALU_4, 0, COND_FBWRITE);

			/* Errata workaround -- the above write can sometimes
			 * fail -.- */

			EMIT(fmov, 0, blank_alu_src, 0, true);
			EMIT(alu_br_compact_cond, midgard_jmp_writeout_op_writeout, TAG_ALU_4, -1, COND_FBWRITE);

			/* Finally, register allocation! Must be done after everything else */
			allocate_registers(ctx);

			break; /* TODO: Multi-block shaders */
		}

		break; /* TODO: Multi-function shaders */
	}

	struct util_dynarray tags;

	util_dynarray_init(compiled, NULL);
	util_dynarray_init(&tags, NULL);

	/* XXX: Workaround hardware errata where shaders must start with a load/store instruction */
	if (unlikely(*((uint8_t *) ctx->current_block.data) != TAG_LOAD_STORE_4)) {
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
	.lower_fpow = true,
	.lower_fsat = true,
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
	//.native_integers = true,
};

int main(int argc, char **argv)
{
	struct gl_shader_program *prog;
	nir_shader *nir;

	struct standalone_options options = {
		.glsl_version = 140,
		.do_link = true,
	};

	if (argc != 2) {
		printf("Must pass exactly two GLSL files\n");
		exit(1);
	}

	prog = standalone_compile_shader(&options, 1, &argv[1]);
	prog->_LinkedShaders[MESA_SHADER_FRAGMENT]->Program->info.stage = MESA_SHADER_FRAGMENT;

	struct util_dynarray compiled;

#if 0
	nir = glsl_to_nir(prog, MESA_SHADER_VERTEX, &nir_options);
	midgard_compile_shader_nir(nir, &compiled);
	finalise_to_disk("/dev/shm/vertex.bin", &compiled);
#endif

	nir = glsl_to_nir(prog, MESA_SHADER_FRAGMENT, &nir_options);
	midgard_compile_shader_nir(nir, &compiled);
	finalise_to_disk("/dev/shm/fragment.bin", &compiled);
}
