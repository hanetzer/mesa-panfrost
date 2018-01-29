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

/* Generic in-memory data type repesenting a single logical instruction, rather
 * than a single instruction group. This is the preferred form for code gen.
 * Multiple midgard_insturctions will later be combined during scheduling,
 * though this is not represented in this structure.  Its format bridges
 * the low-level binary representation with the higher level semantic meaning.
 */

typedef struct midgard_instruction {
	midgard_word_type type; /* ALU, load/store, texture */

	/* Special fields for an ALU instruction */
	bool vector; 
	alu_register_word registers;

	bool has_constants;
	float constants[4];

	union {
		midgard_load_store_word_t load_store;
		midgard_scalar_alu_t scalar_alu;
		midgard_vector_alu_t vector_alu;
		/* TODO Texture */
	};
} midgard_instruction;

/* Helpers to generate midgard_instruction's using macro magic, since every
 * driver seems to do it that way */

#define M_LOAD_STORE(name) \
	static midgard_instruction m_##name(unsigned reg, unsigned address) { \
		midgard_instruction i = { \
			.type = TAG_LOAD_STORE_4, \
			.load_store = { \
				.op = midgard_op_##name, \
				.mask = 0xF, \
				.swizzle = SWIZZLE(COMPONENT_X, COMPONENT_Y, COMPONENT_Z, COMPONENT_W), \
				.reg = reg, \
				.address = address \
			} \
		}; \
		\
		return i; \
	}

const midgard_vector_alu_src_t blank_alu_src = {
	.abs = 0,
	.negate = 0,
	.rep_low = 0,
	.rep_high = 0,
	.half = 0,
	.swizzle = SWIZZLE(COMPONENT_X, COMPONENT_Y, COMPONENT_Z, COMPONENT_W)
};

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
m_alu_vector(midgard_alu_op_e op, unsigned reg1, midgard_vector_alu_src_t mod1, unsigned reg2, midgard_vector_alu_src_t mod2, unsigned reg3)
{
	midgard_instruction ins = {
		.type = TAG_ALU_4,
		.registers = {
			.input1_reg = reg1,
			.input2_reg = reg2,
			.output_reg = reg3
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

#define M_ALU_VECTOR_1(name) \
	static midgard_instruction m_##name(unsigned src, midgard_vector_alu_src_t mod1, unsigned dest) { \
		return m_alu_vector(midgard_alu_op_##name, src, mod1, REGISTER_UNUSED, blank_alu_src, dest); \
	}

#define M_ALU_VECTOR_2(name) \
	static midgard_instruction m_##name(unsigned src1, midgard_vector_alu_src_t mod1, unsigned src2, midgard_vector_alu_src_t mod2, unsigned dest) { \
		return m_alu_vector(midgard_alu_op_##name, src1, mod1, src2, mod2, dest); \
	}

M_LOAD_STORE(ld_st_noop);
M_LOAD_STORE(load_attr_16);
M_LOAD_STORE(load_attr_32);
M_LOAD_STORE(load_vary_16);
M_LOAD_STORE(load_vary_32);
M_LOAD_STORE(load_uniform_16);
M_LOAD_STORE(load_uniform_32);
M_LOAD_STORE(store_vary_16);
M_LOAD_STORE(store_vary_32);

M_ALU_VECTOR_2(fadd);
M_ALU_VECTOR_2(fmul);
M_ALU_VECTOR_2(fmin);
M_ALU_VECTOR_2(fmax);
M_ALU_VECTOR_1(fmov);
M_ALU_VECTOR_1(ffloor);
M_ALU_VECTOR_1(fceil);
//M_ALU_VECTOR_2(fdot3);
//M_ALU_VECTOR_2(fdot3r);
//M_ALU_VECTOR_2(fdot4);
//M_ALU_VECTOR_2(freduce);
M_ALU_VECTOR_2(iadd);
M_ALU_VECTOR_2(isub);
M_ALU_VECTOR_2(imul);
M_ALU_VECTOR_2(imov);
M_ALU_VECTOR_2(feq);
M_ALU_VECTOR_2(fne);
M_ALU_VECTOR_2(flt);
//M_ALU_VECTOR_2(fle);
M_ALU_VECTOR_1(f2i);
M_ALU_VECTOR_2(ieq);
M_ALU_VECTOR_2(ine);
M_ALU_VECTOR_2(ilt);
//M_ALU_VECTOR_2(ile);
//M_ALU_VECTOR_2(csel);
M_ALU_VECTOR_1(i2f);
//M_ALU_VECTOR_2(fatan_pt2);
M_ALU_VECTOR_1(frcp);
M_ALU_VECTOR_1(frsqrt);
M_ALU_VECTOR_1(fsqrt);
M_ALU_VECTOR_1(fexp2);
M_ALU_VECTOR_1(flog2);
M_ALU_VECTOR_1(fsin);
M_ALU_VECTOR_1(fcos);
//M_ALU_VECTOR_2(fatan_pt1);

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

		/* Midgard does not support I/O->I/O copies; lower these */
		NIR_PASS(progress, nir, nir_lower_var_copies);

		NIR_PASS(progress, nir, nir_lower_io, nir_var_all, glsl_type_size, 0);
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

	NIR_PASS_V(nir, nir_convert_from_ssa, false);
}

/* TODO: REGISTER ALLOCATION!!! */

static unsigned
ssa_to_register(nir_ssa_def *def)
{
	return 8 + def->index;
}

static unsigned
resolve_source_register(nir_src src)
{
	if (src.is_ssa) {
		printf("--TODO: RESOLVE REGISTER!--\n");
		return ssa_to_register(src.ssa);
	} else {
		return src.reg.reg->index;
	}
}

static unsigned
resolve_destination_register(nir_dest src)
{
	if (src.is_ssa) {
		printf("--TODO: RESOLVE REGISTER!--\n");
		return ssa_to_register(&src.ssa);
	} else {
		return src.reg.reg->index;
	}
}

static void
emit_load_const(compiler_context *ctx, nir_load_const_instr *instr)
{
	nir_ssa_def def = instr->def;

	midgard_instruction ins = m_fmov(REGISTER_CONSTANT, blank_alu_src, ssa_to_register(&def));
	attach_constants(&ins, &instr->value.f32);
	util_dynarray_append(&ctx->current_block, midgard_instruction, ins);
}

#define EMIT_ALU_CASE_1(op_nir, op_midgard) \
	case nir_op_##op_nir: \
		ins = m_##op_midgard( \
			resolve_source_register(instr->src[0].src), \
			n2m_alu_modifiers(&instr->src[0]), \
			resolve_destination_register(instr->dest.dest)); \
		util_dynarray_append(&ctx->current_block, midgard_instruction, ins); \
		break;

#define EMIT_ALU_CASE_2(op_nir, op_midgard) \
	case nir_op_##op_nir: \
		ins = m_##op_midgard( \
			resolve_source_register(instr->src[0].src), \
			n2m_alu_modifiers(&instr->src[0]), \
			resolve_source_register(instr->src[1].src), \
			n2m_alu_modifiers(&instr->src[1]), \
			resolve_destination_register(instr->dest.dest)); \
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

			reg = resolve_destination_register(instr->dest);

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

			reg = resolve_source_register(instr->src[0]);

			util_dynarray_append(&ctx->current_block, midgard_instruction, m_store_vary_32(reg, offset));

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

/* Midgard prefetches instruction types, so during emission we need to
 * lookahead too. Unless this is the last instruction, in which we return 1. Or
 * if this is the second to last and the last is an ALU, then it's also 1... */

#define IN_ARRAY(n, arr) ((void*)n < (void*)(arr.data + arr.size))
#define IS_ALU(tag) (tag == TAG_ALU_4 || tag == TAG_ALU_8 ||  \
		     tag == TAG_ALU_12 || tag == TAG_ALU_16)

static int
get_lookahead_type(struct util_dynarray block, midgard_instruction *ins)
{
	midgard_instruction *n = ins + 1;

	if (IN_ARRAY(n, block)) {
		if (!IN_ARRAY(n + 1, block) && IS_ALU(n->type))
			return 1;

		return n->type;
	}

	return 1;
}

#define EMIT_AND_COUNT(type, val) util_dynarray_append(emission, type, val); \
				  bytes_emitted += sizeof(type)

static void
emit_binary_instruction(compiler_context *ctx, midgard_instruction *ins, struct util_dynarray *emission)
{
	if (ins->type == TAG_ALU_4 && ins->has_constants) ins->type = TAG_ALU_8;
	uint8_t tag = ins->type | (get_lookahead_type(ctx->current_block, ins) << 4);

	switch(ins->type) {
		case TAG_ALU_4:
		case TAG_ALU_8:
		case TAG_ALU_12:
		case TAG_ALU_16: {
			uint32_t control = tag;
			size_t bytes_emitted = 0;
			
			/* TODO: Determine which units need to be enabled */

			if (ins->vector) {
				control |= ALU_ENAB_VEC_ADD;

				/* TODO */
				EMIT_AND_COUNT(uint32_t, control);
				EMIT_AND_COUNT(alu_register_word, ins->registers);
				EMIT_AND_COUNT(midgard_vector_alu_t, ins->vector_alu);

			} else {
				control |= ALU_ENAB_SCAL_ADD;
			}

			/* Pad ALU op to nearest word */

			if (bytes_emitted & 15)
				util_dynarray_grow(emission, 16 - (bytes_emitted & 15));

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
			/* Load store instructions have two words at once. We
			 * only have one queued up, so we need to NOP pad.
			 * TODO: Make less bad. */

			midgard_load_store_word_t actual = ins->load_store;
			midgard_load_store_word_t fake = m_ld_st_noop(0, 0).load_store;

			midgard_load_store_t instruction = {
				.tag = tag,
				.word1 = *(uint64_t*) &actual,
				.word2 = *(uint64_t*) &fake
			};

			util_dynarray_append(emission, midgard_load_store_t, instruction);

			break;
		}

		default:
			printf("Unknown midgard instruction type\n");
			break;
	}
}

static int
midgard_compile_shader_nir(nir_shader *nir, struct util_dynarray *compiled)
{
	compiler_context ctx = {
		.stage = nir->info.stage
	};

	optimise_nir(nir);
	nir_print_shader(nir, stdout);

	nir_foreach_function(func, nir) {
		if (!func->impl)
			continue;

		nir_foreach_block(block, func->impl) {
			util_dynarray_init(&ctx.current_block, NULL);

			nir_foreach_instr(instr, block) {
				emit_instr(&ctx, instr);
			}

			break; /* TODO: Multi-block shaders */
		}

		break; /* TODO: Multi-function shaders */
	}

	util_dynarray_init(compiled, NULL);

	util_dynarray_foreach(&ctx.current_block, midgard_instruction, ins) {
		emit_binary_instruction(&ctx, ins, compiled);
	}

	util_dynarray_fini(&ctx.current_block);

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
		.fuse_ffma = true,
		.native_integers = true,
		.vertex_id_zero_based = true,
		.lower_extract_byte = true,
		.lower_extract_word = true,
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
	finalise_to_disk("vertex.bin", &compiled);

	nir = glsl_to_nir(prog, MESA_SHADER_FRAGMENT, &nir_options);
	midgard_compile_shader_nir(nir, &compiled);
	finalise_to_disk("fragment.bin", &compiled);
}
