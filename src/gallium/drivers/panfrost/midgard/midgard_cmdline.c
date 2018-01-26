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

/* Generic in-memory data type repesenting an instruction.  Its format bridges
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

static unsigned
alu_src_to_unsigned(midgard_vector_alu_src_t src)
{
	unsigned u;
	memcpy(&u, &src, sizeof(src));
	return u;
}

static midgard_instruction
m_alu_vector(midgard_alu_op_e op, unsigned reg1, unsigned reg2, unsigned reg3)
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
			.src1 = alu_src_to_unsigned(blank_alu_src),
			.src2 = alu_src_to_unsigned(blank_alu_src)
		},
	};

	return ins;
}

#define M_ALU_VECTOR_1(name) \
	static midgard_instruction m_##name(unsigned src, unsigned dest) { \
		return m_alu_vector(midgard_alu_op_##name, src, REGISTER_UNUSED, dest); \
	}

#define M_ALU_VECTOR_2(name) \
	static midgard_instruction m_##name(unsigned src1, unsigned src2, unsigned dest) { \
		return m_alu_vector(midgard_alu_op_##name, src1, src2, dest); \
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

M_ALU_VECTOR_1(fmov);
M_ALU_VECTOR_2(fadd);

static void
attach_constants(midgard_instruction *ins, void *constants)
{
	ins->has_constants = true;
	memcpy(&ins->constants, constants, 16);
}

typedef struct compiler_context {
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
	/* TODO: Proper optimisation loop */

	NIR_PASS_V(nir, nir_lower_vars_to_ssa);

	/* Midgard does not support I/O->I/O copies; lower these */
	NIR_PASS_V(nir, nir_lower_var_copies);

	NIR_PASS_V(nir, nir_lower_io, nir_var_all, glsl_type_size, 0);
	NIR_PASS_V(nir, nir_copy_prop);
	NIR_PASS_V(nir, nir_opt_dce);
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
		printf("SSA index: %d\n", src.ssa->index);
		printf("--TODO: RESOLVE REGISTER!--\n");
		return ssa_to_register(src.ssa);
	} else {
		printf("Reg offset: %d\n", src.reg.base_offset);
		return 0;
	}
}

static unsigned
resolve_destination_register(nir_dest src)
{
	if (src.is_ssa) {
		printf("SSA index: %d\n", src.ssa.index);
		printf("--TODO: RESOLVE REGISTER!--\n");
		return ssa_to_register(&src.ssa);
	} else {
		printf("Reg offset: %d\n", src.reg.base_offset);
		return 0;
	}
}

static void
emit_load_const(compiler_context *ctx, nir_load_const_instr *instr)
{
	nir_ssa_def def = instr->def;

	if (def.num_components == 4 && def.bit_size == 32) {
		midgard_instruction ins = m_fmov(REGISTER_CONSTANT, ssa_to_register(&def));
		attach_constants(&ins, &instr->value.f32);
		util_dynarray_append(&ctx->current_block, midgard_instruction, ins);
	} else {
		printf("Unknown configuration in load_const %d x %d\n", def.num_components, def.bit_size);
	}
}

static void
emit_alu(compiler_context *ctx, nir_alu_instr *instr)
{
	midgard_instruction ins;

	switch(instr->op) {
		case nir_op_fadd:
			ins = m_fadd(resolve_source_register(instr->src[0].src), resolve_source_register(instr->src[1].src), resolve_destination_register(instr->dest.dest));
			util_dynarray_append(&ctx->current_block, midgard_instruction, ins);
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
			const_offset = nir_src_as_const_value(instr->src[0]);
			assert (const_offset && "no indirect inputs");

			offset = nir_intrinsic_base(instr) + const_offset->u32[0];
			assert(offset % 4 == 0);
			offset = offset / 4;

			reg = resolve_destination_register(instr->dest);

			util_dynarray_append(&ctx->current_block, midgard_instruction, m_load_uniform_32(reg, offset));

			break;

		case nir_intrinsic_store_output:
			const_offset = nir_src_as_const_value(instr->src[1]);
			assert(const_offset && "no indirect outputs");

			offset = nir_intrinsic_base(instr) + const_offset->u32[0];
			offset = offset * 4 + nir_intrinsic_component(instr);

			reg = resolve_source_register(instr->src[0]);

			printf("Store output to offset %d\n", offset);
			util_dynarray_append(&ctx->current_block, midgard_instruction, m_store_vary_32(reg, offset));

			break;

		default:
			printf ("Unhandled intrinsic\n");
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
	if (ins->type == TAG_ALU_4) ins->type = TAG_ALU_8;
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

			printf("ALU instruction\n");
			break;
		 }

		case TAG_LOAD_STORE_4: {
			printf("Load store\n");
			printf("Op: %d\n", ins->load_store.op);

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
	compiler_context ctx;

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
