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

static void
emit_load_const(nir_load_const_instr *instr)
{
	nir_ssa_def def = instr->def;

	if (def.num_components == 4 && def.bit_size == 32) {
		printf("Vector ALU with inline constant\n");
		printf("<%f, %f, %f, %f>\n",
			instr->value.f32[0],
			instr->value.f32[1],
			instr->value.f32[2],
			instr->value.f32[3]);

	} else {
		printf("Unknown configuration in load_const %d x %d\n", def.num_components, def.bit_size);
	}
}

static void
get_src(nir_src src)
{
	if (src.is_ssa) {
		printf("SSA index: %d\n", src.ssa->index);
	} else {
		printf("Reg offset: %d\n", src.reg.base_offset);
	}
}

static void
get_dest(nir_dest src)
{
	if (src.is_ssa) {
		printf("SSA index: %d\n", src.ssa.index);
	} else {
		printf("Reg offset: %d\n", src.reg.base_offset);
	}
}

static void
emit_intrinsic(nir_intrinsic_instr *instr)
{
        nir_const_value *const_offset;
        unsigned offset;

	switch(instr->intrinsic) {
		case nir_intrinsic_load_uniform:
			const_offset = nir_src_as_const_value(instr->src[0]);
			assert (const_offset && "no indirect inputs");

			offset = nir_intrinsic_base(instr) + const_offset->u32[0];
			assert(offset % 4 == 0);
			offset = offset / 4;

			printf("Load uniform offset %d\n", offset);
			get_dest(instr->dest);

			break;

		case nir_intrinsic_store_output:
			const_offset = nir_src_as_const_value(instr->src[1]);
			assert(const_offset && "no indirect outputs");

			offset = nir_intrinsic_base(instr) + const_offset->u32[0];
			offset = offset * 4 + nir_intrinsic_component(instr);

			get_src(instr->src[0]);

			printf("Store output to offset %d\n", offset);

			break;

		default:
			printf ("Unhandled intrinsic\n");
	}
}

static void
emit_instr(struct nir_instr *instr)
{
	nir_print_instr(instr, stdout);
	putchar('\n');

	switch(instr->type) {
		case nir_instr_type_load_const:
			emit_load_const(nir_instr_as_load_const(instr));
			break;

		case nir_instr_type_intrinsic:
			emit_intrinsic(nir_instr_as_intrinsic(instr));
			break;
		default:
			printf("Unhandled instruction type\n");
			break;
	}
}

static int
midgard_compile_shader_nir(nir_shader *nir)
{
	optimise_nir(nir);

	nir_foreach_function(func, nir) {
		if (!func->impl)
			continue;

		nir_foreach_block(block, func->impl) {
			nir_foreach_instr(instr, block) {
				emit_instr(instr);
			}
		}
	}

	nir_print_shader(nir, stdout);
	return 0;
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

	nir = glsl_to_nir(prog, MESA_SHADER_FRAGMENT, &nir_options);
	midgard_compile_shader_nir(nir);

	nir = glsl_to_nir(prog, MESA_SHADER_VERTEX, &nir_options);
	midgard_compile_shader_nir(nir);
}
