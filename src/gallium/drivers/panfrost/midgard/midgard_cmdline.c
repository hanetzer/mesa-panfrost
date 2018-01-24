/*
 * Copyright (C) 2018 Alyssa Rosenzweig <alyssa@rosenzweig.io>
 * Copyright (C) 2014 Rob Clark <robclark@freedesktop.org>
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
 *
 * Authors:
 *    Alyssa Rosenzweig <alyssa@rosenzweig.io>
 *    Rob Clark <robclark@freedesktop.org>
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
midgard_compile_shader_nir(nir_shader *nir)
{
	nir_foreach_function(func, nir) {
		if (!func->impl)
			continue;

		nir_foreach_block(block, func->impl) {
			nir_foreach_instr(instr, block) {
				nir_print_instr(instr, stdout);
				putchar('\n');
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

	if (argc != 2) {
		printf("Must pass exactly one GLSL file\n");
		exit(1);
	}

	prog = standalone_compile_shader(&options, 1, &argv[1]);
	prog->_LinkedShaders[MESA_SHADER_FRAGMENT]->Program->info.stage = MESA_SHADER_FRAGMENT;
	nir = glsl_to_nir(prog, MESA_SHADER_FRAGMENT, &nir_options);

	midgard_compile_shader_nir(nir);
}
