/*
 * Copyright (C) 2014 Rob Clark <robclark@freedesktop.org>
 * Copyright (C) 2018 Alyssa Rosenzweig <alyssa@rosenzweig.io>
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
 *    Rob Clark <robclark@freedesktop.org>
 *    Alyssa Rosenzweig <alyssa@rosenzweig.io>
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

static int
midgard_compile_shader_nir(nir_shader *nir)
{
	printf("TODO: Compile from nir :)\n");

	nir_foreach_function(func, nir) {
		printf("Function: %s\n", func->name);

		nir_function_impl *impl = func->impl;

		if (!impl) {
			printf("No implementation?\n");
			break;
		}

		nir_foreach_block(block, impl) {
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

static nir_shader *
load_glsl(unsigned num_files, char* const* files, gl_shader_stage stage)
{
	static const struct standalone_options options = {
			.glsl_version = 140,
			.do_link = true,
	};
	struct gl_shader_program *prog;

	prog = standalone_compile_shader(&options, num_files, files);
	if (!prog)
		errx(1, "couldn't parse `%s'", files[0]);

	nir_shader *nir = glsl_to_nir(prog, stage, &nir_options);

	return nir;
}

static void print_usage(void)
{
	printf("Usage: midgard_compiler [OPTIONS]... <(file.vert | file.frag)*>\n");
}

int main(int argc, char **argv)
{
	int ret = 0, n = 1;
	char *filenames[2];
	int num_files = 0;
	unsigned stage = 0;

	while (n < argc) {
		char *filename = argv[n];
		char *ext = rindex(filename, '.');

		if (strcmp(ext, ".frag") == 0) {
			if (num_files >= ARRAY_SIZE(filenames))
				errx(1, "too many GLSL files");
			stage = MESA_SHADER_FRAGMENT;
		} else if (strcmp(ext, ".vert") == 0) {
			if (num_files >= ARRAY_SIZE(filenames))
				errx(1, "too many GLSL files");
			stage = MESA_SHADER_VERTEX;
		} else {
			print_usage();
			return -1;
		}

		filenames[num_files++] = filename;

		n++;
	}

	nir_shader *nir;

	if (num_files > 0) {
		nir = load_glsl(num_files, filenames, stage);
	} else {
		print_usage();
		return -1;
	}

	ret = midgard_compile_shader_nir(nir);
	if (ret) {
		fprintf(stderr, "compiler failed!\n");
		return ret;
	}
}
