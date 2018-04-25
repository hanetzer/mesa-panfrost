/**************************************************************************
 * 
 * Copyright 2007 VMware, Inc.
 * All Rights Reserved.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 **************************************************************************/

#include "sp_context.h"
#include "sp_state.h"
#include "sp_fs.h"
#include "sp_texture.h"

#include "pipe/p_defines.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"

static void *
softpipe_create_fs_state(struct pipe_context *pipe,
                         const struct pipe_shader_state *templ)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   return softpipe->panfrost->create_fs_state(softpipe->panfrost, templ);
}


static void
softpipe_bind_fs_state(struct pipe_context *pipe, void *fs)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   softpipe->panfrost->bind_fs_state(softpipe->panfrost, fs);
}


static void
softpipe_delete_fs_state(struct pipe_context *pipe, void *fs)
{
   FREE(fs);
}


static void *
softpipe_create_vs_state(struct pipe_context *pipe,
                         const struct pipe_shader_state *templ)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   return softpipe->panfrost->create_vs_state(softpipe->panfrost, templ);
}


static void
softpipe_bind_vs_state(struct pipe_context *pipe, void *vs)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   softpipe->panfrost->bind_vs_state(softpipe->panfrost, vs);
}


static void
softpipe_delete_vs_state(struct pipe_context *pipe, void *vs)
{
   FREE( vs );
}

static void
softpipe_set_constant_buffer(struct pipe_context *pipe,
                             enum pipe_shader_type shader, uint index,
                             const struct pipe_constant_buffer *cb)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   softpipe->panfrost->set_constant_buffer(softpipe->panfrost, shader, index, cb);
}

void
softpipe_init_shader_funcs(struct pipe_context *pipe)
{
   pipe->create_fs_state = softpipe_create_fs_state;
   pipe->bind_fs_state   = softpipe_bind_fs_state;
   pipe->delete_fs_state = softpipe_delete_fs_state;

   pipe->create_vs_state = softpipe_create_vs_state;
   pipe->bind_vs_state   = softpipe_bind_vs_state;
   pipe->delete_vs_state = softpipe_delete_vs_state;

   pipe->set_constant_buffer = softpipe_set_constant_buffer;
}
