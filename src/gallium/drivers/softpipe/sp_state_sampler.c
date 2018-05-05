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

/* Authors:
 *  Brian Paul
 */

#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_format.h"

#include "sp_context.h"
#include "sp_state.h"
#include "sp_texture.h"
#include "sp_screen.h"
#include "state_tracker/sw_winsys.h"

static void
softpipe_bind_sampler_states(struct pipe_context *pipe,
                             enum pipe_shader_type shader,
                             unsigned start,
                             unsigned num,
                             void **samplers)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   softpipe->panfrost->bind_sampler_states(softpipe->panfrost, shader, start, num, samplers);
}


static void
softpipe_sampler_view_destroy(struct pipe_context *pipe,
                              struct pipe_sampler_view *view)
{
   pipe_resource_reference(&view->texture, NULL);
   FREE(view);
}


void
softpipe_set_sampler_views(struct pipe_context *pipe,
                           enum pipe_shader_type shader,
                           unsigned start,
                           unsigned num,
                           struct pipe_sampler_view **views)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   softpipe->panfrost->set_sampler_views(softpipe->panfrost, shader, start, num, views);
}


static void
softpipe_delete_sampler_state(struct pipe_context *pipe,
                              void *sampler)
{
   FREE( sampler );
}

static void *
softpipe_create_sampler_state(struct pipe_context *pipe,
                              const struct pipe_sampler_state *sampler)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   return softpipe->panfrost->create_sampler_state(softpipe->panfrost, sampler);
}

static struct pipe_sampler_view *
softpipe_create_sampler_view(struct pipe_context *pipe,
                             struct pipe_resource *resource,
                             const struct pipe_sampler_view *templ)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   return softpipe->panfrost->create_sampler_view(pipe, resource, templ);
}

void
softpipe_init_sampler_funcs(struct pipe_context *pipe)
{
   pipe->create_sampler_state = softpipe_create_sampler_state;
   pipe->bind_sampler_states = softpipe_bind_sampler_states;
   pipe->delete_sampler_state = softpipe_delete_sampler_state;

   pipe->create_sampler_view = softpipe_create_sampler_view;
   pipe->set_sampler_views = softpipe_set_sampler_views;
   pipe->sampler_view_destroy = softpipe_sampler_view_destroy;
}

