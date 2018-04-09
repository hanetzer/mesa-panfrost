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

/* Authors:  Keith Whitwell <keithw@vmware.com>
 */


#include "sp_context.h"
#include "sp_state.h"

#include "util/u_memory.h"
#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_transfer.h"
#include "draw/draw_context.h"


static void *
softpipe_create_vertex_elements_state(struct pipe_context *pipe,
                                      unsigned count,
                                      const struct pipe_vertex_element *attribs)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   return softpipe->panfrost->create_vertex_elements_state(softpipe->panfrost, count, attribs);
}


static void
softpipe_bind_vertex_elements_state(struct pipe_context *pipe,
                                    void *velems)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   softpipe->panfrost->bind_vertex_elements_state(softpipe->panfrost, velems);
}


static void
softpipe_delete_vertex_elements_state(struct pipe_context *pipe, void *velems)
{
   FREE( velems );
}


static void
softpipe_set_vertex_buffers(struct pipe_context *pipe,
                            unsigned start_slot, unsigned count,
                            const struct pipe_vertex_buffer *buffers)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   softpipe->panfrost->set_vertex_buffers(softpipe->panfrost, start_slot, count, buffers);
}


void
softpipe_init_vertex_funcs(struct pipe_context *pipe)
{
   pipe->create_vertex_elements_state = softpipe_create_vertex_elements_state;
   pipe->bind_vertex_elements_state = softpipe_bind_vertex_elements_state;
   pipe->delete_vertex_elements_state = softpipe_delete_vertex_elements_state;

   pipe->set_vertex_buffers = softpipe_set_vertex_buffers;
}
