/**************************************************************************
 * 
 * Copyright 2007 VMware, Inc.
 * All Rights Reserved.
 * Copyright 2008 VMware, Inc.  All rights reserved.
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

/* Author:
 *    Keith Whitwell <keithw@vmware.com>
 */

#include "draw/draw_context.h"
#include "draw/draw_vbuf.h"
#include "pipe/p_defines.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_pstipple.h"
#include "util/u_inlines.h"
#include "util/u_upload_mgr.h"
#include "sp_clear.h"
#include "sp_context.h"
#include "sp_flush.h"
#include "sp_state.h"
#include "sp_surface.h"
#include "sp_texture.h"
#include "sp_query.h"
#include "sp_screen.h"

static void
softpipe_destroy( struct pipe_context *pipe )
{
   struct softpipe_context *softpipe = softpipe_context( pipe );
   uint i, sh;

   if (softpipe->blitter) {
      util_blitter_destroy(softpipe->blitter);
   }

   if (softpipe->draw)
      draw_destroy( softpipe->draw );

   if (softpipe->pipe.stream_uploader)
      u_upload_destroy(softpipe->pipe.stream_uploader);

   for (i = 0; i < PIPE_MAX_COLOR_BUFS; i++) {
      pipe_surface_reference(&softpipe->framebuffer.cbufs[i], NULL);
   }

   pipe_surface_reference(&softpipe->framebuffer.zsbuf, NULL);

   for (sh = 0; sh < ARRAY_SIZE(softpipe->constants); sh++) {
      for (i = 0; i < ARRAY_SIZE(softpipe->constants[0]); i++) {
         if (softpipe->constants[sh][i]) {
            pipe_resource_reference(&softpipe->constants[sh][i], NULL);
         }
      }
   }

   for (i = 0; i < softpipe->num_vertex_buffers; i++) {
      pipe_vertex_buffer_unreference(&softpipe->vertex_buffer[i]);
   }

   FREE( softpipe );
}

/**
 * if (the texture is being used as a framebuffer surface)
 *    return SP_REFERENCED_FOR_WRITE
 * else if (the texture is a bound texture source)
 *    return SP_REFERENCED_FOR_READ
 * else
 *    return SP_UNREFERENCED
 */
unsigned int
softpipe_is_resource_referenced( struct pipe_context *pipe,
                                 struct pipe_resource *texture,
                                 unsigned level, int layer)
{
   struct softpipe_context *softpipe = softpipe_context( pipe );
   unsigned i, sh;

   if (texture->target == PIPE_BUFFER)
      return SP_UNREFERENCED;

   /* check if any of the bound drawing surfaces are this texture */
   if (softpipe->dirty_render_cache) {
      for (i = 0; i < softpipe->framebuffer.nr_cbufs; i++) {
         if (softpipe->framebuffer.cbufs[i] && 
             softpipe->framebuffer.cbufs[i]->texture == texture) {
            return SP_REFERENCED_FOR_WRITE;
         }
      }
      if (softpipe->framebuffer.zsbuf && 
          softpipe->framebuffer.zsbuf->texture == texture) {
         return SP_REFERENCED_FOR_WRITE;
      }
   }
   
   return SP_UNREFERENCED;
}




static void
softpipe_render_condition(struct pipe_context *pipe,
                          struct pipe_query *query,
                          boolean condition,
                          enum pipe_render_cond_flag mode)
{
   struct softpipe_context *softpipe = softpipe_context( pipe );

   softpipe->render_cond_query = query;
   softpipe->render_cond_mode = mode;
   softpipe->render_cond_cond = condition;
}

#include <trans-builder.h>

struct pipe_context *
softpipe_create_context(struct pipe_screen *screen,
			void *priv, unsigned flags)
{
   struct softpipe_screen *sp_screen = softpipe_screen(screen);
   struct softpipe_context *softpipe = CALLOC_STRUCT(softpipe_context);
   uint i, sh;

   softpipe->panfrost = panfrost_create_context(screen, priv, flags);

   util_init_math();

   softpipe->pipe.screen = screen;
   softpipe->pipe.destroy = softpipe_destroy;
   softpipe->pipe.priv = priv;

   /* state setters */
   softpipe_init_blend_funcs(&softpipe->pipe);
   softpipe_init_clip_funcs(&softpipe->pipe);
   softpipe_init_query_funcs( softpipe );
   softpipe_init_rasterizer_funcs(&softpipe->pipe);
   softpipe_init_streamout_funcs(&softpipe->pipe);
   softpipe_init_texture_funcs( &softpipe->pipe );
   softpipe_init_vertex_funcs(&softpipe->pipe);
   softpipe_init_shader_funcs(&softpipe->pipe);
   softpipe_init_sampler_funcs(&softpipe->pipe);

   softpipe->pipe.set_framebuffer_state = softpipe_set_framebuffer_state;

   softpipe->pipe.draw_vbo = softpipe_draw_vbo;

   softpipe->pipe.clear = softpipe_clear;
   softpipe->pipe.flush = softpipe_flush_wrapped;
   softpipe->pipe.texture_barrier = softpipe_texture_barrier;
   softpipe->pipe.memory_barrier = softpipe_memory_barrier;
   softpipe->pipe.render_condition = softpipe_render_condition;
   
   softpipe->pipe.stream_uploader = u_upload_create_default(&softpipe->pipe);
   if (!softpipe->pipe.stream_uploader)
      goto fail;
   softpipe->pipe.const_uploader = softpipe->pipe.stream_uploader;

   sp_init_surface_functions(softpipe);

   return &softpipe->pipe;

 fail:
   softpipe_destroy(&softpipe->pipe);
   return NULL;
}
