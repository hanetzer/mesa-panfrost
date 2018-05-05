/**************************************************************************
 * 
 * Copyright 2006 VMware, Inc.
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
 /*
  * Authors:
  *   Keith Whitwell <keithw@vmware.com>
  *   Michel Dänzer <daenzer@vmware.com>
  */

#include "pipe/p_defines.h"
#include "util/u_inlines.h"

#include "util/u_format.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_transfer.h"
#include "util/u_surface.h"

#include "sp_context.h"
#include "sp_flush.h"
#include "sp_texture.h"
#include "sp_screen.h"

#include "state_tracker/sw_winsys.h"

static boolean
softpipe_can_create_resource(struct pipe_screen *screen,
                             const struct pipe_resource *res)
{
   return TRUE;
}


/**
 * Texture layout for simple color buffers.
 */
static boolean
softpipe_displaytarget_layout(struct pipe_screen *screen,
                              struct softpipe_resource *spr,
                              const void *map_front_private)
{
   struct sw_winsys *winsys = softpipe_screen(screen)->winsys;

   /* Round up the surface size to a multiple of the tile size?
    */
   printf("Surface created\n");
   spr->dt = winsys->displaytarget_create(winsys,
                                          spr->base.bind,
                                          spr->base.format,
                                          spr->base.width0, 
                                          spr->base.height0,
                                          64,
                                          map_front_private,
                                          &spr->stride[0] );

   return spr->dt != NULL;
}

#define __PAN_GALLIUM
#include <trans-builder.h>

/**
 * Create new pipe_resource given the template information.
 */
static struct pipe_resource *
softpipe_resource_create_front(struct pipe_screen *screen,
                               const struct pipe_resource *templat,
                               const void *map_front_private)
{
	if (!(templat->bind & PIPE_BIND_DISPLAY_TARGET))
		return panfrost_resource_create_front(screen, templat, map_front_private);

	printf("Disp\n");
   struct softpipe_resource *spr = CALLOC_STRUCT(softpipe_resource);
   if (!spr)
      return NULL;

   assert(templat->format != PIPE_FORMAT_NONE);

   spr->base = *templat;
   pipe_reference_init(&spr->base.reference, 1);
   spr->base.screen = screen;

   spr->pot = (util_is_power_of_two(templat->width0) &&
               util_is_power_of_two(templat->height0) &&
               util_is_power_of_two(templat->depth0));

   if (spr->base.bind & (PIPE_BIND_DISPLAY_TARGET |
			 PIPE_BIND_SCANOUT |
			 PIPE_BIND_SHARED)) {
      if (!softpipe_displaytarget_layout(screen, spr, map_front_private))
         goto fail;
   }
    
   return &spr->base;

 fail:
   FREE(spr);
   return NULL;
}

static struct pipe_resource *
softpipe_resource_create(struct pipe_screen *screen,
                         const struct pipe_resource *templat)
{
   return softpipe_resource_create_front(screen, templat, NULL);
}

static void
softpipe_resource_destroy(struct pipe_screen *pscreen,
			  struct pipe_resource *pt)
{
   struct softpipe_screen *screen = softpipe_screen(pscreen);
   struct softpipe_resource *spr = softpipe_resource(pt);

#if 0
   if (spr->dt) {
      /* display target */
      struct sw_winsys *winsys = screen->winsys;
      winsys->displaytarget_destroy(winsys, spr->dt);
   }
   else if (!spr->userBuffer) {
      /* regular texture */
      align_free(spr->data);
   }
#endif
   printf("Resource destroy...\n");

   FREE(spr);
}


static struct pipe_resource *
softpipe_resource_from_handle(struct pipe_screen *screen,
                              const struct pipe_resource *templat,
                              struct winsys_handle *whandle,
                              unsigned usage)
{
   struct sw_winsys *winsys = softpipe_screen(screen)->winsys;
   struct softpipe_resource *spr = CALLOC_STRUCT(softpipe_resource);
   if (!spr)
      return NULL;

   spr->base = *templat;
   pipe_reference_init(&spr->base.reference, 1);
   spr->base.screen = screen;

   spr->pot = (util_is_power_of_two(templat->width0) &&
               util_is_power_of_two(templat->height0) &&
               util_is_power_of_two(templat->depth0));

   printf("Create from handle\n");
   spr->dt = winsys->displaytarget_from_handle(winsys,
                                               templat,
                                               whandle,
                                               &spr->stride[0]);
   if (!spr->dt)
      goto fail;

   return &spr->base;

 fail:
   FREE(spr);
   return NULL;
}


static boolean
softpipe_resource_get_handle(struct pipe_screen *screen,
                             struct pipe_context *ctx,
                             struct pipe_resource *pt,
                             struct winsys_handle *whandle,
                             unsigned usage)
{
   struct sw_winsys *winsys = softpipe_screen(screen)->winsys;
   struct softpipe_resource *spr = softpipe_resource(pt);

   assert(spr->dt);
   if (!spr->dt)
      return FALSE;

   return winsys->displaytarget_get_handle(winsys, spr->dt, whandle);
}


/**
 * Helper function to compute offset (in bytes) for a particular
 * texture level/face/slice from the start of the buffer.
 */
unsigned
softpipe_get_tex_image_offset(const struct softpipe_resource *spr,
                              unsigned level, unsigned layer)
{
   unsigned offset = spr->level_offset[level];

   offset += layer * spr->img_stride[level];

   return offset;
}


/**
 * Get a pipe_surface "view" into a texture resource.
 */
static struct pipe_surface *
softpipe_create_surface(struct pipe_context *pipe,
                        struct pipe_resource *pt,
                        const struct pipe_surface *surf_tmpl)
{
   struct pipe_surface *ps;

   ps = CALLOC_STRUCT(pipe_surface);
   if (ps) {
      pipe_reference_init(&ps->reference, 1);
      pipe_resource_reference(&ps->texture, pt);
      ps->context = pipe;
      ps->format = surf_tmpl->format;
      if (pt->target != PIPE_BUFFER) {
         assert(surf_tmpl->u.tex.level <= pt->last_level);
         ps->width = u_minify(pt->width0, surf_tmpl->u.tex.level);
         ps->height = u_minify(pt->height0, surf_tmpl->u.tex.level);
         ps->u.tex.level = surf_tmpl->u.tex.level;
         ps->u.tex.first_layer = surf_tmpl->u.tex.first_layer;
         ps->u.tex.last_layer = surf_tmpl->u.tex.last_layer;
         if (ps->u.tex.first_layer != ps->u.tex.last_layer) {
            debug_printf("creating surface with multiple layers, rendering to first layer only\n");
         }
      }
      else {
         /* setting width as number of elements should get us correct renderbuffer width */
         ps->width = surf_tmpl->u.buf.last_element - surf_tmpl->u.buf.first_element + 1;
         ps->height = pt->height0;
         ps->u.buf.first_element = surf_tmpl->u.buf.first_element;
         ps->u.buf.last_element = surf_tmpl->u.buf.last_element;
         assert(ps->u.buf.first_element <= ps->u.buf.last_element);
         assert(ps->u.buf.last_element < ps->width);
      }
   }
   return ps;
}


/**
 * Free a pipe_surface which was created with softpipe_create_surface().
 */
static void 
softpipe_surface_destroy(struct pipe_context *pipe,
                         struct pipe_surface *surf)
{
   /* Effectively do the texture_update work here - if texture images
    * needed post-processing to put them into hardware layout, this is
    * where it would happen.  For softpipe, nothing to do.
    */
   assert(surf->texture);
   pipe_resource_reference(&surf->texture, NULL);
   FREE(surf);
}


/**
 * Geta pipe_transfer object which is used for moving data in/out of
 * a resource object.
 * \param pipe  rendering context
 * \param resource  the resource to transfer in/out of
 * \param level  which mipmap level
 * \param usage  bitmask of PIPE_TRANSFER_x flags
 * \param box  the 1D/2D/3D region of interest
 */
static void *
softpipe_transfer_map(struct pipe_context *pipe,
                      struct pipe_resource *resource,
                      unsigned level,
                      unsigned usage,
                      const struct pipe_box *box,
                      struct pipe_transfer **transfer)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   return softpipe->panfrost->transfer_map(softpipe->panfrost, resource, level, usage, box, transfer);
}


/**
 * Unmap memory mapping for given pipe_transfer object.
 */
static void
softpipe_transfer_unmap(struct pipe_context *pipe,
                        struct pipe_transfer *transfer)
{
   struct softpipe_context *softpipe = softpipe_context(pipe);
   return softpipe->panfrost->transfer_unmap(softpipe->panfrost, transfer);
}

/**
 * Create buffer which wraps user-space data.
 */
struct pipe_resource *
softpipe_user_buffer_create(struct pipe_screen *screen,
                            void *ptr,
                            unsigned bytes,
			    unsigned bind_flags)
{
   struct softpipe_resource *spr;

   spr = CALLOC_STRUCT(softpipe_resource);
   if (!spr)
      return NULL;

   pipe_reference_init(&spr->base.reference, 1);
   spr->base.screen = screen;
   spr->base.format = PIPE_FORMAT_R8_UNORM; /* ?? */
   spr->base.bind = bind_flags;
   spr->base.usage = PIPE_USAGE_IMMUTABLE;
   spr->base.flags = 0;
   spr->base.width0 = bytes;
   spr->base.height0 = 1;
   spr->base.depth0 = 1;
   spr->base.array_size = 1;
   spr->userBuffer = TRUE;
   spr->data = ptr;

   return &spr->base;
}


void
softpipe_init_texture_funcs(struct pipe_context *pipe)
{
   pipe->transfer_map = softpipe_transfer_map;
   pipe->transfer_unmap = softpipe_transfer_unmap;

   pipe->transfer_flush_region = u_default_transfer_flush_region;
   pipe->buffer_subdata = u_default_buffer_subdata;
   pipe->texture_subdata = u_default_texture_subdata;

   pipe->create_surface = softpipe_create_surface;
   pipe->surface_destroy = softpipe_surface_destroy;
   pipe->clear_texture = util_clear_texture;
}


void
softpipe_init_screen_texture_funcs(struct pipe_screen *screen)
{
   screen->resource_create = softpipe_resource_create;
   screen->resource_create_front = softpipe_resource_create_front;
   screen->resource_destroy = softpipe_resource_destroy;
   screen->resource_from_handle = softpipe_resource_from_handle;
   screen->resource_get_handle = softpipe_resource_get_handle;
   screen->can_create_resource = softpipe_can_create_resource;
}
