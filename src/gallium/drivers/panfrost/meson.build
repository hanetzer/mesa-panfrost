midgard_compiler = executable(
  'midgard_compiler',
  'midgard/midgard_cmdline.c',
  'midgard/cppwrap.cpp',
  include_directories : [inc_common, inc_src, inc_include, inc_gallium, inc_gallium_aux, include_directories('midgard')],
  dependencies : [
    dep_thread,
    idep_nir
  ],
  link_with : [
    libgallium,
    libglsl_standalone,
    libmesa_util
  ],
  build_by_default : true
)
