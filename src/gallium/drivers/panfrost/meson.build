midgard_compiler = executable(
  'midgard_compiler',
  'midgard/midgard_cmdline.c',
  include_directories : [inc_gallium, inc_gallium_aux, include_directories('midgard')],
  dependencies : [
    idep_nir
  ],
  link_with : [
    libgallium,
    libglsl_standalone,
    libmesa_util
  ],
  build_by_default : true
)
