midgard_nir_algebraic_c = custom_target(
  'midgard_nir_algebraic.c',
  input : 'midgard/midgard_nir_algebraic.py',
  output : 'midgard_nir_algebraic.c',
  command : [
    prog_python2, '@INPUT@',
    '-p', join_paths(meson.source_root(), 'src/compiler/nir/'),
  ],
  capture : true,
  depend_files : nir_algebraic_py,
)

files_midgard = files(
  'midgard/midgard_cmdline.c',
  'midgard/cppwrap.cpp',
)

midgard_compiler = executable(
  'midgard_compiler',
  [files_midgard, midgard_nir_algebraic_c],
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
