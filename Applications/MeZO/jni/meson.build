mezo_example_sources = [
  'main.cpp',
]

mezo_example_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  mezo_example_dependencies += [gtest_dep]
endif

e = executable('mezo_example',
  mezo_example_sources,
  include_directories: [include_directories('.')],
  dependencies: mezo_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
