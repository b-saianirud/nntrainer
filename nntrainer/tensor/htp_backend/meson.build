htp_backend_sources = [
  'htp_backend.cpp',
  'htp_compute_ops.cpp',
]

htp_backend_headers = [
  'htp_backend.h',
  'sdkl_compat.h',
]

foreach s : htp_backend_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : htp_backend_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
