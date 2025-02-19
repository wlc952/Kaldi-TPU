include_directories(${CMAKE_SOURCE_DIR})

if(SHERPA_ONNX_ENABLE_PYTHON)
  message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; print('.'.join(sys.version.split('.')[:2]))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_VERSION
  )
  message(STATUS "PYTHON_VERSION: ${PYTHON_VERSION}")
endif()

set(sources
  base64-decode.cc
  cat.cc
  circular-buffer.cc
  context-graph.cc
  endpoint.cc
  features.cc
  file-utils.cc
  hypothesis.cc
  online-model-config.cc
  online-recognizer-impl.cc
  online-recognizer.cc
  online-stream.cc
  online-transducer-decoder.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-model-config.cc
  online-transducer-model.cc
  online-zipformer-transducer-model.cc
  online-zipformer2-ctc-model-config.cc
  online-zipformer2-ctc-model.cc
  online-zipformer2-transducer-model.cc
  online-ctc-fst-decoder-config.cc
  online-ctc-fst-decoder.cc
  online-ctc-greedy-search-decoder.cc
  online-ctc-model.cc
  onnx-utils.cc
  onnx-to-bm.cc
  pad-sequence.cc
  parse-options.cc
  provider.cc
  resample.cc
  session.cc
  symbol-table.cc
  text-utils.cc
  transpose.cc
  unbind.cc
  utils.cc
  wave-reader.cc
)

if(SHERPA_ONNX_ENABLE_CHECK)
  list(APPEND sources log.cc)
endif()
add_library(sherpa-onnx-core ${sources})
if(APPLE)
  target_compile_options(sherpa-onnx-core PRIVATE
    -Wno-deprecated-declarations
  )
endif()


target_link_libraries(sherpa-onnx-core
  kaldi-native-fbank-core
  kaldi-decoder-core
)


if(BUILD_SHARED_LIBS)
  target_link_libraries(sherpa-onnx-core onnxruntime)
else()
  target_link_libraries(sherpa-onnx-core ${onnxruntime_lib_files})
endif()

# target_include_directories(sherpa-onnx-core PUBLIC ${CMAKE_SOURCE_DIR})

# target_include_directories(sherpa-onnx-core PUBLIC /opt/sophon/libsophon-0.5.0/include)
# link_directories(/opt/sophon/libsophon-0.5.0/lib)

target_include_directories(sherpa-onnx-core PUBLIC ${PROJECT_SOURCE_DIR}/debug_utils/include)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  link_directories(${CMAKE_SOURCE_DIR}/debug_utils/lib_soc)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  link_directories(${CMAKE_SOURCE_DIR}/debug_utils/lib_pcie)
endif()

if (CMAKE_TYPE STREQUAL "DUMP")
    add_definitions(-DDUMP_TENSOR)
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -lcnpy) # -lcryptopp)
    include_directories(${PROJECT_SOURCE_DIR}/sherpa-onnx/csrc/third_party/cnpy) # ln -s /data/Kaldi-TPU/third_party /data/Kaldi-TPU/sherpa-onnx/csrc
    add_subdirectory(${PROJECT_SOURCE_DIR}/sherpa-onnx/csrc/third_party/cnpy) 
else()
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall) # -lcryptopp)
endif()

# target_link_libraries(sherpa-onnx-core -lm)
if (CMAKE_TYPE STREQUAL "DUMP")
    target_link_libraries(sherpa-onnx-core bmrt bmlib cnpy) # cryptopp) # PUBLIC
else()
    target_link_libraries(sherpa-onnx-core bmrt bmlib) # cryptopp) # PUBLIC
endif()

# target_link_libraries(sherpa-onnx-core bmrt bmlib)

if(SHERPA_ONNX_ENABLE_TTS)
  target_link_libraries(sherpa-onnx-core piper_phonemize)
endif()

if(SHERPA_ONNX_ENABLE_CHECK)
  target_compile_definitions(sherpa-onnx-core PUBLIC SHERPA_ONNX_ENABLE_CHECK=1)

  if(SHERPA_ONNX_HAVE_EXECINFO_H)
    target_compile_definitions(sherpa-onnx-core PRIVATE SHERPA_ONNX_HAVE_EXECINFO_H=1)
  endif()

  if(SHERPA_ONNX_HAVE_CXXABI_H)
    target_compile_definitions(sherpa-onnx-core PRIVATE SHERPA_ONNX_HAVE_CXXABI_H=1)
  endif()
endif()

if(NOT BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  target_link_libraries(sherpa-onnx-core -ldl)
endif()

if(NOT WIN32 AND NOT SHERPA_ONNX_ENABLE_WASM AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  target_link_libraries(sherpa-onnx-core -pthread)
endif()

if(SHERPA_ONNX_ENABLE_BINARY)
  add_executable(sherpa-onnx sherpa-onnx.cc)

  set(main_exes
    sherpa-onnx
  )

  foreach(exe IN LISTS main_exes)
    target_link_libraries(${exe} sherpa-onnx-core)
    #target_include_directories(${exe} PRIVATE /data/untool/include)
  endforeach()

  if(NOT WIN32)
    foreach(exe IN LISTS main_exes)
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

      if(SHERPA_ONNX_ENABLE_PYTHON)
        target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      endif()
    endforeach()
  endif()
endif()

if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
  install(TARGETS sherpa-onnx-core DESTINATION ..)
else()
  install(TARGETS sherpa-onnx-core DESTINATION lib)
endif()

if(WIN32 AND BUILD_SHARED_LIBS)
  install(TARGETS sherpa-onnx-core DESTINATION bin)
endif()

if(SHERPA_ONNX_ENABLE_BINARY)
  install(
    TARGETS
      ${main_exes}
    DESTINATION
      bin
  )
endif()

if(SHERPA_ONNX_HAS_ALSA AND SHERPA_ONNX_ENABLE_BINARY)
  message("add alsa ==========================================")
  add_executable(sherpa-onnx-alsa sherpa-onnx-alsa.cc alsa.cc)

  set(exes
    sherpa-onnx-alsa
  )

  foreach(exe IN LISTS exes)
    target_link_libraries(${exe} sherpa-onnx-core)
    #target_include_directories(${exe} PRIVATE /data/untool/include)
  endforeach()

  foreach(exe IN LISTS exes)
    if(DEFINED ENV{SHERPA_ONNX_ALSA_LIB_DIR})
      target_link_libraries(${exe} -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)
    else()
      target_link_libraries(${exe} asound)
    endif()
  endforeach()

  if(NOT WIN32)
    foreach(exe IN LISTS exes)
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")
    endforeach()

    if(SHERPA_ONNX_ENABLE_PYTHON)
      foreach(exe IN LISTS exes)
        target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      endforeach()
    endif()
  endif()

  install(
    TARGETS ${exes}
    DESTINATION
      bin
  )
endif()

if(SHERPA_ONNX_ENABLE_TESTS)
  set(sherpa_onnx_test_srcs
    cat-test.cc
    circular-buffer-test.cc
    context-graph-test.cc
    pad-sequence-test.cc
    transpose-test.cc
    unbind-test.cc
    utfcpp-test.cc
  )

  list(APPEND sherpa_onnx_test_srcs
    speaker-embedding-manager-test.cc
  )

  function(sherpa_onnx_add_test source)
    get_filename_component(name ${source} NAME_WE)
    set(target_name ${name})
    add_executable(${target_name} "${source}")

    target_link_libraries(${target_name}
      PRIVATE
        gtest
        gtest_main
        sherpa-onnx-core
    )

    add_test(NAME "${target_name}"
      COMMAND
        $<TARGET_FILE:${target_name}>
    )
  endfunction()

  foreach(source IN LISTS sherpa_onnx_test_srcs)
    sherpa_onnx_add_test(${source})
  endforeach()
endif()
