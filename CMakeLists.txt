cmake_minimum_required(VERSION 3.16)
project(LoiaconoSpectrogram LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

# ── Qt ──
find_package(Qt6 REQUIRED COMPONENTS Widgets Network OpenGLWidgets)

# ── RtAudio (submodule) ──
set(RTAUDIO_BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(rtaudio)

# ── Application ──
add_executable(loiacono_spectrogram
    main.cpp
    gl_spectrogram_canvas.cpp
    loiacono_gpu_compute.cpp
    loiacono_gpu_rolling_compute.cpp
    loiacono_rolling.cpp
    loiacono_parallel.cpp
    spectrogram_widget.cpp
    api_server.cpp
)

target_link_libraries(loiacono_spectrogram PRIVATE
    Qt6::Widgets
    Qt6::Network
    Qt6::OpenGLWidgets
    rtaudio
)

target_include_directories(loiacono_spectrogram PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)

# ── Tests ──
add_executable(test_cpp_consistency
    tests/test_cpp_consistency.cpp
    loiacono_gpu_compute.cpp
    loiacono_gpu_rolling_compute.cpp
    loiacono_rolling.cpp
    loiacono_parallel.cpp
)

target_link_libraries(test_cpp_consistency PRIVATE
    Qt6::Widgets
    Qt6::Network
    Qt6::OpenGLWidgets
    rtaudio
)

target_include_directories(test_cpp_consistency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)

add_executable(test_simple_consistency
    tests/test_simple_consistency.cpp
    loiacono_gpu_compute.cpp
    loiacono_gpu_rolling_compute.cpp
    loiacono_rolling.cpp
    loiacono_parallel.cpp
)

target_link_libraries(test_simple_consistency PRIVATE
    Qt6::Widgets
    Qt6::Network
    Qt6::OpenGLWidgets
    rtaudio
)

target_include_directories(test_simple_consistency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)
