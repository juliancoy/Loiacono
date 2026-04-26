cmake_minimum_required(VERSION 3.16)
project(LoiaconoSpectrogram LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

option(LOIACONO_ENABLE_DESKTOP_AUDIO "Enable experimental desktop-audio device enumeration" OFF)
option(LOIACONO_ENABLE_VULKAN_COMPUTE "Enable experimental Vulkan compute backend" OFF)

# ── Qt ──
find_package(Qt6 REQUIRED COMPONENTS Widgets Network OpenGLWidgets)
find_package(Vulkan REQUIRED)

# ── RtAudio (submodule) ──
set(RTAUDIO_BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(rtaudio)

# ── Application ──
add_executable(loiacono_spectrogram
    audio_device_labels.cpp
    main.cpp
    tone_curve_editor.cpp
    gl_spectrogram_canvas.cpp
    loiacono_gpu_compute.cpp
    loiacono_vulkan_compute.cpp
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
    Vulkan::Vulkan
    rtaudio
)

target_include_directories(loiacono_spectrogram PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)
target_compile_definitions(loiacono_spectrogram PRIVATE
    LOIACONO_ENABLE_DESKTOP_AUDIO=$<BOOL:${LOIACONO_ENABLE_DESKTOP_AUDIO}>
    LOIACONO_ENABLE_VULKAN_COMPUTE=$<BOOL:${LOIACONO_ENABLE_VULKAN_COMPUTE}>
)

# ── Tests ──
add_executable(test_cpp_consistency
    audio_device_labels.cpp
    tests/test_cpp_consistency.cpp
    loiacono_gpu_compute.cpp
    loiacono_vulkan_compute.cpp
    loiacono_gpu_rolling_compute.cpp
    loiacono_rolling.cpp
    loiacono_parallel.cpp
)

target_link_libraries(test_cpp_consistency PRIVATE
    Qt6::Widgets
    Qt6::Network
    Qt6::OpenGLWidgets
    Vulkan::Vulkan
    rtaudio
)

target_include_directories(test_cpp_consistency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)
target_compile_definitions(test_cpp_consistency PRIVATE
    LOIACONO_ENABLE_DESKTOP_AUDIO=$<BOOL:${LOIACONO_ENABLE_DESKTOP_AUDIO}>
    LOIACONO_ENABLE_VULKAN_COMPUTE=$<BOOL:${LOIACONO_ENABLE_VULKAN_COMPUTE}>
)

add_executable(test_simple_consistency
    audio_device_labels.cpp
    tests/test_simple_consistency.cpp
    loiacono_gpu_compute.cpp
    loiacono_vulkan_compute.cpp
    loiacono_gpu_rolling_compute.cpp
    loiacono_rolling.cpp
    loiacono_parallel.cpp
)

target_link_libraries(test_simple_consistency PRIVATE
    Qt6::Widgets
    Qt6::Network
    Qt6::OpenGLWidgets
    Vulkan::Vulkan
    rtaudio
)

target_include_directories(test_simple_consistency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)
target_compile_definitions(test_simple_consistency PRIVATE
    LOIACONO_ENABLE_DESKTOP_AUDIO=$<BOOL:${LOIACONO_ENABLE_DESKTOP_AUDIO}>
    LOIACONO_ENABLE_VULKAN_COMPUTE=$<BOOL:${LOIACONO_ENABLE_VULKAN_COMPUTE}>
)

# Headless test (no GUI/OpenGL required)
add_executable(test_modes_headless
    audio_device_labels.cpp
    tests/test_modes_headless.cpp
    loiacono_rolling.cpp
    loiacono_parallel.cpp
    loiacono_gpu_compute.cpp
    loiacono_vulkan_compute.cpp
    loiacono_gpu_rolling_compute.cpp
)

target_link_libraries(test_modes_headless PRIVATE
    Qt6::Core
    Qt6::OpenGLWidgets
    Vulkan::Vulkan
)

target_include_directories(test_modes_headless PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)
target_compile_definitions(test_modes_headless PRIVATE
    LOIACONO_ENABLE_DESKTOP_AUDIO=$<BOOL:${LOIACONO_ENABLE_DESKTOP_AUDIO}>
    LOIACONO_ENABLE_VULKAN_COMPUTE=$<BOOL:${LOIACONO_ENABLE_VULKAN_COMPUTE}>
)

add_executable(test_pitch_tuning_headless
    tests/test_pitch_tuning_headless.cpp
    loiacono_rolling.cpp
    loiacono_parallel.cpp
    loiacono_gpu_compute.cpp
    loiacono_vulkan_compute.cpp
    loiacono_gpu_rolling_compute.cpp
)

target_link_libraries(test_pitch_tuning_headless PRIVATE
    Qt6::Core
    Qt6::OpenGLWidgets
    Vulkan::Vulkan
)

target_include_directories(test_pitch_tuning_headless PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)
target_compile_definitions(test_pitch_tuning_headless PRIVATE
    LOIACONO_ENABLE_DESKTOP_AUDIO=$<BOOL:${LOIACONO_ENABLE_DESKTOP_AUDIO}>
    LOIACONO_ENABLE_VULKAN_COMPUTE=$<BOOL:${LOIACONO_ENABLE_VULKAN_COMPUTE}>
)

# Display consistency test (generates images for comparison)
add_executable(test_display_consistency
    audio_device_labels.cpp
    tests/test_display_consistency.cpp
    loiacono_rolling.cpp
    loiacono_parallel.cpp
    loiacono_gpu_compute.cpp
    loiacono_vulkan_compute.cpp
    loiacono_gpu_rolling_compute.cpp
)

target_link_libraries(test_display_consistency PRIVATE
    Qt6::Widgets
    Qt6::OpenGLWidgets
    Vulkan::Vulkan
)

target_include_directories(test_display_consistency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)
target_compile_definitions(test_display_consistency PRIVATE
    LOIACONO_ENABLE_DESKTOP_AUDIO=$<BOOL:${LOIACONO_ENABLE_DESKTOP_AUDIO}>
    LOIACONO_ENABLE_VULKAN_COMPUTE=$<BOOL:${LOIACONO_ENABLE_VULKAN_COMPUTE}>
)

add_executable(test_audio_device_labels
    audio_device_labels.cpp
    tests/test_audio_device_labels.cpp
)

target_link_libraries(test_audio_device_labels PRIVATE
    Qt6::Core
)
target_compile_definitions(test_audio_device_labels PRIVATE
    LOIACONO_ENABLE_DESKTOP_AUDIO=$<BOOL:${LOIACONO_ENABLE_DESKTOP_AUDIO}>
    LOIACONO_ENABLE_VULKAN_COMPUTE=$<BOOL:${LOIACONO_ENABLE_VULKAN_COMPUTE}>
)

add_executable(test_list_audio_devices
    audio_device_labels.cpp
    tests/test_list_audio_devices.cpp
)

target_link_libraries(test_list_audio_devices PRIVATE
    Qt6::Core
    Vulkan::Vulkan
    rtaudio
)

target_include_directories(test_list_audio_devices PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rtaudio
)
target_compile_definitions(test_list_audio_devices PRIVATE
    LOIACONO_ENABLE_DESKTOP_AUDIO=$<BOOL:${LOIACONO_ENABLE_DESKTOP_AUDIO}>
    LOIACONO_ENABLE_VULKAN_COMPUTE=$<BOOL:${LOIACONO_ENABLE_VULKAN_COMPUTE}>
)
