cmake_minimum_required(VERSION 3.24)
project(LoiaconoSpectrogram LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

option(LOIACONO_ENABLE_DESKTOP_AUDIO "Enable experimental desktop-audio device enumeration" OFF)
option(LOIACONO_ENABLE_VULKAN_COMPUTE "Enable experimental Vulkan compute backend" OFF)
set(LOIACONO_QT_PREFIX "" CACHE PATH "Complete Qt 6 installation used for libraries and host tools")

# ── Qt ──
# Keep Qt libraries and host tools on one version. Prefer an explicit prefix,
# then the parent JSynth toolchain, then the qtpaths6 selected by PATH.
if(NOT LOIACONO_QT_PREFIX AND NOT Qt6_DIR)
    set(_loiacono_parent_qt "${CMAKE_CURRENT_SOURCE_DIR}/../build-local/native/qt5/install-linux")
    if(EXISTS "${_loiacono_parent_qt}/lib/cmake/Qt6/Qt6Config.cmake")
        set(LOIACONO_QT_PREFIX "${_loiacono_parent_qt}")
    else()
        find_program(LOIACONO_QTPATHS_EXECUTABLE NAMES qtpaths6 qtpaths)
        if(LOIACONO_QTPATHS_EXECUTABLE)
            execute_process(
                COMMAND "${LOIACONO_QTPATHS_EXECUTABLE}" --query QT_INSTALL_PREFIX
                OUTPUT_VARIABLE LOIACONO_QT_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _loiacono_qtpaths_result
            )
            if(NOT _loiacono_qtpaths_result EQUAL 0)
                set(LOIACONO_QT_PREFIX "")
            endif()
        endif()
    endif()
endif()

if(LOIACONO_QT_PREFIX)
    file(REAL_PATH "${LOIACONO_QT_PREFIX}" LOIACONO_QT_PREFIX)
    set(LOIACONO_QT_PREFIX "${LOIACONO_QT_PREFIX}" CACHE PATH
        "Complete Qt 6 installation used for libraries and host tools" FORCE)
    list(PREPEND CMAKE_PREFIX_PATH "${LOIACONO_QT_PREFIX}")
    execute_process(
        COMMAND "${LOIACONO_QT_PREFIX}/bin/qtpaths6" --query QT_INSTALL_LIBS
        OUTPUT_VARIABLE _loiacono_qt_lib_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _loiacono_qt_lib_result
    )
    if(_loiacono_qt_lib_result EQUAL 0 AND
       EXISTS "${_loiacono_qt_lib_dir}/cmake/Qt6/Qt6Config.cmake")
        set(Qt6_DIR "${_loiacono_qt_lib_dir}/cmake/Qt6" CACHE PATH "Qt 6 package directory" FORCE)
    elseif(EXISTS "${LOIACONO_QT_PREFIX}/lib/cmake/Qt6/Qt6Config.cmake")
        set(Qt6_DIR "${LOIACONO_QT_PREFIX}/lib/cmake/Qt6" CACHE PATH "Qt 6 package directory" FORCE)
    else()
        message(FATAL_ERROR "LOIACONO_QT_PREFIX does not contain a complete Qt 6 installation: ${LOIACONO_QT_PREFIX}")
    endif()
endif()

find_package(Qt6 REQUIRED COMPONENTS Widgets Network OpenGLWidgets)
message(STATUS "Loiacono Qt ${Qt6_VERSION}: ${Qt6_DIR}")
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
    loiacono_pitch_shift.cpp
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
    loiacono_pitch_shift.cpp
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
    loiacono_pitch_shift.cpp
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
    loiacono_pitch_shift.cpp
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
    loiacono_pitch_shift.cpp
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
    loiacono_pitch_shift.cpp
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
