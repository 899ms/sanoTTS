# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ampixa
#
# The sanoTTS inference runtime is MIT; see LICENSE.MIT for the exact file
# list and why the split is sound. The repository as a whole is GPL-3.0,
# because the grapheme-to-phoneme layer embeds espeak-ng. This file does not.
# CMakeLists.txt -- Android (NDK) and any other CMake consumer.
#
# From an app's build.gradle:
#   externalNativeBuild { cmake { path "../../mobile/CMakeLists.txt" } }
#
# Produces libsanotts.so with the JNI shims, which is what
# System.loadLibrary("sanotts") in SanoTTS.kt expects.
cmake_minimum_required(VERSION 3.18)
project(sanotts C)

set(CMAKE_C_STANDARD 99)
set(SANOTTS_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)
# Which voice's operator header gets compiled in. The lineage selects
# LayerNorm-vs-DyT and GELU-vs-ReLU, so this is not interchangeable.
set(SANOTTS_MODEL "${SANOTTS_ROOT}/mcu/models/en_us_e12nano" CACHE PATH "model header dir")

add_library(sanotts SHARED
    ${SANOTTS_ROOT}/mobile/sanotts.c
    ${SANOTTS_ROOT}/mcu/src/snt_nano.c
    ${SANOTTS_ROOT}/mcu/src/snt_kernels_ref.c
    ${SANOTTS_ROOT}/mcu/ports/host/snt_port_host.c
)
if(ANDROID)
    target_sources(sanotts PRIVATE ${SANOTTS_ROOT}/mobile/kotlin/sanotts_jni.c)
    target_link_libraries(sanotts log)
endif()
target_include_directories(sanotts PUBLIC
    ${SANOTTS_ROOT}/mobile
    ${SANOTTS_ROOT}/mcu/include
    ${SANOTTS_ROOT}/mcu/src
    ${SANOTTS_MODEL}
)
target_compile_definitions(sanotts PRIVATE FSD_FAST_MATH)
target_compile_options(sanotts PRIVATE -O2)
