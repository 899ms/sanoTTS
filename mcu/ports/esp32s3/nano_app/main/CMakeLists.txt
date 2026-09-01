idf_component_register(
    SRCS "nano_dev_main.c" "snt_nano.c" "snt_port_esp32s3.c" "sn_matvec_s8_esp32s3.S"
    INCLUDE_DIRS "."
    EMBED_FILES "golden/front_q8.bin" "golden/model_q8.bin"
                "golden/r00_ids.bin" "golden/r00_durs.bin" "golden/r00_audio.bin"
                "golden/r06_ids.bin" "golden/r06_durs.bin" "golden/r06_audio.bin"
                "golden/e2e_uniform.bin" "golden/e2e_noise.bin"
    PRIV_REQUIRES esp_timer espressif__esp-nn esp_psram esp_hw_support
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE SNT_NANO_PROF SNT_NANO_FAST_MATH)
