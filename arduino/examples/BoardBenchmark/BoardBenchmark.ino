/* BoardBenchmark.ino -- run saanoTTS on a board nobody has measured yet,
 * and print one block of numbers you can paste into a GitHub issue.
 *
 * WHY THIS SKETCH EXISTS
 *   We have measured exactly two chips: ESP32-S3 (0.22x real time) and
 *   ESP32-C3 (5.72x). Everything else in docs/mcu-classes-and-porting.md is a
 *   projection. If you have a Teensy 4.x, an STM32H7, an RA8, a P4, an RP2040
 *   -- your one flash turns a projection into a measurement.
 *
 * WHY IT HAS NO PERIPHERALS
 *   No I2S, no DAC, no speaker, no SD card, no LittleFS, no partition scheme.
 *   Weights, phoneme ids and the reference waveform are all compiled into
 *   flash by extras/gen_board_benchmark_data.py. The only thing you need
 *   working is Serial. Fewer moving parts, fewer first-try failures.
 *
 * WHAT IT REPORTS
 *   Correctness FIRST (correlation + RMS ratio against the shipped golden
 *   fixture), then speed. A fast number from a wrong build is worse than no
 *   number, so the sketch prints FAIL and refuses to call it a result.
 *
 * REQUIREMENTS
 *   About 820 KB of flash (731 KB of it embedded data) and at least 88 KB of
 *   free heap. The sketch walks a ladder of arena sizes and takes the largest
 *   one your board will actually give it, so it runs unchanged on a 264 KB
 *   RP2040 and a 1 MB Teensy 4.1, and reports which size it got -- see the
 *   arena note below for why that matters to your RTF. A part with only
 *   512 KB of flash cannot hold the data at all; see BOARDS.md.
 *
 * SETUP
 *   1. Open this sketch and flash it. sanotts_bench_data.h is checked in, so
 *      there is nothing to generate and no file to copy onto the board.
 *   2. Open Serial Monitor at 115200.
 *   3. Paste the REPORT block into:
 *        https://github.com/Ampixa/sanoTTS/issues/new?template=board-report.yml
 *
 *   To re-derive the header from a different fixture, from arduino/:
 *        python3 extras/gen_board_benchmark_data.py \
 *          --fixture ../mcu/test/fixtures/en_us_r7 \
 *          --out examples/BoardBenchmark/sanotts_bench_data.h
 *   and re-check it with extras/bench_host_check.sh, which compiles THIS
 *   sketch on your workstation and gates it against the golden fixture.
 *
 * BUILD-VERIFIED on ESP32-S3/C3/classic, Teensy 4.1/4.0/3.6, RP2040, RP2350
 * and Nucleo-H743ZI2 (arduino-cli 1.5.2). See BOARDS.md.
 */
#include <SanoTTS.h>

/* The generated header says which stack it holds; the two runtimes have the
 * same shape but are not the same API, and the nano decoder is noise-fed so
 * it additionally needs the row's seed. Selecting here rather than by hand
 * means regenerating the header is the only step to switch model. */
#include "sanotts_bench_data.h"
#if SANOTTS_BENCH_NANO
#  include <snt_nano.h>
#else
#  include <snt_tts.h>
#endif

/* ---- board identity -------------------------------------------------
 * Auto-detected, so a first flash produces an attributable report with no
 * edits. Override by editing SANOTTS_BOARD below, or by building with
 * -DSANOTTS_BOARD="\"my board\"".
 *
 * Do NOT name a variable BOARD_NAME here: the arduino-pico and stm32duino
 * cores both pass -DBOARD_NAME="<board>" on the compiler command line, so
 * the declaration expands to a string literal and every one of their targets
 * fails with `expected unqualified-id before string constant` (measured on
 * rp2040 and STM32 before this was renamed). */
#ifndef SANOTTS_BOARD
   /* esp32 cores pass -DARDUINO_BOARD; arduino-pico and stm32duino pass
    * -DBOARD_NAME; Teensyduino passes neither, only -DARDUINO_<board>, so
    * those are spelled out (values from teensy boards.txt build.board). */
#  if defined(ARDUINO_TEENSY41)
#    define SANOTTS_BOARD "Teensy 4.1"
#  elif defined(ARDUINO_TEENSY40)
#    define SANOTTS_BOARD "Teensy 4.0"
#  elif defined(ARDUINO_TEENSY_MICROMOD)
#    define SANOTTS_BOARD "Teensy MicroMod"
#  elif defined(ARDUINO_TEENSY36)
#    define SANOTTS_BOARD "Teensy 3.6"
#  elif defined(ARDUINO_TEENSY35)
#    define SANOTTS_BOARD "Teensy 3.5"
#  elif defined(ARDUINO_TEENSY32)
#    define SANOTTS_BOARD "Teensy 3.2"
#  elif defined(ARDUINO_BOARD)
#    define SANOTTS_BOARD ARDUINO_BOARD
#  elif defined(BOARD_NAME)
#    define SANOTTS_BOARD BOARD_NAME
#  else
#    define SANOTTS_BOARD "UNKNOWN -- edit SANOTTS_BOARD in this sketch"
#  endif
#endif

static const char *g_board = SANOTTS_BOARD;

/* The Arduino mbed cores -- GIGA R1, Portenta H7, Nicla, Opta, Nano 33 BLE,
 * Nano RP2040 Connect -- define no F_CPU at all, so taking it unguarded fails
 * to compile on every one of them. The clock is only metadata here: RTF is
 * measured directly, not derived from it, so an unknown clock costs nothing.
 * Set -DF_CPU=<hz> if you want it filled in. */
#ifndef F_CPU
#define F_CPU 0L
#endif
static const long CPU_CLOCK_HZ = F_CPU;

/* ---- working arena ----------------------------------------------------
 * Measured on the host against this exact fixture (mcu/test/fixtures/en_us_r7,
 * 134 frames): the runtime produces BIT-IDENTICAL output at every arena size
 * from 88 KB to 320 KB -- corr 0.989148, rms_ratio 0.935022, all the way
 * down. At 86 KB it prints ARENA OOM and aborts. So 88 KB is a hard floor
 * and anything above it is pure speed, not correctness.
 *
 * The extra space goes to opportunistic weight-residency buffers (aa_try in
 * snt_tts.c). They cache decoder weights out of flash; on a flash-bound MCU
 * that is worth several times the runtime. That is why the report includes
 * arena_bytes: an RTF measured with a 96 KB arena is NOT comparable to one
 * measured with 320 KB, even on the same chip.
 *
 * Heap, not a static array: a 320 KB .bss overflows ESP32-S3 internal DRAM
 * under the Arduino core (verified -- the linker rejects it by 109,944 bytes),
 * and a board that cannot link is a board nobody reports from. malloc lets
 * the same sketch degrade instead of failing to build. The runtime aligns the
 * base pointer up to 16 bytes itself and charges that to arena_size, so the
 * request carries 16 spare bytes. */
static const size_t ARENA_LADDER[] = {
  320u * 1024u, 256u * 1024u, 192u * 1024u, 160u * 1024u,
  144u * 1024u, 136u * 1024u, 128u * 1024u, 112u * 1024u,
   96u * 1024u,  88u * 1024u,
};
#if SANOTTS_BENCH_NANO
/* Measured on the host for this row (415 frames): peak 128,944 B. The nano
 * arena is 46.5 KB fixed + ~196 B/frame, so a longer row costs more; this
 * floor is for the row actually embedded above. */
static const size_t ARENA_FLOOR = 136u * 1024u;
#else
static const size_t ARENA_FLOOR = 88u * 1024u;
#endif

/* Build with -DSANOTTS_BENCH_MAX_ARENA=<bytes> to skip the larger rungs.
 * Useful if malloc on your core succeeds but leaves the rest of the sketch
 * with no heap, and it is how extras/bench_host_check.sh exercises the
 * lower rungs rather than only the one this machine happens to take. */
#ifndef SANOTTS_BENCH_MAX_ARENA
#define SANOTTS_BENCH_MAX_ARENA ((size_t)-1)
#endif

/* Streaming correlation against the golden reference. Accumulating in the
 * callback means we never hold the whole waveform: the utterance is 34,304
 * samples, which as float32 would be another 137 KB of RAM for nothing. */
typedef struct {
  uint32_t pos;
  double sa, sb, saa, sbb, sab;
} CorrSink;

static int corr_cb(const float *pcm, int n, void *user) {
  CorrSink *c = (CorrSink *)user;
  for (int i = 0; i < n && c->pos < (uint32_t)SANOTTS_BENCH_N_REF; i++, c->pos++) {
    double a = pcm[i];
    double b = (double)SANOTTS_BENCH_REF[c->pos] / (double)SANOTTS_BENCH_REF_SCALE;
    c->sa += a;  c->sb += b;
    c->saa += a * a;  c->sbb += b * b;  c->sab += a * b;
  }
  return 0;
}

static void run_benchmark() {
  Serial.println();
  Serial.println(F("saanoTTS BoardBenchmark"));
  Serial.println(F("======================="));

  /* Take the largest arena this board will actually hand over. Leave some
   * heap behind: Serial/USB CDC on several cores allocates its own buffers,
   * and an arena that starves them turns a clean number into a hang. */
  uint8_t *arena = NULL;
  size_t arena_size = 0;
  for (size_t i = 0; i < sizeof ARENA_LADDER / sizeof ARENA_LADDER[0]; i++) {
    if (ARENA_LADDER[i] > (size_t)SANOTTS_BENCH_MAX_ARENA) continue;
    if (ARENA_LADDER[i] < ARENA_FLOOR) break;
    arena = (uint8_t *)malloc(ARENA_LADDER[i] + 16);
    if (arena) { arena_size = ARENA_LADDER[i] + 16; break; }
  }
  if (!arena) {
    Serial.print(F("FATAL: could not allocate the "));
    Serial.print((unsigned long)ARENA_FLOOR);
    Serial.println(F(" byte minimum arena."));
    Serial.println(F("This board does not have enough free RAM for the"));
    Serial.println(F("whole-utterance path. See docs/mcu-classes-and-porting.md"));
    Serial.println(F("for the streaming variant, and please still open an issue."));
    return;
  }

  CorrSink sink;
  memset(&sink, 0, sizeof sink);

#if SANOTTS_BENCH_NANO
  snt_nano_config cfg;
  snt_nano_stats  st;
#else
  snt_config cfg;
  snt_stats  st;
#endif
  memset(&cfg, 0, sizeof cfg);
  cfg.front_blob = SANOTTS_BENCH_FRONT_Q8;
  cfg.dec_blob   = SANOTTS_BENCH_MODEL_Q8;
  cfg.arena      = arena;
  cfg.arena_size = arena_size;
#if SANOTTS_BENCH_NANO
  /* Noise-fed decoder: with any other seed the output is a different, equally
   * valid waveform and the correlation against the reference is meaningless. */
  cfg.noise_seed = SANOTTS_BENCH_SEED;
#endif
  /* Pass the golden durations so every board synthesizes the SAME 134
   * frames. Letting the duration model predict its own timing would change
   * the output length per build and make both the correlation gate and the
   * cross-board timing incomparable. This is what mcu/test/golden_main.c
   * does, and it is why the numbers below can be compared at all. */
  cfg.dur_override = SANOTTS_BENCH_DURS;

  memset(&st, 0, sizeof st);

  const uint32_t t_start = micros();
#if SANOTTS_BENCH_NANO
  const int rc = snt_nano_synthesize(&cfg, SANOTTS_BENCH_IDS, SANOTTS_BENCH_N_IDS,
                                     corr_cb, &sink, &st);
#else
  const int rc = snt_synthesize(&cfg, SANOTTS_BENCH_IDS, SANOTTS_BENCH_N_IDS,
                                corr_cb, &sink, &st);
#endif
  const uint32_t t_end = micros();

  const double elapsed_s = (double)(t_end - t_start) / 1e6;
  const double audio_s   = (double)st.samples / (double)SANOTTS_BENCH_SAMPLE_RATE;
  const double rtf       = (audio_s > 0.0) ? elapsed_s / audio_s : 0.0;

  /* Effective int8 throughput = (MACs per second of audio) / RTF. The
   * workload constant belongs to the STACK, not the project: R7 is 45 MMAC/s
   * and the 294k nano is 19. Using one for the other reports a throughput the
   * chip never delivered, so the generator emits the right one and this
   * prints n/a when the lineage has no measured figure. */
  const double eff_mmacs = (rtf > 0.0 && SANOTTS_BENCH_MMAC_PER_S > 0.0f)
                             ? (double)SANOTTS_BENCH_MMAC_PER_S / rtf : 0.0;

  const double n = (double)sink.pos;
  double corr = 0.0, rms_ratio = 0.0;
  if (n > 1.0) {
    const double cov = sink.sab - sink.sa * sink.sb / n;
    corr = cov / sqrt((sink.saa - sink.sa * sink.sa / n) *
                      (sink.sbb - sink.sb * sink.sb / n) + 1e-30);
    rms_ratio = sqrt((sink.saa + 1e-30) / (sink.sbb + 1e-30));
  }
  /* Both gates matter. Correlation is scale-invariant, so on its own it
   * cannot see a gain error -- a defective integer iFFT once scored 0.989
   * while emitting samples ~500x hot. */
  const bool pass = (rc == 0) && (corr > 0.98) &&
                    (rms_ratio > 0.80) && (rms_ratio < 1.25);

  Serial.println();
  Serial.println(F("---- REPORT (paste this whole block) ----"));
  Serial.print(F("board:        ")); Serial.println(g_board);
  Serial.print(F("model:        ")); Serial.println(F(SANOTTS_BENCH_MODEL));
  Serial.print(F("cpu_hz:       "));
  if (CPU_CLOCK_HZ > 0) Serial.println(CPU_CLOCK_HZ);
  else Serial.println(F("unknown -- please add your board's clock"));
  Serial.print(F("arena_bytes:  ")); Serial.println((unsigned long)arena_size);
  Serial.print(F("rc:           ")); Serial.println(rc);
  Serial.print(F("frames:       ")); Serial.println(st.frames);
  Serial.print(F("samples:      ")); Serial.println(st.samples);
  Serial.print(F("compared:     ")); Serial.println((unsigned long)sink.pos);
#if SANOTTS_BENCH_NANO
  Serial.print(F("arena_peak:   ")); Serial.println((unsigned long)st.arena_peak);
#endif
  Serial.print(F("audio_s:      ")); Serial.println(audio_s, 4);
  Serial.print(F("elapsed_s:    ")); Serial.println(elapsed_s, 4);
  Serial.print(F("RTF:          ")); Serial.println(rtf, 4);
  Serial.print(F("eff_MMAC_s:   "));
  if (eff_mmacs > 0.0) Serial.println(eff_mmacs, 1);
  else Serial.println(F("n/a (no measured MAC count for this stack)"));
  Serial.print(F("golden_corr:  ")); Serial.println(corr, 6);
  Serial.print(F("rms_ratio:    ")); Serial.println(rms_ratio, 6);
  Serial.print(F("verdict:      ")); Serial.println(pass ? F("PASS") : F("FAIL"));
  Serial.println(F("---- END REPORT ----"));
  Serial.println();

  if (!pass) {
    Serial.println(F("FAIL means the numbers above are NOT a valid measurement."));
    Serial.println(F("  rc != 0        -> synthesis aborted (arena too small?)"));
    Serial.println(F("  corr <= 0.98   -> kernels disagree with the reference"));
    Serial.println(F("  rms_ratio off  -> gain/scaling bug; speed is meaningless"));
    Serial.println(F("Please report it anyway -- a reproducible FAIL is useful."));
  } else if (rtf <= 1.0) {
    Serial.println(F("RTF <= 1.0: this board synthesizes faster than playback."));
  } else {
    Serial.println(F("RTF > 1.0: functional but offline -- slower than playback."));
  }
  Serial.println();
  Serial.println(F("Press Enter in the serial monitor to run again."));
  free(arena);
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 4000) { /* USB CDC boards need a moment */ }
  /* No-op unless the build enabled the ESP32 second-core worker. Declared
   * unconditionally, so this line is safe on every core. */
  snt_port_dualcore_start();
  run_benchmark();
}

/* The report is easy to miss if the monitor opens after boot -- USB-CDC ports
 * and ST-Link UARTs both drop output nobody was listening to. Any keypress
 * re-runs the whole thing, so "I see nothing" has a one-key fix. */
void loop() {
  if (Serial.available()) {
    while (Serial.available()) Serial.read();
    run_benchmark();
  }
}
