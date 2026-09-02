/* bench_host_main.cpp -- run examples/BoardBenchmark/BoardBenchmark.ino on the
 * host and gate on what it prints.
 *
 * The point is that this compiles the REAL sketch source. If someone changes
 * the arena handling, the golden gate or the REPORT block in the .ino, this
 * check moves with it. It exits nonzero unless the sketch prints PASS.
 *
 * The reference numbers come from mcu/test/golden_main.c on the same fixture:
 * corr 0.989148, rms_ratio 0.935022, 134 frames, 34304 samples. */
#include "bench_host_shim.h"

#include <string>

/* The sketch defines setup()/loop() and pulls in the generated data header. */
#include "../examples/BoardBenchmark/BoardBenchmark.ino"

int main() {
    setup();
    loop();
    return 0;
}
