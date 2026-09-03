/* bench_host_shim.h -- the smallest Arduino surface BoardBenchmark.ino uses,
 * so the SKETCH ITSELF (not a re-typed copy of it) can be compiled and run on
 * a workstation. Re-typing the logic is how a host check ends up passing while
 * the shipped sketch is broken; including the real .ino makes that impossible.
 *
 * Only what BoardBenchmark.ino actually touches: Serial print/println with the
 * numeric-precision overload, millis/micros, F(), and F_CPU. */
#ifndef BENCH_HOST_SHIM_H
#define BENCH_HOST_SHIM_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define F(x) (x)
#ifndef F_CPU
#define F_CPU 0L
#endif

inline uint64_t shim_now_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count();
}

inline unsigned long millis() { return (unsigned long)(shim_now_us() / 1000ull); }
inline uint32_t micros() { return (uint32_t)shim_now_us(); }

class ShimSerial {
public:
    void begin(long) {}
    explicit operator bool() const { return true; }
    void println() { std::printf("\n"); }
    void print(const char *s) { std::printf("%s", s); }
    void println(const char *s) { std::printf("%s\n", s); }
    void print(int v) { std::printf("%d", v); }
    void println(int v) { std::printf("%d\n", v); }
    void print(long v) { std::printf("%ld", v); }
    void println(long v) { std::printf("%ld\n", v); }
    void print(unsigned long v) { std::printf("%lu", v); }
    void println(unsigned long v) { std::printf("%lu\n", v); }
    void print(double v, int digits) { std::printf("%.*f", digits, v); }
    void println(double v, int digits) { std::printf("%.*f\n", digits, v); }
    size_t write(const uint8_t *b, size_t n) { return std::fwrite(b, 1, n, stdout); }
    int available() { return 0; }
    int read() { return -1; }
};
static ShimSerial Serial;

#endif
