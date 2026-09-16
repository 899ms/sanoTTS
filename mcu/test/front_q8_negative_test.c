/* front_q8_negative_test.c -- the blob checks, exercised.
 * A quantised runtime that reads a wrong or stale blob does not crash; it
 * synthesizes plausible garbage. These are the cases init() must refuse. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "snt_front_q8.h"

static void *xload(const char *p, size_t *nb) {
    FILE *fh = fopen(p, "rb"); long sz; void *b;
    if (!fh) { fprintf(stderr, "missing %s\n", p); exit(1); }
    fseek(fh, 0, SEEK_END); sz = ftell(fh); fseek(fh, 0, SEEK_SET);
    b = malloc((size_t)sz); if (!b || fread(b, 1, (size_t)sz, fh) != (size_t)sz) exit(1);
    fclose(fh); *nb = (size_t)sz; return b;
}

static int expect(const char *what, int rc, int want_nonzero) {
    int ok = want_nonzero ? (rc != 0) : (rc == 0);
    printf("  %-44s rc %-4d %s\n", what, rc, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    size_t mnb, wnb, other_nb;
    unsigned char *meta, *other;
    int8_t *w;
    snt_front_q8_model m;
    int bad = 0;
    if (argc < 3) { fprintf(stderr, "usage: %s <front_dir> <dec_dir>\n", argv[0]); return 2; }
    {
        char p[512];
        snprintf(p, sizeof p, "%s/front_meta_q8.bin", argv[1]); meta = xload(p, &mnb);
        snprintf(p, sizeof p, "%s/front_weights_q8.bin", argv[1]); w = (int8_t *)xload(p, &wnb);
        snprintf(p, sizeof p, "%s/meta_q8.bin", argv[2]); other = xload(p, &other_nb);
    }
    printf("== blob validation\n");
    bad += expect("the real blob loads", snt_front_q8_init(&m, meta, mnb, w, wnb), 0);
    bad += expect("decoder meta rejected (wrong magic)",
                  snt_front_q8_init(&m, other, other_nb, w, wnb), 1);
    bad += expect("truncated meta rejected",
                  snt_front_q8_init(&m, meta, mnb / 2, w, wnb), 1);
    bad += expect("short weight blob rejected",
                  snt_front_q8_init(&m, meta, mnb, w, wnb / 2), 1);
    bad += expect("NULL weights rejected",
                  snt_front_q8_init(&m, meta, mnb, NULL, wnb), 1);
    {   /* a dimension edited after export must not be tolerated */
        unsigned char *t = malloc(mnb); memcpy(t, meta, mnb);
        t[36] = (unsigned char)(t[36] + 1);   /* a_hidden */
        bad += expect("tampered a_hidden rejected",
                      snt_front_q8_init(&m, t, mnb, w, wnb), 1);
        free(t);
    }
    {   /* version bump must fail closed, not be ignored */
        unsigned char *t = malloc(mnb); memcpy(t, meta, mnb);
        t[4] = 2;
        bad += expect("future version rejected",
                      snt_front_q8_init(&m, t, mnb, w, wnb), 1);
        free(t);
    }
    printf("%s\n", bad ? "NEGATIVE TESTS FAILED" : "all negative tests passed");
    return bad ? 1 : 0;
}
