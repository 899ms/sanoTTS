#!/usr/bin/env python3
"""Generate the Saanotts voice-picker mascot set as crisp 32 px pixel art."""

from pathlib import Path

from PIL import Image, ImageDraw, PngImagePlugin


HERE = Path(__file__).resolve().parent
LOGICAL = 32
SCALE = 8
SIZE = LOGICAL * SCALE

# Palette sampled/derived from docs/assets/saanotts-mcu-hero.png.
BG = "#FAF8F4"
GRID = "#EEEAE4"
GRID_BOLD = "#E8E3DC"
OUTLINE = "#252525"
INK = "#3C3A37"
WHITE = "#FFFDF8"
CREAM = "#F1E5CF"
SHADOW = "#D8D3CC"
CRIMSON = "#DC143C"
CRIMSON_DARK = "#A90F2E"
RED = "#C62E3E"
BLUE = "#2855A6"
NAVY = "#153B6B"
YELLOW = "#F7C843"
ORANGE = "#F07B28"
ORANGE_DARK = "#B64A24"
BROWN = "#8D4A34"
BROWN_DARK = "#5B332C"
TAN = "#C98958"
GREEN = "#278B59"
TEAL = "#147C79"
SILVER = "#C9C8C3"
SILVER_DARK = "#888985"
GREY = "#8E9296"
GREY_DARK = "#5E6367"
SKY = "#3E8FD0"
GOLD = "#E9A72C"
GREEN_DARK = "#186B44"
PINK = "#E39A9A"


def rect(d, box, color, outline=None):
    d.rectangle(box, fill=color, outline=outline)


def px(d, x, y, color):
    d.point((x, y), fill=color)


def ellipse(d, box, color, outline=None, width=1):
    d.ellipse(box, fill=color, outline=outline, width=width)


def poly(d, points, color, outline=None):
    d.polygon(points, fill=color)
    if outline:
        d.line(points + [points[0]], fill=outline, width=1, joint="curve")


def common_canvas():
    """32 px transparent sprite layer; background is added after scaling."""
    im = Image.new("RGBA", (LOGICAL, LOGICAL), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def floor_shadow(d, x0=7, x1=25, y=27):
    rect(d, (x0 + 2, y + 1, x1 + 1, y + 2), SHADOW)
    rect(d, (x0, y, x1, y + 1), "#E2DED8")


def face(d, eyes, smile_y=None):
    for x, y in eyes:
        px(d, x, y, OUTLINE)
        if x + 1 < LOGICAL:
            px(d, x + 1, y - 1, WHITE)
    if smile_y is not None:
        px(d, 15, smile_y, OUTLINE)
        px(d, 16, smile_y + 1, OUTLINE)
        px(d, 17, smile_y, OUTLINE)


def us_pin(d, x, y):
    """6x5 outlined US lapel pin with canton and alternating stripes."""
    rect(d, (x, y, x + 5, y + 4), OUTLINE)
    rect(d, (x + 1, y + 1, x + 4, y + 3), WHITE)
    rect(d, (x + 1, y + 1, x + 4, y + 1), RED)
    rect(d, (x + 1, y + 3, x + 4, y + 3), RED)
    rect(d, (x + 1, y + 1, x + 2, y + 2), BLUE)
    px(d, x + 1, y + 1, WHITE)


def indonesia_badge(d, x, y):
    rect(d, (x, y, x + 5, y + 4), OUTLINE)
    rect(d, (x + 1, y + 1, x + 4, y + 2), RED)
    rect(d, (x + 1, y + 3, x + 4, y + 3), WHITE)


def india_badge(d, x, y):
    rect(d, (x, y, x + 5, y + 5), OUTLINE)
    rect(d, (x + 1, y + 1, x + 4, y + 1), "#E88B2D")
    rect(d, (x + 1, y + 2, x + 4, y + 3), WHITE)
    rect(d, (x + 1, y + 4, x + 4, y + 4), GREEN)
    px(d, x + 2, y + 2, BLUE)


def china_badge(d, x, y):
    rect(d, (x, y, x + 5, y + 5), OUTLINE)
    rect(d, (x + 1, y + 1, x + 4, y + 4), CRIMSON)
    px(d, x + 2, y + 2, YELLOW)
    px(d, x + 1, y + 2, YELLOW)
    px(d, x + 2, y + 1, YELLOW)
    px(d, x + 3, y + 3, YELLOW)


def nepal_flag(d, x, y):
    """Nepal's double-pennon: blue border, crimson inset, white emblems."""
    outer = [(x, y), (x + 6, y + 4), (x + 3, y + 4),
             (x + 7, y + 9), (x, y + 9)]
    poly(d, outer, NAVY, OUTLINE)
    inner = [(x + 1, y + 1), (x + 5, y + 3), (x + 2, y + 3),
             (x + 6, y + 8), (x + 1, y + 8)]
    poly(d, inner, CRIMSON)
    px(d, x + 2, y + 2, WHITE)
    px(d, x + 2, y + 6, WHITE)
    px(d, x + 3, y + 6, WHITE)


def hband_badge(d, x, y, top, mid, bot):
    """6x5 outlined badge, three horizontal bands (Germany, Spain, Russia)."""
    rect(d, (x, y, x + 5, y + 4), OUTLINE)
    rect(d, (x + 1, y + 1, x + 4, y + 1), top)
    rect(d, (x + 1, y + 2, x + 4, y + 2), mid)
    rect(d, (x + 1, y + 3, x + 4, y + 3), bot)


def vband_badge(d, x, y, left, mid, right):
    """6x5 outlined badge, three vertical bands (France, Italy, Romania).
    The inner field is 4 px wide, so the centre band takes two of them."""
    rect(d, (x, y, x + 5, y + 4), OUTLINE)
    rect(d, (x + 1, y + 1, x + 1, y + 3), left)
    rect(d, (x + 2, y + 1, x + 3, y + 3), mid)
    rect(d, (x + 4, y + 1, x + 4, y + 3), right)


def brazil_badge(d, x, y):
    """Green field, yellow lozenge, blue disc."""
    rect(d, (x, y, x + 5, y + 4), OUTLINE)
    rect(d, (x + 1, y + 1, x + 4, y + 3), GREEN)
    px(d, x + 2, y + 1, YELLOW)
    px(d, x + 3, y + 1, YELLOW)
    rect(d, (x + 1, y + 2, x + 4, y + 2), YELLOW)
    px(d, x + 2, y + 3, YELLOW)
    px(d, x + 3, y + 3, YELLOW)
    rect(d, (x + 2, y + 2, x + 3, y + 2), BLUE)


def czech_badge(d, x, y):
    """White over red with the blue hoist wedge."""
    rect(d, (x, y, x + 5, y + 4), OUTLINE)
    rect(d, (x + 1, y + 1, x + 4, y + 2), WHITE)
    rect(d, (x + 1, y + 3, x + 4, y + 3), RED)
    poly(d, [(x + 1, y + 1), (x + 2, y + 2), (x + 1, y + 3)], BLUE)


def turkey_badge(d, x, y):
    """Red field with a white crescent and star, at four pixels of headroom."""
    rect(d, (x, y, x + 5, y + 4), OUTLINE)
    rect(d, (x + 1, y + 1, x + 4, y + 3), CRIMSON)
    px(d, x + 2, y + 1, WHITE)
    px(d, x + 1, y + 2, WHITE)
    px(d, x + 2, y + 3, WHITE)
    px(d, x + 4, y + 2, WHITE)


def jordan_badge(d, x, y):
    """Black/white/green bands with the red hoist triangle."""
    rect(d, (x, y, x + 5, y + 4), OUTLINE)
    rect(d, (x + 1, y + 1, x + 4, y + 1), OUTLINE)
    rect(d, (x + 1, y + 2, x + 4, y + 2), WHITE)
    rect(d, (x + 1, y + 3, x + 4, y + 3), GREEN)
    poly(d, [(x + 1, y + 1), (x + 2, y + 2), (x + 1, y + 3)], CRIMSON)


def german_badge(d, x, y):
    hband_badge(d, x, y, OUTLINE, RED, YELLOW)


def spain_badge(d, x, y):
    hband_badge(d, x, y, RED, YELLOW, RED)


def russia_badge(d, x, y):
    hband_badge(d, x, y, WHITE, BLUE, RED)


def france_badge(d, x, y):
    vband_badge(d, x, y, BLUE, WHITE, RED)


def italy_badge(d, x, y):
    vband_badge(d, x, y, GREEN, WHITE, RED)


def romania_badge(d, x, y):
    vband_badge(d, x, y, BLUE, YELLOW, RED)


def amy():
    im, d = common_canvas()
    floor_shadow(d, 6, 26)
    # Tail behind a plump red-brown body.
    poly(d, [(8, 21), (3, 22), (7, 24), (3, 26), (11, 26)], BROWN_DARK, OUTLINE)
    ellipse(d, (8, 10, 24, 27), BROWN, OUTLINE)
    ellipse(d, (10, 6, 23, 18), BROWN, OUTLINE)
    # Warm crown and cheek.
    rect(d, (13, 7, 20, 8), "#A94A37")
    rect(d, (19, 12, 22, 15), TAN)
    # Breast and wing.
    poly(d, [(12, 16), (16, 14), (21, 17), (20, 25), (14, 26), (11, 22)], CREAM)
    poly(d, [(9, 16), (14, 14), (18, 18), (15, 23), (10, 21)], BROWN_DARK, OUTLINE)
    rect(d, (10, 17, 14, 18), "#B65B40")
    rect(d, (11, 20, 14, 21), TAN)
    # Beak, eye, feet, and a tiny song note.
    poly(d, [(22, 12), (27, 14), (22, 15)], YELLOW, OUTLINE)
    face(d, [(18, 11)])
    rect(d, (13, 27, 14, 28), ORANGE_DARK)
    rect(d, (19, 27, 20, 28), ORANGE_DARK)
    px(d, 6, 9, CRIMSON)
    rect(d, (7, 7, 8, 8), CRIMSON)
    px(d, 9, 9, CRIMSON)
    us_pin(d, 18, 20)
    return im


def amy_small():
    im, d = common_canvas()
    floor_shadow(d, 9, 24)
    # Same markings/palette as Amy, but a larger head and tiny round body.
    poly(d, [(10, 22), (6, 23), (10, 25)], BROWN_DARK, OUTLINE)
    ellipse(d, (10, 14, 23, 27), BROWN, OUTLINE)
    ellipse(d, (9, 7, 24, 20), BROWN, OUTLINE)
    rect(d, (12, 8, 20, 9), "#A94A37")
    rect(d, (19, 13, 23, 16), TAN)
    ellipse(d, (13, 17, 21, 26), CREAM)
    poly(d, [(10, 17), (15, 16), (17, 21), (13, 23)], BROWN_DARK, OUTLINE)
    poly(d, [(23, 13), (27, 15), (23, 16)], YELLOW, OUTLINE)
    face(d, [(18, 12)])
    rect(d, (13, 27, 14, 28), ORANGE_DARK)
    rect(d, (19, 27, 20, 28), ORANGE_DARK)
    us_pin(d, 18, 20)
    # Mini-size cue.
    px(d, 6, 11, CRIMSON)
    px(d, 7, 10, CRIMSON)
    px(d, 8, 11, CRIMSON)
    return im


def kristin():
    im, d = common_canvas()
    floor_shadow(d, 5, 27)
    # Large curled tail behind the seated fox.
    poly(d, [(9, 18), (4, 20), (3, 25), (7, 28), (13, 25),
             (10, 22), (7, 24), (7, 21)], ORANGE_DARK, OUTLINE)
    poly(d, [(4, 25), (7, 28), (11, 26), (8, 23)], WHITE)
    ellipse(d, (11, 15, 24, 28), ORANGE, OUTLINE)
    # Pointed ears and head.
    poly(d, [(10, 11), (9, 4), (15, 8)], ORANGE, OUTLINE)
    poly(d, [(20, 8), (25, 4), (24, 12)], ORANGE, OUTLINE)
    poly(d, [(11, 8), (11, 6), (14, 9)], CREAM)
    poly(d, [(21, 9), (24, 6), (23, 11)], CREAM)
    poly(d, [(11, 9), (17, 6), (24, 10), (24, 17), (18, 21),
             (11, 17)], ORANGE, OUTLINE)
    # White fox mask/muzzle.
    poly(d, [(12, 13), (16, 12), (17, 18), (13, 17)], WHITE)
    poly(d, [(22, 13), (18, 12), (17, 18), (22, 17)], WHITE)
    ellipse(d, (14, 15, 21, 20), CREAM, OUTLINE)
    face(d, [(14, 12), (20, 12)])
    px(d, 17, 16, OUTLINE)
    px(d, 16, 18, OUTLINE)
    px(d, 18, 18, OUTLINE)
    rect(d, (16, 21, 19, 27), WHITE)
    us_pin(d, 20, 21)
    return im


def hfc():
    im, d = common_canvas()
    floor_shadow(d, 6, 26)
    # Calm barn owl: broad wings, heart face, level posture.
    ellipse(d, (7, 10, 25, 28), TAN, OUTLINE)
    poly(d, [(8, 13), (4, 20), (7, 26), (12, 23), (13, 15)], BROWN_DARK, OUTLINE)
    poly(d, [(24, 13), (28, 20), (25, 26), (20, 23), (19, 15)], BROWN_DARK, OUTLINE)
    rect(d, (7, 19, 10, 20), BROWN)
    rect(d, (22, 19, 25, 20), BROWN)
    # Pixel-heart facial disk.
    poly(d, [(10, 8), (15, 7), (16, 10), (17, 7), (22, 8),
             (24, 12), (22, 18), (16, 23), (10, 18), (8, 12)], CREAM, OUTLINE)
    poly(d, [(11, 11), (15, 10), (16, 14), (17, 10), (21, 11),
             (20, 17), (16, 20), (12, 17)], WHITE)
    ellipse(d, (11, 12, 14, 15), OUTLINE)
    ellipse(d, (18, 12, 21, 15), OUTLINE)
    px(d, 12, 12, WHITE)
    px(d, 19, 12, WHITE)
    poly(d, [(15, 16), (17, 16), (16, 19)], YELLOW, OUTLINE)
    rect(d, (12, 27, 14, 28), YELLOW)
    rect(d, (19, 27, 21, 28), YELLOW)
    us_pin(d, 17, 22)
    return im


def vietnamese():
    im, d = common_canvas()
    floor_shadow(d, 4, 28)
    # Low, sturdy water-buffalo body.
    ellipse(d, (5, 15, 25, 27), INK, OUTLINE)
    rect(d, (8, 23, 11, 28), INK, outline=OUTLINE)
    rect(d, (20, 23, 23, 28), INK, outline=OUTLINE)
    rect(d, (7, 27, 12, 28), OUTLINE)
    rect(d, (19, 27, 24, 28), OUTLINE)
    # Horns frame the broad head.
    poly(d, [(10, 14), (5, 13), (2, 9), (3, 7), (6, 11), (12, 11)], CREAM, OUTLINE)
    poly(d, [(22, 14), (27, 13), (30, 9), (29, 7), (26, 11), (20, 11)], CREAM, OUTLINE)
    poly(d, [(9, 12), (11, 8), (16, 7), (21, 8), (23, 12),
             (22, 20), (16, 23), (10, 20)], "#55534F", OUTLINE)
    poly(d, [(10, 10), (7, 9), (8, 14), (11, 14)], INK, OUTLINE)
    poly(d, [(22, 10), (25, 9), (24, 14), (21, 14)], INK, OUTLINE)
    ellipse(d, (11, 16, 21, 21), SILVER_DARK, OUTLINE)
    face(d, [(12, 13), (19, 13)])
    px(d, 14, 18, OUTLINE)
    px(d, 18, 18, OUTLINE)
    # Vietnam scarf: red wrap and end, gold five-pixel star.
    rect(d, (9, 21, 23, 23), CRIMSON, outline=OUTLINE)
    poly(d, [(20, 23), (25, 23), (24, 29), (21, 27)], CRIMSON, OUTLINE)
    px(d, 16, 21, YELLOW)
    px(d, 15, 22, YELLOW)
    px(d, 16, 22, YELLOW)
    px(d, 17, 22, YELLOW)
    return im


def indonesian():
    im, d = common_canvas()
    floor_shadow(d, 5, 27)
    # Long shaggy arms create the orangutan silhouette.
    ellipse(d, (10, 7, 23, 19), ORANGE_DARK, OUTLINE)
    ellipse(d, (9, 15, 23, 27), ORANGE, OUTLINE)
    poly(d, [(10, 15), (6, 16), (3, 26), (6, 28), (11, 22)], ORANGE_DARK, OUTLINE)
    poly(d, [(23, 15), (27, 17), (29, 26), (26, 28), (21, 22)], ORANGE_DARK, OUTLINE)
    rect(d, (4, 26, 8, 28), BROWN_DARK)
    rect(d, (25, 26, 29, 28), BROWN_DARK)
    # Round pale face with orange crest.
    ellipse(d, (11, 10, 22, 20), TAN, OUTLINE)
    rect(d, (12, 7, 14, 10), ORANGE)
    rect(d, (16, 6, 18, 9), ORANGE)
    rect(d, (20, 7, 22, 10), ORANGE)
    ellipse(d, (13, 14, 20, 20), CREAM)
    face(d, [(13, 13), (20, 13)], 18)
    px(d, 16, 16, BROWN_DARK)
    px(d, 17, 16, BROWN_DARK)
    indonesia_badge(d, 18, 21)
    return im


def nepali():
    im, d = common_canvas()
    floor_shadow(d, 4, 28)
    # Large striped tail behind the seated red panda.
    poly(d, [(12, 21), (7, 20), (3, 22), (2, 25), (6, 28), (13, 26)],
         ORANGE_DARK, OUTLINE)
    rect(d, (4, 21, 6, 27), CREAM)
    rect(d, (8, 20, 10, 27), BROWN_DARK)
    ellipse(d, (10, 15, 23, 28), ORANGE_DARK, OUTLINE)
    # Ears and round head.
    ellipse(d, (8, 6, 13, 12), ORANGE_DARK, OUTLINE)
    ellipse(d, (21, 6, 26, 12), ORANGE_DARK, OUTLINE)
    ellipse(d, (10, 7, 24, 21), ORANGE, OUTLINE)
    ellipse(d, (9, 7, 12, 11), CREAM)
    ellipse(d, (22, 7, 25, 11), CREAM)
    # Signature white brows/cheeks and dark mask.
    poly(d, [(11, 11), (16, 10), (15, 14), (11, 16)], CREAM)
    poly(d, [(23, 11), (18, 10), (19, 14), (23, 16)], CREAM)
    poly(d, [(11, 13), (15, 12), (15, 16), (11, 17)], BROWN_DARK)
    poly(d, [(23, 13), (19, 12), (19, 16), (23, 17)], BROWN_DARK)
    ellipse(d, (14, 14, 20, 20), WHITE, OUTLINE)
    face(d, [(13, 14), (21, 14)])
    px(d, 17, 16, OUTLINE)
    px(d, 16, 18, OUTLINE)
    px(d, 18, 18, OUTLINE)
    rect(d, (12, 20, 15, 27), BROWN_DARK)
    # Accessory is explicitly Nepal's two-triangle national flag.
    nepal_flag(d, 21, 18)
    return im


def hindi():
    im, d = common_canvas()
    floor_shadow(d, 3, 29)
    # Peacock fan tail with pixel eyespots.
    poly(d, [(16, 6), (9, 4), (5, 8), (3, 15), (6, 22),
             (12, 24), (16, 22), (20, 24), (27, 21), (29, 14),
             (26, 7), (21, 4)], GREEN, OUTLINE)
    for x, y in [(9, 9), (16, 7), (23, 9), (7, 15), (13, 14),
                 (20, 14), (25, 16), (11, 20), (21, 20)]:
        ellipse(d, (x - 1, y - 1, x + 1, y + 1), YELLOW)
        px(d, x, y, BLUE)
    # Blue body and elegant neck/head.
    ellipse(d, (12, 14, 22, 27), TEAL, OUTLINE)
    rect(d, (14, 9, 19, 20), BLUE, outline=OUTLINE)
    ellipse(d, (13, 6, 20, 13), BLUE, OUTLINE)
    poly(d, [(20, 9), (25, 10), (20, 12)], YELLOW, OUTLINE)
    face(d, [(17, 9)])
    # Three-pixel crest.
    rect(d, (15, 3, 15, 6), OUTLINE)
    rect(d, (17, 2, 17, 6), OUTLINE)
    rect(d, (19, 3, 19, 6), OUTLINE)
    px(d, 15, 2, CRIMSON)
    px(d, 17, 1, CRIMSON)
    px(d, 19, 2, CRIMSON)
    rect(d, (14, 27, 15, 29), YELLOW)
    rect(d, (19, 27, 20, 29), YELLOW)
    india_badge(d, 18, 21)
    return im


def chinese():
    im, d = common_canvas()
    floor_shadow(d, 6, 26)
    # Panda body, ears, limbs.
    ellipse(d, (9, 15, 24, 28), OUTLINE, OUTLINE)
    ellipse(d, (8, 5, 14, 11), OUTLINE)
    ellipse(d, (21, 5, 27, 11), OUTLINE)
    ellipse(d, (9, 7, 26, 22), WHITE, OUTLINE)
    ellipse(d, (7, 18, 12, 27), OUTLINE)
    ellipse(d, (23, 18, 28, 27), OUTLINE)
    ellipse(d, (11, 22, 16, 29), OUTLINE)
    ellipse(d, (19, 22, 24, 29), OUTLINE)
    # Eye patches angle toward a friendly muzzle.
    poly(d, [(11, 12), (15, 10), (17, 13), (14, 17), (11, 16)], OUTLINE)
    poly(d, [(24, 12), (20, 10), (18, 13), (21, 17), (24, 16)], OUTLINE)
    ellipse(d, (14, 14, 21, 21), CREAM, OUTLINE)
    face(d, [(14, 13), (21, 13)])
    px(d, 17, 16, OUTLINE)
    px(d, 16, 18, OUTLINE)
    px(d, 18, 18, OUTLINE)
    china_badge(d, 19, 21)
    return im


def mcu():
    im, d = common_canvas()
    floor_shadow(d, 6, 26)
    # Crimson antenna and bright signal spark.
    rect(d, (16, 2, 17, 7), CRIMSON, outline=OUTLINE)
    px(d, 14, 2, CRIMSON)
    rect(d, (15, 1, 18, 2), CRIMSON)
    px(d, 19, 2, CRIMSON)
    # Friendly robot head.
    rect(d, (10, 6, 23, 13), SILVER, outline=OUTLINE)
    rect(d, (12, 8, 14, 10), OUTLINE)
    rect(d, (19, 8, 21, 10), OUTLINE)
    px(d, 13, 8, WHITE)
    px(d, 20, 8, WHITE)
    rect(d, (15, 11, 18, 11), CRIMSON)
    # ESP32 development board torso, metal radio can, pins and PCB traces.
    rect(d, (9, 13, 24, 27), OUTLINE, outline="#111111")
    rect(d, (11, 15, 22, 23), INK)
    rect(d, (13, 14, 20, 20), SILVER, outline=WHITE)
    rect(d, (14, 15, 19, 18), "#DAD9D4")
    # Antenna trace and small components.
    d.line([(14, 15), (14, 13), (16, 13), (16, 15), (18, 15),
            (18, 13), (20, 13), (20, 15)], fill=OUTLINE, width=1)
    rect(d, (12, 22, 14, 24), "#171717", outline=SILVER_DARK)
    px(d, 18, 22, CRIMSON)
    rect(d, (20, 22, 21, 24), SILVER_DARK)
    for y in (15, 18, 21, 24):
        rect(d, (8, y, 9, y + 1), SILVER)
        rect(d, (24, y, 25, y + 1), SILVER)
    # Blocky arms and feet.
    poly(d, [(8, 15), (5, 16), (4, 22), (7, 23), (10, 19)], SILVER_DARK, OUTLINE)
    poly(d, [(25, 15), (28, 16), (29, 22), (26, 23), (23, 19)], SILVER_DARK, OUTLINE)
    rect(d, (11, 27, 15, 29), SILVER_DARK, outline=OUTLINE)
    rect(d, (19, 27, 23, 29), SILVER_DARK, outline=OUTLINE)
    return im


def trellis():
    """Spider on a woven web -- the factorized trellis the model is named for."""
    im, d = common_canvas()
    floor_shadow(d, 8, 24)
    # Anchor threads radiating from the top corners, then the woven rings.
    for end in ((3, 14), (10, 17), (21, 17), (28, 14)):
        d.line([(16, 1), end], fill=SILVER_DARK, width=1)
    d.line([(3, 3), (28, 3)], fill=SILVER_DARK, width=1)
    for ring in (5, 8, 11):
        d.line([(16 - ring, 3 + ring // 2), (16, 3 + ring),
                (16 + ring, 3 + ring // 2)], fill=SILVER, width=1)
    # Dew beads where the strands cross, the crimson accent of the set.
    px(d, 11, 8, CRIMSON)
    px(d, 21, 8, CRIMSON)
    # Spider: round body, bright eyes, eight blocky legs.
    ellipse(d, (12, 15, 19, 22), OUTLINE)
    ellipse(d, (13, 16, 18, 21), INK)
    px(d, 14, 17, WHITE)
    px(d, 17, 17, WHITE)
    px(d, 14, 18, OUTLINE)
    px(d, 17, 18, OUTLINE)
    rect(d, (15, 20, 16, 20), CRIMSON)
    for y, drop in ((16, 2), (18, 0), (20, 1), (22, 3)):
        d.line([(12, y), (9, y - drop), (7, y - drop + 1)], fill=OUTLINE, width=1)
        d.line([(19, y), (22, y - drop), (24, y - drop + 1)], fill=OUTLINE, width=1)
    # Silk thread anchoring the spider back up into the web.
    d.line([(16, 12), (16, 15)], fill=SILVER_DARK, width=1)
    us_pin(d, 22, 24)
    return im


def _heart_shape(d, cx, top, w, fill, outline):
    """Blocky pixel heart: two round lobes over a pointed base, outlined."""
    r = w // 4
    ellipse(d, (cx - 2 * r, top, cx, top + 2 * r), outline)
    ellipse(d, (cx, top, cx + 2 * r, top + 2 * r), outline)
    poly(d, [(cx - 2 * r, top + r), (cx + 2 * r, top + r), (cx, top + 2 * r + w // 2)], outline)
    ellipse(d, (cx - 2 * r + 1, top + 1, cx - 1, top + 2 * r - 1), fill)
    ellipse(d, (cx + 1, top + 1, cx + 2 * r - 1, top + 2 * r - 1), fill)
    poly(d, [(cx - 2 * r + 1, top + r), (cx + 2 * r - 1, top + r), (cx, top + 2 * r + w // 2 - 2)], fill)
    rect(d, (cx - 2 * r + 1, top + r - 1, cx + 2 * r - 1, top + r + 1), fill)


def heart():
    """A big crimson heart with a face -- the voice is named for its speaker."""
    im, d = common_canvas()
    floor_shadow(d, 6, 26)
    _heart_shape(d, 16, 6, 24, CRIMSON, OUTLINE)
    # highlight on the left lobe
    rect(d, (9, 8, 10, 9), "#F26A84")
    px(d, 11, 7, "#F26A84")
    face(d, [(12, 13), (19, 13)], smile_y=17)
    us_pin(d, 22, 22)
    return im


def heartnano():
    """The same heart, shrunk: a tiny heart standing on a 24 kHz waveform."""
    im, d = common_canvas()
    floor_shadow(d, 8, 24)
    _heart_shape(d, 16, 6, 16, CRIMSON, OUTLINE)
    rect(d, (11, 8, 12, 8), "#F26A84")
    face(d, [(13, 11), (18, 11)], smile_y=14)
    # a short waveform underneath: the 337 KB stack still makes sound
    for i, h in enumerate((1, 3, 2, 4, 2, 3, 1)):
        x = 10 + i * 2
        d.line([(x, 25), (x, 25 - h)], fill=CRIMSON_DARK, width=1)
    us_pin(d, 23, 3)
    return im


# --- the ten languages added on 2026-09-08 ---------------------------------
# Same rules as the set above: a 32 px animal on a transparent layer, one
# national badge, the shared outline/face helpers, nothing anti-aliased.


def german():
    """Stag -- antlers give it a silhouette nothing else in the set has."""
    im, d = common_canvas()
    floor_shadow(d, 6, 26)
    # Antlers first, so the head paints over their roots.
    for sx, dirn in ((12, -1), (20, 1)):
        d.line([(sx, 9), (sx + 2 * dirn, 5), (sx + 3 * dirn, 2)], fill=BROWN_DARK, width=1)
        d.line([(sx + 2 * dirn, 5), (sx + 5 * dirn, 4)], fill=BROWN_DARK, width=1)
        d.line([(sx + 1 * dirn, 7), (sx + 4 * dirn, 6)], fill=BROWN_DARK, width=1)
    ellipse(d, (9, 15, 24, 28), BROWN, OUTLINE)
    poly(d, [(12, 8), (20, 8), (22, 13), (20, 19), (12, 19), (10, 13)], TAN, OUTLINE)
    # Muzzle, ears, dappled flank.
    ellipse(d, (13, 15, 20, 20), CREAM, OUTLINE)
    poly(d, [(10, 10), (6, 8), (9, 13)], TAN, OUTLINE)
    poly(d, [(22, 10), (26, 8), (23, 13)], TAN, OUTLINE)
    px(d, 15, 17, OUTLINE)
    px(d, 18, 17, OUTLINE)
    face(d, [(12, 11), (19, 11)])
    for dx, dy in ((11, 19), (15, 21), (19, 19), (13, 24), (18, 24)):
        px(d, dx, dy, CREAM)
    rect(d, (11, 27, 13, 28), BROWN_DARK)
    rect(d, (20, 27, 22, 28), BROWN_DARK)
    german_badge(d, 20, 21)
    return im


def french():
    """Gallic rooster: comb, wattle, and a fan of tail feathers."""
    im, d = common_canvas()
    floor_shadow(d, 7, 25)
    poly(d, [(9, 18), (3, 12), (2, 18), (5, 22), (10, 23)], NAVY, OUTLINE)
    poly(d, [(9, 17), (4, 14), (4, 19), (10, 21)], BLUE)
    ellipse(d, (8, 12, 23, 27), WHITE, OUTLINE)
    poly(d, [(11, 15), (17, 14), (19, 20), (13, 23)], SILVER)
    ellipse(d, (15, 4, 25, 15), WHITE, OUTLINE)
    # Comb, beak, wattle.
    for cx in (17, 19, 21):
        rect(d, (cx, 2, cx + 1, 4), CRIMSON)
    rect(d, (17, 4, 22, 5), CRIMSON)
    poly(d, [(25, 8), (29, 10), (25, 11)], YELLOW, OUTLINE)
    rect(d, (23, 12, 24, 14), CRIMSON)
    face(d, [(21, 8)])
    rect(d, (12, 27, 14, 28), YELLOW)
    rect(d, (18, 27, 20, 28), YELLOW)
    france_badge(d, 12, 19)
    return im


def spanish():
    """Fighting bull -- black, low-slung, with the horns doing the reading."""
    im, d = common_canvas()
    floor_shadow(d, 3, 26)
    # Body and legs.
    ellipse(d, (3, 15, 20, 26), "#2E2A28", OUTLINE)
    rect(d, (5, 23, 7, 28), "#2E2A28", outline=OUTLINE)
    rect(d, (10, 24, 12, 28), "#221F1D", outline=OUTLINE)
    rect(d, (15, 23, 17, 28), "#2E2A28", outline=OUTLINE)
    poly(d, [(6, 15), (11, 11), (16, 14)], "#3B3634")     # shoulder hump
    d.line([(3, 17), (1, 22), (2, 26)], fill="#2E2A28", width=1)
    px(d, 2, 27, "#3B3634")
    # Head, dropped low the way a bull carries it.
    poly(d, [(16, 13), (23, 11), (28, 14), (29, 20), (25, 24), (19, 22), (16, 18)],
         "#2E2A28", OUTLINE)
    # Horns: two thick cream sweeps, well clear of the skull.
    poly(d, [(17, 12), (14, 6), (12, 4), (14, 4), (17, 8), (20, 11)], CREAM, OUTLINE)
    poly(d, [(26, 12), (29, 6), (31, 5), (31, 7), (29, 9), (28, 12)], CREAM, OUTLINE)
    # Muzzle, nostrils, eye, ear.
    ellipse(d, (22, 18, 29, 24), "#4A4340", OUTLINE)
    px(d, 25, 20, PINK)
    px(d, 27, 20, PINK)
    poly(d, [(16, 14), (12, 15), (16, 18)], "#221F1D", OUTLINE)
    px(d, 21, 15, WHITE)
    px(d, 21, 16, OUTLINE)
    spain_badge(d, 7, 18)
    return im


def italian():
    """Grey wolf, upright and alert -- the Capitoline she-wolf's descendant."""
    im, d = common_canvas()
    floor_shadow(d, 5, 26)
    poly(d, [(10, 20), (4, 19), (2, 24), (7, 27), (12, 25)], GREY_DARK, OUTLINE)
    ellipse(d, (9, 14, 24, 28), GREY, OUTLINE)
    poly(d, [(11, 16), (17, 15), (21, 19), (17, 26), (12, 24)], SILVER)
    poly(d, [(10, 9), (9, 2), (15, 7)], GREY, OUTLINE)
    poly(d, [(20, 7), (26, 2), (25, 9)], GREY, OUTLINE)
    poly(d, [(11, 8), (11, 4), (14, 8)], GREY_DARK)
    poly(d, [(21, 8), (24, 4), (24, 8)], GREY_DARK)
    poly(d, [(10, 8), (17, 5), (25, 8), (24, 16), (17, 20), (10, 16)], GREY, OUTLINE)
    # Long muzzle, the wolf tell.
    poly(d, [(14, 14), (20, 14), (21, 20), (17, 22), (13, 20)], SILVER, OUTLINE)
    rect(d, (16, 18, 18, 19), OUTLINE)
    d.line([(15, 21), (17, 20), (19, 21)], fill=OUTLINE, width=1)
    face(d, [(13, 11), (20, 11)])
    italy_badge(d, 20, 21)
    return im


def portuguese():
    """Blue-and-gold macaw, wings folded, long tail."""
    im, d = common_canvas()
    floor_shadow(d, 8, 24)
    poly(d, [(13, 22), (9, 29), (12, 30), (17, 24)], SKY, OUTLINE)
    poly(d, [(14, 23), (11, 28), (16, 25)], BLUE)
    ellipse(d, (9, 11, 23, 27), SKY, OUTLINE)
    poly(d, [(12, 16), (18, 15), (20, 24), (14, 26)], YELLOW)
    poly(d, [(9, 13), (14, 12), (17, 20), (11, 23)], BLUE, OUTLINE)
    ellipse(d, (12, 4, 24, 16), SKY, OUTLINE)
    # Pale cheek patch, hooked bill.
    ellipse(d, (17, 8, 23, 14), CREAM)
    d.line([(19, 10), (22, 10)], fill=SILVER_DARK, width=1)
    poly(d, [(23, 8), (28, 10), (27, 14), (23, 13)], OUTLINE)
    poly(d, [(23, 9), (26, 11), (23, 12)], "#4A4340")
    face(d, [(19, 9)])
    rect(d, (13, 27, 15, 28), SILVER_DARK)
    rect(d, (18, 27, 20, 28), SILVER_DARK)
    brazil_badge(d, 10, 19)
    return im


def russian():
    """Brown bear: round, heavy, small ears."""
    im, d = common_canvas()
    floor_shadow(d, 5, 27)
    ellipse(d, (6, 13, 26, 28), BROWN, OUTLINE)
    ellipse(d, (11, 18, 21, 27), TAN)
    ellipse(d, (8, 4, 14, 11), BROWN, OUTLINE)
    ellipse(d, (18, 4, 24, 11), BROWN, OUTLINE)
    ellipse(d, (9, 5, 13, 10), BROWN_DARK)
    ellipse(d, (19, 5, 23, 10), BROWN_DARK)
    ellipse(d, (8, 5, 24, 20), BROWN, OUTLINE)
    ellipse(d, (12, 12, 20, 19), TAN, OUTLINE)
    rect(d, (15, 14, 17, 15), OUTLINE)
    d.line([(14, 17), (16, 18), (18, 17)], fill=OUTLINE, width=1)
    face(d, [(11, 10), (20, 10)])
    rect(d, (9, 26, 12, 28), BROWN_DARK)
    rect(d, (20, 26, 23, 28), BROWN_DARK)
    russia_badge(d, 20, 21)
    return im


def czech():
    """Hedgehog -- a spine field no other mascot has."""
    im, d = common_canvas()
    floor_shadow(d, 5, 27)
    poly(d, [(4, 24), (7, 13), (14, 9), (22, 11), (27, 19), (27, 25), (4, 25)],
         BROWN_DARK, OUTLINE)
    # Spines: alternating short strokes over the dome.
    for i, (sx, sy) in enumerate(((7, 17), (9, 13), (12, 11), (15, 10), (18, 11),
                                  (21, 13), (24, 16), (25, 20), (6, 21))):
        d.line([(sx, sy), (sx - 2 + (i % 3), sy - 3)], fill=INK, width=1)
    for sx, sy in ((10, 18), (14, 15), (18, 15), (22, 19), (12, 22), (19, 22)):
        px(d, sx, sy, INK)
    # Pale face pokes out at the front-left.
    poly(d, [(4, 18), (10, 16), (12, 22), (7, 26), (3, 24)], TAN, OUTLINE)
    poly(d, [(3, 21), (7, 20), (7, 24), (3, 24)], CREAM)
    px(d, 3, 22, OUTLINE)
    px(d, 4, 22, OUTLINE)
    face(d, [(7, 20)])
    rect(d, (8, 26, 10, 28), BROWN_DARK)
    rect(d, (18, 26, 20, 28), BROWN_DARK)
    czech_badge(d, 19, 20)
    return im


def romanian():
    """Sheep -- the Miori\u021ba ewe: a cloud of fleece on dark legs."""
    im, d = common_canvas()
    floor_shadow(d, 6, 26)
    for cx, cy, r in ((11, 16, 5), (16, 14, 6), (21, 16, 5),
                      (12, 21, 5), (19, 21, 5), (16, 19, 6)):
        ellipse(d, (cx - r, cy - r, cx + r, cy + r), WHITE, OUTLINE)
    for cx, cy, r in ((11, 16, 4), (16, 14, 5), (21, 16, 4),
                      (12, 21, 4), (19, 21, 4), (16, 19, 5)):
        ellipse(d, (cx - r, cy - r, cx + r, cy + r), WHITE)
    for sx, sy in ((12, 17), (19, 17), (15, 21), (22, 20), (10, 21)):
        px(d, sx, sy, SHADOW)
    # Dark face and ears at the right.
    poly(d, [(21, 12), (27, 11), (28, 18), (23, 20), (20, 17)], INK, OUTLINE)
    poly(d, [(21, 12), (18, 10), (22, 15)], INK, OUTLINE)
    poly(d, [(27, 11), (30, 12), (27, 16)], INK, OUTLINE)
    px(d, 23, 14, WHITE)
    px(d, 26, 14, WHITE)
    rect(d, (25, 17, 27, 18), "#3B3634")
    rect(d, (11, 25, 12, 28), INK)
    rect(d, (18, 25, 19, 28), INK)
    romania_badge(d, 8, 22)
    return im


def turkish():
    """Van cat: white, one dark ear and a dark ringed tail."""
    im, d = common_canvas()
    floor_shadow(d, 6, 26)
    poly(d, [(22, 22), (27, 18), (29, 12), (27, 11), (25, 17), (21, 20)],
         "#C7623A", OUTLINE)
    for ty in (12, 14, 16):
        px(d, 27, ty, BROWN_DARK)
    ellipse(d, (8, 14, 24, 28), WHITE, OUTLINE)
    ellipse(d, (10, 17, 21, 27), WHITE)
    poly(d, [(9, 10), (8, 3), (15, 8)], WHITE, OUTLINE)
    poly(d, [(19, 8), (25, 3), (25, 10)], "#C7623A", OUTLINE)
    poly(d, [(10, 9), (10, 5), (13, 9)], PINK)
    poly(d, [(21, 9), (24, 5), (24, 9)], PINK)
    ellipse(d, (8, 6, 25, 20), WHITE, OUTLINE)
    ellipse(d, (13, 13, 20, 19), WHITE)
    px(d, 16, 15, PINK)
    px(d, 17, 15, PINK)
    d.line([(15, 17), (16, 18), (17, 17)], fill=SILVER_DARK, width=1)
    # The Van cat's odd eyes: one blue, one amber.
    rect(d, (12, 11, 13, 12), BLUE)
    rect(d, (20, 11, 21, 12), GOLD)
    px(d, 12, 11, WHITE)
    px(d, 20, 11, WHITE)
    for wx in (6, 5):
        px(d, wx, 15, SILVER_DARK)
        px(d, wx, 17, SILVER_DARK)
    turkey_badge(d, 15, 21)
    return im


def arabic():
    """Dromedary camel: one hump, long neck, sand palette."""
    im, d = common_canvas()
    floor_shadow(d, 4, 27)
    ellipse(d, (5, 16, 22, 26), TAN, OUTLINE)
    poly(d, [(8, 18), (12, 11), (17, 11), (21, 18)], TAN, OUTLINE)
    poly(d, [(9, 17), (13, 12), (17, 12), (20, 17)], "#D9A472")
    rect(d, (7, 24, 9, 28), TAN, outline=OUTLINE)
    rect(d, (12, 24, 14, 28), "#B9834F", outline=OUTLINE)
    rect(d, (17, 24, 19, 28), TAN, outline=OUTLINE)
    # Neck and small head.
    poly(d, [(19, 20), (21, 11), (25, 8), (27, 9), (24, 13), (23, 21)],
         TAN, OUTLINE)
    poly(d, [(24, 6), (29, 7), (30, 11), (26, 12), (23, 10)], TAN, OUTLINE)
    poly(d, [(24, 5), (26, 3), (26, 6)], "#B9834F", OUTLINE)
    px(d, 29, 9, OUTLINE)
    px(d, 29, 10, OUTLINE)
    d.line([(26, 11), (29, 11)], fill=BROWN_DARK, width=1)
    face(d, [(25, 8)])
    d.line([(5, 19), (2, 23)], fill=TAN, width=1)
    px(d, 2, 24, BROWN_DARK)
    jordan_badge(d, 9, 19)
    return im


def render(sprite, filename, description):
    # Scale only once, with no antialiasing, to retain the 32 px grid.
    sprite = sprite.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    out = Image.new("RGB", (SIZE, SIZE), BG)
    bgd = ImageDraw.Draw(out)
    # Subtle graph-paper treatment from the hero reference.
    for pos in range(0, SIZE, 16):
        color = GRID_BOLD if pos % 64 == 0 else GRID
        bgd.line((pos, 0, pos, SIZE - 1), fill=color, width=1)
        bgd.line((0, pos, SIZE - 1, pos), fill=color, width=1)
    out.paste(sprite.convert("RGB"), mask=sprite.getchannel("A"))
    info = PngImagePlugin.PngInfo()
    info.add_text("Title", description)
    info.add_text("Style", "32x32 logical pixel art; 8x nearest-neighbor; Saanotts palette")
    info.add_text("Generator", "web/assets/mascots/generate_mascots.py")
    out.save(HERE / filename, format="PNG", compress_level=6, pnginfo=info)


def main():
    mascots = {
        "amy.png": (amy, "Amy — warm red-brown songbird with United States flag pin"),
        "kristin.png": (kristin, "Kristin — bright orange fox with United States flag pin"),
        "hfc.png": (hfc, "HFC — calm barn owl with United States flag pin"),
        "amy-small.png": (amy_small, "Amy Small — chibi songbird with United States flag pin"),
        "vietnamese.png": (vietnamese, "Vietnamese — water buffalo with Vietnam flag scarf"),
        "indonesian.png": (indonesian, "Indonesian — orangutan with Indonesia flag badge"),
        "nepali.png": (nepali, "Nepali — red panda with Nepal double-pennon flag"),
        "hindi.png": (hindi, "Hindi — peacock with India flag badge"),
        "chinese.png": (chinese, "Chinese — giant panda with China flag badge"),
        "mcu.png": (mcu, "MCU — tiny robot with ESP32 development board torso"),
        "trellis.png": (trellis, "Trellis — spider on a woven web with United States flag pin"),
        "heart.png": (heart, "Heart — crimson heart with a face and United States flag pin"),
        "heartnano.png": (heartnano, "Heart-nano — tiny crimson heart over a waveform, United States flag pin"),
        "german.png": (german, "German — stag with Germany flag badge"),
        "french.png": (french, "French — Gallic rooster with France flag badge"),
        "spanish.png": (spanish, "Spanish — black bull with Spain flag badge"),
        "italian.png": (italian, "Italian — grey wolf with Italy flag badge"),
        "portuguese.png": (portuguese, "Portuguese — blue-and-gold macaw with Brazil flag badge"),
        "russian.png": (russian, "Russian — brown bear with Russia flag badge"),
        "czech.png": (czech, "Czech — hedgehog with Czech Republic flag badge"),
        "romanian.png": (romanian, "Romanian — fleecy ewe with Romania flag badge"),
        "turkish.png": (turkish, "Turkish — odd-eyed Van cat with Turkey flag badge"),
        "arabic.png": (arabic, "Arabic — dromedary camel with Jordan flag badge"),
    }
    for filename, (builder, description) in mascots.items():
        render(builder(), filename, description)


if __name__ == "__main__":
    main()
