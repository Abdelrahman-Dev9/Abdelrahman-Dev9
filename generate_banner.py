#!/usr/bin/env python3
"""
Master-Prompt Phase 1 banner generator.
Floyd–Steinberg dithered dot portrait → dark.svg / light.svg (1180×610).
"""
from __future__ import annotations

import html
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

ROOT = Path(__file__).resolve().parent
AVATAR = Path("/tmp/gh-avatar.png")
W, H = 1180, 610
# Portrait grid (slightly coarser than 300×340 for README moiré / file size)
GRID_W, GRID_H = 160, 200
DOT = 1.55  # square half-size-ish via rect size

# Profile data (from CV + user)
NAME = "Abdelrahman Mahmoud"
HANDLE = "Abdelrahman-Dev9"
ROLE = "React Native Developer"
ORIGIN = "Mansoura, Egypt"
EDU = "CS & IT · Suez Canal · 2021–2025"
STATUS = "Building · Learning · Shipping"
TOOLCHAIN = "VS Code · Git · Expo · Bun"
LANGS = "TypeScript · JavaScript"
FRONTEND = "React · Next.js · Tailwind"
BACKEND = "Node · Express · NestJS"
DATABASE = "PostgreSQL · Prisma"
INFRA = "Vercel · Supabase · Neon"
MAIL = "abdoelnagar198@gmail.com"
PORTFOLIO = "portfolio · vercel.app"
LINKEDIN = "linkedin.com/in/abdelrahman-elnagar"
GITHUB = "github.com/Abdelrahman-Dev9"

LINKS = {
    "GitHub": "https://github.com/Abdelrahman-Dev9",
    "LinkedIn": "https://www.linkedin.com/in/abdelrahman-elnagar-ab602b291/",
    "Portfolio": "https://my-portfolio-olive-theta-35.vercel.app/",
    "Email": "mailto:abdoelnagar198@gmail.com",
}


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """1-bit Floyd–Steinberg, serpentine."""
    img = gray.astype(np.float64).copy()
    h, w = img.shape
    for y in range(h):
        left_to_right = y % 2 == 0
        xs = range(w) if left_to_right else range(w - 1, -1, -1)
        for x in xs:
            old = img[y, x]
            new = 255.0 if old >= 128 else 0.0
            img[y, x] = new
            err = old - new
            if left_to_right:
                if x + 1 < w:
                    img[y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        img[y + 1, x - 1] += err * 3 / 16
                    img[y + 1, x] += err * 5 / 16
                    if x + 1 < w:
                        img[y + 1, x + 1] += err * 1 / 16
            else:
                if x - 1 >= 0:
                    img[y, x - 1] += err * 7 / 16
                if y + 1 < h:
                    if x + 1 < w:
                        img[y + 1, x + 1] += err * 3 / 16
                    img[y + 1, x] += err * 5 / 16
                    if x - 1 >= 0:
                        img[y + 1, x - 1] += err * 1 / 16
    return img >= 128


def segment_subject(rgb: Image.Image) -> np.ndarray:
    """Keep largest bright/subject component; clear flat background."""
    arr = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    # Assume background is near corners
    corners = np.concatenate(
        [
            arr[:8, :8].reshape(-1, 3),
            arr[:8, -8:].reshape(-1, 3),
            arr[-8:, :8].reshape(-1, 3),
            arr[-8:, -8:].reshape(-1, 3),
        ]
    )
    bg = corners.mean(axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    mask = dist > 28  # subject farther from bg colour
    # Morphological close via max/min filters
    m = Image.fromarray((mask * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(5))
    m = m.filter(ImageFilter.MinFilter(5))
    mask = np.asarray(m) > 128
    # Keep largest connected component (simple flood)
    from collections import deque

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = None
    best_size = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            comp = []
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > best_size:
                best_size = len(comp)
                best = comp
    out = np.zeros_like(mask)
    if best:
        for y, x in best:
            out[y, x] = True
    return out


def prepare_portrait(theme: str):
    im = Image.open(AVATAR).convert("RGB")
    w, h = im.size
    # Head + shoulders
    im = im.crop((int(w * 0.08), int(h * 0.0), int(w * 0.92), int(h * 0.95)))
    im = im.resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)
    mask = segment_subject(im)

    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.3)
    g = g.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    gray = np.asarray(g, dtype=np.float64)

    if theme == "dark":
        # Dots = lit subject → invert so bright face = True after dither of inverted?
        # Spec: dark mode draws lit subject. So dither the subject luminance;
        # ON dots where face is bright.
        work = gray.copy()
        work[~mask] = 0  # force bg black → no dots after invert path
        # Invert so bright areas become high for FS → white bits
        work = work
        bits = floyd_steinberg(work)
        bits = bits & mask
        # Hard clear bleed outside mask
        bits[~mask] = False
    else:
        # Light mode: dots draw dark parts; keep background
        work = 255 - gray  # dark features → high
        bits = floyd_steinberg(work)

    ys, xs = np.where(bits)
    return list(zip(xs.tolist(), ys.tolist())), mask


def dots_to_path_runs(dots: list[tuple[int, int]], ox: float, oy: float, cell: float) -> str:
    """Pack horizontal runs of adjacent dots into path commands."""
    by_row: dict[int, list[int]] = {}
    for x, y in dots:
        by_row.setdefault(y, []).append(x)
    parts = []
    s = cell * 0.82  # dot size
    for y in sorted(by_row):
        xs = sorted(by_row[y])
        run_start = xs[0]
        prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            # emit run run_start..prev
            n = prev - run_start + 1
            px = ox + run_start * cell
            py = oy + y * cell
            # one rect per run via path H
            parts.append(f"M{px:.2f},{py:.2f}h{n * cell * 0.9:.2f}v{s:.2f}h{-n * cell * 0.9:.2f}z")
            if x is not None:
                run_start = prev = x
    return "".join(parts)


def make_intro_groups(dots: list[tuple[int, int]], n_groups: int = 48) -> list[list[tuple[int, int]]]:
    """Scatter dots into interleaved random groups for shimmer intro."""
    rng = random.Random(42)
    shuffled = dots[:]
    rng.shuffle(shuffled)
    groups = [[] for _ in range(n_groups)]
    for i, d in enumerate(shuffled):
        groups[i % n_groups].append(d)
    return groups


def info_row(label: str, value: str, y: float, label_c: str, value_c: str, muted: str) -> str:
    # Fixed columns with textLength for alignment
    lx, vx = 520, 780
    leader_x0, leader_x1 = 620, 770
    # dotted leader as small circles or path
    dots = []
    x = leader_x0
    while x < leader_x1:
        dots.append(f'<circle cx="{x:.0f}" cy="{y - 4}" r="0.9" fill="{muted}" opacity="0.55"/>')
        x += 5
    return f'''
    <text x="{lx}" y="{y}" fill="{label_c}"
      font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
      font-size="13" textLength="95" lengthAdjust="spacingAndGlyphs">{esc(label)}</text>
    {''.join(dots)}
    <text x="{vx}" y="{y}" fill="{value_c}"
      font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
      font-size="13" textLength="340" lengthAdjust="spacingAndGlyphs">{esc(value)}</text>
    '''


def build_svg(theme: str) -> str:
    dark = theme == "dark"
    if dark:
        bg, panel, stroke = "#0A101F", "#0F172A", "#1E293B"
        text, muted, label = "#F8FAFC", "#94A3B8", "#64748B"
        portrait = "#A78BFA"
        chrome = "#22D3EE"
        accent = "#10B981"
        header = "#070B14"
        live = "#EF4444"
    else:
        bg, panel, stroke = "#F8FAFC", "#FFFFFF", "#E2E8F0"
        text, muted, label = "#0F172A", "#64748B", "#94A3B8"
        portrait = "#7C3AED"
        chrome = "#0891B2"
        accent = "#059669"
        header = "#EEF2FF"
        live = "#DC2626"

    dots, _mask = prepare_portrait(theme)
    print(f"  {theme}: {len(dots)} dots")

    # Portrait frame geometry — left ~38%
    frame_x, frame_y = 28, 58
    frame_w, frame_h = 430, 520
    # Map grid into frame with padding
    pad = 18
    cell = min((frame_w - 2 * pad) / GRID_W, (frame_h - 40 - 2 * pad) / GRID_H)
    ox = frame_x + pad + (frame_w - 2 * pad - GRID_W * cell) / 2
    oy = frame_y + 36 + pad

    groups = make_intro_groups(dots, 48)
    # Intro: each group fades 0→1 staggered over ~2s within 3.2s total
    # Keep base layer fully visible for GitHub (no SMIL) + animated overlay groups
    base_path = dots_to_path_runs(dots, ox, oy, cell)

    group_paths = []
    for gi, gdots in enumerate(groups):
        if not gdots:
            continue
        d = dots_to_path_runs(gdots, ox, oy, cell)
        begin = 0.15 + (gi / max(len(groups) - 1, 1)) * 2.0
        # For GitHub: start visible. Animation only pulses opacity slightly.
        group_paths.append(
            f'<path d="{d}" fill="{portrait}" shape-rendering="crispEdges" opacity="0">'
            f'<animate attributeName="opacity" values="0;1" dur="0.35s" begin="{begin:.2f}s" fill="freeze"/>'
            f"</path>"
        )

    # Visible-by-default base (GitHub-safe). Intro groups also freeze to 1.
    # After intro, hide base? No — keep base always visible; intro groups are additive shimmer only if we start base at 0.
    # Spec wants shimmer in. For GitHub: show base always. For browsers with SMIL: start base opacity 0 then show groups then show base.
    portrait_layer = f'''
    <g id="portrait-base">
      <path d="{base_path}" fill="{portrait}" shape-rendering="crispEdges" opacity="1"/>
    </g>
    <g id="portrait-intro" opacity="0">
      <!-- intro shimmer overlays (visible when SMIL runs; harmless if not) -->
      {''.join(group_paths)}
    </g>
    '''

    # Subtle loop float on whole portrait
    portrait_anim = f'''
    <g id="visual-map">
      <animateTransform attributeName="transform" type="translate"
        values="0,0;0,-1.5;0,0" dur="8s" repeatCount="indefinite"/>
      {portrait_layer}
    </g>
    '''

    # Info rows
    rows = [
        ("Subject", NAME),
        ("Role", ROLE),
        ("Origin", ORIGIN),
        ("Education", EDU),
        ("Status", STATUS),
        ("ToolChain", TOOLCHAIN),
        ("Core.Lang", LANGS),
        ("Core.Frontend", FRONTEND),
        ("Core.Backend", BACKEND),
        ("Core.Database", DATABASE),
        ("Core.Infra", INFRA),
        ("Grid.Mail", MAIL),
        ("Grid.Portfolio", PORTFOLIO),
        ("Grid.LinkedIn", LINKEDIN),
        ("Grid.GitHub", GITHUB),
    ]
    y0 = 118
    row_xml = []
    for i, (lab, val) in enumerate(rows):
        row_xml.append(info_row(lab, val, y0 + i * 23, label, text, muted))

    aria = (
        f"{NAME}, {ROLE}. {ORIGIN}. {EDU}. "
        f"Stack: {LANGS}; {FRONTEND}; {BACKEND}; {DATABASE}."
    )

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"
  role="img" aria-label="{esc(aria)}">
  <title>{esc(NAME)} — {esc(ROLE)}</title>
  <desc>{esc(aria)}</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{bg}"/>
      <stop offset="100%" stop-color="{panel if dark else "#EEF2FF"}"/>
    </linearGradient>
    <radialGradient id="glow" cx="20%" cy="30%" r="50%">
      <stop offset="0%" stop-color="{portrait}" stop-opacity="{0.18 if dark else 0.08}"/>
      <stop offset="100%" stop-color="{portrait}" stop-opacity="0"/>
    </radialGradient>
    <filter id="soft" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="6" stdDeviation="10" flood-color="#000" flood-opacity="{0.4 if dark else 0.08}"/>
    </filter>
    <clipPath id="frame"><rect width="{W}" height="{H}" rx="18" ry="18"/></clipPath>
  </defs>

  <g clip-path="url(#frame)">
    <rect width="{W}" height="{H}" fill="url(#bg)"/>
    <rect width="{W}" height="{H}" fill="url(#glow)"/>

    <!-- Terminal chrome -->
    <g filter="url(#soft)">
      <rect x="16" y="14" width="1148" height="582" rx="16" fill="{panel}" stroke="{stroke}" stroke-width="1"/>
      <rect x="16" y="14" width="1148" height="36" rx="16" fill="{header}"/>
      <rect x="16" y="32" width="1148" height="18" fill="{header}"/>
      <line x1="16" y1="50" x2="1164" y2="50" stroke="{stroke}"/>
      <circle cx="40" cy="32" r="5" fill="#FF5F57"/>
      <circle cx="58" cy="32" r="5" fill="#FEBC2E"/>
      <circle cx="76" cy="32" r="5" fill="#28C840"/>
      <text x="590" y="37" text-anchor="middle" fill="{muted}"
        font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace" font-size="12"
        >profile.sh --live</text>

      <!-- LEFT: VISUAL.MAP -->
      <rect x="{frame_x}" y="{frame_y}" width="{frame_w}" height="{frame_h}" rx="14"
        fill="{header if dark else "#F1F5F9"}" stroke="{stroke}"/>
      <text x="{frame_x + 18}" y="{frame_y + 26}" fill="{chrome}"
        font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
        font-size="11" letter-spacing="2" font-weight="700">VISUAL.MAP</text>
      <text x="{frame_x + frame_w - 18}" y="{frame_y + 26}" text-anchor="end" fill="{label}"
        font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace" font-size="10"
        >dither · portrait</text>
      {portrait_anim}

      <!-- RIGHT: SYSTEM.INFO -->
      <text x="500" y="78" fill="{chrome}"
        font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
        font-size="13" letter-spacing="2" font-weight="700">SYSTEM.INFO</text>

      <!-- LIVE badge -->
      <rect x="1088" y="62" width="52" height="20" rx="6" fill="{live}" opacity="0.15"/>
      <circle cx="1100" cy="72" r="3.5" fill="{live}">
        <animate attributeName="opacity" values="1;0.25;1" dur="1.4s" repeatCount="indefinite"/>
      </circle>
      <text x="1108" y="76" fill="{live}"
        font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
        font-size="11" font-weight="700">LIVE</text>

      <!-- Handle pill -->
      <rect x="500" y="88" width="200" height="22" rx="11" fill="{chrome}" opacity="0.12" stroke="{chrome}" stroke-width="1"/>
      <text x="600" y="103" text-anchor="middle" fill="{chrome}"
        font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
        font-size="12" font-weight="600">@{esc(HANDLE)}</text>

      {''.join(row_xml)}

      <!-- Footer links (README has clickable badges; SVG links often stripped) -->
      <text x="500" y="560" fill="{label}"
        font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace" font-size="11"
        >$ ready · open to opportunities</text>
      <text x="1148" y="560" text-anchor="end" fill="{accent}"
        font-family="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace" font-size="11"
        >Better code -&gt; Better tomorrow</text>
    </g>
  </g>
</svg>
'''


def main() -> None:
    for theme in ("dark", "light"):
        svg = build_svg(theme)
        out = ROOT / f"{theme}.svg"
        out.write_text(svg, encoding="utf-8")
        kb = out.stat().st_size / 1024
        print(f"wrote {out.name} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
