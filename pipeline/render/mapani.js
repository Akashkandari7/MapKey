// mapani.js: the map-explainer drawing engine. Plugs into kit.js (ReelMimic engine-kit runtime).
//
// Everything is a pure function of t (frames render in parallel and out of order) and every style value comes from
// TOKENS (style/tokens.json); the shot list comes from SCENES (work/<episode>/scenes.json). World coordinates are Web
// Mercator metres; a camera is {x, y, mpp} (metres per output pixel). One camera draws the basemap AND every overlay,
// so highlights, routes and labels stay locked to the geography while the camera moves.

const MAPANI = (() => {
  const LOOK = () => TOKENS.looks[SCENES.look];
  const SAFE = () => TOKENS.format.safe_area;

  // ---------- easing (keep in sync with EASES in pipeline/plan.py) ----------
  const EASE = {
    // easeOutSine with a soft (C1) start over the first 20 %: no jolt when a move begins during the drift
    softOutSine: u => { const a = .2, g = u < a ? u * u / (2 * a) : u - a / 2; return Math.sin(Math.PI / 2 * g / (1 - a / 2)); },
    linear: u => u,
    easeOutSine: u => Math.sin(u * Math.PI / 2),
    easeInOutSine: u => -(Math.cos(Math.PI * u) - 1) / 2,
    easeOutCubic: u => 1 - Math.pow(1 - u, 3),
    easeInOutCubic: u => u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2,
    backOut: u => { const s = 1.70158; return 1 + (s + 1) * Math.pow(u - 1, 3) + s * Math.pow(u - 1, 2); },
  };

  // ---------- camera ----------
  // Keep identical to camera_at in pipeline/plan.py: each move eases from wherever the camera is (drift included)
  // to its framing; between moves the camera drifts in from where it arrived (look A: never fully still).
  function camAt(shot, t) { return camKeys(shot.camera, shot.camera.length, t, shot.drift_per_s || 0); }
  function camKeys(keys, n, t, d) {
    const k = keys[n - 1];
    if (n === 1) return { x: k.x, y: k.y, mpp: k.mpp * Math.exp(-d * Math.max(0, t - k.t)) };
    const t0 = k.t0 !== undefined ? k.t0 : keys[n - 2].t;
    if (t <= t0) return camKeys(keys, n - 1, t, d);
    const s = camKeys(keys, n - 1, t0, d);  // a move starts from wherever the camera really is: no jolt
    if (t < k.t) {
      const e = (EASE[k.ease] || EASE.easeOutSine)((t - t0) / Math.max(1e-6, k.t - t0));
      return { x: s.x + (k.x - s.x) * e, y: s.y + (k.y - s.y) * e, mpp: Math.exp(Math.log(s.mpp) + (Math.log(k.mpp) - Math.log(s.mpp)) * e) };
    }
    return { x: k.x, y: k.y, mpp: k.mpp * Math.exp(-d * (t - k.t)) };
  }
  const sx = (c, X) => (X - c.x) / c.mpp + W / 2;
  const sy = (c, Y) => (c.y - Y) / c.mpp + H / 2;

  // ---------- assets ----------
  const IMG = {};
  async function setup() {
    // createImageBitmap, not img.decode(): Chrome defers decode() in background tabs, and render.mjs runs several
    // tabs in parallel; a deferred decode left a worker waiting forever for window.ready
    for (const lv of SCENES.imagery || []) {
      const im = new Image();
      await new Promise((ok, bad) => { im.onload = ok; im.onerror = () => bad(new Error('basemap failed to load: ' + lv.file)); im.src = lv.file; });
      IMG[lv.file] = await createImageBitmap(im);
    }
    for (const s of SCENES.shots) for (const L of s.layers || []) if (L.type === 'picture' && !IMG[L.file]) {
      const im = new Image();
      await new Promise((ok, bad) => { im.onload = ok; im.onerror = () => bad(new Error('picture failed to load: ' + L.file)); im.src = L.file; });
      IMG[L.file] = await createImageBitmap(im);
    }
    // per-geometry cumulative lengths for progressive route drawing
    for (const g of Object.values(SCENES.geometry)) if (g.kind === 'polygon') {
      let x0 = Infinity, x1 = -Infinity;
      for (const poly of g.coords) for (const [X] of poly[0]) { x0 = Math.min(x0, X); x1 = Math.max(x1, X); }
      g.width = x1 - x0;
    }
    for (const g of Object.values(SCENES.geometry)) if (g.kind === 'line') {
      const L = [0]; for (let i = 1; i < g.coords.length; i++) L.push(L[i - 1] + Math.hypot(g.coords[i][0] - g.coords[i - 1][0], g.coords[i][1] - g.coords[i - 1][1]));
      g.len = L;
    }
    const fams = Object.values(TOKENS.fonts).filter(f => f.family);
    await Promise.all(fams.map(f => document.fonts.load(`${f.style || 'normal'} ${f.weight} 40px "${f.family}"`, 'Ab12')));
  }

  // ---------- basemap ----------
  function drawBasemap(g, c) {
    g.fillStyle = '#0b0d0c'; g.fillRect(0, 0, W, H);
    // coarse -> fine: every level that overlaps the view is drawn; finer levels cover coarser ones
    const levels = (SCENES.imagery || []).slice().sort((a, b) => b.res_m - a.res_m);
    for (const lv of levels) {
      if (lv.res_m > c.mpp * 4 && lv !== levels[0]) continue;  // much coarser than needed and not the base: skip
      const im = IMG[lv.file], [bx0, by0, bx1, by1] = lv.bounds_3857, r = (bx1 - bx0) / im.width;
      const vx0 = c.x - W / 2 * c.mpp, vx1 = c.x + W / 2 * c.mpp, vy0 = c.y - H / 2 * c.mpp, vy1 = c.y + H / 2 * c.mpp;
      const ix0 = Math.max(bx0, vx0), ix1 = Math.min(bx1, vx1), iy0 = Math.max(by0, vy0), iy1 = Math.min(by1, vy1);
      if (ix1 <= ix0 || iy1 <= iy0) continue;
      const srcX = (ix0 - bx0) / r, srcY = (by1 - iy1) / r, srcW = (ix1 - ix0) / r, srcH = (iy1 - iy0) / r;
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      // a finer level fades in as the camera zooms towards it (no hard switch, no visible edge of its area)
      g.globalAlpha = lv === levels[0] ? 1 : Math.min(1, Math.max(0, (4 * lv.res_m - c.mpp) / (2 * lv.res_m)));
      g.drawImage(im, srcX, srcY, srcW, srcH, sx(c, ix0), sy(c, iy1), (ix1 - ix0) / c.mpp, (iy1 - iy0) / c.mpp);
      g.globalAlpha = 1;
    }
  }

  function polyPath(g, c, geom) {
    const p = new Path2D();
    for (const poly of geom.coords) for (const ring of poly) {
      ring.forEach(([X, Y], i) => i ? p.lineTo(sx(c, X), sy(c, Y)) : p.moveTo(sx(c, X), sy(c, Y)));
      p.closePath();
    }
    return p;
  }

  let DIM = null;
  function drawFocus(g, c, geomId) {
    const fd = LOOK().basemap.focus_dim;
    if (!fd || !fd.enabled || !geomId) return;
    // deep inside a big place (a country seen at city scale) its coarse outline no longer matches the coast: fade out
    const [r0, r1] = fd.fade_ratio || [1e9, 2e9], ratio = SCENES.geometry[geomId].width / (W * c.mpp);
    const k = Math.min(1, Math.max(0, (r1 - ratio) / (r1 - r0)));
    if (k <= 0) return;
    DIM = DIM || makeCanvas(W, H);
    const d = DIM.getContext('2d');
    d.setTransform(1, 0, 0, 1, 0, 0); d.globalCompositeOperation = 'source-over'; d.filter = 'none';
    d.clearRect(0, 0, W, H);
    d.fillStyle = `rgba(0,0,0,${1 - fd.outside_brightness})`; d.fillRect(0, 0, W, H);
    d.globalCompositeOperation = 'destination-out'; d.filter = `blur(${fd.feather_px}px)`;
    d.fillStyle = '#000'; d.fill(polyPath(d, c, SCENES.geometry[geomId]), 'evenodd');
    d.filter = 'none'; d.globalCompositeOperation = 'source-over';
    g.save(); g.globalAlpha = k; g.drawImage(DIM, 0, 0);
    drawLift(g, c, SCENES.geometry[geomId]); g.restore();
  }

  // The subject looks raised off the map (r1 Gurugram): a soft shadow falls outside its edge, down-right (light from
  // the top-left), and a thin light rim catches the edge. Shadow only outside: the cut-out keeps the area clean.
  let LIFT = null;
  function drawLift(g, c, geom) {
    const L = LOOK().basemap.lift; if (!L || !L.enabled) return;
    LIFT = LIFT || makeCanvas(W, H);
    const d = LIFT.getContext('2d'), path = polyPath(d, c, geom);
    d.setTransform(1, 0, 0, 1, 0, 0); d.globalCompositeOperation = 'source-over'; d.clearRect(0, 0, W, H);
    d.filter = `blur(${L.shadow_blur}px)`; d.fillStyle = `rgba(0,0,0,${L.shadow_opacity})`;
    d.save(); d.translate(L.shadow_dx, L.shadow_dy); d.fill(path, 'evenodd'); d.restore();
    d.filter = 'none'; d.globalCompositeOperation = 'destination-out'; d.fill(path, 'evenodd');
    d.globalCompositeOperation = 'source-over';
    g.drawImage(LIFT, 0, 0);
    if (L.rim_opacity > 0) {
      g.save(); g.lineJoin = 'round'; g.lineWidth = L.rim_width; g.strokeStyle = `rgba(255,255,255,${L.rim_opacity * g.globalAlpha})`; g.globalAlpha = 1;
      g.stroke(path); g.restore();
    }
  }

  function drawVignette(g) {
    const rings = LOOK().basemap.vignette.rings_rel, k = LOOK().basemap.vignette.strength ?? 1;
    const R = Math.hypot(W, H) / 2, grd = g.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, R);
    rings.forEach((rel, i) => grd.addColorStop(Math.min(1, (i + .5) / rings.length), `rgba(0,0,0,${Math.max(0, (1 - rel) * k)})`));
    grd.addColorStop(1, `rgba(0,0,0,${Math.max(0, (1 - rings[rings.length - 1]) * k)})`);
    g.fillStyle = grd; g.fillRect(0, 0, W, H);
  }

  // ---------- overlays ----------
  function drawRoute(g, c, L, t) {
    const geom = SCENES.geometry[L.geom], R = LOOK().route;
    const u = Math.min(1, Math.max(0, (t - L.in) / L.draw_s));
    if (u <= 0) return;
    const total = geom.len[geom.len.length - 1] * (EASE[L.ease] || EASE.easeOutCubic)(u);
    const p = new Path2D();
    const pts = geom.coords, cum = geom.len;
    p.moveTo(sx(c, pts[0][0]), sy(c, pts[0][1]));
    for (let i = 1; i < pts.length; i++) {
      if (cum[i] <= total) { p.lineTo(sx(c, pts[i][0]), sy(c, pts[i][1])); continue; }
      const f = (total - cum[i - 1]) / (cum[i] - cum[i - 1]);
      p.lineTo(sx(c, pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * f), sy(c, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * f));
      break;
    }
    const col = R.colors[L.color] || L.color;
    g.save(); g.lineCap = L.dash ? 'butt' : R.cap; g.lineJoin = R.join; g.lineWidth = L.width || R.width;
    if (L.dash) g.setLineDash(L.dash);  // roads: white dashed (r1 Dwarka Expressway), metro lines solid
    g.shadowColor = rgba('#000000', R.shadow.opacity); g.shadowBlur = R.shadow.blur; g.shadowOffsetY = R.shadow.offset_y;
    g.strokeStyle = col; g.stroke(p);
    g.shadowColor = rgba(col, R.glow.opacity); g.shadowBlur = R.glow.blur; g.shadowOffsetY = 0;
    g.stroke(p); g.restore();
  }

  function drawTint(g, c, L, t) {
    const tint = LOOK().region_tint || LOOK().basemap.region_tint; if (!tint) return;
    const a = Math.min(1, Math.max(0, (t - L.in) / (TOKENS.motion.fill_in[SCENES.look]?.duration || .25)));
    if (a <= 0) return;
    g.save(); g.globalAlpha = a * tint.opacity; g.globalCompositeOperation = tint.blend || 'color';
    g.fillStyle = tint.color; g.fill(polyPath(g, c, SCENES.geometry[L.geom]), 'evenodd'); g.restore();
  }

  // A counted group's unit (r4: states fill one by one on their names), in look A: translucent colour over the
  // satellite, a bright glowing edge and a short flash when it arrives; dimmed when the next group takes over, back at
  // full strength for the recap. outline: true = claimed land (dashed edge only, e.g. Taiwan).
  function drawFill(g, c, L, t) {
    if (t < L.in) return;
    const F = LOOK().fill, col = L.color;
    let k = EASE.easeOutCubic(Math.min(1, (t - L.in) / F.in_s));
    if (L.dim_at !== undefined && t > L.dim_at) {
      let d = 1 - (1 - F.dim_to) * Math.min(1, (t - L.dim_at) / F.dim_s);
      if (L.undim_at !== undefined && t > L.undim_at) d += (1 - d) * Math.min(1, (t - L.undim_at) / F.dim_s);
      k *= d;
    }
    if (k <= 0) return;
    const p = polyPath(g, c, SCENES.geometry[L.geom]);
    g.save(); g.lineJoin = 'round';
    if (!L.outline) {
      const flash = Math.max(0, 1 - (t - L.in) / F.flash_s);
      g.globalAlpha = Math.min(1, k * F.opacity + flash * F.flash_opacity); g.fillStyle = col; g.fill(p, 'evenodd');
    }
    g.globalAlpha = k; g.lineWidth = L.width || F.edge_width; g.strokeStyle = col;
    if (L.dash) g.setLineDash(L.dash);
    g.shadowColor = rgba(col, F.glow_opacity); g.shadowBlur = F.glow_blur; g.stroke(p);
    g.restore();
  }

  function drawOutline(g, c, L, t) {
    if (t < L.in) return;
    const O = LOOK().outline, k = Math.min(1, (t - L.in) / O.in_s);
    g.save(); g.globalAlpha = k * O.opacity; g.lineJoin = 'round'; g.lineWidth = O.width; g.strokeStyle = O.color;
    g.stroke(polyPath(g, c, SCENES.geometry[L.geom])); g.restore();
  }

  // sets font + letter spacing (tokens tracking is in em) on a context
  function setFont(g, styleTok, sizePx) {
    const f = TOKENS.fonts[styleTok.font], s = sizePx ?? styleTok.size;
    g.font = `${f.style || 'normal'} ${f.weight} ${s}px "${f.family}"`;
    g.letterSpacing = `${((f.tracking || 0) * s).toFixed(2)}px`;
  }
  // Text comes in with a soft left-to-right wipe (a feathered edge sweeping across the whole line) and a slight rise,
  // instead of letters popping in one by one (2026-10-19 user: the typing looked too fast and jerky).
  let OFFC = null;
  function revealText(g, txt, t, tin, dur, sizePx, x, y, drawFn) {
    const R = TOKENS.motion.text_reveal;
    const raw = Math.min(1, Math.max(0, (t - tin) / (dur || R.duration)));
    if (raw <= 0 || !txt) return;
    const u = EASE[R.ease](raw), rise = (1 - u) * R.rise_em * sizePx;
    if (raw >= 1) return drawFn(g, x, y);
    OFFC = OFFC || makeCanvas(W, H);
    const o = OFFC.getContext('2d');
    o.setTransform(1, 0, 0, 1, 0, 0); o.globalCompositeOperation = 'source-over'; o.clearRect(0, 0, W, H);
    o.setTransform(g.getTransform());
    o.font = g.font; o.letterSpacing = g.letterSpacing; o.textAlign = g.textAlign; o.textBaseline = g.textBaseline;
    o.globalAlpha = 1; o.filter = 'none';
    drawFn(o, x, y + rise);
    const w = g.measureText(txt).width, F = R.feather_em * sizePx;
    const left = g.textAlign === 'center' ? x - w / 2 : g.textAlign === 'right' ? x - w : x;
    const edge = left - F + u * (w + 2 * F);
    const grd = o.createLinearGradient(edge - F, 0, edge, 0);
    grd.addColorStop(0, 'rgba(0,0,0,1)'); grd.addColorStop(1, 'rgba(0,0,0,0)');
    o.globalCompositeOperation = 'destination-in'; o.fillStyle = grd;
    o.fillRect(left - 4 * sizePx, y - 4 * sizePx, w + 8 * sizePx, 8 * sizePx);
    o.globalCompositeOperation = 'source-over';
    g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(OFFC, 0, 0); g.restore();
  }
  function fadeOut(t, out) {
    if (out === undefined || out === null) return 1;
    const d = TOKENS.motion.text_out.duration;
    return Math.min(1, Math.max(0, (out + d - t) / d));
  }
  // Reference text treatment (r1 close-up): no hard outline - a wide soft dark shadow that lifts the text off the
  // map, a tight one for edge definition, and a vertical gradient fill (bright top, deeper bottom).
  // baseline: 'top' (stats: glyphs span y..y+size) or 'alphabetic' (titles: glyphs span y-0.75size..y)
  function softText(g, txt, x, y, sizePx, color, fx, baseline) {
    if (!txt) return;
    fx = fx || {};
    const top = baseline === 'top' ? y + sizePx * .08 : y - sizePx * .74, bot = baseline === 'top' ? y + sizePx * .82 : y;
    g.save();
    if (fx.halo) {  // smoky dark halo hugging the glyphs: a thick stroke, blurred (no hard edge anywhere)
      g.save(); g.filter = `blur(${(fx.halo.blur_em * sizePx).toFixed(1)}px)`; g.lineJoin = 'round';
      g.lineWidth = fx.halo.width_em * sizePx; g.strokeStyle = `rgba(0,0,0,${fx.halo.opacity})`; g.fillStyle = g.strokeStyle;
      g.strokeText(txt, x, y + fx.halo.offset_em * sizePx); g.fillText(txt, x, y + fx.halo.offset_em * sizePx); g.restore();
    }
    for (const sh of [fx.shadow_wide, fx.shadow_tight]) {
      if (!sh) continue;
      g.shadowColor = `rgba(0,0,0,${sh.opacity})`; g.shadowBlur = sh.blur_em * sizePx; g.shadowOffsetY = sh.offset_em * sizePx;
      g.fillStyle = color; g.fillText(txt, x, y);
    }
    g.shadowColor = 'transparent';
    if (fx.gradient) {
      const gr = g.createLinearGradient(0, top, 0, bot);
      gr.addColorStop(0, shade(color, fx.gradient.top_lighten)); gr.addColorStop(1, shade(color, -fx.gradient.bottom_darken));
      g.fillStyle = gr;
    } else g.fillStyle = color;
    g.fillText(txt, x, y);
    g.restore();
  }
  // lighten (k > 0, towards white) or darken (k < 0, towards black) a hex colour
  function shade(hex, k) {
    const n = parseInt(hex.slice(1), 16), ch = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    const f = v => Math.round(k >= 0 ? v + (255 - v) * k : v * (1 + k));
    return `rgb(${ch.map(f).join(',')})`;
  }

  // Text either sits on the screen (x, y) or lives on the map (X, Y world point + mpp_in = zoom when it appeared):
  // map text scales with the camera, as in the reference reels.
  function anchorOf(c, L) {
    if (L.X === undefined) return { x: L.x, y: L.y, k: 1 };
    return { x: sx(c, L.X), y: sy(c, L.Y), k: L.mpp_in / c.mpp };
  }

  function drawTitle(g, c, L, t) {
    if (t < L.in) return;
    const st = LOOK()[L.style] || LOOK().title_on_map, a = fadeOut(t, L.out);
    if (a <= 0) return;
    // the hook title is on screen almost at once (r1), so its wipe is quicker
    const dur = (L.type_s ?? 1) < 0.3 ? TOKENS.motion.text_reveal.hook_duration : TOKENS.motion.text_reveal.duration;
    const p = anchorOf(c, L), grow = L.X === undefined ? 1 + (LOOK().title.grow_per_s || 0) * (t - L.in) : p.k;
    g.save(); g.globalAlpha = a; g.translate(p.x, p.y); g.scale(grow, grow);
    setFont(g, st); g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    revealText(g, L.text, t, L.in, dur, st.size, 0, 0, (c, x, y) => softText(c, L.text, x, y, st.size, st.color, st.text_fx, 'alphabetic'));
    g.restore();
  }

  function drawStat(g, c, L, t) {
    const a = fadeOut(t, L.out); if (a <= 0) return;
    const S = LOOK().stat, gapPx = TOKENS.plan.stat_line_gap * H;
    // Each line's height in WORLD metres = its token size at the zoom it appears with (line.mpp_in); the block is
    // centred on its anchor. Screen-anchored stats use plain pixels (mpp = 1 everywhere).
    const world = L.X !== undefined, m0 = world ? L.mpp_in : 1;
    // h_world (planner): the line's height on the map, set so it reads at its token size once the camera settles
    const hw = L.lines.map(ln => world ? (ln.h_world ?? S[ln.style].size * (ln.mpp_in || m0)) : S[ln.style].size), gw = gapPx * m0;
    const total = hw.reduce((s, h) => s + h, 0) + gw * (hw.length - 1);
    const p = anchorOf(c, L), toPx = world ? 1 / c.mpp : 1;
    let y = p.y - total / 2 * toPx;
    g.save(); g.globalAlpha = a; g.textAlign = L.align || 'center'; g.textBaseline = 'top';
    L.lines.forEach((ln, i) => {
      const st = S[ln.style], px = hw[i] * toPx;
      if (t >= ln.in) {
        setFont(g, st, px);
        const txt = st.case === 'upper' ? ln.text.toUpperCase() : ln.text;
        revealText(g, txt, t, ln.in, null, px, p.x, y, (c, x, yy) => softText(c, txt, x, yy, px, ln.color || st.color, S.text_fx, 'top'));
      }
      y += px + gw * toPx;
    });
    g.restore();
  }

  function drawPin(g, c, L, t) {
    if (t < L.in) return;
    const a = fadeOut(t, L.out); if (a <= 0) return;
    const st = LOOK().station, pill = LOOK().label_pill, m = TOKENS.motion.label_in;
    const u = Math.min(1, (t - L.in) / m.duration), s = m.from_scale + (1 - m.from_scale) * EASE[m.ease](u);
    const [X, Y] = SCENES.geometry[L.geom].coords, x = sx(c, X), y = sy(c, Y);
    g.save(); g.globalAlpha = a; g.translate(x, y); g.scale(s, s);
    g.beginPath(); g.arc(0, 0, st.dot_radius, 0, Math.PI * 2); g.fillStyle = st.dot_fill; g.fill();
    g.lineWidth = st.dot_stroke_width; g.strokeStyle = st.dot_stroke; g.stroke();
    setFont(g, { font: pill.font, size: pill.size }); g.textBaseline = 'middle';
    const w = g.measureText(L.label).width + pill.pad_x * 2, h = pill.size + pill.pad_y * 2, gap = st.dot_radius + 10;
    // the planner picks the free side so labels never overlap (above is the default, as in r1)
    const side = L.side || 'above';
    const [bx, by] = { above: [-w / 2, -gap - h], below: [-w / 2, gap], right: [gap, -h / 2], left: [-gap - w, -h / 2] }[side];
    g.fillStyle = pill.fill; g.beginPath(); g.roundRect(bx, by, w, h, pill.radius); g.fill();
    if (pill.pointer) {
      g.beginPath();
      if (side === 'above') { g.moveTo(-6, by + h); g.lineTo(6, by + h); g.lineTo(0, by + h + 7); }
      else if (side === 'below') { g.moveTo(-6, by); g.lineTo(6, by); g.lineTo(0, by - 7); }
      else if (side === 'right') { g.moveTo(bx, -6); g.lineTo(bx, 6); g.lineTo(bx - 7, 0); }
      else { g.moveTo(bx + w, -6); g.lineTo(bx + w, 6); g.lineTo(bx + w + 7, 0); }
      g.fill();
    }
    g.fillStyle = pill.text; g.textAlign = 'center'; g.fillText(L.label, bx + w / 2, by + h / 2 + 1);
    g.restore();
  }

  // A photo of the place, popping up on its name: rounded card, thin white border, deep soft shadow, a slow push-in
  // inside the frame, the place name on a dark fade at the bottom and the photo's credit (author + licence).
  function drawPicture(g, L, t) {
    if (t < L.in) return;
    const a = fadeOut(t, L.out); if (a <= 0) return;
    const P = LOOK().picture, im = IMG[L.file];
    const u = Math.min(1, (t - L.in) / P.pop_s), e = EASE.easeOutCubic(u), s = 0.9 + 0.1 * EASE.backOut(u);
    const w = L.w, h = L.h, r = P.radius;
    g.save(); g.globalAlpha = a * Math.min(1, u * 1.6);
    g.translate(L.x, L.y + (1 - e) * 24); g.scale(s, s);
    g.shadowColor = 'rgba(0,0,0,.6)'; g.shadowBlur = 40; g.shadowOffsetY = 16;
    g.fillStyle = '#ffffff'; g.beginPath(); g.roundRect(-w / 2 - P.border, -h / 2 - P.border, w + 2 * P.border, h + 2 * P.border, r + P.border); g.fill();
    g.shadowColor = 'transparent';
    g.save(); g.beginPath(); g.roundRect(-w / 2, -h / 2, w, h, r); g.clip();
    const life = Math.max(0.5, (L.out ?? L.in + 3) - L.in + 0.3), k = 1 + P.ken_burns * Math.min(1, (t - L.in) / life);
    const sc = Math.max(w / im.width, h / im.height) * k;
    g.drawImage(im, -im.width * sc / 2, -im.height * sc / 2, im.width * sc, im.height * sc);
    const grd = g.createLinearGradient(0, h / 2 - h * .42, 0, h / 2);
    grd.addColorStop(0, 'rgba(0,0,0,0)'); grd.addColorStop(1, 'rgba(0,0,0,.72)');
    g.fillStyle = grd; g.fillRect(-w / 2, -h / 2, w, h);
    g.restore();
    if (L.caption) {
      setFont(g, { font: TOKENS.looks[SCENES.look].title_on_map.font, size: P.caption_size }); g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      softText(g, L.caption, -w / 2 + 24, h / 2 - 24, P.caption_size, '#FFFFFF', LOOK().title_on_map.text_fx, 'alphabetic');
    }
    if (L.credit) {
      g.font = `500 ${P.credit_size}px "Inter"`; g.textAlign = 'right'; g.textBaseline = 'alphabetic'; g.fillStyle = 'rgba(255,255,255,.72)';
      g.fillText(L.credit, w / 2 - 16, h / 2 - 14);
    }
    g.restore();
  }

  function drawCredit(g) {
    if (!SCENES.credits || !SCENES.credits.length) return;
    g.save(); g.font = `500 15px "Inter"`; g.fillStyle = 'rgba(255,255,255,.38)'; g.textAlign = 'left'; g.textBaseline = 'bottom';
    g.fillText(SCENES.credits.join(' · '), SAFE().left * .5, H - SAFE().bottom * .5); g.restore();
  }

  function rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  // ---------- shots ----------
  function drawMapShot(shot, t) {
    const g = G, c = camAt(shot, t);
    drawBasemap(g, c);
    drawFocus(g, c, shot.focus);
    for (const L of shot.layers) if (L.type === 'tint') drawTint(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'fill') drawFill(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'outline') drawOutline(g, c, L, t);
    drawVignette(g);
    for (const L of shot.layers) if (L.type === 'route') drawRoute(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'pin') drawPin(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'title') drawTitle(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'stat') drawStat(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'picture') drawPicture(g, L, t);
    drawCredit(g);
  }

  // Follow / COMMENT buttons on the face-cam (r5): pill with a pop-in
  function drawButton(g, L, t) {
    if (t < L.in) return;
    const a = fadeOut(t, L.out); if (a <= 0) return;
    const B = TOKENS.facecam[L.style], P = TOKENS.facecam.follow_button.pop;
    const u = Math.min(1, (t - L.in) / P.duration), s = P.from_scale + (1 - P.from_scale) * EASE[P.ease](u);
    setFont(g, { font: B.font, size: B.size });
    const icon = B.icon === 'plus' ? B.size * .8 : 0;
    const w = g.measureText(B.text).width + 2 * B.pad_x + icon, h = B.size + 2 * B.pad_y;
    g.save(); g.globalAlpha = a; g.translate(W / 2, B.y * H); g.scale(s, s);
    g.shadowColor = 'rgba(0,0,0,.35)'; g.shadowBlur = 18; g.shadowOffsetY = 6;
    if (B.gradient) { const gr = g.createLinearGradient(-w / 2, 0, w / 2, 0); gr.addColorStop(0, B.gradient[0]); gr.addColorStop(1, B.gradient[1]); g.fillStyle = gr; }
    else g.fillStyle = B.fill;
    g.beginPath(); g.roundRect(-w / 2, -h / 2, w, h, B.radius); g.fill();
    g.shadowColor = 'transparent'; g.fillStyle = B.text_color; g.textAlign = 'center'; g.textBaseline = 'middle';
    if (icon) {  // a plus sign before the word, like Instagram's follow button
      const ix = -w / 2 + B.pad_x + icon * .35, sz = icon * .32; g.lineWidth = B.size * .12; g.strokeStyle = B.text_color; g.lineCap = 'round';
      g.beginPath(); g.moveTo(ix - sz, 0); g.lineTo(ix + sz, 0); g.moveTo(ix, -sz); g.lineTo(ix, sz); g.stroke();
    }
    g.fillText(B.text, icon / 2, 2);
    g.restore();
  }

  // face-cam shot: in the normal pass a placeholder (the clip covers it in assemble.py); in the overlay pass
  // (?overlay) only the popups and buttons, on a transparent frame, composited over the zoomed clip
  function drawFacecam(shot, t) {
    const overlay = location.search.includes('overlay');
    if (!overlay) return drawPlaceholder(shot, t, 'FACE-CAM CLIP HERE');
    G.clearRect(0, 0, W, H);
    for (const L of shot.layers || []) {
      if (L.type === 'stat') drawCamStat(G, L, t);
      if (L.type === 'button') drawButton(G, L, t);
    }
  }

  // numbers said on camera: the map's stat style, bigger, on a soft dark box (a white wall kills yellow text)
  function drawCamStat(g, L, t) {
    const a = fadeOut(t, L.out); if (a <= 0 || t < L.lines[0].in) return;
    const S = LOOK().stat, Bx = TOKENS.facecam.stat_box;
    const sizes = L.lines.map(ln => S[ln.style].size * Bx.size_scale);
    const widths = L.lines.map((ln, i) => { setFont(g, S[ln.style], sizes[i]); return g.measureText(ln.text).width; });
    const bw = Math.max(...widths) + 2 * Bx.pad_x, bh = sizes.reduce((s, v) => s + v * 1.1, 0) + 2 * Bx.pad_y;
    const u = Math.min(1, (t - L.lines[0].in) / .25), s = .85 + .15 * EASE.backOut(u);
    g.save(); g.globalAlpha = a; g.translate(L.x, L.y); g.scale(s, s);
    g.fillStyle = rgba(Bx.fill, Bx.opacity * Math.min(1, u * 2)); g.beginPath(); g.roundRect(-bw / 2, -bh / 2, bw, bh, Bx.radius); g.fill();
    let y = -bh / 2 + Bx.pad_y;
    L.lines.forEach((ln, i) => {
      const st = S[ln.style]; setFont(g, st, sizes[i]); g.fillStyle = st.color; g.textAlign = 'center'; g.textBaseline = 'top';
      const txt = st.case === 'upper' ? ln.text.toUpperCase() : ln.text, sz = sizes[i];
      revealText(g, txt, t, ln.in, null, sz, 0, y, (c, x, yy) => softText(c, txt, x, yy, sz, st.color, S.text_fx, 'top'));
      y += sizes[i] * 1.1;
    });
    g.restore();
  }

  function drawPlaceholder(shot, t, label) {
    G.fillStyle = '#16161a'; G.fillRect(0, 0, W, H);
    G.fillStyle = '#ffffff'; G.globalAlpha = .8; G.font = '700 54px "Inter"'; G.textAlign = 'center';
    G.fillText(label, W / 2, H / 2 - 20); G.font = '500 34px "Inter"'; G.globalAlpha = .55;
    G.fillText(`${shot.start.toFixed(1)}s – ${shot.end.toFixed(1)}s`, W / 2, H / 2 + 40); G.globalAlpha = 1;
  }

  // register one shot: called from work/<ep>/render/scenes/<id>.js; t is absolute (episode) time
  function shot(s) {
    const off = SCENES.offset || 0;
    shots([[s.start - off, (lt) => {
      const t = lt + off;
      if (s.kind === 'map') drawMapShot(s, t);
      else if (s.kind === 'facecam') drawFacecam(s, t);
      else drawPlaceholder(s, t, 'B-ROLL HERE');
    }]]);
  }

  return { setup, shot, camAt };
})();

window.ENGINE = { setup: MAPANI.setup };
