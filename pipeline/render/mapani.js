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
  function camAt(shot, t) {
    const keys = shot.camera;
    let c;
    if (t <= keys[0].t) c = { x: keys[0].x, y: keys[0].y, mpp: keys[0].mpp };
    else {
      const last = keys[keys.length - 1];
      c = { x: last.x, y: last.y, mpp: last.mpp };
      for (let i = 1; i < keys.length; i++) {
        const a = keys[i - 1], b = keys[i];
        if (t < b.t) {
          const t0 = b.t0 !== undefined ? b.t0 : a.t;
          const u = t <= t0 ? 0 : Math.min(1, (t - t0) / Math.max(1e-6, b.t - t0));
          const e = (EASE[b.ease] || EASE.easeOutSine)(u);
          const lm = Math.log(a.mpp) + (Math.log(b.mpp) - Math.log(a.mpp)) * e;
          c = { x: a.x + (b.x - a.x) * e, y: a.y + (b.y - a.y) * e, mpp: Math.exp(lm) };
          break;
        }
      }
    }
    c.mpp *= Math.exp(-(shot.drift_per_s || 0) * t);  // look A: the camera never stops pushing in
    return c;
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
    // per-geometry cumulative lengths for progressive route drawing
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
      g.drawImage(im, srcX, srcY, srcW, srcH, sx(c, ix0), sy(c, iy1), (ix1 - ix0) / c.mpp, (iy1 - iy0) / c.mpp);
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
    DIM = DIM || makeCanvas(W, H);
    const d = DIM.getContext('2d');
    d.setTransform(1, 0, 0, 1, 0, 0); d.globalCompositeOperation = 'source-over'; d.filter = 'none';
    d.clearRect(0, 0, W, H);
    d.fillStyle = `rgba(0,0,0,${1 - fd.outside_brightness})`; d.fillRect(0, 0, W, H);
    d.globalCompositeOperation = 'destination-out'; d.filter = `blur(${fd.feather_px}px)`;
    d.fillStyle = '#000'; d.fill(polyPath(d, c, SCENES.geometry[geomId]), 'evenodd');
    d.filter = 'none'; d.globalCompositeOperation = 'source-over';
    g.drawImage(DIM, 0, 0);
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
    g.save(); g.lineCap = R.cap; g.lineJoin = R.join; g.lineWidth = R.width;
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

  // sets font + letter spacing (tokens tracking is in em) on a context
  function setFont(g, styleTok, sizePx) {
    const f = TOKENS.fonts[styleTok.font], s = sizePx ?? styleTok.size;
    g.font = `${f.style || 'normal'} ${f.weight} ${s}px "${f.family}"`;
    g.letterSpacing = `${((f.tracking || 0) * s).toFixed(2)}px`;
  }
  function typed(text, t, tin, dur) {
    const u = Math.min(1, Math.max(0, (t - tin) / dur));
    return text.slice(0, Math.round(u * text.length));
  }
  function fadeOut(t, out) {
    if (out === undefined || out === null) return 1;
    const d = TOKENS.motion.text_out.duration;
    return Math.min(1, Math.max(0, (out + d - t) / d));
  }
  function shadowText(g, st) {
    const sh = LOOK().stat.shadow;
    g.shadowColor = rgba(sh.color, sh.opacity); g.shadowBlur = sh.blur; g.shadowOffsetY = sh.offset_y;
  }
  // dark halo first (stroke under the fill), then the coloured text with its drop shadow, as in r1
  function haloText(g, txt, x, y, sizePx, outline) {
    if (outline) {
      g.save(); g.shadowColor = 'transparent'; g.lineJoin = 'round';
      g.lineWidth = outline.width_em * sizePx; g.strokeStyle = rgba(outline.color, outline.opacity);
      g.strokeText(txt, x, y); g.restore();
    }
    g.fillText(txt, x, y);
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
    const txt = typed(L.text, t, L.in, L.type_s ?? TOKENS.motion.text_in.duration);
    const p = anchorOf(c, L), grow = L.X === undefined ? 1 + (LOOK().title.grow_per_s || 0) * (t - L.in) : p.k;
    g.save(); g.globalAlpha = a; g.translate(p.x, p.y); g.scale(grow, grow);
    setFont(g, st); g.textAlign = 'center'; g.textBaseline = 'alphabetic'; g.fillStyle = st.color;
    shadowText(g); haloText(g, txt, 0, 0, st.size, st.outline); g.restore();
  }

  function drawStat(g, c, L, t) {
    const a = fadeOut(t, L.out); if (a <= 0) return;
    const S = LOOK().stat, gapPx = TOKENS.plan.stat_line_gap * H;
    // Each line's height in WORLD metres = its token size at the zoom it appears with (line.mpp_in); the block is
    // centred on its anchor. Screen-anchored stats use plain pixels (mpp = 1 everywhere).
    const world = L.X !== undefined, m0 = world ? L.mpp_in : 1;
    const hw = L.lines.map(ln => S[ln.style].size * (world ? (ln.mpp_in || m0) : 1)), gw = gapPx * m0;
    const total = hw.reduce((s, h) => s + h, 0) + gw * (hw.length - 1);
    const p = anchorOf(c, L), toPx = world ? 1 / c.mpp : 1;
    let y = p.y - total / 2 * toPx;
    g.save(); g.globalAlpha = a; g.textAlign = L.align || 'center'; g.textBaseline = 'top';
    L.lines.forEach((ln, i) => {
      const st = S[ln.style], px = hw[i] * toPx;
      if (t >= ln.in) {
        setFont(g, st, px); g.fillStyle = st.color; shadowText(g);
        const txt = st.case === 'upper' ? ln.text.toUpperCase() : ln.text;
        haloText(g, typed(txt, t, ln.in, TOKENS.motion.number_in.duration), p.x, y, px, S.outline);
      }
      y += px + gw * toPx;
    });
    g.restore();
  }

  function drawPin(g, c, L, t) {
    if (t < L.in) return;
    const st = LOOK().station, pill = LOOK().label_pill, m = TOKENS.motion.label_in;
    const u = Math.min(1, (t - L.in) / m.duration), s = m.from_scale + (1 - m.from_scale) * EASE[m.ease](u);
    const [X, Y] = SCENES.geometry[L.geom].coords, x = sx(c, X), y = sy(c, Y);
    g.save(); g.translate(x, y); g.scale(s, s);
    g.beginPath(); g.arc(0, 0, st.dot_radius, 0, Math.PI * 2); g.fillStyle = st.dot_fill; g.fill();
    g.lineWidth = st.dot_stroke_width; g.strokeStyle = st.dot_stroke; g.stroke();
    setFont(g, { font: pill.font, size: pill.size }); g.textBaseline = 'middle';
    const w = g.measureText(L.label).width + pill.pad_x * 2, h = pill.size + pill.pad_y * 2;
    const bx = -w / 2, by = -st.dot_radius - 10 - h;
    g.fillStyle = pill.fill; g.beginPath(); g.roundRect(bx, by, w, h, pill.radius); g.fill();
    if (pill.pointer) { g.beginPath(); g.moveTo(-6, by + h); g.lineTo(6, by + h); g.lineTo(0, by + h + 7); g.fill(); }
    g.fillStyle = pill.text; g.textAlign = 'center'; g.fillText(L.label, 0, by + h / 2 + 1);
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
    drawVignette(g);
    for (const L of shot.layers) if (L.type === 'route') drawRoute(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'pin') drawPin(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'title') drawTitle(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'stat') drawStat(g, c, L, t);
    drawCredit(g);
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
      else drawPlaceholder(s, t, s.kind === 'facecam' ? 'FACE-CAM CLIP HERE' : 'B-ROLL HERE');
    }]]);
  }

  return { setup, shot, camAt };
})();

window.ENGINE = { setup: MAPANI.setup };
