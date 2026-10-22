// story.js: story mode (the @adarshh.gupta format, style/tokens.json "story"). Loaded after mapani.js.
//
// A story reel is a list of shots that change every 2.5-6 s: AI/stock photos that never stand still, real news
// clippings floating in 3D with a marker sweeping over the spoken line, maps (mapani.js camera + story layers), the
// creator's face-cam, a torn-paper CTA and an Instagram "Follow" end card. Shots overlap by half a transition on each
// side of their cut; both are drawn into buffers and blended (fade, defocus, zoom-blur, whip, flash, clouds, glitch).
// Every frame is a pure function of t, like the rest of the engine.

const STORY = (() => {
  const S = () => TOKENS.story;
  const E = MAPANI.EASE;
  const IM = {};
  const LIST = [];
  const cl = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const smooth = (a, b, x) => { const u = cl((x - a) / (b - a)); return u * u * (3 - 2 * u); };

  // ---------- buffers & textures ----------
  const BUF = {};
  let TILTED = false;  // drawing into a map that will be put in perspective
  function buf(name, w = W, h = H) {
    let c = BUF[name];
    if (!c || c.width !== w || c.height !== h) c = BUF[name] = makeCanvas(w, h);
    const g = c.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0); g.globalAlpha = 1; g.globalCompositeOperation = 'source-over'; g.filter = 'none';
    g.shadowColor = 'transparent'; g.shadowBlur = 0; g.shadowOffsetX = g.shadowOffsetY = 0;
    return [c, g];
  }
  const TEX = {};
  // value noise, fractal: clouds, paper fibre, the red news backdrop
  function noiseCanvas(w, h, seed, octaves, base, fn) {
    const c = makeCanvas(w, h), g = c.getContext('2d'), img = g.createImageData(w, h);
    const r = rnd(seed), G0 = 64, grid = [];
    for (let i = 0; i < G0 * G0; i++) grid.push(r());
    const at = (x, y) => grid[((y % G0 + G0) % G0) * G0 + ((x % G0 + G0) % G0)];
    const vn = (x, y) => {
      const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi, u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
      return lerp(lerp(at(xi, yi), at(xi + 1, yi), u), lerp(at(xi, yi + 1), at(xi + 1, yi + 1), u), v);
    };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let a = 0, amp = .5, f = base, tot = 0;
      for (let o = 0; o < octaves; o++) { a += vn(x * f, y * f) * amp; tot += amp; amp *= .5; f *= 2; }
      const px = fn(a / tot, x, y), k = (y * w + x) * 4;
      img.data[k] = px[0]; img.data[k + 1] = px[1]; img.data[k + 2] = px[2]; img.data[k + 3] = px[3];
    }
    g.putImageData(img, 0, 0);
    return c;
  }
  function buildTextures() {
    // grain: 4 tiles, one per frame in turn (moving grain, never the same pattern twice in a row)
    TEX.grain = [0, 1, 2, 3].map(i => {
      const c = makeCanvas(256, 256), g = c.getContext('2d'), img = g.createImageData(256, 256), r = rnd('grain' + i);
      for (let k = 0; k < img.data.length; k += 4) { const v = r() * 255; img.data[k] = img.data[k + 1] = img.data[k + 2] = v; img.data[k + 3] = 255; }
      g.putImageData(img, 0, 0); return c;
    });
    TEX.cloud = noiseCanvas(540, 960, 'cloud', 6, 1 / 90, (v) => {
      const a = smooth(.42, .78, v); const l = 215 + 40 * v;
      return [l, l, l + 6, a * 255];
    });
    TEX.red = noiseCanvas(270, 480, 'red', 5, 1 / 40, (v, x, y) => {
      const k = .35 + .9 * v * v; return [150 * k, 14 * k, 22 * k, 255];
    });
    TEX.paper = noiseCanvas(540, 220, 'paper', 4, 1 / 6, (v) => { const l = 236 + 16 * v; return [l, l, l - 3, 255]; });
  }

  async function load(file) {
    if (!file || IM[file]) return;
    const im = new Image();
    await new Promise((ok, bad) => { im.onload = ok; im.onerror = () => bad(new Error('image failed to load: ' + file)); im.src = file; });
    IM[file] = await createImageBitmap(im);
  }
  async function setup() {
    buildTextures();
    for (const s of SCENES.shots) {
      for (const f of [s.file, s.bg_file, s.avatar_file]) await load(f);
      for (const cd of s.cards || []) await load(cd.file);
      for (const L of s.layers || []) await load(L.file);
    }
    await document.fonts.load(`400 40px "Kaushan Script"`, 'Ab');
  }

  // ---------- 3D plane: an image seen at an angle (perspective-correct mesh of affine triangles) ----------
  function project(p, cx, cy, f) { const z = f + p[2]; return [cx + p[0] * f / z, cy + p[1] * f / z]; }
  function rot3(p, rx, ry, rz) {
    let [x, y, z] = p, c, s;
    c = Math.cos(rx); s = Math.sin(rx); [y, z] = [y * c - z * s, y * s + z * c];
    c = Math.cos(ry); s = Math.sin(ry); [x, z] = [x * c + z * s, -x * s + z * c];
    c = Math.cos(rz); s = Math.sin(rz); [x, y] = [x * c - y * s, x * s + y * c];
    return [x, y, z];
  }
  function tri(g, src, s0, s1, s2, d0, d1, d2) {
    // affine map s -> d, clipped to the (slightly grown) destination triangle: no hairline seams between triangles
    const den = s0[0] * (s1[1] - s2[1]) + s1[0] * (s2[1] - s0[1]) + s2[0] * (s0[1] - s1[1]);
    if (Math.abs(den) < 1e-9) return;
    const a = (d0[0] * (s1[1] - s2[1]) + d1[0] * (s2[1] - s0[1]) + d2[0] * (s0[1] - s1[1])) / den;
    const b = (d0[1] * (s1[1] - s2[1]) + d1[1] * (s2[1] - s0[1]) + d2[1] * (s0[1] - s1[1])) / den;
    const c = (d0[0] * (s2[0] - s1[0]) + d1[0] * (s0[0] - s2[0]) + d2[0] * (s1[0] - s0[0])) / den;
    const d = (d0[1] * (s2[0] - s1[0]) + d1[1] * (s0[0] - s2[0]) + d2[1] * (s1[0] - s0[0])) / den;
    const e = (d0[0] * (s1[0] * s2[1] - s2[0] * s1[1]) + d1[0] * (s2[0] * s0[1] - s0[0] * s2[1]) + d2[0] * (s0[0] * s1[1] - s1[0] * s0[1])) / den;
    const f = (d0[1] * (s1[0] * s2[1] - s2[0] * s1[1]) + d1[1] * (s2[0] * s0[1] - s0[0] * s2[1]) + d2[1] * (s0[0] * s1[1] - s1[0] * s0[1])) / den;
    const mx = (d0[0] + d1[0] + d2[0]) / 3, my = (d0[1] + d1[1] + d2[1]) / 3, grow = p => {
      const dx = p[0] - mx, dy = p[1] - my, l = Math.hypot(dx, dy) || 1; return [p[0] + dx / l * .9, p[1] + dy / l * .9];
    };
    const q0 = grow(d0), q1 = grow(d1), q2 = grow(d2);
    g.save(); g.beginPath(); g.moveTo(q0[0], q0[1]); g.lineTo(q1[0], q1[1]); g.lineTo(q2[0], q2[1]); g.closePath(); g.clip();
    g.transform(a, b, c, d, e, f); g.drawImage(src, 0, 0); g.restore();
  }
  // draw src (w x h px) as a plane of size pw x ph centred at (cx, cy), rotated (degrees), perspective distance f
  function plane(g, src, cx, cy, pw, ph, rx, ry, rz, f, n = 10) {
    const w = src.width, h = src.height, R = Math.PI / 180, pts = [];
    for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) {
      const p = rot3([(i / n - .5) * pw, (j / n - .5) * ph, 0], rx * R, ry * R, rz * R);
      pts.push(project(p, cx, cy, f));
    }
    const P = (i, j) => pts[j * (n + 1) + i], Sx = i => i / n * w, Sy = j => j / n * h;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      tri(g, src, [Sx(i), Sy(j)], [Sx(i + 1), Sy(j)], [Sx(i), Sy(j + 1)], P(i, j), P(i + 1, j), P(i, j + 1));
      tri(g, src, [Sx(i + 1), Sy(j)], [Sx(i + 1), Sy(j + 1)], [Sx(i), Sy(j + 1)], P(i + 1, j), P(i + 1, j + 1), P(i, j + 1));
    }
    return [P(0, 0), P(n, 0), P(n, n), P(0, n)];
  }

  // ---------- text ----------
  const FONT = k => TOKENS.fonts[k];
  function font(g, key, size) {
    const f = FONT(key);
    g.font = `${f.style || 'normal'} ${f.weight} ${size}px "${f.family}"`;
    g.letterSpacing = `${((f.tracking || 0) * size).toFixed(2)}px`;
  }
  const caseOf = (r, s) => r.case === 'upper' ? s.toUpperCase() : s;
  // the line's runs: [{t, color?}] (a run may be coloured differently, e.g. 'Nepal' in orange)
  const runsOf = (ln, role) => (ln.runs || [{ t: ln.text }]).map(r => ({ ...r, t: caseOf(role, r.t) }));
  function fmtCount(ln, u) {
    const c = ln.count, v = c.from + (c.to - c.from) * E.easeOutCubic(u);
    const s = v.toFixed(c.decimals || 0).replace(/\B(?=(\d{3})+(?!\d))/g, c.sep === undefined ? ',' : c.sep);
    return (c.prefix || '') + s + (c.suffix || '');
  }
  // one run of text with the role's treatment: wide soft shadow, coloured glow, extrusion, gradient fill
  function paintRun(g, txt, x, y, size, role, colorOverride) {
    if (!txt) return;
    const top = y - size * .78, bot = y + size * .05;
    if (role.extrude) {
      for (let k = role.extrude.depth; k >= 1; k--) { g.fillStyle = role.extrude.color; g.fillText(txt, x, y + k * size / 100); }
    }
    if (role.shadow) {
      g.save(); g.shadowColor = `rgba(0,0,0,${role.shadow.opacity})`; g.shadowBlur = role.shadow.blur; g.shadowOffsetY = role.shadow.dy || 0;
      g.fillStyle = 'rgba(0,0,0,.6)'; g.fillText(txt, x, y); g.restore();
    }
    if (role.glow) {  // two passes: a wide halo and a tighter core (one shadow pass read weak next to the reference)
      g.save(); g.fillStyle = role.glow.color;
      g.shadowColor = MAPANI.rgba(role.glow.color, role.glow.opacity); g.shadowBlur = role.glow.blur; g.fillText(txt, x, y);
      g.shadowBlur = role.glow.blur * .4; g.fillText(txt, x, y); g.restore();
    }
    if (colorOverride) g.fillStyle = colorOverride;
    else {
      const gr = g.createLinearGradient(0, top, 0, bot); gr.addColorStop(0, role.fill[0]); gr.addColorStop(1, role.fill[1]); g.fillStyle = gr;
    }
    g.fillText(txt, x, y);
  }

  // A text overlay: lines of runs in a preset (tokens story.text.presets). Lines pop in with a blur on their word
  // (impact/chapter/date), type with the voice (body, caption box), or count up (numbers).
  function drawText(g, L, t) {
    if (t < L.in) return;
    const P = S().text.presets[L.preset], TI = S().text.in, out = L.out ?? Infinity;
    const fade = cl((out + S().text.out_s - t) / S().text.out_s);
    if (fade <= 0) return;
    const roles = P.roles, lines = L.lines;
    const sizes = lines.map(ln => (roles[ln.role] || {}).size * (L.scale || 1) * (ln.scale || 1));
    const gap = (P.line_gap || 0) * H;
    let total = sizes.reduce((a, b) => a + b, 0) + gap * (lines.length - 1);
    const cx = L.x ?? W / 2, cy0 = (L.y ?? P.y) * (L.y > 1 ? 1 : H) - total / 2;
    // the whole block slowly grows while it is on screen (reference titles are never static)
    const grow = 1 + .025 * cl((t - L.in) / 3);
    g.save(); g.globalAlpha *= fade; g.translate(cx, cy0 + total / 2); g.scale(grow, grow); g.translate(-cx, -(cy0 + total / 2));
    if (P.backdrop) {  // soft dark cloud behind a stat block, so white text reads on any photo
      g.save(); g.filter = `blur(${P.backdrop.blur}px)`; g.fillStyle = `rgba(0,0,0,${P.backdrop.opacity * cl((t - L.in) / .3)})`;
      g.beginPath(); g.ellipse(cx, cy0 + total / 2, W * .42, total * .75, 0, 0, Math.PI * 2); g.fill(); g.restore();
    }
    if (P.box) drawCaptionBox(g, L, t, P, cx, cy0, total, sizes);
    let y = cy0;
    lines.forEach((ln, i) => {
      const role = roles[ln.role], size = sizes[i], tin = ln.in ?? (L.in + i * TI.stagger_s);
      y += size;
      if (t >= tin && !P.box) {
        font(g, role.font, size); g.textBaseline = 'alphabetic'; g.textAlign = 'left';
        let runs = runsOf(ln, role);
        if (ln.count) runs = [{ t: fmtCount(ln, cl((t - tin) / TI.count_s)) }];
        const widths = runs.map(r => g.measureText(r.t).width), tw = widths.reduce((a, b) => a + b, 0);
        let x = cx - tw / 2;
        const u = cl((t - tin) / TI.pop_s), e = E.easeOutCubic(u);
        g.save();
        if (!role.type) {  // pop: from a little bigger and blurred, sharp and in place
          const s = TI.pop_from + (1 - TI.pop_from) * e;
          g.translate(cx, y - size * .35); g.scale(s, s); if (role.rotate) g.rotate(role.rotate * Math.PI / 180); g.translate(-cx, -(y - size * .35));
          g.globalAlpha *= cl(u * 1.8);
          if (u < 1) g.filter = `blur(${(TI.blur_px * (1 - e)).toFixed(1)}px)`;
        } else if (role.rotate) { g.translate(cx, y); g.rotate(role.rotate * Math.PI / 180); g.translate(-cx, -y); }
        let wi = 0;
        runs.forEach((r, k) => {
          // a run may use another role of the preset (its own gradient) or just a flat colour
          const rr = r.role && roles[r.role] && roles[r.role].fill ? { ...role, ...roles[r.role], font: role.font } : role;
          const col = r.color || (r.role && roles[r.role] && !roles[r.role].fill ? roles[r.role].color : null);
          if (!role.type) paintRun(g, r.t, x, y, size, rr, col);
          else wi = softWords(g, r.t, x, y, size, rr, col, t, tin, wi);  // sentences: word by word, soft, no typing
          x += widths[k];
        });
        if (role.rules) {  // '— SECOND REASON —': thin rules either side
          g.fillStyle = '#FFFFFF'; const rw = size * 1.6, ry = y - size * .36;
          g.fillRect(cx - tw / 2 - rw - size * .4, ry, rw, Math.max(3, size * .07)); g.fillRect(cx + tw / 2 + size * .4, ry, rw, Math.max(3, size * .07));
        }
        g.restore();
      }
      y += gap;
    });
    g.restore();
  }
  // Words of a sentence arrive one after another, each fading up from a little below with a slight blur (2026-10-22
  // user: text must come in smoothly, not typed). Positions come from the whole run's measure, so kerning is kept.
  function softWords(g, txt, x0, y, size, role, col, t, tin, i0) {
    const TI = S().text.in, parts = txt.split(/(\s+)/);
    let pos = 0, i = i0;
    for (const w of parts) {
      if (w.trim()) {
        const u = cl((t - tin - i * TI.word_step_s) / TI.word_s), e = E.easeOutCubic(u);
        if (u > 0) {
          g.save(); g.globalAlpha *= e;
          if (u < 1) g.filter = `blur(${(TI.word_blur_px * (1 - e)).toFixed(1)}px)`;
          paintRun(g, w, x0 + g.measureText(txt.slice(0, pos)).width, y + (1 - e) * size * .28, size, role, col);
          g.restore();
        }
        i++;
      }
      pos += w.length;
    }
    return i;
  }
  // q3: a brown box with an orange border; the narration types into it word by word, wrapped to the box width
  function drawCaptionBox(g, L, t, P, cx, cy0, total, sizes) {
    const B = P.box, role = P.roles.text, size = role.size, w = B.w * W, ln = L.lines[0];
    font(g, role.font, size); g.textBaseline = 'alphabetic'; g.textAlign = 'left';
    const words = (ln.words || ln.text.split(' ').map((w, i) => ({ w, t: L.in + i * .25 })));
    const lines = [[]]; let lw = 0;
    for (const wd of words) {
      const ww = g.measureText(wd.w + ' ').width;
      if (lw + ww > w - 2 * B.pad_x && lines[lines.length - 1].length) { lines.push([]); lw = 0; }
      lines[lines.length - 1].push(wd); lw += ww;
    }
    const lh = size * 1.32, bh = lines.length * lh + 2 * B.pad_y, by = (L.y ?? P.y) * H - bh / 2, bx = cx - w / 2;
    const u = E.easeOutCubic(cl((t - L.in) / .3));
    g.save(); g.globalAlpha *= u;
    g.fillStyle = MAPANI.rgba(B.fill, B.opacity); g.fillRect(bx, by, w, bh);
    g.lineWidth = B.stroke_w; g.strokeStyle = B.stroke; g.strokeRect(bx, by, w, bh);
    let y = by + B.pad_y + size;
    for (const row of lines) {
      let x = bx + B.pad_x;
      for (const wd of row) {
        const a = E.easeOutCubic(cl((t - wd.t) / S().text.in.word_s));
        if (a > 0) { g.globalAlpha = u * a; g.fillStyle = role.fill[0]; g.shadowColor = 'rgba(0,0,0,.6)'; g.shadowBlur = 6; g.fillText(wd.w, x, y + (1 - a) * size * .28); }
        x += g.measureText(wd.w + ' ').width;
      }
      y += lh;
    }
    g.restore();
  }

  // ---------- image shot: a still that keeps moving (push / pull / pan), graded, with its overlays ----------
  function cover(g, im, s, ox, oy, rot) {
    const k = Math.max(W / im.width, H / im.height) * s;
    g.save(); g.translate(W / 2 + ox, H / 2 + oy); if (rot) g.rotate(rot);
    g.drawImage(im, -im.width * k / 2, -im.height * k / 2, im.width * k, im.height * k); g.restore();
  }
  function motionAt(s, t) {
    const m = s.motion || {}, I = S().image, span = Math.max(.5, s.end - s.start + .6);
    const u = (E[m.ease] || E.easeInOutSine)(cl((t - s.start + .3) / span));
    const [s0, s1] = m.scale || I.push, [x0, y0] = m.from || [0, 0], [x1, y1] = m.to || [0, 0];
    return { k: s0 + (s1 - s0) * u, x: x0 + (x1 - x0) * u, y: y0 + (y1 - y0) * u, r: ((m.rot || 0) * (u - .5)) * Math.PI / 180 };
  }
  function drawImageShot(g, s, t) {
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const im = IM[s.file];
    if (im) { const m = motionAt(s, t); cover(g, im, m.k, m.x, m.y, m.r); }
    else placeholder(g, s, 'IMAGE');
    overlays(g, s, t);
  }
  function overlays(g, s, t) {
    for (const L of s.layers || []) {
      if (L.type === 'text') drawText(g, L, t);
      else if (L.type === 'sticker') drawSticker(g, L, t);
      else if (L.type === 'stamp') drawStamp(g, L, t);
    }
  }
  // a cut-out picture (PNG) popping onto the frame: icons, a second photo in a round frame (q4 bus/auto circles)
  function drawSticker(g, L, t) {
    if (t < L.in) return;
    const im = IM[L.file]; if (!im) return;
    const fade = cl(((L.out ?? Infinity) + .2 - t) / .2), u = cl((t - L.in) / .35), s = .6 + .4 * E.backOut(u);
    g.save(); g.globalAlpha *= fade * cl(u * 2); g.translate(L.x, L.y); g.scale(s, s);
    if (L.round) {
      g.save(); g.beginPath(); g.arc(0, 0, L.w / 2, 0, Math.PI * 2); g.shadowColor = 'rgba(0,0,0,.6)'; g.shadowBlur = 30; g.fillStyle = '#fff'; g.fill(); g.restore();
      g.save(); g.beginPath(); g.arc(0, 0, L.w / 2 - 6, 0, Math.PI * 2); g.clip();
      const k = Math.max(L.w / im.width, L.w / im.height); g.drawImage(im, -im.width * k / 2, -im.height * k / 2, im.width * k, im.height * k); g.restore();
    } else {
      const k = L.w / im.width; g.shadowColor = 'rgba(0,0,0,.55)'; g.shadowBlur = 24; g.shadowOffsetY = 10;
      g.drawImage(im, -L.w / 2, -im.height * k / 2, L.w, im.height * k);
    }
    g.restore();
  }

  // ---------- news shot: the real article as a card floating in 3D, a marker sweeping over the spoken line ----------
  function drawNewsShot(g, s, t) {
    const N = S().news, bg = s.bg || N.bg;
    if (bg.kind === 'image' && IM[s.bg_file]) {
      const m = motionAt({ ...s, motion: { scale: [1.08, 1.0] } }, t); cover(g, IM[s.bg_file], m.k, 0, 0, 0);
      g.fillStyle = 'rgba(0,0,0,.35)'; g.fillRect(0, 0, W, H);
    } else {
      g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
      const m = motionAt({ ...s, motion: { scale: [1.15, 1.0] } }, t);
      g.save(); g.translate(W / 2, H / 2); g.scale(m.k, m.k); g.drawImage(TEX.red, -W / 2, -H / 2, W, H); g.restore();
      const vg = g.createRadialGradient(W / 2, H * .45, W * .2, W / 2, H / 2, H * .75);
      vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(0,0,0,.8)'); g.fillStyle = vg; g.fillRect(0, 0, W, H);
    }
    // one card, or a montage of several (the ER original's 'screenshots attached'): each new article flies in on top
    // of the previous ones, slightly turned, the older ones settling back and dimming
    const cards = s.cards && s.cards.length ? s.cards : [{ file: s.file, highlights: s.highlights, focus: s.focus, in: s.start }];
    if (!IM[cards[0].file]) return placeholder(g, s, 'NEWS');
    const live = cards.filter(cd => t >= cd.in - .15 && IM[cd.file]);
    live.forEach((cd, i) => newsCard(g, s, cd, t, i, live.length - 1 - i));  // depth 0 = the newest, on top
    overlays(g, s, t);
  }
  function newsCard(g, s, cd, t, idx, depth) {
    const N = S().news, im = IM[cd.file];
    // the card: screenshot + highlight marks (they sweep in on their words), drawn flat, then put in 3D
    const [card, cg] = buf('card', im.width, im.height);
    cg.drawImage(im, 0, 0);
    for (const h of cd.highlights || []) {
      const u = cl((t - h.in) / N.highlight.sweep_s); if (u <= 0) continue;
      cg.save(); cg.globalCompositeOperation = 'multiply'; cg.fillStyle = MAPANI.rgba(N.highlight.color, 1);
      // rects of one phrase sweep in one after another, like one stroke of a marker across wrapped lines
      const tot = h.rects.reduce((a, r) => a + r[2], 0); let left = u * tot;
      for (const [x, y, w, hh] of h.rects) { const ww = Math.min(w, left); if (ww <= 0) break; cg.fillRect(x - 3, y - 2, ww + 6, hh + 4); left -= w; }
      cg.restore();
    }
    if (N.paper) {  // printed-paper feel (q2 Hindustan Times card): warm tint + paper fibre, multiplied over the page
      cg.save(); cg.globalCompositeOperation = 'multiply';
      cg.fillStyle = N.paper.tint; cg.fillRect(0, 0, card.width, card.height);
      cg.globalAlpha = N.paper.opacity; cg.drawImage(TEX.paper, 0, 0, card.width, card.height); cg.restore();
    }
    const I = N.in, u = E.easeOutCubic(cl((t - cd.in + .15) / I.dur)), life = cl((t - cd.in) / Math.max(1, s.end - cd.in));
    // a tall clipping (stretched to reach the spoken line) is sized by its height: the headline must stay on screen
    const cw = Math.min(N.card_w * W, .8 * H * im.width / im.height), ch = cw * im.height / im.width;
    // after the pop-in the camera keeps pushing toward the highlighted line (the top card only)
    const foc = cd.focus || [.5, .35], push = 1 + (N.push - 1) * E.easeInOutSine(life) * (depth ? 0 : 1);
    const back = 1 - .07 * Math.min(depth, 3);  // older cards sit back
    const sc = (I.from_scale + (1 - I.from_scale) * u) * push * back;
    const rx = lerp(I.from_rx, I.to_rx, u) * (s.tilt ?? 1), ry = lerp(I.from_ry, I.to_ry, u) * (s.tilt ?? 1);
    const fx = (foc[0] - .5) * cw * sc, fy = (foc[1] - .5) * ch * sc;
    const side = idx % 2 ? 1 : -1, many = (s.cards || []).length > 1;
    const cx = W / 2 - fx * (push - 1) / push * 1.2 + (many ? side * 34 * Math.min(depth + 1, 3) * (depth ? 1 : .4) : 0);
    const cy = H * .48 - fy * (push - 1) / push * 1.2 + (1 - u) * (many ? 520 : 160) - (many ? 26 * depth : 0);
    const rz = (s.rz || 0) + (many ? side * (depth ? 4 : 1.5) : 0);
    g.save(); g.globalAlpha *= cl(u * 2.5);
    // soft shadow under the card
    g.save(); g.filter = `blur(${N.shadow.blur}px)`; g.fillStyle = `rgba(0,0,0,${N.shadow.opacity})`;
    g.fillRect(cx - cw * sc / 2, cy - ch * sc / 2 + N.shadow.dy, cw * sc, ch * sc); g.restore();
    const corners = plane(g, card, cx, cy, cw * sc, ch * sc, rx, ry, rz, 2200, 8);
    if (depth) {  // the older cards dim back
      g.fillStyle = `rgba(0,0,0,${.18 * Math.min(depth, 3)})`; g.beginPath();
      corners.forEach(([x, y], k) => k ? g.lineTo(x, y) : g.moveTo(x, y)); g.closePath(); g.fill();
    }
    g.restore();
  }

  // ---------- graphic shot: simple motion graphics on a dark studio background ----------
  // 'connect': the things being linked appear one by one as cards, lines join each to the last ('connect the dots'),
  // then light pulses run along the links. 'list': points slide in one by one with a check mark.
  function studioBg(g, t) {
    const gr = g.createLinearGradient(0, 0, 0, H); gr.addColorStop(0, '#0B1222'); gr.addColorStop(1, '#141C33');
    g.fillStyle = gr; g.fillRect(0, 0, W, H);
    g.save(); g.strokeStyle = 'rgba(120,160,255,.07)'; g.lineWidth = 1.5; const off = (t * 14) % 90;
    for (let x = -90 + off; x < W + 90; x += 90) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
    for (let y = -90 + off; y < H + 90; y += 90) { g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
    g.restore();
    const r = rnd('dust');  // slow drifting specks of light
    g.save();
    for (let i = 0; i < 40; i++) {
      const x = (r() * W + t * (8 + r() * 14)) % W, y = (r() * H - t * (6 + r() * 10) % H + H) % H, a = .15 + .35 * r();
      g.fillStyle = `rgba(150,190,255,${a})`; g.beginPath(); g.arc(x, y, 1.5 + 2 * r(), 0, Math.PI * 2); g.fill();
    }
    g.restore();
  }
  function drawGraphicShot(g, s, t) {
    studioBg(g, t);
    const GR = s.graphic || {}, items = GR.items || [], P = S().graphic;
    const zoom = 1 + .05 * cl((t - s.start) / Math.max(1, s.end - s.start));
    g.save(); g.translate(W / 2, H / 2); g.scale(zoom, zoom); g.translate(-W / 2, -H / 2);
    if (GR.kind === 'list') drawList(g, items, t, P.list);
    else drawConnect(g, items, t, P.connect, s.start);
    g.restore();
    overlays(g, s, t);
  }
  // one card per row, zig-zagging left / right, so wide cards never overlap
  function nodePos(i, n) {
    return [W * (i % 2 ? .6 : .4), H * .52 + (i - (n - 1) / 2) * H * .125];
  }
  function drawConnect(g, items, t, C, t0) {
    const n = items.length, pos = items.map((_, i) => nodePos(i, n));
    for (let i = 1; i < n; i++) {  // links first (under the cards): each new card joined to the one before
      const u = E.easeInOutCubic(cl((t - items[i].in - .1) / C.link_s)); if (u <= 0) continue;
      const [x0, y0] = pos[i - 1], [x1, y1] = pos[i];
      g.save(); g.strokeStyle = C.link; g.lineWidth = 5; g.shadowColor = C.link; g.shadowBlur = 16; g.lineCap = 'round';
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u); g.stroke();
      const all = items[n - 1].in + C.link_s;  // once everything is linked, light runs along the chain
      if (t > all) {
        const k = ((t - all) * .8 + i * .3) % 1; g.fillStyle = '#FFFFFF'; g.shadowColor = '#FFFFFF'; g.shadowBlur = 20;
        g.beginPath(); g.arc(x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, 7, 0, Math.PI * 2); g.fill();
      }
      g.restore();
    }
    items.forEach((it, i) => {
      const u = cl((t - it.in) / .4); if (u <= 0) return;
      const e = E.backOut(u), [x, y] = pos[i], w = C.card_w, h = C.card_h, bob = Math.sin((t - t0) * 1.3 + i) * 5;
      g.save(); g.translate(x, y + bob); g.scale(.6 + .4 * e, .6 + .4 * e); g.globalAlpha *= cl(u * 2);
      g.shadowColor = 'rgba(0,0,0,.5)'; g.shadowBlur = 30; g.shadowOffsetY = 12;
      g.fillStyle = C.card; g.beginPath(); g.roundRect(-w / 2, -h / 2, w, h, 18); g.fill(); g.shadowColor = 'transparent';
      g.strokeStyle = C.link; g.lineWidth = 6; g.beginPath(); g.arc(-w / 2 + 52, 0, 22, -Math.PI * .1, Math.PI * 1.6); g.stroke();
      g.fillStyle = C.link; g.beginPath(); g.arc(-w / 2 + 52, 0, 7, 0, Math.PI * 2); g.fill();
      font(g, 'st_semi', C.size); g.fillStyle = '#121826'; g.textAlign = 'left'; g.textBaseline = 'middle';
      g.fillText(it.text, -w / 2 + 92, 2, w - 110);
      g.restore();
    });
  }
  function drawList(g, items, t, Lt) {
    const n = items.length, y0 = H * .5 - (n - 1) * Lt.gap / 2;
    items.forEach((it, i) => {
      const u = E.easeOutCubic(cl((t - it.in) / .45)); if (u <= 0) return;
      const y = y0 + i * Lt.gap, x = W * .12 + (1 - u) * -80;
      g.save(); g.globalAlpha *= u;
      g.fillStyle = 'rgba(255,255,255,.07)'; g.beginPath(); g.roundRect(x - 20, y - Lt.size, W * .78, Lt.size * 2, 18); g.fill();
      g.fillStyle = Lt.tick; g.beginPath(); g.arc(x + 26, y, 24, 0, Math.PI * 2); g.fill();
      g.strokeStyle = '#0B1222'; g.lineWidth = 6; g.lineCap = 'round'; g.lineJoin = 'round';
      g.beginPath(); g.moveTo(x + 14, y + 1); g.lineTo(x + 23, y + 10); g.lineTo(x + 39, y - 9); g.stroke();
      font(g, 'st_bold', Lt.size); g.fillStyle = '#FFFFFF'; g.textAlign = 'left'; g.textBaseline = 'middle';
      g.fillText(it.text, x + 70, y + 2, W * .66);
      g.restore();
    });
  }

  // a rubber stamp slamming onto the picture ('DECLASSIFIED', 'CONFIRMED'): big, then it hits and the frame shakes once
  function drawStamp(g, L, t) {
    if (t < L.in) return;
    const St = S().stamp, u = cl((t - L.in) / St.slam_s), e = E.easeInOutCubic(u), sc = St.from_scale + (1 - St.from_scale) * e;
    g.save(); g.translate(L.x ?? W / 2, (L.y ?? St.y) * H); g.rotate(St.rotate * Math.PI / 180); g.scale(sc, sc);
    g.globalAlpha *= Math.min(1, u * 3) * cl(((L.out ?? Infinity) + .2 - t) / .2);
    font(g, 'st_impact', St.size);
    const txt = L.text.toUpperCase(), w = g.measureText(txt).width + 2 * St.pad, h = St.size * 1.35;
    g.strokeStyle = St.color; g.lineWidth = 10; g.strokeRect(-w / 2, -h / 2, w, h);
    g.lineWidth = 3; g.strokeRect(-w / 2 + 14, -h / 2 + 14, w - 28, h - 28);
    g.fillStyle = St.color; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(txt, 0, 4);
    g.globalCompositeOperation = 'destination-out'; g.globalAlpha = .35; g.drawImage(TEX.paper, -w / 2, -h / 2, w, h);  // worn ink
    g.restore();
  }
  function stampShake(s, t) {
    for (const L of s.layers || []) if (L.type === 'stamp') {
      const k = t - (L.in + S().stamp.slam_s);
      if (k > 0 && k < .25) { const a = (1 - k / .25) * 14; return [Math.sin(k * 90) * a, Math.cos(k * 70) * a * .6]; }
    }
    return [0, 0];
  }

  // ---------- map shot (mapani camera) + story layers, optionally tilted like Google Earth ----------
  function drawMap(g, s, t) {
    const prev = G;
    if (s.tilt) {
      const [mc, mg] = buf('map'); G = mg; TILTED = true; MAPANI.drawMapShot(s, t); TILTED = false; G = prev;
      g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
      const T3 = S().map.tilt, k = 1.45;
      plane(g, mc, W / 2, H * .52, W * k, H * k, s.tilt.rx ?? T3.rx, 0, s.tilt.rz || 0, T3.persp * 1.2, 14);
    } else { G = g; MAPANI.drawMapShot(s, t); G = prev; }
    if (s.intro === 'clouds') cloudsParting(g, cl((t - s.start + .1) / 1.7));
    overlays(g, s, t);
  }
  // q2 / q4 open on clouds that part over the map as the camera falls through them
  function cloudsParting(g, u) {
    if (u >= 1) return;
    const e = E.easeInOutCubic(u), s = 1.5 + 1.2 * e;
    g.save();
    g.globalAlpha = Math.pow(1 - u, .8);
    for (const side of [-1, 1]) {
      g.save(); g.translate(W / 2 + side * e * W * .75, H / 2); g.scale(side * s, s);
      g.drawImage(TEX.cloud, -W / 2, -H / 2, W, H); g.restore();
    }
    g.globalAlpha = .7 * Math.pow(1 - u, 2.2); g.fillStyle = '#E9EDF0'; g.fillRect(0, 0, W, H);
    g.restore();
  }
  window.STORY_MAP_HOOK = (when, g, c, shot, t) => {
    if (when === 'under') { for (const L of shot.layers) if (L.type === 'region') drawRegion(g, c, L, t); return; }
    for (const L of shot.layers) if (L.type === 'flow') drawFlow(g, c, L, t);
    // magnifiers first: they show the map under them, not the labels drawn after
    for (const L of shot.layers) if (L.type === 'callout') drawCallout(g, c, L, t);
    for (const L of shot.layers) if (L.type === 'infobox') drawInfobox(g, c, L, t);
    for (const L of shot.layers) {
      if (L.type === 'tag') drawTag(g, c, L, t);
      else if (L.type === 'region_label') drawRegionLabel(g, c, L, t);
    }
  };
  // a place coloured in: glowing translucent fill (texture shows), white edge; or a heat gradient (q2 India); or a
  // picture inside the outline. It never just appears: the outline draws itself around the place first, then the
  // colour / picture fills in, and a picture keeps pushing in (2026-10-22 user: our Delhi map felt stuck)
  function perimeter(c, geom) {
    let n = 0;
    for (const poly of geom.coords) { const r = poly[0]; for (let i = 1; i < r.length; i++) n += Math.hypot((r[i][0] - r[i - 1][0]) / c.mpp, (r[i][1] - r[i - 1][1]) / c.mpp); }
    return n;
  }
  function drawRegion(g, c, L, t) {
    if (t < L.in) return;
    const F = S().map.fill_styles, R = S().map.region_in, geom = SCENES.geometry[L.geom], p = MAPANI.polyPath(g, c, geom);
    const out = cl(((L.out ?? Infinity) + .3 - t) / .3);
    if (out <= 0) return;
    if (L.style === 'heat') {  // the heat map washes in over the whole country
      const k = E.easeOutCubic(cl((t - L.in) / (L.in_s || .45))) * out;
      const H0 = F.heat, [X, Y] = L.hot, x = MAPANI.sx(c, X), y = MAPANI.sy(c, Y), r = L.radius_m / c.mpp;
      const gr = g.createRadialGradient(x, y, 0, x, y, r);
      H0.stops.forEach((col, i) => gr.addColorStop(i / (H0.stops.length - 1), col));
      g.save(); g.clip(p, 'evenodd'); g.globalAlpha = k * H0.opacity; g.globalCompositeOperation = 'hard-light';
      g.fillStyle = gr; g.fillRect(0, 0, W, H); g.restore();
      g.save(); g.globalAlpha = k * out; g.lineJoin = 'round'; g.lineWidth = F.glow.edge_w; g.strokeStyle = MAPANI.rgba(F.glow.edge, F.glow.edge_opacity);
      g.shadowColor = 'rgba(255,255,255,.8)'; g.shadowBlur = F.glow.glow_blur; g.stroke(p); g.restore();
      return;
    }
    const ue = E.easeInOutCubic(cl((t - L.in) / R.edge_draw_s));               // outline drawing on
    const uf = E.easeOutCubic(cl((t - L.in - R.fill_after_s) / R.fill_s)) * out; // fill / picture coming in
    const per = perimeter(c, geom);
    if (L.style === 'photo') {  // q2 Delhi: the place's outline is a window onto a picture of the problem
      const im = IM[L.file], pts = geom.coords.flat(2);
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const [X, Y] of pts) { const x = MAPANI.sx(c, X), y = MAPANI.sy(c, Y); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
      if (uf > 0) {
        g.save(); g.globalAlpha = uf; g.clip(p, 'evenodd');
        if (im) {  // the picture rises out of a slight blur and keeps pushing in for the whole shot
          const life = cl((t - L.in) / Math.max(3, (L.out ?? L.in + 6) - L.in));
          const bw = x1 - x0, bh = y1 - y0, kk = Math.max(bw / im.width, bh / im.height) * (R.photo_push[0] + (R.photo_push[1] - R.photo_push[0]) * E.easeOutSine(life));
          if (uf < 1) g.filter = `blur(${(10 * (1 - uf)).toFixed(1)}px)`;
          g.drawImage(im, (x0 + x1) / 2 - im.width * kk / 2, (y0 + y1) / 2 - im.height * kk / 2, im.width * kk, im.height * kk);
          g.filter = 'none';
        }
        g.filter = 'blur(16px)'; g.lineWidth = 70; g.strokeStyle = 'rgba(0,0,0,.6)'; g.stroke(p);  // inner shadow: cut INTO the map
        g.restore();
      }
      const P0 = F.photo_edge;
      g.save(); g.lineJoin = 'round'; g.lineCap = 'round'; g.globalAlpha = out;
      if (ue < 1) {  // one bright red line running around the place
        g.lineWidth = P0.width * .7; g.strokeStyle = P0.color; g.shadowColor = MAPANI.rgba(P0.color, 1); g.shadowBlur = P0.glow;
        g.setLineDash([per * ue, per + 10]); g.stroke(p);
      } else {
        const pulse = 1 + .12 * Math.sin((t - L.in) * 3.2);  // the rim breathes a little: alive, not frozen
        for (let j = 0; j < 3; j++) {  // a ragged red rim: three dashed passes of different rhythm
          g.lineWidth = P0.width * (1 - j * .28) * pulse; g.strokeStyle = MAPANI.rgba(P0.color, .5 + j * .2);
          g.shadowColor = MAPANI.rgba(P0.color, .9); g.shadowBlur = P0.glow * pulse; g.setLineDash([3 + j * 4, 2 + j * 2]); g.stroke(p);
        }
      }
      g.restore();
      return;
    }
    const st = F.glow;
    if (uf > 0) {
      g.save(); g.clip(p, 'evenodd');
      g.globalCompositeOperation = 'overlay'; g.globalAlpha = uf; g.fillStyle = L.color; g.fillRect(0, 0, W, H);
      g.globalCompositeOperation = 'source-over'; g.globalAlpha = uf * st.opacity * (L.opacity ?? 1); g.fill(p, 'evenodd');
      g.restore();
    }
    g.save(); g.globalAlpha = out; g.lineJoin = 'round'; g.lineCap = 'round'; g.lineWidth = L.edge_w ?? st.edge_w;
    g.strokeStyle = MAPANI.rgba(L.edge || st.edge, st.edge_opacity);
    g.shadowColor = MAPANI.rgba(L.color || '#FFFFFF', .9); g.shadowBlur = st.glow_blur;
    if (ue < 1) g.setLineDash([per * ue, per + 10]);
    g.stroke(p); g.restore();
  }
  // q2 water sources: a river / canal glowing cyan, drawing itself toward the city, with light pulses running along it
  // (the water arriving) and an arrowhead at its tip
  function drawFlow(g, c, L, t) {
    if (t < L.in) return;
    const F = S().map.flow, geom = SCENES.geometry[L.geom], pts = geom.coords, cum = geom.len;
    const u = E.easeInOutSine(cl((t - L.in) / (L.draw_s || F.draw_s))), total = cum[cum.length - 1] * u;
    const p = new Path2D(); let a = [MAPANI.sx(c, pts[0][0]), MAPANI.sy(c, pts[0][1])], b = a;
    p.moveTo(a[0], a[1]);
    for (let i = 1; i < pts.length; i++) {
      let X = pts[i][0], Y = pts[i][1];
      if (cum[i] > total) { const f = (total - cum[i - 1]) / (cum[i] - cum[i - 1]); X = pts[i - 1][0] + (X - pts[i - 1][0]) * f; Y = pts[i - 1][1] + (Y - pts[i - 1][1]) * f; }
      a = b; b = [MAPANI.sx(c, X), MAPANI.sy(c, Y)]; p.lineTo(b[0], b[1]);
      if (cum[i] > total) break;
    }
    const col = L.color || F.color;
    g.save(); g.lineCap = 'round'; g.lineJoin = 'round';
    g.shadowColor = MAPANI.rgba(col, .9); g.shadowBlur = F.glow; g.strokeStyle = MAPANI.rgba(col, .45); g.lineWidth = F.width * 2; g.stroke(p);
    g.shadowBlur = F.glow * .5; g.strokeStyle = col; g.lineWidth = F.width; g.stroke(p);
    g.shadowColor = 'rgba(255,255,255,.9)'; g.shadowBlur = 10; g.setLineDash([F.dash, F.gap]); g.lineDashOffset = -t * F.speed;
    g.strokeStyle = 'rgba(235,250,255,.95)'; g.lineWidth = F.width * .42; g.stroke(p); g.setLineDash([]);
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), s = F.width * 1.9;
    g.translate(b[0], b[1]); g.rotate(ang); g.fillStyle = '#EFFBFF'; g.shadowColor = MAPANI.rgba(col, 1); g.shadowBlur = 18;
    g.beginPath(); g.moveTo(s * .9, 0); g.lineTo(-s * .5, -s * .65); g.lineTo(-s * .25, 0); g.lineTo(-s * .5, s * .65); g.closePath(); g.fill();
    g.restore();
  }
  // the source's share in a dark box with a water-drop icon: '57% WATER' / 'FROM HARYANA (RAVI-BEAS + YAMUNA)'
  function drawInfobox(g, c, L, t) {
    if (t < L.in) return;
    const B = S().map.infobox, fade = cl(((L.out ?? Infinity) + .25 - t) / .25), u = cl((t - L.in) / .4), s = .6 + .4 * E.backOut(u);
    if (fade <= 0) return;
    const m = /^(\S*\d\S*)\s*(.*)$/.exec(L.label) || [null, '', L.label], num = m[1], rest = m[2];
    font(g, 'st_impact', B.size); const nw = g.measureText(num + (num ? ' ' : '')).width, rw = g.measureText(rest).width;
    font(g, 'st_semi', B.sub_size); const sw = L.sub ? g.measureText(L.sub).width : 0;
    const icon = B.size * .9, w = Math.max(nw + rw, sw) + icon + 3 * B.pad, h = B.size + (L.sub ? B.sub_size * 1.25 : 0) + 2 * B.pad;
    let x = L.X !== undefined ? MAPANI.sx(c, L.X) + (L.dx || 0) : L.x, y = L.X !== undefined ? MAPANI.sy(c, L.Y) + (L.dy || 0) : L.y;
    // inside a tilted map the frame is magnified ~1.45x and seen in perspective: only its middle reaches the screen
    const mx = TILTED ? 70 + W * .2 : 70, myT = TILTED ? 260 + H * .14 : 260, myB = TILTED ? 420 + H * .12 : 420;
    x = Math.max(mx + w / 2, Math.min(W - mx - w / 2, x)); y = Math.max(myT + h / 2, Math.min(H - myB - h / 2, y));
    g.save(); g.globalAlpha *= fade * cl(u * 2); g.translate(x, y); g.scale(s, s);
    g.shadowColor = 'rgba(0,0,0,.6)'; g.shadowBlur = 24; g.shadowOffsetY = 8;
    g.fillStyle = MAPANI.rgba(B.fill, B.opacity); g.beginPath(); g.roundRect(-w / 2, -h / 2, w, h, B.radius); g.fill();
    g.shadowColor = 'transparent'; g.lineWidth = 2; g.strokeStyle = 'rgba(255,255,255,.28)'; g.stroke();
    const ix = -w / 2 + B.pad + icon / 2, iy = -h / 2 + B.pad + B.size * .45;  // water drop
    g.fillStyle = B.icon; g.beginPath(); g.moveTo(ix, iy - icon * .5);
    g.bezierCurveTo(ix + icon * .45, iy - icon * .02, ix + icon * .42, iy + icon * .45, ix, iy + icon * .45);
    g.bezierCurveTo(ix - icon * .42, iy + icon * .45, ix - icon * .45, iy - icon * .02, ix, iy - icon * .5); g.fill();
    let tx = -w / 2 + 2 * B.pad + icon; const ty = -h / 2 + B.pad + B.size * .82;
    font(g, 'st_impact', B.size); g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    if (num) { g.fillStyle = B.num_color; g.fillText(num, tx, ty); tx += nw; }
    g.fillStyle = '#FFFFFF'; g.fillText(rest, tx, ty);
    if (L.sub) { font(g, 'st_semi', B.sub_size); g.fillStyle = 'rgba(255,255,255,.78)'; g.fillText(L.sub, -w / 2 + 2 * B.pad + icon, ty + B.sub_size * 1.3); }
    g.restore();
  }
  function drawTag(g, c, L, t) {
    if (t < L.in) return;
    const T0 = S().map.tag, X = L.X, Y = L.Y, x = MAPANI.sx(c, X), y = MAPANI.sy(c, Y);
    const fade = cl(((L.out ?? Infinity) + .25 - t) / .25), u = cl((t - L.in) / T0.pop_s), s = .5 + .5 * E.backOut(u);
    if (fade <= 0) return;
    font(g, T0.font, T0.size);
    const w = g.measureText(L.label).width + 2 * T0.pad_x, h = T0.size + 2 * T0.pad_y;
    g.save(); g.globalAlpha *= fade * cl(u * 2); g.translate(x, y); g.scale(s, s);
    const by = -h - T0.pointer;
    g.shadowColor = 'rgba(0,0,0,.55)'; g.shadowBlur = 14; g.shadowOffsetY = 4;
    g.fillStyle = MAPANI.rgba(T0.fill, T0.opacity); g.fillRect(-w / 2, by, w, h);
    g.beginPath(); g.moveTo(-T0.pointer, -T0.pointer); g.lineTo(T0.pointer, -T0.pointer); g.lineTo(0, 0); g.fill();
    g.shadowColor = 'transparent'; g.fillStyle = T0.stripe; g.fillRect(-w / 2, by, w, T0.stripe_h);
    g.fillStyle = T0.text; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(L.label, 0, by + h / 2 + 2);
    g.restore();
  }
  function drawRegionLabel(g, c, L, t) {
    if (t < L.in) return;
    const st = L.style === '3d' ? S().map.label3d : S().map.region_label;
    const fade = cl(((L.out ?? Infinity) + .25 - t) / .25), u = cl((t - L.in) / .4);
    const x = MAPANI.sx(c, L.X), y = MAPANI.sy(c, L.Y), size = st.size * (L.scale || 1) * (L.mpp_in ? L.mpp_in / c.mpp : 1);
    g.save(); g.globalAlpha *= fade * u; g.translate(x, y); g.rotate((L.rot || 0) * Math.PI / 180);
    font(g, st.font, size); g.textAlign = 'center'; g.textBaseline = 'middle';
    const txt = L.text.toUpperCase();
    if (st.depth) for (let k = st.depth; k >= 1; k--) { g.fillStyle = st.side; g.fillText(txt, k * .4, k * size / 60); }
    g.shadowColor = 'rgba(0,0,0,.75)'; g.shadowBlur = size * .35; g.shadowOffsetY = size * .08;
    g.fillStyle = st.color; g.fillText(txt, 0, 0);
    g.restore();
  }
  // a round magnifier on a spot: the map under it zoomed and tinted, white ring, its name in it (q3)
  function drawCallout(g, c, L, t) {
    if (t < L.in) return;
    const C0 = S().map.callout, fade = cl(((L.out ?? Infinity) + .3 - t) / .3), u = cl((t - L.in) / .45), s = .3 + .7 * E.backOut(u);
    if (fade <= 0) return;
    const x = L.x ?? MAPANI.sx(c, L.X), y = L.y ?? MAPANI.sy(c, L.Y), r = (L.r || C0.r);
    g.save(); g.globalAlpha *= fade * cl(u * 2); g.translate(x, y); g.scale(s, s);
    g.save(); g.beginPath(); g.arc(0, 0, r, 0, Math.PI * 2); g.clip();
    const src = g.canvas, z = C0.zoom;  // the frame drawn so far, magnified around the spot
    g.drawImage(src, x - r / z, y - r / z, 2 * r / z, 2 * r / z, -r, -r, 2 * r, 2 * r);
    g.fillStyle = MAPANI.rgba(C0.tint, C0.tint_opacity); g.fillRect(-r, -r, 2 * r, 2 * r);
    g.restore();
    g.lineWidth = C0.ring; g.strokeStyle = '#FFFFFF'; g.shadowColor = 'rgba(0,0,0,.6)'; g.shadowBlur = 18;
    g.beginPath(); g.arc(0, 0, r, 0, Math.PI * 2); g.stroke(); g.shadowColor = 'transparent';
    font(g, C0.font, C0.size); g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = '#FFFFFF';
    g.shadowColor = 'rgba(0,0,0,.8)'; g.shadowBlur = 10;
    const lines = L.label.toUpperCase().split('\n');
    lines.forEach((ln, i) => g.fillText(ln, 0, (i - (lines.length - 1) / 2) * C0.size * 1.08));
    g.restore();
  }

  // ---------- CTA: torn paper strip, the question typing in with the voice, key words red ----------
  function tornPath(w, h, seed) {
    const r = rnd(seed), p = new Path2D(), step = 14;
    p.moveTo(0, 0);
    for (let x = 0; x <= w; x += step) p.lineTo(x, (r() - .5) * 14 + Math.sin(x / 90) * 5);
    for (let x = w; x >= 0; x -= step) p.lineTo(x, h + (r() - .5) * 14 + Math.sin(x / 70 + 2) * 5);
    p.closePath(); return p;
  }
  let TORN = null;
  function drawCtaShot(g, s, t) {
    const C0 = S().cta;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    if (IM[s.bg_file]) { const m = motionAt({ ...s, motion: { scale: [1.05, 1.12] } }, t); cover(g, IM[s.bg_file], m.k, 0, 0, 0); }
    g.fillStyle = `rgba(0,0,0,${C0.bg_dim})`; g.fillRect(0, 0, W, H);
    const ph = C0.paper_h * H, py = C0.y * H, u = cl((t - s.start) / C0.in_s), e = E.backOut(u);
    TORN = TORN || tornPath(W + 40, ph, 'torn');
    g.save(); g.translate(W / 2, py); g.rotate((1 - e) * -.05); g.scale(1, .08 + .92 * e); g.translate(-W / 2 - 20, -ph / 2);
    g.shadowColor = 'rgba(0,0,0,.6)'; g.shadowBlur = 40; g.shadowOffsetY = 18; g.fillStyle = '#F4F4F2'; g.fill(TORN);
    g.shadowColor = 'transparent'; g.save(); g.clip(TORN); g.globalAlpha = .55; g.drawImage(TEX.paper, 0, 0, W + 40, ph); g.restore();
    g.restore();
    if (u < .6) return;
    // text: each word fades up on its spoken time (no typing), wrapped, centred; emphasised words red caps
    const words = s.words || [], size = C0.size, maxw = W * .8, lh = size * 1.3;
    const fw = wd => { font(g, wd.hl ? C0.hl_font : C0.font, size); return g.measureText((wd.hl ? wd.w.toUpperCase() : wd.w) + ' ').width; };
    const rows = [[]]; let lw = 0;
    for (const wd of words) { const ww = fw(wd); if (lw + ww > maxw && rows[rows.length - 1].length) { rows.push([]); lw = 0; } rows[rows.length - 1].push(wd); lw += ww; }
    let y = py - (rows.length - 1) * lh / 2 + size * .35;
    for (const row of rows) {
      const rw = row.reduce((a, wd) => a + fw(wd), 0); let x = W / 2 - rw / 2;
      for (const wd of row) {
        const ww = fw(wd), txt = wd.hl ? wd.w.toUpperCase() : wd.w, a = E.easeOutCubic(cl((t - wd.t) / S().text.in.word_s));
        font(g, wd.hl ? C0.hl_font : C0.font, size); g.textAlign = 'left'; g.textBaseline = 'alphabetic';
        if (a > 0) { g.save(); g.globalAlpha *= a; g.fillStyle = wd.hl ? C0.hl_color : '#141414'; g.fillText(txt, x, y + (1 - a) * size * .3); g.restore(); }
        x += ww;
      }
      y += lh;
    }
  }

  // ---------- Follow end card ----------
  function drawFollowShot(g, s, t) {
    const F = S().follow, lt = t - s.start;
    g.fillStyle = '#0B0B0D'; g.fillRect(0, 0, W, H);
    if (IM[s.bg_file]) { cover(g, IM[s.bg_file], 1.02 + .02 * cl(lt / 4), 0, 0, 0); g.fillStyle = `rgba(0,0,0,${F.bg_dim})`; g.fillRect(0, 0, W, H); }
    const cx = W / 2, cy = F.y * H, R = F.avatar_d / 2, u = E.backOut(cl(lt / .45));
    g.save(); g.translate(cx, cy); g.scale(.6 + .4 * u, .6 + .4 * u); g.globalAlpha = cl(lt / .25);
    g.save(); g.beginPath(); g.arc(0, 0, R, 0, Math.PI * 2); g.clip();
    const av = IM[s.avatar_file];
    if (av) { const k = Math.max(2 * R / av.width, 2 * R / av.height); g.drawImage(av, -av.width * k / 2, -av.height * k / 2, av.width * k, av.height * k); }
    else { g.fillStyle = '#333'; g.fillRect(-R, -R, 2 * R, 2 * R); }
    g.restore();
    const ring = E.easeInOutCubic(cl((lt - .15) / F.ring_s));
    g.lineWidth = F.ring; g.strokeStyle = F.ring_color; g.lineCap = 'round';
    g.beginPath(); g.arc(0, 0, R + F.ring / 2, -Math.PI / 2, -Math.PI / 2 + ring * Math.PI * 2); g.stroke();
    g.restore();
    // button below the avatar, clicked at click_at: Follow (blue) -> Following (dark)
    const B = F.button, by = cy + R - B.h * .15, done = lt >= F.click_at, press = Math.exp(-Math.max(0, lt - F.click_at) * 10) * (done ? 1 : 0);
    const bu = E.backOut(cl((lt - .35) / .35));
    if (lt > .35) {
      g.save(); g.translate(cx, by); g.scale(bu * (1 - .08 * press), bu * (1 - .08 * press));
      g.fillStyle = done ? B.done_fill : B.fill; g.beginPath(); g.roundRect(-B.w / 2, -B.h / 2, B.w, B.h, B.radius); g.fill();
      font(g, B.font, B.size); g.fillStyle = done ? B.done_text : B.text; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(done ? B.done_label : B.label, 0, 3); g.restore();
    }
    // handle box typing in
    // handle box: opens sideways from the centre, the name fades up into it (no typing - user rule)
    const Hh = F.handle, txt = s.handle || '';
    if (lt > .45) {
      font(g, Hh.font, Hh.size);
      const ub = E.easeOutCubic(cl((lt - .45) / .35)), ut = E.easeOutCubic(cl((lt - .6) / .4));
      const tw = g.measureText(txt).width, bw = (tw + 2 * Hh.pad_x) * ub, bh = Hh.size + 2 * Hh.pad_y, hy = by + B.h / 2 + bh / 2 + 4;
      g.fillStyle = Hh.fill; g.fillRect(cx - bw / 2, hy - bh / 2, bw, bh);
      g.save(); g.globalAlpha *= ut; if (ut < 1) g.filter = `blur(${(6 * (1 - ut)).toFixed(1)}px)`;
      g.fillStyle = Hh.text; g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText(txt, cx - tw / 2, hy + 3 + (1 - ut) * 10);
      g.restore();
    }
    // hand cursor: glides in, taps the button
    const cu = E.easeInOutCubic(cl((lt - .7) / (F.click_at - .75)));
    if (lt > .7) drawHand(g, lerp(W * .78, cx + B.w * .22, cu), lerp(H * .78, by + B.h * .12, cu), 1 - .15 * press);
  }
  function drawHand(g, x, y, s) {
    g.save(); g.translate(x, y); g.scale(s * 1.6, s * 1.6);
    const p = new Path2D('M0 0 L0 -20 C0 -24 6 -24 6 -20 L6 -6 L6 -10 C6 -14 12 -14 12 -10 L12 -5 L12 -8 C12 -12 18 -12 18 -8 L18 -3 L18 -5 C18 -9 24 -9 24 -5 L24 10 C24 18 18 24 10 24 L6 24 C2 24 -2 21 -4 18 L-12 6 C-14 2 -9 -1 -6 2 Z');
    g.translate(-4, 20); g.fillStyle = '#FFFFFF'; g.strokeStyle = '#111'; g.lineWidth = 2.2; g.fill(p); g.stroke(p);
    g.restore();
  }

  function placeholder(g, s, label) {
    g.fillStyle = '#16161a'; g.fillRect(0, 0, W, H);
    g.fillStyle = 'rgba(255,255,255,.8)'; g.font = '700 54px "Inter"'; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    g.fillText(label, W / 2, H / 2 - 20); g.font = '500 30px "Inter"'; g.fillStyle = 'rgba(255,255,255,.5)';
    g.fillText((s.note || s.id || '').slice(0, 48), W / 2, H / 2 + 36);
  }

  // ---------- one shot into a context ----------
  function drawShot(g, s, t) {
    const prev = G; G = g;
    g.save();
    const [shx, shy] = stampShake(s, t);
    if (shx || shy) { g.translate(shx, shy); g.translate(W / 2, H / 2); g.scale(1.03, 1.03); g.translate(-W / 2, -H / 2); }
    if (s.kind === 'map') drawMap(g, s, t);
    else if (s.kind === 'graphic') drawGraphicShot(g, s, t);
    else if (s.kind === 'image') drawImageShot(g, s, t);
    else if (s.kind === 'news') drawNewsShot(g, s, t);
    else if (s.kind === 'cta') drawCtaShot(g, s, t);
    else if (s.kind === 'follow') drawFollowShot(g, s, t);
    else if (s.kind === 'facecam') { if (location.search.includes('overlay')) g.clearRect(0, 0, W, H); else placeholder(g, s, 'FACE-CAM CLIP HERE'); overlays(g, s, t); }
    else placeholder(g, s, (s.kind || 'shot').toUpperCase());
    g.restore();
    if (s.kind !== 'facecam' && s.kind !== 'follow') finishShot(g, s, t);
    G = prev;
  }
  function finishShot(g, s, t) {
    const F = S().finish, v = s.vignette ?? F.vignette;
    if (v > 0) {
      const gr = g.createRadialGradient(W / 2, H / 2, H * .28, W / 2, H / 2, H * .62);
      gr.addColorStop(0, 'rgba(0,0,0,0)'); gr.addColorStop(1, `rgba(0,0,0,${v})`); g.fillStyle = gr; g.fillRect(0, 0, W, H);
    }
  }

  // ---------- transitions ----------
  function blend(g, a, b, type, u) {
    const T0 = S().transition;
    g.save();
    if (type === 'blur') {  // defocus dissolve: the old frame goes soft, the new one comes in soft and sharpens
      const k = smooth(0, 1, u);
      g.filter = `blur(${(T0.blur_px * k).toFixed(1)}px)`; g.drawImage(a, 0, 0);
      g.filter = `blur(${(T0.blur_px * (1 - k)).toFixed(1)}px)`; g.globalAlpha = smooth(.2, .8, u); g.drawImage(b, 0, 0);
      g.filter = 'none';
    } else if (type === 'zoom') {
      // push through: the old frame rushes toward the viewer with a zoom blur, the new one settles from big to rest
      const out = E.easeInOutCubic(u), sw = smooth(.38, .62, u), n = T0.zoom_copies;
      const zoomDraw = (src, base, spread, alpha) => {
        for (let i = 0; i < n; i++) {
          const s = base * (1 + spread * i / n); g.globalAlpha = alpha / (i * .55 + 1);
          g.drawImage(src, W / 2 - W * s / 2, H / 2 - H * s / 2, W * s, H * s);
        }
      };
      g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
      if (sw < 1) zoomDraw(a, 1 + T0.zoom_scale * out, .25 * Math.sin(Math.PI * u), 1 - sw);
      if (sw > 0) zoomDraw(b, 1 + T0.zoom_scale * .6 * (1 - out), .25 * Math.sin(Math.PI * u), sw);
      g.globalAlpha = .35 * Math.sin(Math.PI * u) ** 4; g.fillStyle = '#FFFFFF'; g.fillRect(0, 0, W, H);
    } else if (type === 'whip') {
      const e = E.easeInOutCubic(u), n = T0.whip_copies, v = Math.sin(Math.PI * u) * W * .12;
      for (const [src, off] of [[a, -e * W], [b, (1 - e) * W]]) for (let i = 0; i < n; i++) {
        g.globalAlpha = 1 / (i * .7 + 1); g.drawImage(src, off - v * i / n, 0);
      }
    } else if (type === 'flash') {
      g.drawImage(u < .5 ? a : b, 0, 0);
      g.globalAlpha = Math.pow(1 - Math.abs(2 * u - 1), 1.5); g.fillStyle = '#FFFFFF'; g.fillRect(0, 0, W, H);
    } else if (type === 'cloud') {
      g.drawImage(u < .5 ? a : b, 0, 0);
      const c = Math.sin(Math.PI * u), s = 1.2 + .8 * u;
      g.globalAlpha = smooth(0, .45, u) * (1 - smooth(.55, 1, u));
      g.drawImage(TEX.cloud, W / 2 - W * s / 2 - W * .1 * (u - .5), H / 2 - H * s / 2, W * s, H * s);
      g.globalAlpha = .55 * c * c; g.fillStyle = '#E8ECEF'; g.fillRect(0, 0, W, H);
    } else if (type === 'glitch') {
      const src = u < .5 ? a : b, f = Math.floor(T * 30), r = rnd('gl' + f), amp = Math.sin(Math.PI * u) * 60;
      g.drawImage(src, 0, 0);
      for (let i = 0; i < 9; i++) {
        const y = r() * H, h = 20 + r() * 120, dx = (r() - .5) * 2 * amp;
        g.drawImage(src, 0, y, W, h, dx, y, W, h);
      }
      g.globalCompositeOperation = 'screen'; g.globalAlpha = .5 * Math.sin(Math.PI * u);
      g.filter = 'hue-rotate(90deg)'; g.drawImage(src, amp * .3, 0); g.filter = 'none';
    } else {  // fade
      g.drawImage(a, 0, 0); g.globalAlpha = E.easeInOutSine(u); g.drawImage(b, 0, 0);
    }
    g.restore();
  }

  // ---------- frame ----------
  function frame(t) {
    let i = 0; while (i + 1 < LIST.length && t >= LIST[i + 1].start) i++;
    const s = LIST[i], nx = LIST[i + 1], pv = LIST[i - 1];
    const durOf = x => x && x.trans ? (x.trans.dur ?? S().transition.dur[x.trans.type] ?? .4) : 0;
    let tr = null;
    if (nx && durOf(nx) > 0 && t >= nx.start - durOf(nx) / 2) tr = { a: s, b: nx, d: durOf(nx), at: nx.start, type: nx.trans.type };
    else if (pv && durOf(s) > 0 && t < s.start + durOf(s) / 2) tr = { a: pv, b: s, d: durOf(s), at: s.start, type: s.trans.type };
    if (!tr) drawShot(G, s, t);
    else {
      const [ca, ga] = buf('ta'), [cb, gb] = buf('tb');
      drawShot(ga, tr.a, t); drawShot(gb, tr.b, t);
      blend(G, ca, cb, tr.type, cl((t - (tr.at - tr.d / 2)) / tr.d));
    }
    finishFrame(G, t, s);
  }
  function finishFrame(g, t, s) {
    const F = S().finish, overlay = location.search.includes('overlay');
    if (!overlay && F.grain > 0) {
      g.save(); g.globalAlpha = F.grain; g.globalCompositeOperation = 'overlay';
      const tile = TEX.grain[Math.floor(t * 30) % 4]; g.fillStyle = g.createPattern(tile, 'repeat'); g.fillRect(0, 0, W, H); g.restore();
    }
    const Wm = S().watermark;
    if (Wm.enabled && SCENES.handle && !(s && s.kind === 'follow')) {
      g.save(); font(g, Wm.font, Wm.size); g.textAlign = 'center'; g.textBaseline = 'middle';
      g.shadowColor = 'rgba(0,0,0,.45)'; g.shadowBlur = 8; g.fillStyle = `rgba(255,255,255,${Wm.opacity})`;
      g.fillText(SCENES.handle, W / 2, Wm.y * H); g.restore();
    }
  }

  function shot(s) {
    LIST.push(s); LIST.sort((a, b) => a.start - b.start);
    if (LIST.length === 1) { const off = SCENES.offset || 0; shots([[0, (lt) => frame(lt + off)]]); }
  }
  return { setup, shot, drawText, plane };
})();

window.ENGINE = { setup: async () => { await MAPANI.setup(); await STORY.setup(); } };
