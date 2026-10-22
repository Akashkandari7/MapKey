// news_shot.mjs: screenshot of a real news article (headline + first paragraphs) for a story-mode news card, and
// where the phrases to highlight are on it.
//
//   node news_shot.mjs --url=https://... --out=work/ep/news/n1.png [--phrases='["over-exploited","100 per cent"]']
//                      [--height=1500] [--chrome=<path>]
// Prints JSON: {"ok": true, "title": "...", "size": [w, h], "highlight_shots": ["..._hl0.png", null, ...]}
// (one screenshot per phrase with it painted magenta; null = phrase not found; pipeline/news.py turns them into boxes)
//
// The page is shown at a phone-ish article width with 2x pixels (crisp when the card is pushed in). Fixed and sticky
// elements (cookie bars, app banners, sticky headers) and ads are hidden first; the clip starts just above the
// headline (h1) so the card reads like the reference's ThePrint / Tribune clippings.
import puppeteer from 'puppeteer-core';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).map(a => { const i = a.indexOf('='); return i < 0 ? [a.replace(/^--/, ''), true] : [a.slice(2, i), a.slice(i + 1)]; }));
const CHROMES = [args.chrome, process.env.CHROME_PATH, 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome', '/usr/bin/chromium'];
const CHROME = CHROMES.find(p => p && existsSync(p));
const phrases = args.phrases ? JSON.parse(args.phrases) : [];
// a phone-width column at 3x: body text lands ~35 px high on the 970 px card (readable), the headline big
const height = +(args.height || 640), width = +(args.width || 440), DPR = 3;

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox', '--disable-notifications', '--mute-audio'] });
try {
  const page = await browser.newPage();
  // a phone: news sites serve their clean one-column mobile article (a narrow desktop window still got side rails)
  await page.setUserAgent('Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Mobile Safari/537.36');
  await page.setViewport({ width, height: 3200, deviceScaleFactor: DPR, isMobile: true, hasTouch: true });  // tall: the screenshot never resizes the page (that reflowed the text away from the measured spots)
  await page.goto(args.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await new Promise(r => setTimeout(r, 6000));  // consent panels arrive late (AP: ~5 s)
  // some sites redirect once more after load (consent, AMP): retry the page script on a fresh context
  const settle = async fn => { for (let k = 0; k < 3; k++) { try { return await fn(); } catch (e) { if (!/context was destroyed|detached/i.test(e.message)) throw e; await new Promise(r => setTimeout(r, 2500)); } } return fn(); };
  // the paper's logo (masthead), taken before the clean-up hides sticky headers; Python puts it above the headline
  let logo = null;
  try {
    const lb = await settle(() => page.evaluate(() => {
      const c = [...document.querySelectorAll('header img, header svg, [class*="logo" i] img, [class*="logo" i] svg, img[alt*="logo" i], [id*="logo" i] img, [id*="logo" i] svg')]
        .map(e => [e, e.getBoundingClientRect()]).filter(([e, r]) => r.width > 70 && r.height > 14 && r.height < 140 && r.top + window.scrollY < 400 && r.width / r.height > 1.6);
      if (!c.length) return null;
      c.sort((a, b) => b[1].width * b[1].height - a[1].width * a[1].height);
      const r = c[0][1];
      return { x: r.left, y: r.top + window.scrollY, width: r.width, height: r.height };
    }));
    if (lb) { logo = args.out.replace(/\.png$/, '_logo.png'); await page.screenshot({ path: logo, clip: lb, omitBackground: true }); }
  } catch (e) { logo = null; }
  await settle(() => page.evaluate(async (withImages) => {
    // the clean-up runs now and again after lazy content (ads injected on scroll) has arrived
    window.__mapaniClean = () => {
      // hide overlays: anything fixed/sticky, consent dialogs, ads, video players - but never the headline's own box
      const h1 = [...document.querySelectorAll('h1')].find(h => h.innerText.trim().length > 15);
      const keep = el => h1 && (el.contains(h1) || h1.contains(el));
      for (const el of document.querySelectorAll('body *')) {
        const cs = getComputedStyle(el);
        if ((cs.position === 'fixed' || cs.position === 'sticky') && !keep(el)) el.style.setProperty('display', 'none', 'important');
      }
      // consent / privacy panels (OneTrust, Didomi, Sourcepoint, Quantcast...) - also the inline ones (AP 'We Care About
      // Your Privacy' sat in the middle of the article)
      for (const el of document.querySelectorAll('[role="dialog"], [aria-modal="true"], [id*="onetrust" i], [class*="onetrust" i], [id*="didomi" i], [id*="sp_message" i], [class*="qc-cmp" i], [class*="fc-consent" i], [id*="privacy" i], [class*="privacy" i], [class*="gdpr" i], [id*="gdpr" i]'))
        if (!keep(el)) el.style.setProperty('display', 'none', 'important');
      for (const el of document.querySelectorAll('div, section, aside')) {
        const tx = (el.innerText || '').trim();
        if (!keep(el) && tx.length < 1200 && /we care about your privacy|cookie (settings|preferences|policy)|accept all cookies|manage (cookies|preferences)|your privacy choices/i.test(tx)
            && !el.querySelector('h1')) el.style.setProperty('display', 'none', 'important');
      }
      for (const el of document.querySelectorAll('iframe, ins, [id*="ad-"], [class*="advert"], [class*="ad-slot"], [class*="cookie"], [id*="cookie"], [class*="consent"], video, [class*="newsletter"], [class*="share"], [class*="social"], [class*="follow"], [class*="font-size"], [class*="comment"]'))
        if (!keep(el)) el.style.setProperty('display', 'none', 'important');
      document.documentElement.style.overflow = 'visible'; document.body.style.overflow = 'visible';
      // the lead photo pushes the text (and the highlighted line) off the card: hidden unless --images
      if (!withImages) for (const el of document.querySelectorAll('figure, picture, img, [class*="featured-image"], [class*="lead-image"]'))
        if (!keep(el) && el.getBoundingClientRect().height > 160) el.style.setProperty('display', 'none', 'important');
      // ad slots: their 'Advertisement' label goes, and boxes that hold nothing visible collapse (they keep their
      // reserved height otherwise: a white gap in the middle of the card)
      for (const el of document.querySelectorAll('body *')) {
        const t = (el.innerText || '').trim();
        if (keep(el) || t.length > 70 || !/advertisement|sponsored|story continues below|ad[- ]free|too many ads|^ad$/i.test(t)) continue;
        // the ad label's whole box goes (its reserved grey slot too): the largest ancestor that is still only that
        let box = el;
        while (box.parentElement && !keep(box.parentElement) && (box.parentElement.innerText || '').trim().length <= 90
               && box.parentElement.getBoundingClientRect().height < 900) box = box.parentElement;
        box.style.setProperty('display', 'none', 'important');
      }
      for (const el of [...document.querySelectorAll('div, section, aside, figure')].reverse()) {
        if (keep(el)) continue;
        if (el.getBoundingClientRect().height > 30 && !(el.innerText || '').trim() && ![...el.querySelectorAll('img, svg, picture, video, canvas')].some(m => m.getBoundingClientRect().height > 80))  // small icons (gallery arrows) don't count
          el.style.setProperty('display', 'none', 'important');
      }
    };
    window.__mapaniClean();
    // lazy images above the fold must be in before measuring: their size reflows the text around them
    for (const im of document.images) { im.loading = 'eager'; im.decoding = 'sync'; }
    window.scrollTo(0, 1200); await new Promise(r => setTimeout(r, 400)); window.scrollTo(0, 0);
    await document.fonts.ready;
    await Promise.race([Promise.all([...document.images].filter(i => !i.complete).map(i => new Promise(r => { i.onload = i.onerror = r; }))),
      new Promise(r => setTimeout(r, 6000))]);
  }, !!args.images));
  await new Promise(r => setTimeout(r, 800));
  await settle(() => page.evaluate(() => window.__mapaniClean && window.__mapaniClean()));
  await new Promise(r => setTimeout(r, 300));
  const info = await settle(() => page.evaluate((phrases, height) => {
    const h1 = [...document.querySelectorAll('h1')].find(h => h.innerText.trim().length > 15 && h.getBoundingClientRect().height > 0);
    if (!h1) return { ok: false, why: 'no headline (h1) on the page' };
    // the article column: the nearest ancestor of the headline that also holds paragraphs
    let col = h1.parentElement;
    while (col && col !== document.body && col.querySelectorAll('p').length < 3) col = col.parentElement;
    const r0 = h1.getBoundingClientRect(), rc = (col || document.body).getBoundingClientRect();
    // at phone width the article is the whole page: clip edge to edge (a narrow "column" made a giant, cropped card)
    const x = 0, y = window.scrollY + r0.top - 40, w = document.documentElement.clientWidth;
    // phrases: found in the text (whitespace/case-insensitive) and kept as Ranges; they are painted magenta with the
    // CSS Custom Highlight API (no DOM change, no reflow) for one extra screenshot each. Python diffs those against
    // the plain one to get the marker boxes, so they always match the picture, whatever the page's layout does.
    // letters and digits only: 'over-exploited' finds ‘over exploited’, '100 per cent,' finds '100 per cent'
    const sep = ch => !/[\p{L}\p{N}]/u.test(ch);
    const norm = s => s.replace(/[^\p{L}\p{N}]+/gu, ' ').toLowerCase().trim();
    const walker = document.createTreeWalker(col || document.body, NodeFilter.SHOW_TEXT);
    const map = []; let flat = '', prevSpace = false;
    for (let n; (n = walker.nextNode());) for (let i = 0; i < n.nodeValue.length; i++) {
      const ch = n.nodeValue[i], sp = sep(ch);
      if (sp && prevSpace) continue;
      map.push([n, i]); flat += sp ? ' ' : ch.toLowerCase().slice(0, 1); prevSpace = sp;
    }
    // the first occurrence that is really on the card: visible, and inside the clipped area
    // the card is `height` tall, stretched (up to 1.45x) when that brings a spoken phrase onto it
    const clipTop = window.scrollY + r0.top - 40, maxH = height * 1.45;
    window.__ranges = phrases.map(ph => {
      const q = norm(ph);
      for (let k = flat.indexOf(q); k >= 0; k = flat.indexOf(q, k + 1)) {
        const a = map[k], b = map[k + q.length - 1];
        const rg = document.createRange(); rg.setStart(a[0], a[1]); rg.setEnd(b[0], b[1] + 1);
        const el = a[0].parentElement, rs = [...rg.getClientRects()].filter(r => r.width > 1 && r.height > 1);
        const inClip = rs.length && rs.every(r => window.scrollY + r.top >= clipTop && window.scrollY + r.bottom <= clipTop + maxH);
        if (inClip && el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) return rg;
      }
      return null;
    });
    const st = document.createElement('style'); st.textContent = '::highlight(mapani){background-color:#FF00FF}';
    document.head.appendChild(st);
    const found = window.__ranges.map(r => !!r);
    const lowest = Math.max(0, ...window.__ranges.filter(Boolean).map(rg => Math.max(...[...rg.getClientRects()].map(r => window.scrollY + r.bottom))));
    const H = Math.round(Math.min(maxH, Math.max(height, lowest - clipTop + 90)));
    const text = ((col || document.body).innerText || "").replace(/\s+/g, " ").slice(0, 1800);
    return { ok: true, title: h1.innerText.trim(), clip: { x, y, width: w, height: H }, found, text };
  }, phrases, height));
  if (!info.ok) { console.log(JSON.stringify(info)); process.exit(0); }
  mkdirSync(dirname(args.out), { recursive: true });
  // last sweep right before the picture: a consent panel or ad that arrived meanwhile goes too (no reflow risk: it
  // is display:none'd only if it is an overlay-type element)
  await settle(() => page.evaluate(() => { for (const el of document.querySelectorAll('iframe, [role="dialog"], [aria-modal="true"]')) el.style.setProperty('display', 'none', 'important'); }));
  await page.screenshot({ path: args.out, clip: info.clip, captureBeyondViewport: false });
  const hl = [];
  for (let i = 0; i < phrases.length; i++) {
    if (!info.found[i]) { hl.push(null); continue; }
    await page.evaluate(i => new Promise(r => { CSS.highlights.set('mapani', new Highlight(window.__ranges[i])); requestAnimationFrame(() => requestAnimationFrame(r)); }), i);
    const f = args.out.replace(/\.png$/, `_hl${i}.png`);
    await page.screenshot({ path: f, clip: info.clip, captureBeyondViewport: false });
    hl.push(f);
  }
  await page.evaluate(() => CSS.highlights.clear());
  console.log(JSON.stringify({ ok: true, title: info.title, text: info.text, size: [Math.round(info.clip.width * DPR), Math.round(info.clip.height * DPR)], highlight_shots: hl, logo }));
} catch (e) {
  console.log(JSON.stringify({ ok: false, why: String(e.message || e) }));
} finally { await browser.close(); }
