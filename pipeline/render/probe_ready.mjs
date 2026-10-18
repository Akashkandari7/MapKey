// Diagnostic: open studio.html N times in parallel (cwd = work/<ep>/render) and report how long each page takes to
// become ready, plus any console errors, page errors or failed file loads.   node probe_ready.mjs 3
import puppeteer from 'puppeteer-core';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const n = +(process.argv[2] || 3);
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--allow-file-access-from-files', '--use-angle=d3d11', '--enable-gpu-rasterization'],
});
await Promise.all(Array.from({ length: n }, async (_, i) => {
  const page = await browser.newPage(), t0 = Date.now();
  page.on('console', m => console.log(`[p${i}] ${m.type()}: ${m.text()}`));
  page.on('pageerror', e => console.log(`[p${i}] pageerror: ${e.message}`));
  page.on('requestfailed', r => console.log(`[p${i}] failed: ${r.url().slice(-70)} ${r.failure()?.errorText}`));
  await page.goto(pathToFileURL(resolve('studio.html')).href + '?render', { waitUntil: 'networkidle0', timeout: 180000 });
  const t1 = Date.now();
  await page.waitForFunction('window.ready === true', { timeout: 180000 });
  console.log(`page ${i}: loaded ${t1 - t0} ms, ready ${Date.now() - t0} ms`);
}));
await browser.close();
