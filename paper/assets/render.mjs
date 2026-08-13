// HTML -> PDF, waiting on the document's OWN signal instead of a timeout.
// The page sets body[data-render=done] only after MathJax has typeset every
// equation AND paged.js has finished paginating; printing before that yields
// page breaks computed against un-laid-out math (silently wrong, looks fine).
//
// Served over http, not opened as file://: paged.js re-fetches the stylesheet
// by XHR to parse @page rules, and a file:// origin fails that fetch on CORS --
// which silently costs you page numbers, running heads and page geometry.
import { chromium } from 'playwright-core';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve, dirname, basename } from 'node:path';

const [htmlArg, pdf] = process.argv.slice(2);
if (!htmlArg || !pdf) {
    console.error('usage: node render.mjs <input.html> <output.pdf>');
    process.exit(2);
}

// Document root is derived from the HTML's OWN path (build.sh writes it to
// <paper>/out/), never from import.meta.dirname + path.relative(): this
// filesystem is case-insensitive while path.relative is case-SENSITIVE, so a
// cwd spelled `simopd` against a root spelled `SimOPD` yields a url of
// `/../../SimOPD/...` -- served as 404, and every symptom then looks like a
// MathJax problem.
const htmlPath = resolve(htmlArg);
const ROOT = resolve(dirname(htmlPath), '..');         // paper/
const urlPath = `/${basename(dirname(htmlPath))}/${basename(htmlPath)}`;

const TYPES = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
                '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
                '.json': 'application/json', '.woff2': 'font/woff2', '.svg': 'image/svg+xml' };

const server = createServer(async (req, res) => {
    const clean = decodeURIComponent(req.url.split('?')[0]);
    const file = resolve(ROOT, '.' + clean);
    if (!file.startsWith(ROOT)) { res.writeHead(403).end(); return; }   // no path escape
    try {
        const body = await readFile(file);
        res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' });
        res.end(body);
    } catch {
        res.writeHead(404).end('not found');
    }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const CHROME = process.env.CHROME_PATH
    || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage();
const problems = [];
page.on('console', m => { if (m.type() === 'error') problems.push(`console: ${m.text()}`); });
page.on('pageerror', e => problems.push(`pageerror: ${e.message}`));
page.on('requestfailed', r => problems.push(`requestfailed: ${r.url()}`));
// a bare "404" in the console names no url; record which asset was missing
page.on('response', r => {
    // /favicon.ico is requested by the browser, not the document: a 404 there
    // is expected and must not make a clean build look dirty
    if (r.status() >= 400 && !r.url().endsWith('/favicon.ico')) {
        problems.push(`http ${r.status()}: ${r.url()}`);
    }
});

server.on('error', e => problems.push(`server: ${e.message}`));
await page.goto(base + urlPath, { waitUntil: 'load' });
try {
    await page.waitForFunction(() => document.body.dataset.render === 'done', null,
                               { timeout: 120000 });
} catch (e) {
    // a bare "timeout exceeded" hides WHY; dump what the page actually reached
    const state = await page.evaluate(() => ({
        mathjax: typeof window.MathJax, paged: typeof window.PagedPolyfill,
        equations: document.querySelectorAll('mjx-container').length,
        pages: document.querySelectorAll('.pagedjs_page').length,
        flag: document.body.dataset.render,
    })).catch(err => ({ evaluateFailed: String(err).slice(0, 200) }));
    console.error(`FATAL: page never signalled done. url=${base + urlPath}`);
    console.error('  state: ' + JSON.stringify(state));
    problems.slice(0, 12).forEach(p => console.error('  ' + p));
    await browser.close();
    server.close();
    process.exit(1);
}

const stats = await page.evaluate(() => ({
    pages: document.querySelectorAll('.pagedjs_page').length,
    equations: document.querySelectorAll('mjx-container').length,
    // MathJax error nodes: an equation that failed to parse renders as
    // mjx-merror and reads as a red blob in the PDF
    mathErrors: document.querySelectorAll('mjx-merror, .MathJax_Error').length,
    pageNumbers: document.querySelectorAll('.pagedjs_margin-bottom-center .pagedjs_margin-content')
                         .length,
    leftoverDelims: (document.body.innerText.match(/\$\$/g) || []).length,
}));

await page.pdf({ path: pdf, printBackground: true, preferCSSPageSize: true });

// SHOTS=1: png of every paginated sheet. Same DOM the PDF is printed from, so
// it is the cheap way to eyeball CJK glyphs (tofu), math and page furniture.
if (process.env.SHOTS === '1') {
    const sheets = await page.$$('.pagedjs_page');
    for (const [i, sheet] of sheets.entries()) {
        await sheet.screenshot({ path: pdf.replace(/\.pdf$/, `-p${i + 1}.png`) });
    }
    console.log(`shots: ${sheets.length} png next to the pdf`);
}

await browser.close();
server.close();

console.log(`pages=${stats.pages} equations=${stats.equations} ` +
            `math-errors=${stats.mathErrors} page-number-boxes=${stats.pageNumbers} ` +
            `leftover-delims=${stats.leftoverDelims}`);
if (problems.length) {
    console.error('--- page problems ---');
    problems.slice(0, 12).forEach(p => console.error('  ' + p));
}
if (stats.pages === 0 || stats.equations === 0 || stats.mathErrors > 0
    || stats.leftoverDelims > 0) {
    console.error('FATAL: render is not clean (see counters above)');
    process.exit(1);
}
