# paper/ — 排版产线(markdown → PDF,不需要 LaTeX)

```bash
npm install                     # 一次性:mathjax / pagedjs / playwright-core(全本地)
bash build.sh                   # 构建 opd-framework-zh.md → out/opd-framework-zh.pdf
bash build.sh 别的备忘.md        # 同一条链排任何一篇
SHOTS=1 bash build.sh           # 额外输出每页 PNG,用于肉眼验收
```

## 文件

| 路径 | 作用 |
|---|---|
| `opd-framework-zh.md` | **唯一真源**:正文 + YAML 头(标题/副标题/日期/摘要) |
| `assets/template.html` | pandoc 模板:标题页 → 摘要 → 目录 → 正文 |
| `assets/print.css` | 打印样式:A4、页边距、`@page` 页码与页眉、宋体正文 + 苹方标题、booktabs 式表格 |
| `assets/setup.js` | MathJax 与 paged.js 的时序编排(见下) |
| `assets/render.mjs` | 起临时 http 服务 → headless Chrome → `page.pdf()`,并做渲染自检 |
| `out/*.pdf` | 产物(入库);`out/*.html`、`out/*.png` 为中间件,已 gitignore |

## 为什么是这条链

本机没有 LaTeX,而 BasicTeX 需要管理员密码。pandoc + MathJax(SVG)+ paged.js +
Chrome 打印能拿到分页排版真正需要的东西:`@page` 页码与页眉、公式不跨页、
表格不断行、CJK 字体正确嵌入(实测嵌入 STSongti + PingFang + Times)。

## 四个已踩过的坑(改动前先读)

1. **时序**:paged.js 必须等 MathJax 排完再分页,否则页高是按未排版的源文本算的——
   页面看起来正常,分页却是错的。`setup.js` 用 `PagedConfig.auto=false` +
   `MathJax.startup.promise` 串起来,并在最后置 `body[data-render=done]`,
   `render.mjs` 只等这个标志,不用超时猜。
2. **不能用 `file://`**:paged.js 会用 XHR 重新取样式表来解析 `@page`,
   `file://` 源会被 CORS 挡掉——代价是静默丢失页码/页眉/页面几何。故 `render.mjs`
   自带临时 http 服务。
3. **不要重复加载 MathJax 组件**:`tex-svg-full` 已含 ams,再写
   `loader.load:['[tex]/ams']` 会让它去网络取已有组件,离线时 `startup.promise`
   永远不 resolve(表现为打印超时)。
4. **`path.relative` 大小写敏感,而 macOS 文件系统不敏感**:cwd 拼作 `simopd`、
   根目录拼作 `SimOPD` 时会算出 `/../../SimOPD/...` 的 URL → 404,而所有症状
   都长得像 MathJax 出错。`render.mjs` 因此从 HTML 自身路径推导站点根。

## 自检

`render.mjs` 每次构建打印一行计数,任一项异常即非零退出:

```
pages=8 equations=83 math-errors=0 page-number-boxes=8 leftover-delims=0
```

`math-errors>0` = 公式语法错(PDF 里会是红块);`leftover-delims>0` = 有 `$$`
没被排版;`page-number-boxes` 应等于页数(减首页则说明 `@page :first` 生效)。
