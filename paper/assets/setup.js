// MathJax + paged.js wiring. Lives OUTSIDE the pandoc template on purpose:
// pandoc templates treat '$' as a variable delimiter, so JS containing '$$'
// (the display-math delimiter) cannot be inlined there.
window.PagedConfig = { auto: false };   // do not paginate before math is laid out

window.MathJax = {
  // NO loader.load / packages here: tex-svg-full already bundles ams, and
  // re-requesting '[tex]/ams' makes the loader go to the network for a
  // component it already has -- offline that hangs startup.promise forever,
  // i.e. the render flag never fires and the print times out.
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    tags: 'none'          // equations carry explicit \tag{n}
  },
  svg: { fontCache: 'global', scale: 0.94 },
  startup: {
    ready() {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(function () {
        // paginate only now: page heights must be measured against typeset math
        window.PagedPolyfill.preview().then(function () {
          document.body.setAttribute('data-render', 'done');   // printer's signal
        });
      });
    }
  }
};
