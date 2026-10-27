/* Throttle Console: a local, read-only view of `throttle check` history.
   No network requests leave 127.0.0.1; every figure comes from the API,
   which reuses throttle.check's own comparison code. */
(function () {
  "use strict";

  var $ = function (s, el) { return (el || document).querySelector(s); };
  var page = $("#page");
  var navEl = $("#nav");
  var cache = {};

  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function api(path) {
    return fetch(path, { headers: { Accept: "application/json" } }).then(function (r) {
      if (!r.ok) return r.json().catch(function () { return {}; }).then(function (b) {
        throw new Error(b.detail || ("HTTP " + r.status));
      });
      return r.json();
    });
  }
  function money(v) {
    if (v == null || !isFinite(v)) return "n/a";
    var a = Math.abs(v);
    return (v < 0 ? "-$" : "$") + (a >= 1000 ? a.toLocaleString(undefined, { maximumFractionDigits: 0 })
      : a >= 1 ? a.toFixed(2) : a.toFixed(4));
  }
  function pct(v) { return v == null || !isFinite(v) ? "" : (v > 0 ? "+" : "") + v.toFixed(1) + "%"; }
  function when(iso) {
    if (!iso) return "";
    var d = new Date(iso);
    if (isNaN(d)) return esc(iso);
    return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }
  function badge(verdict) {
    var map = { "CHEAPER": "cheaper", "MORE EXPENSIVE": "costlier", "NO WINNER": "nowin", "NOT CALIBRATED": "notcal" };
    if (!verdict) return '<span class="badge first">FIRST CHECK</span>';
    return '<span class="badge ' + (map[verdict] || "first") + '">' + esc(verdict) + "</span>";
  }
  function toast(msg) {
    var t = document.createElement("div");
    t.className = "toast"; t.textContent = msg; document.body.appendChild(t);
    setTimeout(function () { t.remove(); }, 1400);
  }
  function copyText(text, el) {
    var done = function () { toast("Copied"); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { selectText(el); });
    } else { selectText(el); }
  }
  function selectText(el) {
    if (!el) return;
    var r = document.createRange(); r.selectNodeContents(el);
    var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
  }
  function cmd(text) {
    return '<div class="cmd"><code>' + esc(text) + '</code><button class="copy" type="button" data-copy="' + esc(text) + '">copy</button></div>';
  }

  /* ---------- icons ---------- */
  var I = {
    quickstart: '<path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H11v16H5.5A1.5 1.5 0 0 1 4 18.5z"/><path d="M20 5.5A1.5 1.5 0 0 0 18.5 4H13v16h5.5a1.5 1.5 0 0 0 1.5-1.5z"/>',
    checks: '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4 6l1 1 2-2M4 12l1 1 2-2M4 18l1 1 2-2"/>',
    compare: '<path d="M7 4v16M17 4v16"/><path d="M4 9h6M14 15h6"/>',
    endpoints: '<rect x="3" y="4" width="18" height="6" rx="2"/><rect x="3" y="14" width="18" height="6" rx="2"/><path d="M7 7h.01M7 17h.01"/>',
    ci: '<circle cx="6" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="12" r="2.5"/><path d="M6 8.5v7M8.3 7.3l7.4 3.6"/>',
    share: '<path d="M12 15V3M7 8l5-5 5 5"/><path d="M5 13v6a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6"/>',
    settings: '<circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1.2l2-1.5-2-3.4-2.3 1a7 7 0 0 0-2-1.2L14.3 3h-4l-.3 2.7a7 7 0 0 0-2 1.2l-2.3-1-2 3.4 2 1.5a7 7 0 0 0 0 2.4l-2 1.5 2 3.4 2.3-1a7 7 0 0 0 2 1.2l.3 2.7h4l.3-2.7a7 7 0 0 0 2-1.2l2.3 1 2-3.4-2-1.5c.07-.4.1-.8.1-1.2z"/>',
    rocket: '<path d="M5 15c-1.5 1.5-2 5-2 5s3.5-.5 5-2"/><path d="M9 15l-3-3c1-4 5-8 13-9-1 8-5 12-9 13z"/><circle cx="14" cy="10" r="1.6"/>',
    book: '<path d="M4 4h10a4 4 0 0 1 4 4v12H8a4 4 0 0 1-4-4z"/><path d="M8 20a4 4 0 0 1 0-8h10"/>',
    wrench: '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.2L3 17.8V21h3.2l6.3-6.3a4 4 0 0 0 5.2-5.4l-2.6 2.6-2.4-.6-.6-2.4z"/>',
    terminal: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M12 15h5"/>',
    receipt: '<path d="M6 3h12v18l-3-2-3 2-3-2-3 2z"/><path d="M9 8h6M9 12h6"/>',
    clock: '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/>'
  };
  function icon(name) {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + (I[name] || "") + "</svg>";
  }

  var CONSOLE = [
    ["quickstart", "Quickstart", "quickstart"], ["checks", "Checks", "checks"], ["compare", "Compare", "compare"],
    ["endpoints", "Endpoints", "endpoints"], ["ci", "CI Gate", "ci"], ["share", "Share", "share"], ["settings", "Settings", "settings"]
  ];
  var LEARN = [
    ["docs/getting-started/quickstart", "Getting Started", "rocket"], ["docs/concepts/noise-and-calibration", "Concepts", "book"],
    ["docs/guides/vllm", "Guides", "wrench"], ["docs/cli/overview", "CLI Reference", "terminal"],
    ["docs/results/overview", "Results", "receipt"], ["docs/changelog", "Changelog", "clock"]
  ];

  /* ---------- routing ---------- */
  function route() {
    var h = location.hash.replace(/^#\/?/, "") || "quickstart";
    var q = "";
    var i = h.indexOf("?");
    if (i >= 0) { q = h.slice(i + 1); h = h.slice(0, i); }
    return { path: h, query: new URLSearchParams(q) };
  }
  function tabOf(path) {
    if (path.indexOf("docs/cli/") === 0) return "cli";
    if (path.indexOf("docs/") === 0) return "docs";
    return "console";
  }

  function renderNav(r) {
    var tab = tabOf(r.path), html = "";
    document.querySelectorAll(".tab").forEach(function (t) { t.classList.toggle("active", t.dataset.tab === tab); });
    if (tab === "console") {
      html += '<div class="nav-group">' + CONSOLE.map(function (c) {
        var active = r.path === c[0] || r.path.indexOf(c[0] + "/") === 0;
        return '<a class="nav-item' + (active ? " active" : "") + '" href="#/' + c[0] + '">' + icon(c[2]) + esc(c[1]) + "</a>";
      }).join("") + "</div>";
      html += '<div class="nav-sep"></div><div class="nav-title">Learn</div><div class="nav-group">' + LEARN.map(function (c) {
        return '<a class="nav-item" href="#/' + c[0] + '">' + icon(c[2]) + esc(c[1]) + "</a>";
      }).join("") + "</div>";
      navEl.innerHTML = html;
      return;
    }
    getNav().then(function (groups) {
      var slug = r.path.slice(5);
      navEl.innerHTML = groups.filter(function (g) { return g.tab === tab; }).map(function (g) {
        return '<div class="nav-title">' + esc(g.group) + '</div><div class="nav-group">' + g.pages.map(function (p) {
          return '<a class="nav-item sub' + (p.slug === slug ? " active" : "") + '" href="#/docs/' + esc(p.slug) + '">' + esc(p.title) + "</a>";
        }).join("") + "</div>";
      }).join("") + '<div class="nav-sep"></div><a class="nav-item" href="#/quickstart">' + icon("quickstart") + "Back to Console</a>";
    });
  }
  function getNav() { return cache.nav ? Promise.resolve(cache.nav) : api("/api/docs-nav").then(function (n) { cache.nav = n; return n; }); }

  function show(html) { page.innerHTML = html; page.focus({ preventScroll: true }); window.scrollTo(0, 0); }
  function fail(e) { show('<div class="empty">Could not load this page: ' + esc(e.message) + "</div>"); }

  /* ---------- charts ---------- */
  function scale(lo, hi, w, pad) {
    var span = hi - lo || Math.abs(hi) || 1;
    lo -= span * 0.08; hi += span * 0.08;
    return function (v) { return pad + (v - lo) / (hi - lo) * (w - 2 * pad); };
  }
  function ciBar(s, lo, hi, tone) {
    if (!s || s.mean == null) return "";
    var x = scale(lo, hi, 160, 6), c = tone || "#c6ff34";
    var a = s.ci_low != null ? x(s.ci_low) : x(s.mean), b = s.ci_high != null ? x(s.ci_high) : x(s.mean);
    return '<svg class="ci" viewBox="0 0 160 22" role="img" aria-label="95% interval ' + esc(money(s.ci_low)) + " to " + esc(money(s.ci_high)) + '">' +
      '<line x1="6" y1="11" x2="154" y2="11" stroke="#272c35" stroke-width="1"/>' +
      '<rect x="' + a.toFixed(1) + '" y="7" width="' + Math.max(2, b - a).toFixed(1) + '" height="8" rx="4" fill="' + c + '" fill-opacity=".28" stroke="' + c + '" stroke-opacity=".7"/>' +
      '<circle cx="' + x(s.mean).toFixed(1) + '" cy="11" r="3.4" fill="' + c + '"/></svg>';
  }
  function tone(verdict) {
    return verdict === "MORE EXPENSIVE" ? "#ff5d4d" : verdict === "NOT CALIBRATED" ? "#f2c14e" : verdict === "NO WINNER" ? "#a8b0bb" : "#c6ff34";
  }
  function spark(series) {
    var pts = series.filter(function (p) { return p.mean != null; });
    if (!pts.length) return "";
    var w = 220, h = 44, lo = Math.min.apply(null, pts.map(function (p) { return p.ci_low != null ? p.ci_low : p.mean; }));
    var hi = Math.max.apply(null, pts.map(function (p) { return p.ci_high != null ? p.ci_high : p.mean; }));
    var span = hi - lo || hi || 1; lo -= span * 0.15; hi += span * 0.15;
    var x = function (i) { return pts.length === 1 ? w / 2 : 8 + i * (w - 16) / (pts.length - 1); };
    var y = function (v) { return h - 6 - (v - lo) / (hi - lo) * (h - 12); };
    var line = pts.map(function (p, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p.mean).toFixed(1); }).join(" ");
    var area = line + " L" + x(pts.length - 1).toFixed(1) + " " + h + " L" + x(0).toFixed(1) + " " + h + " Z";
    var last = pts[pts.length - 1];
    return '<svg class="spark" viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="$/M trend">' +
      '<path d="' + area + '" fill="#c6ff34" fill-opacity=".08"/><path d="' + line + '" fill="none" stroke="#c6ff34" stroke-width="1.8" stroke-linejoin="round"/>' +
      '<circle cx="' + x(pts.length - 1).toFixed(1) + '" cy="' + y(last.mean).toFixed(1) + '" r="3.5" fill="#c6ff34"/></svg>';
  }
  function compareChart(a, b, comp) {
    var sa = a.summary, sb = b.summary, noise = comp && comp.noise;
    var vals = [sa.ci_low, sa.ci_high, sb.ci_low, sb.ci_high, sa.mean, sb.mean].filter(function (v) { return v != null; });
    var floor = noise && noise.floor_percent != null ? noise.floor_percent : null;
    if (floor != null && sa.mean != null) { vals.push(sa.mean * (1 - floor / 100), sa.mean * (1 + floor / 100)); }
    var W = 760, x = scale(Math.min.apply(null, vals), Math.max.apply(null, vals), W, 150);
    var rows = [[a, sa, "A · " + (a.label || a.id), "#8d939c"], [b, sb, "B · " + (b.label || b.id), tone(comp && comp.verdict)]];
    var svg = '<svg class="chart" viewBox="0 0 ' + W + ' 190" role="img" aria-label="Compare two checks">';
    if (floor != null && sa.mean != null) {
      var l = x(sa.mean * (1 - floor / 100)), r = x(sa.mean * (1 + floor / 100));
      svg += '<rect x="' + l.toFixed(1) + '" y="18" width="' + (r - l).toFixed(1) + '" height="128" fill="#a8b0bb" fill-opacity=".08" stroke="#a8b0bb" stroke-opacity=".35" stroke-dasharray="4 4"/>' +
        '<text x="' + ((l + r) / 2).toFixed(1) + '" y="12" fill="#8d939c" font-size="11" font-family="ui-monospace,monospace" text-anchor="middle">noise bound ±' + floor.toFixed(1) + "%</text>";
    }
    rows.forEach(function (row, i) {
      var s = row[1], yy = 58 + i * 58, c = row[3];
      svg += '<text x="0" y="' + (yy + 4) + '" fill="#eef0ea" font-size="13" font-family="ui-sans-serif,system-ui">' + esc(row[2].slice(0, 20)) + "</text>";
      if (s.mean == null) return;
      var a1 = x(s.ci_low != null ? s.ci_low : s.mean), b1 = x(s.ci_high != null ? s.ci_high : s.mean);
      svg += '<line x1="150" y1="' + yy + '" x2="' + (W - 10) + '" y2="' + yy + '" stroke="#1c2027"/>' +
        '<rect x="' + a1.toFixed(1) + '" y="' + (yy - 9) + '" width="' + Math.max(3, b1 - a1).toFixed(1) + '" height="18" rx="9" fill="' + c + '" fill-opacity=".25" stroke="' + c + '"/>' +
        '<circle cx="' + x(s.mean).toFixed(1) + '" cy="' + yy + '" r="5" fill="' + c + '"/>' +
        '<text x="' + x(s.mean).toFixed(1) + '" y="' + (yy + 28) + '" fill="#c3c8cd" font-size="12" font-family="ui-monospace,monospace" text-anchor="middle">' + esc(money(s.mean)) + "/M</text>";
    });
    return svg + '<text x="150" y="182" fill="#6f7680" font-size="11" font-family="ui-monospace,monospace">bars = 95% CI · dot = mean $ per million ' + esc(a.metric) + " tokens</text></svg>";
  }

  /* ---------- pages ---------- */
  var QS = {
    connect: ["engines.svg", "Connect your server", "Any OpenAI-compatible server works: vLLM, SGLang, Ollama, TGI, LMDeploy. Throttle only needs its URL.", "ollama serve        # or: vllm serve <model>"],
    first: ["quickstart.svg", "Run your first check", "Measure what the server costs per million tokens, with a 95% confidence interval.", "throttle check --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50"],
    calibrate: ["noise.svg", "Calibrate noise", "Run the same check 3 times without changing anything. Then every change gets a verdict it can defend.", "throttle check --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50   # ×3, nothing changed"],
    ci: ["ci.svg", "Gate your CI", "Fail a deploy when a calibrated check says it got costlier. One step in GitHub Actions.", "uses: KushagraKanaujia/throttle@v0.4.2"]
  };

  function pQuickstart() {
    Promise.all([api("/api/quickstart-status"), api("/api/checks"), api("/api/summary")]).then(function (res) {
      var status = res[0], checks = res[1], sum = res[2];
      var firstOpen = status.filter(function (s) { return !s.done; })[0];
      var cards = status.map(function (s) {
        var c = QS[s.key], state = s.done ? "done" : (firstOpen && firstOpen.key === s.key ? "next" : "");
        var label = s.done ? "✓ DONE" : state === "next" ? "NEXT" : "TO DO";
        var href = s.key === "ci" ? "#/ci" : s.key === "calibrate" ? "#/docs/concepts/noise-and-calibration" : s.key === "connect" ? "#/docs/getting-started/installation" : "#/docs/getting-started/first-check";
        return '<div class="card"><a class="card-art" href="' + href + '"><img src="/static/illustrations/' + c[0] + '" alt=""><span class="card-state ' + state + '">' + label + "</span></a>" +
          '<h3><a class="plain" href="' + href + '">' + esc(c[1]) + "</a></h3><p>" + esc(c[2]) + "</p>" + cmd(c[3]) + "</div>";
      }).join("");
      var latest = checks[0];
      var latestHtml = latest ? '<h2>Latest check</h2><div class="panel"><div class="table-wrap"><table><tbody>' + row(latest, bounds(checks)) + "</tbody></table></div></div>" : "";
      show('<p class="eyebrow">Getting Started</p><h1>Quickstart Guide</h1><p class="lede">Know what your LLM server costs per million tokens, and whether your last change really made it cheaper. ' +
        esc(sum.checks) + " check" + (sum.checks === 1 ? "" : "s") + " recorded so far.</p>" + '<div class="cards">' + cards + "</div>" + latestHtml);
    }).catch(fail);
  }

  function bounds(checks) {
    var v = [];
    checks.forEach(function (c) { ["ci_low", "ci_high", "mean"].forEach(function (k) { if (c.summary[k] != null) v.push(c.summary[k]); }); });
    return v.length ? [Math.min.apply(null, v), Math.max.apply(null, v)] : [0, 1];
  }
  function row(c, b) {
    var ch = c.changes.slice(0, 2).map(function (x) { return '<span class="chip">' + esc(x.key) + "</span>"; }).join("");
    if (c.changes.length > 2) ch += '<span class="chip">+' + (c.changes.length - 2) + "</span>";
    return '<tr data-href="#/checks/' + esc(c.id) + '"><td class="num small muted">' + when(c.created_at) + "</td><td><b>" + esc(c.label || "—") + '</b><div class="small muted">' + esc(c.model || "") + " · " + esc(c.endpoint_masked) + "</div></td>" +
      '<td class="num">' + esc(money(c.summary.mean)) + '<div class="small muted">per M ' + esc(c.metric) + "</div></td><td>" + ciBar(c.summary, b[0], b[1], tone(c.verdict)) + '</td><td class="num">' + esc(pct(c.delta_percent)) + "</td><td>" + badge(c.verdict) + "</td><td>" + (ch || '<span class="muted small">—</span>') + "</td></tr>";
  }

  function pChecks() {
    Promise.all([api("/api/checks"), api("/api/summary")]).then(function (res) {
      var checks = res[0], sum = res[1];
      if (!checks.length) return show('<p class="eyebrow">Console</p><h1>Checks</h1>' + emptyHistory(sum));
      var b = bounds(checks), v = function (name) { return checks.filter(function (c) { return c.verdict === name; }).length; };
      show('<p class="eyebrow">Console</p><h1>Checks</h1><p class="lede">Every <code>throttle check</code> you have run, newest first, each judged against its baseline exactly as the CLI judged it.</p>' +
        '<div class="stats"><div class="stat"><span>Checks</span><b>' + checks.length + '</b></div><div class="stat"><span>CHEAPER</span><b class="t-accent">' + v("CHEAPER") +
        '</b></div><div class="stat"><span>NO WINNER</span><b>' + v("NO WINNER") + '</b></div><div class="stat"><span>NOT CALIBRATED</span><b class="t-nc">' + v("NOT CALIBRATED") + "</b></div></div>" +
        '<div class="panel"><div class="table-wrap"><table><thead><tr><th>When</th><th>Check</th><th>$ / M</th><th>95% CI</th><th>Change</th><th>Verdict</th><th>What changed</th></tr></thead><tbody>' +
        checks.map(function (c) { return row(c, b); }).join("") + "</tbody></table></div></div>");
    }).catch(fail);
  }
  function emptyHistory(sum) {
    return '<div class="empty">No checks yet in <code>' + esc(sum.history_file) + "</code>.<br><br>" + cmd(QS.first[3]) + "</div>";
  }

  function pCheck(id) {
    api("/api/checks/" + encodeURIComponent(id)).then(function (c) {
      var comp = c.comparison, noise = comp && comp.noise;
      var blocks = (c.blocks || []).map(function (bl, i) {
        var d = bl.dollars_per_million || {};
        return "<tr><td>" + (i + 1) + '</td><td class="num">' + esc(money(d[c.metric])) + '</td><td class="num">' + esc(money(d.input)) + '</td><td class="num">' + esc(money(d.total)) + "</td></tr>";
      }).join("");
      var changes = comp ? (c.changes.length ? c.changes.map(function (x) {
        return "<tr><td><code>" + esc(x.key) + '</code></td><td class="num muted">' + esc(x.old) + '</td><td class="num">' + esc(x.new) + "</td></tr>";
      }).join("") : '<tr><td colspan="3" class="muted">Nothing changed in the config fingerprint.</td></tr>') : "";
      var fp = Object.keys(c.fingerprint || {}).sort().map(function (k) {
        return "<dt>" + esc(k) + "</dt><dd class=\"num small\">" + esc(c.fingerprint[k]) + "</dd>";
      }).join("");
      show('<p class="eyebrow"><a href="#/checks">Checks</a> / ' + esc(c.id) + "</p><h1>" + esc(c.label || c.id) + "</h1>" +
        '<p class="lede">' + esc(c.model) + " · " + esc(c.endpoint_masked) + " · " + when(c.created_at) + " · Throttle " + esc(c.throttle_version) + "</p>" +
        '<div class="detail-grid"><div class="panel pad"><div class="row between"><div class="big">' + esc(money(c.summary.mean)) + "<small>per million " + esc(c.metric) + " tokens</small></div>" + badge(c.verdict) + "</div>" +
        '<p class="muted small mt10">95% CI ' + esc(money(c.summary.ci_low)) + " to " + esc(money(c.summary.ci_high)) + " across " + esc(c.summary.n_blocks) + " blocks · GPU " + esc(money(c.gpu_hourly_rate_usd)) + "/hr, " + esc(c.rate_source || "") + "</p>" +
        (comp ? '<p class="reason">' + esc(comp.reason) + "</p>" : '<p class="reason">First check for this endpoint: the next one is compared against it.</p>') +
        (comp && c.baseline ? '<p class="small muted mt14">Baseline: <a href="#/checks/' + esc(c.baseline.id) + '">' + esc(c.baseline.label || c.baseline.id) + "</a> · " + esc(money(c.baseline.summary.mean)) + "/M · change " + esc(pct(comp.delta_percent)) +
          (noise && noise.floor_percent != null ? " · noise bound ±" + noise.floor_percent.toFixed(1) + "% (" + noise.degrees_of_freedom + " df)" : "") + ' · <a href="#/compare?a=' + esc(c.baseline.id) + "&b=" + esc(c.id) + '">open in Compare</a></p>' : "") +
        "</div>" + '<div class="panel"><div class="code-head">Share this result (no traffic)<button class="copy" type="button" data-copy="' + esc(c.share_command) + '">copy</button></div><pre class="code wrap">' + esc(c.share_command) +
        '</pre><div class="pad small muted pt0">Prints a markdown summary with URLs, hosts and secrets removed, plus a pre-filled GitHub issue link.</div></div></div>' +
        (comp ? '<h2>What changed</h2><div class="panel"><div class="table-wrap"><table><thead><tr><th>Setting</th><th>Before</th><th>After</th></tr></thead><tbody>' + changes + "</tbody></table></div></div>" : "") +
        '<h2>Blocks</h2><div class="panel"><div class="table-wrap"><table><thead><tr><th>Block</th><th>$ / M ' + esc(c.metric) + "</th><th>$ / M input</th><th>$ / M total</th></tr></thead><tbody>" + blocks + "</tbody></table></div></div>" +
        '<h2>Config fingerprint</h2><div class="panel"><dl class="kv">' + fp + "</dl></div>");
      page.querySelectorAll("tbody tr").forEach(function (tr) { tr.style.cursor = "default"; });
    }).catch(fail);
  }

  function pCompare(q) {
    api("/api/checks").then(function (checks) {
      if (checks.length < 2) return show('<p class="eyebrow">Console</p><h1>Compare</h1><div class="empty">You need at least two checks to compare.</div>');
      var judged = checks.filter(function (c) { return c.baseline_id && ["CHEAPER", "MORE EXPENSIVE", "NO WINNER"].indexOf(c.verdict) >= 0; })[0];
      var a = q.get("a") || (judged ? judged.baseline_id : checks[1].id), b = q.get("b") || (judged ? judged.id : checks[0].id);
      var opts = function (sel) {
        return checks.map(function (c) {
          return '<option value="' + esc(c.id) + '"' + (c.id === sel ? " selected" : "") + ">" + esc((c.label || "—") + " · " + money(c.summary.mean) + "/M · " + when(c.created_at)) + "</option>";
        }).join("");
      };
      show('<p class="eyebrow">Console</p><h1>Compare</h1><p class="lede">Pick a baseline (A) and a later check (B). The verdict uses the same rules as <code>throttle check</code>: the change must beat the run-to-run noise bound and the 95% intervals must not overlap.</p>' +
        '<div class="row mb22"><label class="small muted" for="ca">A</label><select id="ca">' + opts(a) + '</select><label class="small muted" for="cb">B</label><select id="cb">' + opts(b) + "</select></div><div id=\"cres\"></div>");
      var go = function () {
        var va = $("#ca").value, vb = $("#cb").value;
        history.replaceState(null, "", "#/compare?a=" + encodeURIComponent(va) + "&b=" + encodeURIComponent(vb));
        if (va === vb) { $("#cres").innerHTML = '<div class="empty">Pick two different checks.</div>'; return; }
        api("/api/compare?a=" + encodeURIComponent(va) + "&b=" + encodeURIComponent(vb)).then(function (r) {
          var c = r.comparison;
          $("#cres").innerHTML = '<div class="panel pad"><div class="row between mb8"><div class="big big-34">' + esc(pct(c.delta_percent) || "n/a") +
            '<small>' + esc(money(r.a.summary.mean)) + " → " + esc(money(r.b.summary.mean)) + " per M " + esc(r.b.metric) + "</small></div>" + badge(c.verdict) + "</div>" + compareChart(r.a, r.b, c) + '<p class="reason">' + esc(c.reason) + "</p></div>";
        }).catch(function (e) { $("#cres").innerHTML = '<div class="empty">' + esc(e.message) + "</div>"; });
      };
      $("#ca").addEventListener("change", go); $("#cb").addEventListener("change", go); go();
    }).catch(fail);
  }

  function pEndpoints() {
    api("/api/endpoints").then(function (eps) {
      show('<p class="eyebrow">Console</p><h1>Endpoints</h1><p class="lede">Each server and model you have measured, with its cost per million tokens over time.</p>' +
        (eps.length ? '<div class="ep-list">' + eps.map(function (e) {
          var last = e.series[e.series.length - 1];
          return '<div class="panel ep"><div><b>' + esc(e.model) + '</b><div class="small muted">' + esc(e.endpoint_masked) + " · " + e.series.length + " checks</div><div class=\"mt8\">" +
            e.labels.map(function (l) { return '<span class="chip">' + esc(l) + "</span>"; }).join("") + "</div></div>" + spark(e.series) +
            '<div class="num right"><div class="t-20">' + esc(money(last.mean)) + '</div><div class="small muted">latest $/M</div></div></div>';
        }).join("") + "</div>" : '<div class="empty">No endpoints yet.</div>'));
    }).catch(fail);
  }

  var ACTION = [
    "name: cost-check", "on: [pull_request]", "jobs:", "  throttle:", "    runs-on: ubuntu-latest", "    steps:",
    "      - uses: KushagraKanaujia/throttle@v0.4.2", "        with:", "          url: ${{ secrets.STAGING_LLM_URL }}",
    "          model: your-model", "          gpu-hourly-rate: 1.50", "          fail-if-costlier: 5"
  ].join("\n");
  function pCI() {
    show('<p class="eyebrow">Console</p><h1>CI Gate</h1><p class="lede">Run a cost check on every deploy and fail the job when a calibrated check says it got more expensive. Inputs are documented in <a href="#/docs/guides/ci-github-action">Gate costs in CI</a>.</p>' +
      '<div class="panel"><div class="code-head">.github/workflows/cost-check.yml<button class="copy" type="button" data-copy="' + esc(ACTION) + '">copy</button></div><pre class="code">' + esc(ACTION) + "</pre></div>" +
      '<h2>Exit codes</h2><div class="panel"><div class="exit-grid"><b>0</b><span>OK: not costlier past the threshold, or the first check (nothing to compare with yet).</span><b>4</b><span>A calibrated check says MORE EXPENSIVE by more than <code>--fail-if-costlier</code>.</span><b>5</b><span>NOT CALIBRATED while a threshold is set: fewer than 3 recent repeats, so the gate refuses to guess.</span></div></div>' +
      '<p class="small muted mt16">The action page lists every input. Keep endpoint URLs and keys in repository secrets.</p>');
  }

  function pShare() {
    api("/api/checks").then(function (checks) {
      show('<p class="eyebrow">Console</p><h1>Share</h1><p class="lede">Turn any check into a shareable summary. It runs offline, removes URLs, hostnames and anything that looks like a secret, and prints a pre-filled GitHub issue link. Nothing is uploaded unless you submit it.</p>' +
        (checks.length ? '<div class="panel"><div class="table-wrap"><table><thead><tr><th>When</th><th>Check</th><th>Verdict</th><th>Command</th></tr></thead><tbody>' + checks.map(function (c) {
          var s = "throttle check --share-id " + c.id;
          return '<tr><td class="num small muted">' + when(c.created_at) + "</td><td><b>" + esc(c.label || "—") + '</b><div class="small muted">' + esc(money(c.summary.mean)) + "/M</div></td><td>" + badge(c.verdict) + "</td><td>" + cmd(s) + "</td></tr>";
        }).join("") + "</tbody></table></div></div>" : '<div class="empty">No checks to share yet.</div>'));
      page.querySelectorAll("tbody tr").forEach(function (tr) { tr.style.cursor = "default"; });
    }).catch(fail);
  }

  function pSettings() {
    api("/api/summary").then(function (s) {
      show('<p class="eyebrow">Console</p><h1>Settings</h1><p class="lede">Where this console reads from, and what it does not do.</p>' +
        '<div class="panel"><dl class="kv"><dt>Throttle</dt><dd class="num">' + esc(s.version) + '</dd><dt>History file</dt><dd class="num small">' + esc(s.history_file) + "</dd><dt>Checks</dt><dd class=\"num\">" + esc(s.checks) +
        (s.skipped_lines ? " (" + esc(s.skipped_lines) + " unreadable lines skipped)" : "") + '</dd><dt>Endpoints</dt><dd class="num">' + esc(s.endpoints) + '</dd><dt>Change it</dt><dd class="num small">throttle ui --history-dir PATH (or THROTTLE_CHECK_HISTORY_DIR)</dd></dl></div>' +
        '<h2>Privacy</h2><div class="panel pad"><p class="m0b10">The console runs on <b>127.0.0.1</b> and makes no network requests. It reads your local check history and the docs bundled with Throttle. It never contacts your endpoints, and the page loads nothing from the internet.</p><p class="small muted m0">Endpoint hosts other than localhost are masked in lists. Shared summaries strip URLs, hostnames and secrets.</p></div>');
    }).catch(fail);
  }

  function pDoc(slug) {
    Promise.all([api("/api/docs/" + slug), getNav()]).then(function (res) {
      var d = res[0], flat = [];
      res[1].forEach(function (g) { g.pages.forEach(function (p) { flat.push(p); }); });
      var i = flat.map(function (p) { return p.slug; }).indexOf(slug);
      var prev = i > 0 ? flat[i - 1] : null, next = i >= 0 && i < flat.length - 1 ? flat[i + 1] : null;
      var html = window.marked ? window.marked.parse(d.markdown, { gfm: true }) : "<pre>" + esc(d.markdown) + "</pre>";
      show('<article class="doc">' + html + '</article><nav class="doc-foot" aria-label="Pages">' +
        (prev ? '<a href="#/docs/' + esc(prev.slug) + '"><small>Previous</small>' + esc(prev.title) + "</a>" : "<span></span>") +
        (next ? '<a href="#/docs/' + esc(next.slug) + '" class="right"><small>Next</small>' + esc(next.title) + "</a>" : "") + "</nav>");
      fixDocLinks(slug);
    }).catch(fail);
  }
  function fixDocLinks(slug) {
    var base = slug.split("/").slice(0, -1);
    function resolve(rel) {
      var parts = base.slice();
      rel.split("/").forEach(function (seg) { if (seg === "..") parts.pop(); else if (seg && seg !== ".") parts.push(seg); });
      return parts.join("/");
    }
    page.querySelectorAll(".doc a[href]").forEach(function (a) {
      var h = a.getAttribute("href");
      if (/^https?:/.test(h)) { a.target = "_blank"; a.rel = "noopener noreferrer"; return; }
      var m = h.match(/^([^#?]+)\.md(#.*)?$/);
      if (m) a.setAttribute("href", "#/docs/" + resolve(m[1]));
    });
    page.querySelectorAll(".doc img[src]").forEach(function (img) {
      var s = img.getAttribute("src"), m = s.match(/illustrations\/([a-z0-9-]+\.svg)$/);
      if (m) img.setAttribute("src", "/static/illustrations/" + m[1]);
      else if (!/^\/static\//.test(s)) img.remove();
    });
  }

  /* ---------- search / ask ---------- */
  var CONSOLE_HITS = CONSOLE.map(function (c) { return { kind: "page", title: c[1], href: "#/" + c[0], snippet: "Console" }; });
  function openPalette() {
    if ($(".overlay")) return;
    var o = document.createElement("div");
    o.className = "overlay";
    o.innerHTML = '<div class="palette" role="dialog" aria-modal="true" aria-label="Search"><input id="pq" type="search" placeholder="Search docs, commands and your checks…" autocomplete="off" aria-label="Search"><ul id="pr"></ul><div class="hint">Searches the bundled docs and your local check history. Nothing leaves this machine.</div></div>';
    document.body.appendChild(o);
    var input = $("#pq"), list = $("#pr"), sel = 0, items = [], timer = null;
    function paint() {
      list.innerHTML = items.length ? items.map(function (h, i) {
        return '<li><a class="' + (i === sel ? "sel" : "") + '" href="' + esc(h.href) + '"><span class="kind">' + esc(h.kind) + "</span>" + esc(h.title) + (h.snippet ? "<small>" + esc(h.snippet) + "</small>" : "") + "</a></li>";
      }).join("") : '<li><span class="small muted noresult">No results.</span></li>';
    }
    function run() {
      var q = input.value.trim();
      if (q.length < 2) { items = CONSOLE_HITS.slice(); sel = 0; paint(); return; }
      var ql = q.toLowerCase();
      var local = CONSOLE_HITS.filter(function (h) { return h.title.toLowerCase().indexOf(ql) >= 0; });
      api("/api/search?q=" + encodeURIComponent(q)).then(function (hits) {
        items = local.concat(hits.map(function (h) {
          return { kind: h.kind, title: h.title, snippet: h.snippet, href: h.kind === "doc" ? "#/docs/" + h.slug : "#/checks/" + h.id };
        }));
        sel = 0; paint();
      }).catch(function () { items = local; paint(); });
    }
    function close() { o.remove(); document.removeEventListener("keydown", keys); }
    function keys(e) {
      if (e.key === "Escape") close();
      else if (e.key === "ArrowDown") { sel = Math.min(items.length - 1, sel + 1); paint(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); paint(); e.preventDefault(); }
      else if (e.key === "Enter" && items[sel]) { location.hash = items[sel].href.slice(1); close(); }
    }
    o.addEventListener("click", function (e) { if (e.target === o || e.target.closest("a")) close(); });
    input.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(run, 120); });
    document.addEventListener("keydown", keys);
    run(); input.focus();
  }

  /* ---------- wiring ---------- */
  function render() {
    var r = route();
    renderNav(r);
    $("#side").classList.remove("open");
    var p = r.path;
    if (p === "quickstart") pQuickstart();
    else if (p === "checks") pChecks();
    else if (p.indexOf("checks/") === 0) pCheck(decodeURIComponent(p.slice(7)));
    else if (p === "compare") pCompare(r.query);
    else if (p === "endpoints") pEndpoints();
    else if (p === "ci") pCI();
    else if (p === "share") pShare();
    else if (p === "settings") pSettings();
    else if (p.indexOf("docs/") === 0) pDoc(p.slice(5));
    else pQuickstart();
  }
  document.addEventListener("click", function (e) {
    var c = e.target.closest("[data-copy]");
    if (c) { copyText(c.getAttribute("data-copy"), c.previousElementSibling || c); return; }
    var tr = e.target.closest("tr[data-href]");
    if (tr && !e.target.closest("a,button")) location.hash = tr.getAttribute("data-href").slice(1);
    var t = e.target.closest(".tab");
    if (t) location.hash = t.dataset.tab === "console" ? "/quickstart" : t.dataset.tab === "cli" ? "/docs/cli/overview" : "/docs/index";
  });
  document.addEventListener("keydown", function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); }
  });
  $("#searchBtn").addEventListener("click", openPalette);
  $("#askBtn").addEventListener("click", openPalette);
  $("#menu").addEventListener("click", function () { $("#side").classList.toggle("open"); });
  window.addEventListener("hashchange", render);
  api("/api/summary").then(function (s) { $("#ver").textContent = "v" + s.version; }).catch(function () {});
  render();
})();
