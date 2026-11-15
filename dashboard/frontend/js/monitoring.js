"use strict";

/* monitoring.js - Safety Monitoring, Operation Logs and Performance Analysis. */
(function () {
  const LEVEL_RANK = { debug: 0, info: 1, warning: 2, error: 3 };
  const MAX_ROWS = 500;

  const els = {};
  let clearanceChart = null;
  let lastPerfSignature = "";
  let hiddenBefore = 0;          // "Clear view" hides logs up to this id

  // ------------------------------------------------------------------
  // Safety
  // ------------------------------------------------------------------
  function updateSafety(s) {
    const safety = s.safety;
    const state = els.safetyState;
    state.textContent = safety.state;
    state.className = "pill safety-" + safety.state;

    els.estop.textContent = safety.estop ? "ACTIVE" : "Released";
    els.estop.style.color = safety.estop ? "var(--bad)" : "";
    els.count.textContent = String(safety.violation_count);

    const clearance = safety.min_clearance === null || safety.min_clearance === undefined
      ? null
      : safety.min_clearance * 1000;
    els.clearance.textContent = Dash.fmt(clearance, 1);
    if (clearance !== null) clearanceChart.push(s.timestamp, [clearance]);

    els.list.textContent = "";
    const violations = safety.violations || [];
    if (!violations.length) {
      els.list.append(Dash.el("li", { class: "none", text: "None" }));
    } else {
      for (const text of violations) els.list.append(Dash.el("li", { text }));
    }
  }

  // ------------------------------------------------------------------
  // Logs
  // ------------------------------------------------------------------
  function visible(entry) {
    return entry.id > hiddenBefore &&
      LEVEL_RANK[entry.level] >= LEVEL_RANK[els.level.value];
  }

  function rowFor(entry) {
    return Dash.el("div", { class: "log-row lvl-" + entry.level },
      Dash.el("span", { class: "t", text: Dash.formatClock(entry.time) }),
      Dash.el("span", { text: entry.level }),
      Dash.el("span", { class: "src", text: entry.source }),
      Dash.el("span", { text: entry.message }));
  }

  function scrollIfNeeded() {
    if (els.autoscroll.checked) els.logList.scrollTop = els.logList.scrollHeight;
  }

  function renderAll() {
    els.logList.textContent = "";
    for (const entry of Dash.logs) {
      if (visible(entry)) els.logList.append(rowFor(entry));
    }
    scrollIfNeeded();
  }

  function appendEntries(entries) {
    for (const entry of entries) {
      if (visible(entry)) els.logList.append(rowFor(entry));
    }
    while (els.logList.children.length > MAX_ROWS) els.logList.firstChild.remove();
    scrollIfNeeded();
  }

  function csvCell(value) {
    let text = String(value === null || value === undefined ? "" : value);
    if (/^[=+\-@\t\r]/.test(text)) text = "'" + text;      // avoid spreadsheet formula injection
    return '"' + text.replace(/"/g, '""') + '"';
  }

  function downloadCsv() {
    const lines = ["id,time,level,source,message"];
    for (const e of Dash.logs) {
      lines.push([e.id, new Date(e.time * 1000).toISOString(), e.level, e.source, e.message]
        .map(csvCell).join(","));
    }
    Dash.download("operation_log.csv", "text/csv", lines.join("\n"));
  }

  // ------------------------------------------------------------------
  // Performance
  // ------------------------------------------------------------------
  function formatMetric(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function updatePerformance(s) {
    const perf = s.performance || {};
    const signature = JSON.stringify(perf);
    if (signature === lastPerfSignature) return;
    lastPerfSignature = signature;

    els.perfBody.textContent = "";
    const entries = Object.entries(perf);
    if (!entries.length) {
      els.perfBody.append(Dash.el("tr", {},
        Dash.el("td", { class: "dim", text: "No metrics yet (they appear after the evaluation phase)" }),
        Dash.el("td", { text: "" })));
      return;
    }
    for (const [key, value] of entries) {
      els.perfBody.append(Dash.el("tr", {},
        Dash.el("td", { text: Dash.prettyKey(key) }),
        Dash.el("td", { class: "num", text: formatMetric(value) })));
    }
  }

  // ------------------------------------------------------------------
  Dash.registerPanel("monitoring", {
    init() {
      els.safetyState = Dash.$("#safety-state");
      els.estop = Dash.$("#safety-estop");
      els.count = Dash.$("#safety-violations-count");
      els.clearance = Dash.$("#safety-clearance");
      els.list = Dash.$("#safety-violation-list");
      els.perfBody = Dash.$("#perf-body");
      els.logList = Dash.$("#log-list");
      els.level = Dash.$("#log-level");
      els.autoscroll = Dash.$("#log-autoscroll");

      clearanceChart = new Dash.Chart(Dash.$("#safety-chart"), {
        windowSec: 60,
        unit: "mm",
        decimals: 0,
        series: [{ name: "min clearance" }],
      });

      els.level.addEventListener("change", renderAll);
      Dash.$("#btn-log-clear").addEventListener("click", () => {
        hiddenBefore = Dash.logs.length ? Dash.logs[Dash.logs.length - 1].id : 0;
        renderAll();
      });
      Dash.$("#btn-log-csv").addEventListener("click", downloadCsv);
      Dash.$("#btn-download-session").addEventListener("click", () => {
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        Dash.download(`session_${stamp}.json`, "application/json",
          JSON.stringify({ exported_at: new Date().toISOString(), state: Dash.state, logs: Dash.logs }, null, 2));
      });
    },

    onState(s) {
      updateSafety(s);
      updatePerformance(s);
    },

    onLog(entries, options) {
      if (options && options.reset) {
        hiddenBefore = 0;
        renderAll();
      } else {
        appendEntries(entries);
      }
    },
  });
})();