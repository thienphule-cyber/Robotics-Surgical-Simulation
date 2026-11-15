"use strict";

/* robot.js - Robot Monitoring: joint table, TCP position, charts, TCP path. */
(function () {
  const MAX_TRAIL = 3000;
  const els = {};
  let rows = [];
  let chart = null;
  let rowsBuilt = false;
  let trail = [];            // [[x, y], ...] in metres
  let lastSimTime = 0;
  let lastPhase = "";
  let trailCanvas = null;

  function buildRows(names) {
    els.body.textContent = "";
    rows = names.map((name) => {
      const angle = Dash.el("td", { class: "num", text: "—" });
      const speed = Dash.el("td", { class: "num", text: "—" });
      const fill = Dash.el("div", { class: "bar-fill" });
      els.body.append(Dash.el("tr", {},
        Dash.el("td", { text: name }), angle, speed,
        Dash.el("td", {}, Dash.el("div", { class: "bar" }, fill))));
      return { angle, speed, fill };
    });
    chart = new Dash.Chart(Dash.$("#robot-chart"), {
      windowSec: 60,
      unit: "deg",
      decimals: 0,
      series: names.map((_, i) => ({ name: "J" + (i + 1) })),
    });
  }

  function drawTrail() {
    const { ctx, w, h } = Dash.fitCanvas(trailCanvas);
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);

    const grid = Dash.cssVar("--grid", "#2a3753");
    const text = Dash.cssVar("--text-dim", "#8fa0bd");
    const accent = Dash.cssVar("--accent", "#4cc9f0");
    const left = 8;
    const top = 8;
    const pw = w - left - 8;
    const ph = h - top - 22;

    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.strokeRect(left, top, pw, ph);
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = text;

    if (!trail.length) {
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("no data yet", left + pw / 2, top + ph / 2);
      return;
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of trail) {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const span = Math.max(maxX - minX, maxY - minY, 0.1) * 1.2;   // equal aspect, min 10 cm
    const scale = Math.min(pw, ph) / span;
    const px = (x) => left + pw / 2 + (x - cx) * scale;
    const py = (y) => top + ph / 2 - (y - cy) * scale;

    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    trail.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(px(x), py(y)) : ctx.lineTo(px(x), py(y))));
    ctx.stroke();

    const [sx, sy] = trail[0];
    ctx.beginPath();
    ctx.arc(px(sx), py(sy), 4, 0, Math.PI * 2);
    ctx.stroke();

    const [ex, ey] = trail[trail.length - 1];
    ctx.fillStyle = accent;
    ctx.beginPath();
    ctx.arc(px(ex), py(ey), 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = text;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(`x ${(cx - span / 2).toFixed(2)} m`, left, top + ph + 4);
    ctx.textAlign = "right";
    ctx.fillText(`${(cx + span / 2).toFixed(2)} m →  (y up)`, left + pw, top + ph + 4);
  }

  Dash.registerPanel("robot", {
    init() {
      els.body = Dash.$("#joint-table-body");
      els.x = Dash.$("#tcp-x");
      els.y = Dash.$("#tcp-y");
      els.z = Dash.$("#tcp-z");
      els.sim = Dash.$("#sim-time");
      trailCanvas = Dash.$("#tcp-trail");
      Dash.$("#btn-clear-trail").addEventListener("click", () => {
        trail = [];
        drawTrail();
      });
      window.addEventListener("resize", drawTrail);
    },

    onState(s) {
      const robot = s.robot;
      if (!rowsBuilt) {
        buildRows(robot.joint_names || ["J1", "J2", "J3", "J4", "J5", "J6"]);
        rowsBuilt = true;
      }

      const deg = robot.q_deg;
      const qdot = robot.qdot;
      rows.forEach((row, i) => {
        const angle = deg ? deg[i] : null;
        row.angle.textContent = Dash.fmt(angle, 1);
        row.speed.textContent = qdot ? Dash.fmt((qdot[i] * 180) / Math.PI, 1) : "—";
        const pct = angle === null || angle === undefined
          ? 50
          : Math.min(100, Math.max(0, ((angle + 360) / 720) * 100));
        row.fill.style.left = pct + "%";
      });
      if (deg) chart.push(s.timestamp, deg);

      const p = robot.tcp_position;
      els.x.textContent = p ? Dash.fmt(p[0] * 1000, 1) : "—";
      els.y.textContent = p ? Dash.fmt(p[1] * 1000, 1) : "—";
      els.z.textContent = p ? Dash.fmt(p[2] * 1000, 1) : "—";
      els.sim.textContent = Dash.fmt(robot.sim_time, 2);

      // A new run (sim time went backwards or phase restarted) clears the path.
      const phase = s.status.phase;
      if ((robot.sim_time !== null && robot.sim_time < lastSimTime - 1e-6) ||
          (phase !== lastPhase && phase === "initializing")) {
        trail = [];
      }
      lastSimTime = robot.sim_time || 0;
      lastPhase = phase;

      if (p) {
        const last = trail[trail.length - 1];
        if (!last || Math.hypot(p[0] - last[0], p[1] - last[1]) > 0.0005) {
          trail.push([p[0], p[1]]);
          if (trail.length > MAX_TRAIL) trail.shift();
        }
      }
      drawTrail();
    },
  });
})();