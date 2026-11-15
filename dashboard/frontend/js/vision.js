"use strict";

/*
 * vision.js - Vision Monitoring.
 * Shows the feature positions in the image plane (the dashboard receives
 * features, not the raw camera frames; the live image is in the Swift camera panel).
 */
(function () {
  // Must match the camera resolution used by the simulation (workcell image_size).
  const IMG_W = 640;
  const IMG_H = 480;

  const els = {};
  let canvas = null;
  let chart = null;

  const isValid = (row) =>
    Array.isArray(row) && row.length === 2 && Number.isFinite(row[0]) && Number.isFinite(row[1]);

  function draw(vision) {
    const { ctx, w, h } = Dash.fitCanvas(canvas);
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);

    const grid = Dash.cssVar("--grid", "#2a3753");
    const text = Dash.cssVar("--text-dim", "#8fa0bd");
    const ok = Dash.cssVar("--ok", "#3ddc97");
    const bad = Dash.cssVar("--bad", "#ff5d73");
    const warn = Dash.cssVar("--warn", "#ffd166");

    const scale = Math.min(w / IMG_W, h / IMG_H) * 0.96;
    const ox = (w - IMG_W * scale) / 2;
    const oy = (h - IMG_H * scale) / 2;
    const X = (u) => ox + u * scale;
    const Y = (v) => oy + v * scale;

    // Image frame and grid (every 80 px)
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    for (let u = 0; u <= IMG_W; u += 80) {
      ctx.beginPath(); ctx.moveTo(X(u), Y(0)); ctx.lineTo(X(u), Y(IMG_H)); ctx.stroke();
    }
    for (let v = 0; v <= IMG_H; v += 80) {
      ctx.beginPath(); ctx.moveTo(X(0), Y(v)); ctx.lineTo(X(IMG_W), Y(v)); ctx.stroke();
    }
    ctx.strokeStyle = text;
    ctx.strokeRect(X(0), Y(0), IMG_W * scale, IMG_H * scale);

    const current = vision.current_features || [];
    const desired = vision.desired_features || [];
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";

    // Desired features (crosses) and error lines
    desired.forEach((d, i) => {
      if (!isValid(d)) return;
      const x = X(d[0]);
      const y = Y(d[1]);
      ctx.strokeStyle = bad;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 6, y); ctx.lineTo(x + 6, y);
      ctx.moveTo(x, y - 6); ctx.lineTo(x, y + 6);
      ctx.stroke();
      if (isValid(current[i])) {
        ctx.strokeStyle = warn;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(X(current[i][0]), Y(current[i][1]));
        ctx.stroke();
      }
    });

    // Current features (circles)
    current.forEach((c, i) => {
      if (!isValid(c)) return;
      const x = X(c[0]);
      const y = Y(c[1]);
      ctx.strokeStyle = ok;
      ctx.fillStyle = ok;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillText(String(i), x + 10, y - 8);
    });

    ctx.fillStyle = text;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("green circle: current   red cross: desired", X(0) + 6, Y(0) + 6);
  }

  Dash.registerPanel("vision", {
    init() {
      canvas = Dash.$("#vision-canvas");
      ["vision-camera", "vision-nfeat", "vision-status", "vision-fps", "vision-error"]
        .forEach((id) => { els[id] = Dash.$("#" + id); });
      chart = new Dash.Chart(Dash.$("#vision-chart"), {
        windowSec: 60,
        unit: "px",
        decimals: 1,
        yMin: 0,
        series: [{ name: "IBVS error" }],
      });
      window.addEventListener("resize", () => { if (Dash.state) draw(Dash.state.vision); });
    },

    onState(s) {
      const v = s.vision;
      const total = (v.current_features || []).length;
      const lost = (v.current_features || []).filter((row) => !isValid(row)).length;

      els["vision-camera"].textContent = v.camera || "—";
      els["vision-nfeat"].textContent = total ? `${total - lost} / ${total}` : "—";
      els["vision-status"].textContent = v.status || "—";
      els["vision-fps"].textContent = Dash.fmt(v.fps, 1);

      const err = els["vision-error"];
      err.textContent = Dash.fmt(v.error_px, 2);
      const tol = s.config ? s.config.ibvs_tol_px : null;
      err.style.color = v.error_px === null || v.error_px === undefined || tol === null
        ? ""
        : v.error_px <= tol ? "var(--ok)" : "var(--warn)";

      if (Number.isFinite(v.error_px)) chart.push(s.timestamp, [v.error_px]);
      draw(v);
    },
  });
})();