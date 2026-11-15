"use strict";

/*
 * dashboard.js - core of the dashboard.
 *
 *  - Dash.state / Dash.logs: latest data received from the server
 *  - Panels register with Dash.registerPanel(name, {init, onState, onLog, onConnection})
 *  - Connection: WebSocket stream, automatic reconnect, REST polling fallback
 *  - Shared helpers: Dash.api, Dash.el, Dash.fmt, Dash.Chart, Dash.toast, ...
 */
(function () {
  const PALETTE = ["#4cc9f0", "#f72585", "#b5e48c", "#ffd166", "#a78bfa", "#fb923c"];
  const MAX_LOGS = 500;

  const CONFIG_LABELS = {
    ibvs_gain: "IBVS gain (λ)",
    ibvs_tol_px: "IBVS tolerance (px)",
    tcp_speed_max: "Max TCP speed (m/s)",
    qdot_max: "Max joint speed (rad/s)",
    warn_distance: "Warning distance (m)",
  };

  const PHASE_LABELS = {
    idle: "Idle",
    initializing: "Initializing",
    planning: "Planning",
    vision_init: "Vision init",
    positioning: "Positioning",
    servoing: "Visual servoing",
    evaluation: "Evaluation",
    completed: "Completed",
    aborted: "Aborted",
  };

  const Dash = {
    state: null,
    logs: [],
    info: null,
    connection: "connecting",
    panels: {},
    charts: [],
    PALETTE,
    PHASE_LABELS,
  };
  window.Dash = Dash;

  // ------------------------------------------------------------------
  // Small helpers
  // ------------------------------------------------------------------
  Dash.$ = (selector, root) => (root || document).querySelector(selector);

  /** Create an element. Children that are strings become text nodes (safe). */
  Dash.el = function (tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else if (value !== false && value !== null && value !== undefined) node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children) node.append(child);
    return node;
  };

  Dash.fmt = (value, digits = 2, unit = "") =>
    value === null || value === undefined || Number.isNaN(value) ? "—" : Number(value).toFixed(digits) + unit;

  Dash.formatDuration = function (seconds) {
    const s = Math.max(0, Math.floor(seconds || 0));
    const mm = String(Math.floor(s / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  };

  Dash.formatClock = (epochSeconds) =>
    new Date(epochSeconds * 1000).toLocaleTimeString([], { hour12: false });

  Dash.prettyKey = (key) => String(key).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

  Dash.cssVar = (name, fallback) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

  Dash.toast = function (message, kind) {
    let box = Dash.$("#toasts");
    if (!box) {
      box = Dash.el("div", { id: "toasts", "aria-live": "polite" });
      document.body.append(box);
    }
    const toast = Dash.el("div", { class: "toast " + (kind || "info"), text: message });
    box.append(toast);
    setTimeout(() => toast.remove(), 4000);
  };

  Dash.download = function (filename, mime, text) {
    const url = URL.createObjectURL(new Blob([text], { type: mime }));
    const link = Dash.el("a", { href: url, download: filename });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  /** JSON request to the REST API. Throws Error(message) on failure. */
  Dash.api = async function (method, path, body) {
    const options = { method, headers: {} };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(path, options);
    } catch (err) {
      throw new Error("Cannot reach the dashboard server");
    }
    let data = null;
    try { data = await response.json(); } catch (err) { /* empty body */ }
    if (!response.ok) throw new Error((data && data.error) || `HTTP ${response.status}`);
    return data;
  };

  /** Match the canvas bitmap to its CSS size (HiDPI aware). */
  Dash.fitCanvas = function (canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  };

  // ------------------------------------------------------------------
  // Time-series chart (last N seconds, several series)
  // ------------------------------------------------------------------
  class Chart {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.series = opts.series || [{ name: "value" }];
      this.windowSec = opts.windowSec || 60;
      this.unit = opts.unit || "";
      this.yMin = opts.yMin;
      this.yMax = opts.yMax;
      this.decimals = opts.decimals === undefined ? 1 : opts.decimals;
      this.compact = !!opts.compact;
      this.times = [];
      this.data = this.series.map(() => []);
      this.dirty = true;
      Dash.charts.push(this);
    }

    push(t, values) {
      this.times.push(t);
      values.forEach((v, i) => this.data[i].push(Number.isFinite(v) ? v : null));
      const cutoff = t - this.windowSec;
      while (this.times.length > 1 && this.times[0] < cutoff) {
        this.times.shift();
        this.data.forEach((d) => d.shift());
      }
      this.dirty = true;
    }

    clear() {
      this.times = [];
      this.data = this.series.map(() => []);
      this.dirty = true;
    }

    draw() {
      const { ctx, w, h } = Dash.fitCanvas(this.canvas);
      if (!w || !h) return;
      ctx.clearRect(0, 0, w, h);

      const grid = Dash.cssVar("--grid", "#2a3753");
      const text = Dash.cssVar("--text-dim", "#8fa0bd");
      const pad = this.compact ? { l: 2, r: 2, t: 3, b: 3 } : { l: 46, r: 8, t: 8, b: 20 };
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;

      let lo = this.yMin;
      let hi = this.yMax;
      if (lo === undefined || hi === undefined) {
        let mn = Infinity;
        let mx = -Infinity;
        for (const d of this.data) for (const v of d) if (v !== null) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
        if (!Number.isFinite(mn)) { mn = 0; mx = 1; }
        if (mx - mn < 1e-9) { mn -= 0.5; mx += 0.5; }
        const margin = (mx - mn) * 0.08;
        if (lo === undefined) lo = mn - margin;
        if (hi === undefined) hi = mx + margin;
      }

      const tMax = this.times.length ? this.times[this.times.length - 1] : 0;
      const tMin = tMax - this.windowSec;
      const X = (t) => pad.l + ((t - tMin) / this.windowSec) * pw;
      const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * ph;

      if (!this.compact) {
        ctx.strokeStyle = grid;
        ctx.fillStyle = text;
        ctx.lineWidth = 1;
        ctx.font = "11px system-ui, sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (let i = 0; i <= 4; i++) {
          const v = lo + ((hi - lo) * i) / 4;
          const y = Y(v);
          ctx.beginPath();
          ctx.moveTo(pad.l, y);
          ctx.lineTo(w - pad.r, y);
          ctx.stroke();
          ctx.fillText(v.toFixed(this.decimals), pad.l - 4, y);
        }
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(`last ${this.windowSec} s` + (this.unit ? `  ·  ${this.unit}` : ""), pad.l + pw / 2, h - pad.b + 5);
      }

      this.data.forEach((d, si) => {
        ctx.strokeStyle = this.series[si].color || PALETTE[si % PALETTE.length];
        ctx.lineWidth = this.compact ? 1.5 : 1.6;
        ctx.beginPath();
        let penDown = false;
        for (let i = 0; i < d.length; i++) {
          if (d[i] === null) { penDown = false; continue; }
          const x = X(this.times[i]);
          const y = Y(d[i]);
          if (!penDown) { ctx.moveTo(x, y); penDown = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });

      if (!this.compact && this.series.length > 1) {
        ctx.font = "11px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        let x = pad.l + 6;
        this.series.forEach((s, i) => {
          ctx.fillStyle = s.color || PALETTE[i % PALETTE.length];
          ctx.fillRect(x, pad.t + 4, 8, 8);
          ctx.fillStyle = text;
          ctx.fillText(s.name, x + 11, pad.t + 8);
          x += 11 + ctx.measureText(s.name).width + 10;
        });
      }
    }
  }
  Dash.Chart = Chart;

  function frame() {
    for (const chart of Dash.charts) {
      if (chart.dirty) {
        chart.dirty = false;
        chart.draw();
      }
    }
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------------
  // Panels
  // ------------------------------------------------------------------
  Dash.registerPanel = function (name, handlers) {
    Dash.panels[name] = handlers;
  };

  function notify(method, ...args) {
    for (const [name, panel] of Object.entries(Dash.panels)) {
      if (typeof panel[method] === "function") {
        try {
          panel[method](...args);
        } catch (err) {
          console.error(`Panel '${name}' failed in ${method}:`, err);
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // Header, controls, parameters
  // ------------------------------------------------------------------
  function updateHeader(state) {
    const phase = state.status.phase;
    const badge = Dash.$("#phase-badge");
    badge.textContent = PHASE_LABELS[phase] || phase;
    badge.className = "pill phase-" + phase;
    Dash.$("#elapsed").textContent = Dash.formatDuration(state.status.elapsed_s);

    const banner = Dash.$("#estop-banner");
    banner.hidden = !state.safety.estop;
    Dash.$("#estop-reason").textContent = (state.safety.violations || []).join("; ");
  }

  function updateControls(state) {
    const st = state && state.status;
    const safety = state && state.safety;
    const connected = Dash.connection === "live" || Dash.connection === "polling";
    const rules = {
      start: () => !st.running,
      pause: () => st.running && !st.paused,
      resume: () => st.running && st.paused,
      stop: () => st.running,
      estop: () => true,
      reset_estop: () => safety.estop,
      reset: () => !st.running,
      start_recording: () => !st.recording,
      stop_recording: () => st.recording,
    };
    document.querySelectorAll("[data-command]").forEach((button) => {
      const rule = rules[button.dataset.command];
      button.disabled = !(connected && st && safety && rule && rule());
    });
    const apply = Dash.$("#config-apply");
    if (apply) apply.disabled = !connected;
  }

  Dash.sendCommand = async function (command) {
    try {
      await Dash.api("POST", "/api/command", { command });
      Dash.toast(`Command sent: ${command}`, "ok");
    } catch (err) {
      Dash.toast(err.message, "error");
    }
  };

  function setupControls() {
    document.querySelectorAll("[data-command]").forEach((button) => {
      button.addEventListener("click", () => Dash.sendCommand(button.dataset.command));
    });
    Dash.$("#config-apply").addEventListener("click", applyConfig);
  }

  function setupConfig() {
    const form = Dash.$("#config-form");
    const spec = (Dash.info && Dash.info.config_spec) || {};
    form.textContent = "";
    for (const [key, s] of Object.entries(spec)) {
      const input = Dash.el("input", { type: "number", id: `cfg-${key}`, min: s.min, max: s.max, step: "any", "data-key": key });
      input.addEventListener("input", () => { input.dataset.dirty = "1"; });
      form.append(Dash.el("label", { class: "field" },
        Dash.el("span", { text: CONFIG_LABELS[key] || Dash.prettyKey(key) }),
        input,
        Dash.el("small", { text: `allowed range ${s.min} to ${s.max}` })));
    }
    if (Dash.state) updateConfigInputs(Dash.state);
  }

  function updateConfigInputs(state) {
    document.querySelectorAll("#config-form input").forEach((input) => {
      if (document.activeElement === input || input.dataset.dirty) return;
      const value = state.config[input.dataset.key];
      if (value !== undefined) input.value = value;
    });
  }

  async function applyConfig() {
    const updates = {};
    document.querySelectorAll("#config-form input").forEach((input) => {
      if (input.dataset.dirty && input.value !== "") updates[input.dataset.key] = Number(input.value);
    });
    if (!Object.keys(updates).length) {
      Dash.toast("No parameter changed");
      return;
    }
    try {
      const result = await Dash.api("PUT", "/api/config", updates);
      document.querySelectorAll("#config-form input").forEach((input) => {
        delete input.dataset.dirty;
        input.value = result.config[input.dataset.key];
      });
      Dash.toast("Parameters updated", "ok");
    } catch (err) {
      Dash.toast(err.message, "error");
    }
  }

  // ------------------------------------------------------------------
  // Data handling
  // ------------------------------------------------------------------
  let lastLogId = 0;

  function addLogs(entries, reset) {
    const fresh = [];
    for (const entry of entries || []) {
      if (entry.id > lastLogId) {
        lastLogId = entry.id;
        fresh.push(entry);
      }
    }
    if (fresh.length) {
      Dash.logs.push(...fresh);
      if (Dash.logs.length > MAX_LOGS) Dash.logs.splice(0, Dash.logs.length - MAX_LOGS);
    }
    if (reset) notify("onLog", Dash.logs.slice(), { reset: true });
    else if (fresh.length) notify("onLog", fresh, { reset: false });
  }

  function applySnapshot(data, resetLogs) {
    Dash.state = data;
    if (data.logs) {
      if (resetLogs) {
        Dash.logs = [];
        lastLogId = 0;
        addLogs(data.logs, true);
      } else {
        addLogs(data.logs, false);
      }
    }
    updateHeader(data);
    updateControls(data);
    updateConfigInputs(data);
    notify("onState", data);
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case "snapshot": applySnapshot(msg.data, true); break;
      case "state": applySnapshot(msg.data, false); break;
      case "log": addLogs(msg.data.entries, false); break;
      default: break;      // "pong"
    }
  }

  // ------------------------------------------------------------------
  // Connection: WebSocket with reconnect, REST polling as a fallback
  // ------------------------------------------------------------------
  let socket = null;
  let retry = 0;
  let reconnectTimer = null;
  let pingTimer = null;

  function setConnection(status) {
    if (Dash.connection === status) return;
    Dash.connection = status;
    const labels = {
      live: ["Live", "pill-ok"],
      polling: ["Polling (no WebSocket)", "pill-warn"],
      connecting: ["Connecting…", "pill-warn"],
      offline: ["Offline", "pill-bad"],
    };
    const [text, cls] = labels[status] || labels.offline;
    const pill = Dash.$("#conn-indicator");
    pill.textContent = text;
    pill.className = "pill " + cls;
    updateControls(Dash.state);
    notify("onConnection", status);
  }

  async function ensureInfo() {
    if (Dash.info) return;
    try {
      Dash.info = await Dash.api("GET", "/api/info");
      setupConfig();
    } catch (err) { /* retried on the next successful connection */ }
  }

  function scheduleReconnect() {
    clearTimeout(reconnectTimer);
    const delay = Math.min(1000 * 2 ** retry, 10000);
    retry += 1;
    reconnectTimer = setTimeout(connect, delay);
  }

  function connect() {
    const port = (Dash.info && Dash.info.ws_port) || 8765;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    try {
      socket = new WebSocket(`${scheme}://${location.hostname}:${port}`);
    } catch (err) {
      scheduleReconnect();
      return;
    }
    socket.onopen = () => {
      retry = 0;
      setConnection("live");
      ensureInfo();
      clearInterval(pingTimer);
      pingTimer = setInterval(() => {
        if (socket && socket.readyState === WebSocket.OPEN) socket.send("ping");
      }, 15000);
    };
    socket.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch (err) { return; }
      handleMessage(msg);
    };
    socket.onclose = () => {
      clearInterval(pingTimer);
      if (Dash.connection === "live") setConnection("offline");
      scheduleReconnect();
    };
    socket.onerror = () => {
      try { socket.close(); } catch (err) { /* already closed */ }
    };
  }

  async function poll() {
    if (Dash.connection === "live") return;
    try {
      const data = await Dash.api("GET", "/api/state");
      setConnection("polling");
      ensureInfo();
      applySnapshot(data, false);
    } catch (err) {
      setConnection("offline");
    }
  }

  async function start() {
    notify("init");
    setupControls();
    await ensureInfo();
    connect();
    setInterval(poll, 2000);
    window.addEventListener("resize", () => Dash.charts.forEach((c) => { c.dirty = true; }));
    requestAnimationFrame(frame);
  }

  // deferred scripts have all run (panels registered) before this event fires
  document.addEventListener("DOMContentLoaded", start);
})();