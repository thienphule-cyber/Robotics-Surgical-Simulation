"use strict";

/* patient.js - Patient Overview and Surgical Status panels. */
(function () {
  const STEPS = [
    ["initializing", "Initialization"],
    ["planning", "Surgical planning"],
    ["vision_init", "Vision initialization"],
    ["positioning", "Robot positioning"],
    ["servoing", "Visual servoing"],
    ["evaluation", "Evaluation"],
    ["completed", "Completed"],
  ];

  const VITALS = [
    { key: "heart_rate_bpm", label: "Heart rate", unit: "bpm", digits: 0 },
    { key: "spo2_percent", label: "SpO₂", unit: "%", digits: 1 },
    { key: "systolic_mmhg", label: "Systolic BP", unit: "mmHg", digits: 0 },
    { key: "diastolic_mmhg", label: "Diastolic BP", unit: "mmHg", digits: 0 },
    { key: "resp_rate_bpm", label: "Resp. rate", unit: "/min", digits: 0 },
  ];

  const els = {};
  const vitalUi = {};
  let lastRegions = "";

  Dash.registerPanel("patient", {
    init() {
      ["patient-id", "patient-desc", "patient-posture", "region-select", "site-position",
        "vitals-grid", "vitals-badge", "phase-steps", "status-message",
        "status-chips", "status-elapsed"].forEach((id) => { els[id] = Dash.$("#" + id); });

      for (const vital of VITALS) {
        const value = Dash.el("span", { class: "tile-value", text: "—" });
        const canvas = Dash.el("canvas", { class: "spark" });
        els["vitals-grid"].append(Dash.el("div", { class: "tile" },
          Dash.el("span", { class: "tile-label", text: `${vital.label} (${vital.unit})` }),
          value, canvas));
        vitalUi[vital.key] = {
          value,
          chart: new Dash.Chart(canvas, { compact: true, windowSec: 60, series: [{ name: vital.label }] }),
        };
      }

      for (const [key, label] of STEPS) {
        els["phase-steps"].append(Dash.el("li", { "data-phase": key, text: label }));
      }

      els["region-select"].addEventListener("change", async () => {
        const wanted = els["region-select"].value;
        try {
          await Dash.api("PUT", "/api/patient/region", { region: wanted });
          Dash.toast(`Surgical region: ${wanted}`, "ok");
        } catch (err) {
          Dash.toast(err.message, "error");
          if (Dash.state) els["region-select"].value = Dash.state.patient.selected_region;
        }
      });
    },

    onState(s) {
      const patient = s.patient;
      const status = s.status;

      els["patient-id"].textContent = patient.patient_id;
      els["patient-desc"].textContent = patient.description;
      els["patient-posture"].textContent = patient.posture;
      els["vitals-badge"].hidden = !patient.vitals_simulated;

      const select = els["region-select"];
      const signature = patient.available_regions.join("|");
      if (signature !== lastRegions) {
        select.textContent = "";
        for (const region of patient.available_regions) {
          select.append(Dash.el("option", { value: region, text: Dash.prettyKey(region) }));
        }
        lastRegions = signature;
      }
      if (document.activeElement !== select) select.value = patient.selected_region;
      select.disabled = status.phase === "positioning" || status.phase === "servoing";

      els["site-position"].textContent = patient.site_position
        ? patient.site_position.map((v) => (v * 1000).toFixed(0)).join(", ") + " mm"
        : "—";

      for (const vital of VITALS) {
        const value = patient.vitals[vital.key];
        vitalUi[vital.key].value.textContent = Dash.fmt(value, vital.digits);
        vitalUi[vital.key].chart.push(s.timestamp, [value]);
      }

      // Surgical status
      const index = STEPS.findIndex((step) => step[0] === status.phase);
      els["phase-steps"].classList.toggle("aborted", status.phase === "aborted");
      Array.from(els["phase-steps"].children).forEach((li, i) => {
        if (index < 0) li.className = "";
        else if (i < index) li.className = "done";
        else if (i === index) li.className = status.phase === "completed" ? "done" : "active";
        else li.className = "";
      });
      els["status-message"].textContent =
        status.message || (status.phase === "aborted" ? "Procedure aborted" : "");
      els["status-elapsed"].textContent = Dash.formatDuration(status.elapsed_s);

      const chips = els["status-chips"];
      chips.textContent = "";
      const addChip = (text, cls) => chips.append(Dash.el("span", { class: "pill " + cls, text }));
      if (status.running) addChip(status.paused ? "paused" : "running", status.paused ? "pill-warn" : "pill-ok");
      else addChip("stopped", "pill-muted");
      if (status.recording) addChip("● recording", "pill-bad");
    },
  });
})();