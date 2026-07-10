document.addEventListener("DOMContentLoaded", () => {
  const q = selector => document.querySelector(selector);
  const menu = q(".menu");
  const links = q(".nav-links");
  if (menu && links) menu.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    menu.setAttribute("aria-expanded", String(open));
  });
  const form = q("#sequence-form");
  if (!form) return;
  const sequenceInput = q("#sequence");
  const apiInput = q("#api-url");
  const status = q("#sequence-status");
  const submit = q("#predict-button");
  const results = q("#prediction-results");
  const runBinary = q("#run-binary");
  const runSigma = q("#run-sigma");
  const cleanUrl = value => value.trim().replace(/\/+$/, "");
  function gradioSource(value) {
    if (/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(value)) return value;
    try {
      const url = new URL(value);
      const spaceMatch = url.pathname.match(/^\/spaces\/([^/]+)\/([^/]+)/);
      if (url.hostname === "huggingface.co" && spaceMatch) {
        return `${spaceMatch[1]}/${spaceMatch[2]}`;
      }
      if (url.hostname.endsWith(".hf.space")) return url.origin;
    } catch (_) {
      return null;
    }
    return null;
  }
  async function requestPrediction(target, payload) {
    const source = gradioSource(target);
    if (source) {
      const { Client } = await import("https://cdn.jsdelivr.net/npm/@gradio/client/+esm");
      const client = await Client.connect(source);
      const result = await client.predict("/predict", [
        payload.sequence,
        payload.run_binary,
        payload.run_sigma
      ]);
      const output = result.data[0];
      return typeof output === "string" ? JSON.parse(output) : output;
    }
    const response = await fetch(`${target}/predict`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map(item => item.msg).join("; ") : data.detail;
      throw new Error(detail || `Backend returned HTTP ${response.status}.`);
    }
    return data;
  }
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[character]));
  apiInput.value = localStorage.getItem("bayessigmaApiUrl") || "";
  apiInput.addEventListener("change", () => localStorage.setItem("bayessigmaApiUrl", cleanUrl(apiInput.value)));

  function probabilityTable(probabilities) {
    const entries = Object.entries(probabilities || {});
    if (!entries.length) return "<p>Unavailable.</p>";
    const rows = entries.map(([label, value]) => `<tr><td>${escapeHtml(label)}</td><td>${(Number(value) * 100).toFixed(2)}%</td></tr>`).join("");
    return `<div class="table-wrap"><table><thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  function badge(available) {
    return `<span class="availability ${available ? "available" : "unavailable"}">${available ? "Available" : "Unavailable"}</span>`;
  }
  function renderTask(task, title, uncertaintyRequired) {
    if (!task || !task.available) return `<h3>${title}</h3>${badge(false)}<p>This model/output is unavailable from the configured backend.</p>`;
    const calibrated = task.calibration_available ? `<h4>Temperature-scaled probabilities</h4>${probabilityTable(task.calibrated_probabilities)}` : "<p>Calibration: unavailable.</p>";
    const conformal = task.conformal_available ? `<p><strong>90% conformal set:</strong> ${escapeHtml(task.conformal_set_90.join(", ") || "Empty set")}</p><p><strong>95% conformal set:</strong> ${escapeHtml(task.conformal_set_95.join(", ") || "Empty set")}</p>` : "<p>Conformal prediction: unavailable.</p>";
    let uncertainty = "";
    if (uncertaintyRequired) uncertainty = task.uncertainty && task.uncertainty.available ? `<h4>MC-dropout uncertainty</h4><p>Predictive entropy: ${Number(task.uncertainty.entropy).toFixed(6)}<br>Mutual information: ${Number(task.uncertainty.mutual_information).toFixed(6)}</p><small>${escapeHtml(task.uncertainty.note)}</small>` : "<p>MC-dropout uncertainty: unavailable.</p>";
    return `<h3>${title}</h3>${badge(true)}<p class="prediction-label">${escapeHtml(task.predicted_label)}</p><p>Model: <code>${escapeHtml(task.model_name)}</code><br>Model confidence: ${(Number(task.confidence) * 100).toFixed(2)}%</p><h4>Raw probabilities</h4>${probabilityTable(task.probabilities)}${calibrated}${conformal}${uncertainty}`;
  }
  function render(data) {
    q("#validation-summary").innerHTML = `<h3>Validated input</h3><p><strong>${data.sequence_length} bp</strong> · GC content <strong>${(Number(data.gc_content) * 100).toFixed(2)}%</strong></p><code class="sequence-display">${escapeHtml(data.sequence)}</code>`;
    q("#warning-list").innerHTML = (data.warnings || []).length ? `<div class="api-warnings"><h3>Scientific warnings</h3><ul>${data.warnings.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : "";
    q("#binary-result").innerHTML = renderTask(data.binary, "Binary promoter", false);
    q("#sigma-result").innerHTML = renderTask(data.sigma, "Sigma factor", true);
    results.hidden = false;
    results.scrollIntoView({behavior:"smooth", block:"start"});
  }
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const sequence = sequenceInput.value.replace(/\s/g, "").toUpperCase();
    sequenceInput.value = sequence;
    status.className = "status";
    results.hidden = true;
    if (sequence.length !== 81) {
      status.textContent = `Sequence length is ${sequence.length} bp; exactly 81 bp are required.`;
      status.classList.add("error");
      return;
    }
    if (!/^[ACGT]+$/.test(sequence)) {
      status.textContent = "Use only A, C, G, and T.";
      status.classList.add("error");
      return;
    }
    if (!runBinary.checked && !runSigma.checked) {
      status.textContent = "Select at least one prediction task.";
      status.classList.add("error");
      return;
    }
    const api = cleanUrl(apiInput.value);
    if (!api) {
      status.textContent = "Backend API URL is required for live prediction.";
      status.classList.add("error");
      return;
    }
    localStorage.setItem("bayessigmaApiUrl", api);
    status.textContent = "Running artifact-backed inference...";
    submit.disabled = true;
    try {
      const data = await requestPrediction(api, {
        sequence,
        run_binary: runBinary.checked,
        run_sigma: runSigma.checked
      });
      render(data);
      status.textContent = "Prediction completed.";
      status.classList.add("success");
    } catch (error) {
      status.textContent = `Prediction failed: ${error.message}`;
      status.classList.add("error");
    } finally {
      submit.disabled = false;
    }
  });
});
