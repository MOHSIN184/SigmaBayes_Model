"use strict";

const $ = (id) => document.getElementById(id);
const MAX_BYTES = 10 * 1024 * 1024;
const MAX_RECORDS = 5000;
const EXAMPLE = $("example-sequence").textContent.trim();

const ui = {
  sequence: $("sequence-input"),
  file: $("file-input"),
  binary: $("binary-input"),
  sigma: $("sigma-input"),
  submit: $("submit-button"),
  reset: $("reset-button"),
  example: $("example-button"),
  status: $("status"),
  body: $("result-body"),
  summary: $("summary"),
  chart: {
    empty: $("chart-empty"),
    content: $("chart-content"),
    visual: $("chart-visual"),
    tooltip: $("chart-tooltip"),
    totalValue: $("chart-total-value"),
    segments: {
      total: $("chart-total"),
      promoters: $("chart-promoters"),
      nonPromoters: $("chart-non-promoters"),
      sigma: $("chart-sigma"),
    },
    legends: {
      total: $("legend-total"),
      promoters: $("legend-promoters"),
      nonPromoters: $("legend-non-promoters"),
      sigma: $("legend-sigma"),
    },
  },
};

function alertMessage(message, type = "info") {
  ui.status.className = `alert ${type}`;
  ui.status.textContent = message;
}

function setLoading(active) {
  ui.submit.disabled = active;
  ui.submit.classList.toggle("loading", active);
  ui.submit.querySelector(".button-label").textContent = active
    ? "Predicting…"
    : "Predict sequences";
}

function resetPredictionChart() {
  ui.chart.empty.hidden = false;
  ui.chart.content.hidden = true;
  ui.chart.content.classList.remove("is-animated");
  ui.chart.tooltip.hidden = true;
}

function resetResults() {
  ui.body.replaceChildren();
  const row = ui.body.insertRow();
  row.className = "empty";
  const resultCell = row.insertCell();
  resultCell.colSpan = 7;
  resultCell.textContent = "Results will appear here after prediction.";
  ui.summary.hidden = true;
  ui.summary.replaceChildren();
  resetPredictionChart();
}

function cleanId(value, fallback) {
  return (value.trim().split(/\s+/)[0] || fallback).slice(0, 200);
}

function lineParser(source) {
  const records = [];
  let mode = null;
  let current = null;
  let index = 0;

  return {
    push(raw) {
      const line = raw.trim();
      if (!line) return;
      if (mode === null) mode = line.startsWith(">") ? "fasta" : "plain";
      if (line.startsWith(">")) {
        if (mode !== "fasta") {
          throw new Error("Invalid FASTA format. Headers cannot follow plain sequence data.");
        }
        if (current) records.push(current);
        current = { id: cleanId(line.slice(1), `Sequence_${++index}`), sequence: "" };
        return;
      }
      if (mode === "fasta") {
        if (!current) {
          throw new Error(
            "Invalid FASTA format. Sequence data must follow a header beginning with '>'.",
          );
        }
        current.sequence += line;
      } else {
        records.push({ id: `${source}_${++index}`, sequence: line });
      }
    },
    finish() {
      if (current) records.push(current);
      return records;
    },
  };
}

function parseLines(lines, source) {
  const parser = lineParser(source);
  for (const line of lines) parser.push(line);
  return parser.finish();
}

async function parseFile(file) {
  if (!file) return [];
  if (file.size > MAX_BYTES) throw new Error("The uploaded file exceeds the 10 MB limit.");
  if (!/\.(fasta|fa|txt)$/i.test(file.name)) {
    throw new Error("Upload a .fasta, .fa, or .txt file.");
  }

  const reader = file.stream().getReader();
  const decoder = new TextDecoder();
  const parser = lineParser("File");
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) parser.push(line);
    if (done) break;
  }
  if (buffer) parser.push(buffer);
  return parser.finish();
}

function uniqueRecords(records) {
  const seen = new Map();
  return records.map((record) => {
    const count = (seen.get(record.id) || 0) + 1;
    seen.set(record.id, count);
    return { id: count === 1 ? record.id : `${record.id}_${count}`, sequence: record.sequence };
  });
}

async function collectRecords() {
  let records = parseLines(ui.sequence.value.split(/\r?\n/), "Sequence");
  records = records.concat(await parseFile(ui.file.files[0]));
  if (!records.length) {
    throw new Error("Please enter a DNA sequence or upload a FASTA file.");
  }
  if (records.length > MAX_RECORDS) {
    throw new Error("A maximum of 5,000 sequences is allowed per batch.");
  }
  return uniqueRecords(records);
}

function cell(row, value, className = "") {
  const node = row.insertCell();
  node.textContent = value;
  if (className) node.className = className;
  return node;
}

function percentage(count, total) {
  return total ? (count / total) * 100 : 0;
}

function configureSegment(segment, label, count, percent, offset = 0) {
  const boundedPercent = Math.max(0, Math.min(100, percent));
  segment.style.strokeDasharray = `${boundedPercent} ${100 - boundedPercent}`;
  segment.style.transform = `rotate(${-90 + offset * 3.6}deg)`;
  const tooltip = `${label}: ${count.toLocaleString()} (${boundedPercent.toFixed(1)}%)`;
  segment.dataset.tooltip = tooltip;
  segment.setAttribute("aria-label", tooltip);

  let title = segment.querySelector("title");
  if (!title) {
    title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    segment.append(title);
  }
  title.textContent = tooltip;
}

function renderPredictionChart(data) {
  const results = Array.isArray(data.results) ? data.results : [];
  const total = results.length;
  const promoters = results.filter((result) => result.prediction === "Promoter").length;
  const nonPromoters = results.filter(
    (result) => result.prediction === "Non-Promoter",
  ).length;
  const sigmaFactors = results.filter((result) => Boolean(result.sigma_factor)).length;

  const promoterPercent = percentage(promoters, total);
  const nonPromoterPercent = percentage(nonPromoters, total);
  const sigmaPercent = percentage(sigmaFactors, total);

  configureSegment(ui.chart.segments.total, "Total Sequences", total, 100);
  configureSegment(ui.chart.segments.promoters, "Promoters", promoters, promoterPercent);
  configureSegment(
    ui.chart.segments.nonPromoters,
    "Non-Promoters",
    nonPromoters,
    nonPromoterPercent,
    promoterPercent,
  );
  configureSegment(ui.chart.segments.sigma, "Sigma Factors", sigmaFactors, sigmaPercent);

  ui.chart.totalValue.textContent = total.toLocaleString();
  ui.chart.legends.total.textContent = total.toLocaleString();
  ui.chart.legends.promoters.textContent = promoters.toLocaleString();
  ui.chart.legends.nonPromoters.textContent = nonPromoters.toLocaleString();
  ui.chart.legends.sigma.textContent = sigmaFactors.toLocaleString();

  ui.chart.empty.hidden = true;
  ui.chart.content.hidden = false;
  ui.chart.content.classList.remove("is-animated");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => ui.chart.content.classList.add("is-animated"));
  });
}

function showChartTooltip(segment, event) {
  const bounds = ui.chart.visual.getBoundingClientRect();
  const x = event?.clientX ? event.clientX - bounds.left : bounds.width / 2;
  const y = event?.clientY ? event.clientY - bounds.top : bounds.height * .28;
  ui.chart.tooltip.textContent = segment.dataset.tooltip;
  ui.chart.tooltip.style.left = `${x}px`;
  ui.chart.tooltip.style.top = `${y}px`;
  ui.chart.tooltip.hidden = false;
}

function hideChartTooltip() {
  ui.chart.tooltip.hidden = true;
}

function initializeChartInteractions() {
  for (const segment of Object.values(ui.chart.segments)) {
    segment.addEventListener("pointerenter", (event) => showChartTooltip(segment, event));
    segment.addEventListener("pointermove", (event) => showChartTooltip(segment, event));
    segment.addEventListener("pointerleave", hideChartTooltip);
    segment.addEventListener("focus", () => showChartTooltip(segment));
    segment.addEventListener("blur", hideChartTooltip);
  }
}

function renderResults(data) {
  renderPredictionChart(data);
  ui.body.replaceChildren();
  for (const result of data.results) {
    const row = ui.body.insertRow();
    cell(row, result.id);
    cell(row, String(result.length));
    const prediction = cell(row, result.prediction || "—");
    if (result.prediction) {
      const badge = document.createElement("span");
      badge.className = `badge ${result.prediction === "Promoter" ? "promoter" : "non-promoter"}`;
      badge.textContent = result.prediction;
      prediction.replaceChildren(badge);
    }
    cell(row, result.confidence == null ? "—" : `${(result.confidence * 100).toFixed(2)}%`);
    cell(row, result.sigma_factor || "");
    cell(row, `${result.processing_time_ms.toFixed(1)} ms`);
    cell(row, result.error || result.status, result.error ? "status-error" : "");
  }

  if (data.results.length === 1 && data.results[0].status === "Success") {
    renderSummary(data.results[0]);
  } else {
    ui.summary.hidden = false;
    ui.summary.className = "alert info";
    ui.summary.textContent =
      `Processed ${data.processed} sequence(s); ${data.failed} failed. ` +
      `Total server time: ${(data.total_time_ms / 1000).toFixed(2)} s.`;
  }
}

function renderSummary(result) {
  ui.summary.hidden = false;
  ui.summary.className = "summary-card";
  ui.summary.replaceChildren();
  const values = [
    ["Predicted class", result.prediction || "—"],
    ["Confidence", result.confidence == null ? "—" : `${(result.confidence * 100).toFixed(2)}%`],
    ["Sequence length", `${result.length} bp`],
    ["Processing time", `${result.processing_time_ms.toFixed(1)} ms`],
  ];
  for (const [label, value] of values) {
    const metric = document.createElement("div");
    metric.className = "metric";
    const small = document.createElement("small");
    const strong = document.createElement("strong");
    small.textContent = label;
    strong.textContent = value;
    metric.append(small, strong);
    ui.summary.append(metric);
  }
}

async function submit() {
  if (ui.submit.disabled) return;
  setLoading(true);
  alertMessage("Validating and analyzing sequences…", "info");
  try {
    const records = await collectRecords();
    const response = await fetch("/predict-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        records,
        run_binary: ui.binary.checked,
        run_sigma: ui.sigma.checked,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : (data.detail?.[0]?.msg || "Prediction request failed.");
      throw new Error(detail);
    }
    renderResults(data);
    alertMessage(
      data.failed
        ? `Processed ${data.processed} sequence(s); ${data.failed} record(s) need attention.`
        : `Successfully processed ${data.processed} sequence(s).`,
      data.failed ? "info" : "success",
    );
  } catch (error) {
    resetResults();
    alertMessage(error.message || "Prediction request failed.", "error");
  } finally {
    setLoading(false);
  }
}

ui.submit.addEventListener("click", submit);
ui.example.addEventListener("click", () => {
  ui.sequence.value = EXAMPLE;
  ui.file.value = "";
  resetResults();
  alertMessage("Verified 81 bp example loaded.", "info");
  ui.sequence.focus();
});
ui.reset.addEventListener("click", () => {
  ui.sequence.value = "";
  ui.file.value = "";
  ui.binary.checked = true;
  ui.sigma.checked = false;
  resetResults();
  alertMessage("Inputs are ready for a new analysis.", "info");
  ui.sequence.focus();
});

initializeChartInteractions();
