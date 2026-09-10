// TCA Viewer frontend — vanilla JS, native Canvas 2D only (no D3/Three.js).
// Nothing here polls automatically; every fetch is triggered by a user
// action (tab switch, dropdown change, or the manual Refresh button) so the
// tool never generates background HTTP/CPU load of its own.

const state = {
  status: null,
  segments: [],
  activeTab: "graph",
};

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

// ── Status bar / >3D disable ─────────────────────────────────────────────

async function loadStatus() {
  state.status = await fetchJSON("/api/status");
  const s = state.status;
  const info = document.getElementById("status-info");
  info.textContent = `dir=${s.run_dir}  dimensions=${s.dimensions ?? "unknown"}  ` +
    `judge_state=${s.has_judge_state ? "yes" : "no"}`;

  const banner = document.getElementById("viz-disabled-banner");
  if (s.visualization_enabled) {
    banner.classList.add("hidden");
  } else {
    // visualization_enabled is false for two very different reasons --
    // don't claim ">3 dimensions" when the real issue is "no segment data
    // found yet" (dimensions is null, e.g. wrong --dir or nothing trained).
    if (s.dimensions === null || s.dimensions === undefined) {
      banner.textContent = `Visualization disabled: no trained segment (.nexseg) files found in ` +
        `"${s.run_dir}". Point --dir at a run directory that has segment_0.nexseg etc., or train first.`;
    } else {
      banner.textContent = `Visualization disabled: this run has ${s.dimensions} dimensions, more than ` +
        `${s.max_visualizable_dimensions}. Geometric rendering isn't meaningful past that — use the ` +
        `Logs and Confidence tabs instead.`;
    }
    banner.classList.remove("hidden");
  }
  // Graph and Signal Paths tabs need geometric rendering — disable their
  // canvases (not the tabs entirely, so Confidence/Curves/Logs still work)
  // when dimensions exceed the visualizable limit.
  const geometryDisabled = !s.visualization_enabled;
  document.getElementById("graph-canvas").style.opacity = geometryDisabled ? 0.15 : 1;
  document.getElementById("paths-canvas").style.opacity = geometryDisabled ? 0.15 : 1;
}

// ── Tabs ──────────────────────────────────────────────────────────────────

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      document.getElementById(`tab-${tab}`).classList.add("active");
      state.activeTab = tab;
      refreshActiveTab();
    });
  });
}

function refreshActiveTab() {
  switch (state.activeTab) {
    case "graph": loadGraphTab(); break;
    case "confidence": loadConfidenceTab(); break;
    case "paths": loadPathsTab(); break;
    case "curves": loadCurvesTab(); break;
    case "logs": loadLogsTab(); break;
    case "query": loadQueryTab(); break;
  }
}

// ── Projection helpers (2D direct, simple isometric for 3D) ─────────────

function project(position, maxX, canvasW, canvasH) {
  const pad = 40;
  const scale = (Math.min(canvasW, canvasH) - 2 * pad) / (2 * Math.max(maxX, 1));
  const cx = canvasW / 2, cy = canvasH / 2;
  if (position.length <= 2) {
    const [x, y = 0] = position;
    return [cx + x * scale, cy - y * scale];
  }
  // Simple isometric projection for 3D — no external 3D library needed.
  const [x, y, z] = position;
  const cos30 = Math.cos(Math.PI / 6), sin30 = Math.sin(Math.PI / 6);
  const sx = (x - z) * cos30;
  const sy = (x + z) * sin30 - y;
  return [cx + sx * scale, cy + sy * scale];
}

// ── Graph tab ─────────────────────────────────────────────────────────────

async function loadGraphTab() {
  if (state.segments.length === 0) {
    state.segments = await fetchJSON("/api/segments");
    const sel = document.getElementById("graph-segment-select");
    sel.innerHTML = state.segments
      .map(s => `<option value="${s.segment_id}">segment ${s.segment_id} (${s.n_nodes} nodes)</option>`)
      .join("");
  }
  if (state.segments.length === 0) {
    document.getElementById("graph-node-count").textContent = "no .nexseg files found in this directory";
    return;
  }
  await drawSelectedSegment();
}

async function drawSelectedSegment() {
  const sel = document.getElementById("graph-segment-select");
  const segId = sel.value || sel.options[0]?.value;
  if (segId === undefined) return;
  const seg = await fetchJSON(`/api/segment/${segId}`);
  document.getElementById("graph-node-count").textContent =
    `${seg.processing_nodes.length} processing nodes, ${seg.reviewers.length} reviewers, max_x=${seg.max_x}, dim=${seg.dimensions}`;

  const canvas = document.getElementById("graph-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const showConn = document.getElementById("graph-show-connections").checked;
  const showRev = document.getElementById("graph-show-reviewers").checked;
  const maxX = seg.max_x || 10;

  const posOf = p => project(p, maxX, canvas.width, canvas.height);

  // Connections first (under the nodes)
  if (showConn) {
    ctx.strokeStyle = "rgba(77,171,247,0.18)";
    ctx.lineWidth = 1;
    for (const node of seg.processing_nodes) {
      const [nx, ny] = posOf(node.position);
      for (const target of node.connected_positions) {
        const [tx, ty] = posOf(target);
        ctx.beginPath();
        ctx.moveTo(nx, ny);
        ctx.lineTo(tx, ty);
        ctx.stroke();
      }
    }
    ctx.strokeStyle = "rgba(255,146,44,0.35)";
    const [spx, spy] = posOf(seg.splitter.position);
    for (const target of seg.splitter.connected_positions) {
      const [tx, ty] = posOf(target);
      ctx.beginPath();
      ctx.moveTo(spx, spy);
      ctx.lineTo(tx, ty);
      ctx.stroke();
    }
  }

  // Processing nodes
  ctx.fillStyle = "#4dabf7";
  const nodePositions = [];
  for (const node of seg.processing_nodes) {
    const [x, y] = posOf(node.position);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
    nodePositions.push({ x, y, position: node.position });
  }

  // Splitter (diamond)
  const [spx, spy] = posOf(seg.splitter.position);
  ctx.fillStyle = "#ff922b";
  ctx.save();
  ctx.translate(spx, spy);
  ctx.rotate(Math.PI / 4);
  ctx.fillRect(-6, -6, 12, 12);
  ctx.restore();

  // Reviewers (stars, approximated as larger circles + outline)
  if (showRev) {
    ctx.fillStyle = "#51cf66";
    ctx.strokeStyle = "#eaffea";
    ctx.lineWidth = 1.5;
    for (const revPos of seg.reviewers) {
      const [x, y] = posOf(revPos);
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  // Hover tooltip (recomputed on mousemove, cheap nearest-point search over
  // a few hundred nodes — negligible cost, only runs while hovering).
  const tooltip = document.getElementById("graph-tooltip");
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let nearest = null, bestDist = 14;
    for (const p of nodePositions) {
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < bestDist) { bestDist = d; nearest = p; }
    }
    if (nearest) {
      tooltip.textContent = `pos=[${nearest.position.map(v => v.toFixed(2)).join(", ")}]`;
      tooltip.style.left = `${e.clientX + 12}px`;
      tooltip.style.top = `${e.clientY + 12}px`;
      tooltip.classList.remove("hidden");
    } else {
      tooltip.classList.add("hidden");
    }
  };
  canvas.onmouseleave = () => tooltip.classList.add("hidden");
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("graph-segment-select").addEventListener("change", drawSelectedSegment);
  document.getElementById("graph-show-connections").addEventListener("change", drawSelectedSegment);
  document.getElementById("graph-show-reviewers").addEventListener("change", drawSelectedSegment);
});

// ── Confidence tab ─────────────────────────────────────────────────────────

async function loadConfidenceTab() {
  const container = document.getElementById("confidence-content");
  container.innerHTML = "";

  const judge = await fetchJSON("/api/judge");
  const clusters = judge.segment_weights?.clusters || [];

  const trace = await fetchJSON("/api/trace?limit=50");

  // Aggregate observed confidence/weight per segment from trace records.
  const perSeg = {};
  for (const rec of trace) {
    const segs = rec.breakdown?.segments || {};
    for (const [segId, info] of Object.entries(segs)) {
      if (!perSeg[segId]) perSeg[segId] = { weights: [], means: [], relevance: info.relevance };
      perSeg[segId].weights.push(info.weight);
      perSeg[segId].means.push(info.mean);
    }
  }

  const segIds = new Set([
    ...clusters.map(c => c.segment_id).filter(x => x !== null && x !== undefined),
    ...Object.keys(perSeg).map(Number),
  ]);

  if (segIds.size === 0) {
    container.innerHTML = `<p class="hint">No JudgeNode state or trace data found in this directory yet. ` +
      `Train with the normal pipeline (writes judge_node.judgestate) and/or enable ` +
      `settings.infer.trace_enabled to populate this view.</p>`;
    return;
  }

  for (const segId of [...segIds].sort((a, b) => a - b)) {
    const obs = perSeg[segId];
    const card = document.createElement("div");
    card.className = "seg-card";
    const avgWeight = obs ? obs.weights.reduce((a, b) => a + b, 0) / obs.weights.length : null;
    const avgMean = obs ? obs.means.reduce((a, b) => a + b, 0) / obs.means.length : null;
    card.innerHTML = `
      <h3>Segment ${segId}</h3>
      ${bar("cluster relevance", obs?.relevance ?? "—", obs?.relevance ?? 0)}
      ${obs ? bar("avg agg. weight (last " + obs.weights.length + ")", avgWeight.toFixed(4), avgWeight) : `<p class="hint">no trace records selected this segment</p>`}
      ${obs ? bar("avg predicted mean", avgMean.toFixed(2), Math.min(avgMean / 100, 1)) : ""}
    `;
    container.appendChild(card);
  }
}

function bar(label, valueText, frac) {
  const pct = Math.max(0, Math.min(1, Number(frac) || 0)) * 100;
  return `<div class="bar-row">
    <span class="bar-label">${label}</span>
    <span class="bar-track"><span class="bar-fill" style="width:${pct}%"></span></span>
    <span class="bar-value">${valueText}</span>
  </div>`;
}

// ── Signal Paths tab ─────────────────────────────────────────────────────

async function loadPathsTab() {
  const trace = await fetchJSON("/api/trace?limit=50");
  const recSel = document.getElementById("trace-record-select");
  const segSel = document.getElementById("trace-segment-select");
  const hint = document.getElementById("trace-hint");

  if (trace.length === 0) {
    recSel.innerHTML = "";
    segSel.innerHTML = "";
    hint.textContent = "No trace.jsonl records found — enable settings.infer.trace_enabled and run inference.";
    document.getElementById("paths-canvas").getContext("2d").clearRect(0, 0, 900, 640);
    return;
  }
  hint.textContent = `${trace.length} recent inference record(s) loaded.`;

  recSel.innerHTML = trace.map((r, i) =>
    `<option value="${i}">#${i}  score=${Number(r.score).toFixed(2)}  seg=${r.segment_id}  archetype=${r.archetype}</option>`
  ).join("");
  recSel.selectedIndex = trace.length - 1;

  async function drawForSelection() {
    const rec = trace[Number(recSel.value)];
    const segIds = Object.keys(rec.paths || {});
    segSel.innerHTML = segIds.map(id => `<option value="${id}">segment ${id}</option>`).join("");
    if (segIds.length === 0) {
      document.getElementById("paths-detail").innerHTML = "<p class=\"hint\">This record has no path data.</p>";
      return;
    }
    await drawPathsForSegment(rec);
  }

  recSel.onchange = drawForSelection;
  segSel.onchange = () => drawPathsForSegment(trace[Number(recSel.value)]);
  await drawForSelection();
}

// ids lets the Query tab reuse this exact rendering against its own
// canvas/select/detail elements instead of duplicating the drawing logic.
const PATHS_TAB_IDS = { segSelectId: "trace-segment-select", canvasId: "paths-canvas", detailId: "paths-detail" };

async function drawPathsForSegment(rec, ids = PATHS_TAB_IDS) {
  const segSel = document.getElementById(ids.segSelectId);
  const segId = segSel.value;
  if (segId === undefined || segId === "") return;
  const seg = await fetchJSON(`/api/segment/${segId}`);
  const maxX = seg.max_x || 10;
  const canvas = document.getElementById(ids.canvasId);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const posOf = p => project(p, maxX, canvas.width, canvas.height);

  // Base graph, dim
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  for (const node of seg.processing_nodes) {
    const [nx, ny] = posOf(node.position);
    for (const target of node.connected_positions) {
      const [tx, ty] = posOf(target);
      ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(tx, ty); ctx.stroke();
    }
  }
  ctx.fillStyle = "rgba(255,255,255,0.25)";
  for (const node of seg.processing_nodes) {
    const [x, y] = posOf(node.position);
    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  }

  // Highlighted signal paths for this segment
  const paths = (rec.paths && rec.paths[segId]) || [];
  const colors = ["#ff922b", "#51cf66", "#4dabf7", "#e64980", "#fab005"];
  const detail = document.getElementById(ids.detailId);
  let detailHtml = `<table class="simple"><tr><th>#</th><th>hops</th><th>final prediction</th><th>variance</th></tr>`;
  paths.forEach((p, i) => {
    const color = colors[i % colors.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    p.positions.forEach((pos, hopIdx) => {
      const [x, y] = posOf(pos);
      if (hopIdx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // Mark start/end
    if (p.positions.length > 0) {
      const [sx, sy] = posOf(p.positions[0]);
      const [ex, ey] = posOf(p.positions[p.positions.length - 1]);
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(ex, ey, 6, 0, Math.PI * 2); ctx.fill();
    }
    detailHtml += `<tr><td style="color:${color}">signal ${i}</td><td>${p.positions.length}</td>` +
      `<td>${Number(p.final_prediction).toFixed(3)}</td><td>${Number(p.variance).toFixed(4)}</td></tr>`;
  });
  detailHtml += `</table>`;
  detail.innerHTML = paths.length ? detailHtml : `<p class="hint">No collected signal paths for this segment on this record.</p>`;
}

// ── Training curves tab ────────────────────────────────────────────────

async function loadCurvesTab() {
  const rows = await fetchJSON("/api/epoch_metrics");
  const hint = document.getElementById("curves-hint");
  const canvas = document.getElementById("curves-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (rows.length === 0) {
    hint.textContent = "No error-epoch.csv found in this directory.";
    return;
  }
  hint.textContent = `${rows.length} epoch record(s).`;

  const r2 = rows.map(r => Number(r.test_r2 ?? r.r2)).filter(v => !Number.isNaN(v));
  if (r2.length === 0) {
    hint.textContent += " (no r2 column found to plot)";
    return;
  }

  const pad = 40, w = canvas.width - 2 * pad, h = canvas.height - 2 * pad;
  const minY = Math.min(0, ...r2), maxY = Math.max(...r2, 0.1);
  const xAt = i => pad + (i / Math.max(r2.length - 1, 1)) * w;
  const yAt = v => pad + h - ((v - minY) / (maxY - minY)) * h;

  ctx.strokeStyle = "#2a2f3a";
  ctx.beginPath(); ctx.moveTo(pad, yAt(0)); ctx.lineTo(pad + w, yAt(0)); ctx.stroke();

  ctx.strokeStyle = "#4dabf7";
  ctx.lineWidth = 2;
  ctx.beginPath();
  r2.forEach((v, i) => { const x = xAt(i), y = yAt(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.stroke();

  ctx.fillStyle = "#8a8f9c";
  ctx.font = "11px monospace";
  ctx.fillText("R² over epochs", pad, 16);
  ctx.fillText(maxY.toFixed(2), 4, yAt(maxY));
  ctx.fillText(minY.toFixed(2), 4, yAt(minY));
}

// ── Logs tab ──────────────────────────────────────────────────────────────

async function loadLogsTab() {
  const fileSel = document.getElementById("log-file-select");
  if (fileSel.options.length === 0) {
    const files = await fetchJSON("/api/log_files");
    fileSel.innerHTML = files.map(f => `<option value="${f}">${f}</option>`).join("");
    if (files.length === 0) {
      document.getElementById("log-content").innerHTML = `<p class="hint">No .log files found in logs/.</p>`;
      return;
    }
  }
  await renderLogs();
}

async function renderLogs() {
  const file = document.getElementById("log-file-select").value;
  if (!file) return;
  const level = document.getElementById("log-level-select").value;
  const limit = document.getElementById("log-limit-select").value;
  const url = `/api/logs?file=${encodeURIComponent(file)}&limit=${limit}` + (level ? `&min_level=${level}` : "");
  const entries = await fetchJSON(url);
  const container = document.getElementById("log-content");
  container.innerHTML = entries
    .map(e => `<div class="log-line log-${e.level}">[${e.level}] ${escapeHtml(e.message)}</div>`)
    .join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("log-file-select").addEventListener("change", renderLogs);
  document.getElementById("log-level-select").addEventListener("change", renderLogs);
  document.getElementById("log-limit-select").addEventListener("change", renderLogs);
});

// ── Query (live inference) ────────────────────────────────────────────────

let querySchema = null; // cached — schema doesn't change without retraining

async function loadQueryTab() {
  const hint = document.getElementById("query-schema-hint");
  if (!querySchema) {
    querySchema = await fetchJSON("/api/schema");
  }
  if (querySchema.error || !querySchema.columns || querySchema.columns.length === 0) {
    hint.textContent = querySchema.error || "No input columns found.";
    document.getElementById("query-form").innerHTML = "";
    return;
  }
  hint.textContent = `${querySchema.columns.length} input field(s), derived from the dataset CSV ` +
    `(target "${querySchema.target_column}" excluded).`;
  buildQueryForm(querySchema.columns);
}

function buildQueryForm(columns) {
  const form = document.getElementById("query-form");
  form.innerHTML = columns.map(col => {
    const id = `query-field-${col.name}`;
    if (col.type === "numeric") {
      const mid = (Number(col.min) + Number(col.max)) / 2;
      const hint = (col.min !== null && col.max !== null) ? ` (${col.min}–${col.max})` : "";
      return `<div class="controls"><label style="width:14em;display:inline-block">${col.name}${hint}
        <input type="number" id="${id}" data-type="numeric" data-name="${col.name}"
               value="${Number.isFinite(mid) ? mid.toFixed(2) : ""}" step="any"></label></div>`;
    }
    if (col.type === "categorical") {
      return `<div class="controls"><label style="width:14em;display:inline-block">${col.name}
        <select id="${id}" data-type="categorical" data-name="${col.name}">
          ${col.options.map(o => `<option value="${escapeHtml(String(o))}">${escapeHtml(String(o))}</option>`).join("")}
        </select></label></div>`;
    }
    return `<div class="controls"><label style="width:14em;display:inline-block">${col.name}
      <input type="text" id="${id}" data-type="text" data-name="${col.name}"></label></div>`;
  }).join("");
}

async function runQuery() {
  const errBox = document.getElementById("query-error");
  errBox.classList.add("hidden");
  const inputs = document.querySelectorAll("#query-form [data-name]");
  const feature_values = {};
  for (const el of inputs) {
    const name = el.dataset.name;
    if (el.dataset.type === "numeric") {
      const v = parseFloat(el.value);
      if (Number.isNaN(v)) { errBox.textContent = `"${name}" needs a number.`; errBox.classList.remove("hidden"); return; }
      feature_values[name] = v;
    } else {
      feature_values[name] = el.value;
    }
  }
  const aggregation_mode = document.getElementById("query-aggregation-mode").value;
  const selection_percentage = parseFloat(document.getElementById("query-selection-pct").value);

  const btn = document.getElementById("query-run-btn");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feature_values, aggregation_mode, selection_percentage }),
    });
    const record = await res.json();
    if (!res.ok) {
      errBox.textContent = record.error || `request failed (${res.status})`;
      errBox.classList.remove("hidden");
      return;
    }
    renderQueryResult(record);
  } catch (e) {
    errBox.textContent = String(e);
    errBox.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Query";
  }
}

const QUERY_TAB_IDS = { segSelectId: "query-segment-select", canvasId: "query-canvas", detailId: "query-detail" };

function renderQueryResult(record) {
  document.getElementById("query-result").innerHTML = `
    <div class="seg-card">
      <h3>Result — score = ${Number(record.score).toFixed(3)}</h3>
      ${bar("confidence", Number(record.confidence).toFixed(3), record.confidence)}
      <div class="hint">dominant segment_id=${record.segment_id}  archetype=${record.archetype}
        aggregation_mode=${record.breakdown?.aggregation_mode ?? ""}</div>
    </div>`;

  const segs = record.breakdown?.segments || {};
  document.getElementById("query-confidence").innerHTML = Object.entries(segs).map(([segId, info]) => `
    <div class="seg-card">
      <h3>Segment ${segId}</h3>
      ${bar("agg. weight", Number(info.weight).toFixed(4), info.weight)}
      ${bar("cluster relevance", Number(info.relevance).toFixed(4), info.relevance)}
      ${bar("predicted mean", Number(info.mean).toFixed(2), Math.min(info.mean / 100, 1))}
      <div class="hint">n_reviewers=${info.n_reviewers}</div>
    </div>`).join("");

  const segIds = Object.keys(record.paths || {});
  const segSel = document.getElementById("query-segment-select");
  segSel.innerHTML = segIds.map(id => `<option value="${id}">segment ${id}</option>`).join("");
  if (segIds.length === 0) {
    document.getElementById("query-canvas").getContext("2d").clearRect(0, 0, 900, 480);
    document.getElementById("query-detail").innerHTML = "<p class=\"hint\">No path data returned.</p>";
    return;
  }
  segSel.onchange = () => drawPathsForSegment(record, QUERY_TAB_IDS);
  drawPathsForSegment(record, QUERY_TAB_IDS);
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("query-run-btn").addEventListener("click", runQuery);
});

// ── Boot ──────────────────────────────────────────────────────────────────

document.getElementById("refresh-btn").addEventListener("click", async () => {
  await loadStatus();
  refreshActiveTab();
});

setupTabs();
loadStatus().then(refreshActiveTab);
