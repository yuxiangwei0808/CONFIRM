const $ = (selector) => document.querySelector(selector);
const esc = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const steps = ["Question", "Plan", "Results", "Checks", "Decision"];
const replaySteps = ["Diagnose", "Propose", "Validate", "Evaluate"];
const standards = ["screen", "replicate", "full"];
const names = { screen: "Screen", replicate: "Replicate", full: "Full" };
const params = new URLSearchParams(location.search);
const state = {
  example: params.get("example") || "nonreplicating",
  step: Math.min(4, Math.max(0, Number(params.get("step")) || 0)),
  standard: standards.includes(params.get("standard"))
    ? params.get("standard")
    : "full",
  replay: 0,
};
let data, followup;
const pValue = (value) => (value < 0.001 ? "< 0.001" : `= ${value.toFixed(3)}`);
const number = (value) =>
  Math.abs(value) < 0.00005 ? "0.0000" : value.toFixed(4);
const badge = (pass) =>
  `<span class="status ${pass ? "pass" : "fail"}">${pass ? "✓ Pass" : "− Fail"}</span>`;
const decisionBadge = (pass) =>
  `<span class="status ${pass ? "pass" : "fail"}">${pass ? "Supported" : "Withheld"}</span>`;
const current = () => data.examples.find((x) => x.id === state.example);

function updateURL() {
  const url = new URL(location.href);
  url.searchParams.set("example", state.example);
  url.searchParams.set("step", state.step);
  url.searchParams.set("standard", state.standard);
  history.replaceState(null, "", url);
}

function picker() {
  return `<div class="standard-picker"><div class="standards-label">Compare reporting standards</div><div class="standards" role="group" aria-label="Reporting standard">${standards.map((s) => `<button type="button" data-standard="${s}" aria-pressed="${state.standard === s}">${names[s]}</button>`).join("")}</div></div>`;
}

function scatter(example, cohort, label) {
  const all = Object.values(example.points).flat();
  const ys = all.map((p) => p[1]);
  const xs = all.map((p) => p[0]);
  const ymin = Math.min(...ys),
    ymax = Math.max(...ys),
    xmin = 15,
    xmax = 85;
  const low = ymin - (ymax - ymin) * 0.08,
    high = ymax + (ymax - ymin) * 0.08;
  const X = (x) => 43 + ((x - xmin) / (xmax - xmin)) * 310,
    Y = (y) => 202 - ((y - low) / (high - low)) * 164;
  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((t) => {
      const y = low + t * (high - low);
      return `<line x1="43" x2="353" y1="${Y(y)}" y2="${Y(y)}" stroke="#e4e9e0"/><text x="35" y="${Y(y) + 3}" text-anchor="end">${y.toFixed(1)}</text>`;
    })
    .join("");
  const ticks = [20, 40, 60, 80]
    .map((x) => `<text x="${X(x)}" y="222" text-anchor="middle">${x}</text>`)
    .join("");
  const points = example.points[cohort]
    .map(
      ([x, y, site]) =>
        `<circle cx="${X(x).toFixed(2)}" cy="${Y(y).toFixed(2)}" r="2.5" fill="${site === "A" ? "#467b6a" : "#c28b65"}" opacity=".65"/>`,
    )
    .join("");
  return `<div class="plot-panel"><div class="plot-header"><h4>${label}</h4><span>n = 600</span></div><svg class="scatter" viewBox="0 0 375 248" role="img" aria-label="${esc(label)}: raw synthetic measurement against age, colored by site"><text x="43" y="20">Measurement (arbitrary units)</text>${grid}${points}${ticks}<text x="199" y="244" text-anchor="middle">Age (years)</text></svg><div class="plot-footer"><span>180 of 600 observations shown</span><span><span class="legend"><i></i>Site A</span> &nbsp; <span class="legend site-b"><i></i>Site B</span></span></div></div>`;
}

function gates(example) {
  const r = example.results,
    g = r.verdict.gates,
    c = example.contract;
  const family = g.multiplicity_effective_family_size;
  return [
    {
      name: "Search history & multiplicity",
      pass: g.search_provenance && g.multiplicity,
      required: true,
      explanation:
        "The plan records how the hypothesis was selected. The significance threshold accounts for the declared search family.",
      observed: `Family size: ${family} · p ${pValue(r.primary.p)} · threshold: ${(c.gates.multiplicity.alpha / family).toFixed(3)}`,
    },
    {
      name: "Measured confounds",
      pass: g.confound && g.confound_completeness,
      required: true,
      explanation: g.confound_completeness
        ? "The model includes the required adjustment variables. The audit finds no measured confound that is associated with the predictor and omitted from the plan."
        : "Site is associated with age and omitted from the plan. This check fails even though the apparent association is significant.",
      observed: `Adjustment variables: ${c.covariates.join(", ")}${g.confound_completeness ? "" : " · omitted: site"}`,
    },
    {
      name: "Statistical power",
      pass: g.power,
      required: state.standard === "full",
      explanation:
        "The power calculation asks whether the planned sample can detect a reference effect chosen before analysis. This synthetic example uses a reference effect of 0.5.",
      observed: `Estimated power: ${(r.power.achieved_power * 100).toFixed(1)}% · required: ${(c.gates.power.min_power * 100).toFixed(0)}%`,
    },
    {
      name: "Analytic stability",
      pass: g.multiverse,
      required: state.standard === "full",
      explanation:
        "The analysis is repeated across specified choices of adjustment, outlier handling, and model. A consistent result must appear in enough of these alternatives.",
      observed: `Consistent analyses: ${(r.multiverse.fraction_consistent * 100).toFixed(1)}% · required: ${(c.gates.multiverse.min_fraction_consistent * 100).toFixed(0)}%`,
    },
    {
      name: "Independent replication",
      pass: g.replication,
      required: state.standard !== "screen",
      explanation: g.replication
        ? "The association has the expected direction and passes the significance threshold in the independent replication cohort."
        : "The association is absent in the independent replication sample. The discovery result alone does not satisfy this requirement.",
      observed: `Replication p ${pValue(r.replication.cohort_results[0].effect.p)} · required: same direction and p < ${c.gates.replication.alpha}`,
    },
  ];
}

function question(example) {
  return `<div class="two-col"><div class="question-copy"><p class="stage-kicker">Start with a scientific question</p><h3>${esc(example.question)}</h3><p>We compare two independent samples. The analysis plan fixes the measurement, adjustment variables, expected direction, and checks before the result is evaluated.</p><div class="sample-line"><div><b>600</b><span>Discovery observations</span></div><div><b>600</b><span>Replication observations</span></div></div><details class="ground-truth"><summary>How this example was constructed</summary><p>${esc(example.setup)}</p></details></div><div class="question-plot">${scatter(example, "DEMO_DISC", "A first look at discovery")}<p class="plot-description">A visible pattern is the starting point. Next, inspect the plan used to test it.</p></div></div>`;
}

function plan(example) {
  const c = example.contract;
  return `<div class="stage-header"><div><p class="stage-kicker">Specify before testing</p><h3>The analysis choices are fixed.</h3><p>This example starts from a prepared plan. In an agent run, the model drafts a plan for approval before execution.</p></div></div><div class="two-col"><div class="plan-block"><h4>What will be tested</h4><div class="plan-row"><span class="plan-label">Predictor</span><span>Age</span></div><div class="plan-row"><span class="plan-label">Measurement</span><span>Synthetic imaging measurement</span></div><div class="plan-row"><span class="plan-label">Direction</span><span>${c.estimand.direction === "negative" ? "Decreases" : "Increases"} with age</span></div><div class="plan-row"><span class="plan-label">Adjust for</span><span class="chips">${c.covariates.map((x) => `<span class="chip">${x === "eTIV" ? "Head size" : esc(x[0].toUpperCase() + x.slice(1))}</span>`).join("")}</span></div><div class="plan-row"><span class="plan-label">Samples</span><span>Independent discovery and replication cohorts</span></div></div><div class="plan-block"><h4>What the checks require</h4><div class="plan-row"><span class="plan-label">Significance</span><span>p &lt; 0.05, adjusted for the search family</span></div><div class="plan-row"><span class="plan-label">Power</span><span>At least 80% for a reference effect of 0.5</span></div><div class="plan-row"><span class="plan-label">Stability</span><span>At least 60% consistent alternative analyses</span></div><div class="plan-row"><span class="plan-label">Replication</span><span>Same direction and p &lt; 0.05 in new data</span></div><div class="plan-row"><span class="plan-label">Search family</span><span>1 pre-specified hypothesis</span></div></div></div><div class="plan-foot"><span class="hash">Plan SHA-256: ${example.contract_sha256.slice(0, 24)}…</span><a href="data/${example.id}-plan.json" download class="text-link">Download the full plan ↓</a></div>`;
}

function results(example) {
  const a = example.results.primary,
    b = example.results.replication.cohort_results[0].effect;
  return `<div class="stage-header"><div><p class="stage-kicker">Computed by the analysis code</p><h3>Does the finding appear in new data?</h3><p>Both plots use the same scales. Dots show raw measurements; the effect estimates below account for the variables in the plan.</p></div></div><div class="two-col results-grid">${scatter(example, "DEMO_DISC", "Discovery")}${scatter(example, "DEMO_REP", "Replication")}</div><div class="metrics">${[
    ["Discovery", a],
    ["Replication", b],
  ]
    .map(
      ([label, e]) =>
        `<div><div class="metric-line"><span>${label} effect</span><b>${number(e.beta)}</b><span>p ${esc(pValue(e.p))}</span></div><p class="ci-note">95% CI [${number(e.ci_low)}, ${number(e.ci_high)}]</p></div>`,
    )
    .join(
      "",
    )}</div><p class="plot-description">Effects are measurement units per year of age, fitted using all 600 observations per cohort. A confidence interval describes uncertainty around the estimated effect.</p>`;
}

function checks(example) {
  return `<div class="stage-header"><div><p class="stage-kicker">Five evidence checks</p><h3>Open a check to see the evidence.</h3><p>The reporting standard selects which checks are required.</p></div>${picker()}</div><div class="gate-list">${gates(
    example,
  )
    .map(
      (g, i) =>
        `<details class="gate-row" ${!g.pass ? "open" : ""}><summary><span class="gate-index">0${i + 1}</span><span class="gate-name">${g.name}</span><span class="gate-required">${g.required ? "Required" : "Not required"}</span>${badge(g.pass)}</summary><div class="gate-detail">${g.explanation}<span class="observed">${esc(g.observed)}</span></div></details>`,
    )
    .join(
      "",
    )}</div><p class="optional-note">All five checks are computed. A failed check withholds the claim when that check is required by the selected standard.</p>`;
}

function decision(example) {
  const supported = example.support[state.standard],
    failed = gates(example).filter((g) => g.required && !g.pass);
  const why = supported
    ? state.standard === "screen" && !example.support.full
      ? "The discovery requirements pass. Screen does not require replication; this claim is withheld under Replicate and Full."
      : "All checks required by this reporting standard pass."
    : example.id === "site_confounded"
      ? "The plan omits site, a measured confound associated with age. The apparent association does not pass the confound check."
      : "The association does not repeat in the independent cohort. Replication is required by this reporting standard.";
  return `<div class="stage-header"><div><p class="stage-kicker">A decision tied to the evidence</p><h3>What can this analysis report?</h3></div>${picker()}</div><div class="decision-layout"><div class="decision-card ${supported ? "" : "withheld"}"><span class="decision-symbol" aria-hidden="true">${supported ? "✓" : "−"}</span><h3>${supported ? "Supported" : "Withheld"}</h3><p>${why}</p><span class="decision-scope">${names[state.standard]} reporting standard${supported && state.standard === "full" ? " · Confirmed" : ""}</span></div><div><p class="quiet">Same computed results, different requirements</p><table class="comparison"><thead><tr><th scope="col">Standard</th><th scope="col">Reporting decision</th></tr></thead><tbody>${standards.map((s) => `<tr class="${s === state.standard ? "active" : ""}"><td>${names[s]}</td><td>${decisionBadge(example.support[s])}</td></tr>`).join("")}</tbody></table><p class="decision-note">${failed.length ? `Required check that failed: ${failed.map((g) => g.name.toLowerCase()).join(", ")}.` : "Support means the selected requirements passed."} Choose the standard before running a new analysis.</p><p class="decision-note">${esc(example.lesson)}</p></div></div>`;
}

function render() {
  const example = current();
  $("#case-picker").innerHTML = data.examples
    .map(
      (e, i) =>
        `<button type="button" class="case-button" data-case="${e.id}" aria-pressed="${e.id === state.example}"><span class="case-num">0${i + 1}</span><span><span class="case-title">${esc(e.title)}</span><span class="case-short">${esc(e.short)}</span></span></button>`,
    )
    .join("");
  $("#step-picker").innerHTML = steps
    .map(
      (label, i) =>
        `<button type="button" class="step-button" data-step="${i}" ${i === state.step ? 'aria-current="step"' : ""}><span>0${i + 1}</span>${label}</button>`,
    )
    .join("");
  $("#stage-content").innerHTML = [question, plan, results, checks, decision][
    state.step
  ](example);
  $("#record-link").href = `data/${example.id}-record.json`;
  $("#previous").disabled = state.step === 0;
  $("#next").textContent =
    state.step === 4
      ? "Watch agent follow-up ↓"
      : `Next: ${steps[state.step + 1].toLowerCase()} →`;
  $("#step-count").textContent =
    `${String(state.step + 1).padStart(2, "0")} / 05`;
  updateURL();
}

function renderReplay() {
  const c = followup.candidates;
  $("#replay-picker").innerHTML = replaySteps
    .map(
      (s, i) =>
        `<button type="button" class="replay-step" data-replay="${i}" ${state.replay === i ? 'aria-current="step"' : ""}><span>0${i + 1}</span>${s}</button>`,
    )
    .join("");
  $("#replay-count").textContent = `0${state.replay + 1} / 04`;
  $("#replay-actor").textContent = [
    "CONFIRM · Failure diagnosis",
    "Agent · Recorded proposals",
    "CONFIRM · Plan validation",
    "CONFIRM · Computed results",
  ][state.replay];
  const panels = [
    `<h3>The groups differ by site.</h3><p>The original plan compares a functional-connectivity measurement between two constructed groups, adjusting for age and sex. Site is associated with group membership but is missing from the plan.</p><div class="replay-callout"><strong>Failed check: measured confounds</strong><br>The claim is withheld. The agent receives this diagnosis and the original analysis plan.</div>`,
    `<h3>Two proposals address the diagnosis.</h3><div class="proposal"><span>01</span><div><h4>Add the missing site adjustment</h4><p>Keep the original group comparison and connectivity measurement. Add site to the adjustment variables.</p><div class="chips"><span class="chip">Age</span><span class="chip">Sex</span><span class="chip">+ Site</span></div></div></div><div class="proposal"><span>02</span><div><h4>Explore a related connectivity pattern</h4><p>Test ${c[1].region_count} related connectivity measurements with the same group comparison, adding site adjustment.</p></div></div>`,
    `<h3>Both plans pass validation.</h3><p>The code checks the proposed changes before execution. A valid plan still needs statistical support.</p><ul class="validation-list"><li>Group comparison and original cohorts retained</li><li>Existing adjustment variables retained; site added</li><li>Gate thresholds preserved</li><li>Related outcomes and executable designs checked</li></ul><p>The second proposal changes both the outcome scope and adjustment variables. This combined change is recorded in its validation warning.</p>`,
    `<h3>Neither follow-up receives support.</h3><div class="proposal"><span>01</span><div><h4>Site-adjusted original measurement</h4><p>Discovery p ${esc(pValue(c[0].primary.p))}. Adjusting for site removes the apparent group difference.</p>${decisionBadge(c[0].supported)}</div></div><div class="proposal"><span>02</span><div><h4>Related connectivity pattern</h4><p>${c[1].significant_regions} of ${c[1].region_count} measurements pass the recorded regional significance check.</p>${decisionBadge(c[1].supported)}</div></div><p class="replay-callout">The final correction family contains ${followup.final_family_size} hypotheses, including the original test. The original claim remains withheld.</p>`,
  ];
  $("#replay-content").innerHTML = panels[state.replay];
  $("#replay-prev").disabled = state.replay === 0;
  $("#replay-next").textContent =
    state.replay === 3 ? "Replay from start ↺" : "Next →";
}

document.addEventListener("click", (event) => {
  const target = event.target.closest("button");
  if (!target || !data) return;
  if (target.dataset.case) {
    state.example = target.dataset.case;
    render();
    document
      .querySelector(`[data-case="${state.example}"]`)
      .focus({ preventScroll: true });
  }
  if (target.dataset.step !== undefined) {
    state.step = Number(target.dataset.step);
    render();
    document
      .querySelector(`[data-step="${state.step}"]`)
      .focus({ preventScroll: true });
  }
  if (target.dataset.standard) {
    state.standard = target.dataset.standard;
    render();
    document
      .querySelector(`[data-standard="${state.standard}"]`)
      .focus({ preventScroll: true });
  }
  if (target.dataset.replay !== undefined) {
    state.replay = Number(target.dataset.replay);
    renderReplay();
    document
      .querySelector(`[data-replay="${state.replay}"]`)
      .focus({ preventScroll: true });
  }
});
$("#previous").addEventListener("click", () => {
  if (data && state.step > 0) {
    state.step--;
    render();
  }
});
$("#next").addEventListener("click", () => {
  if (!data) return;
  if (state.step < 4) {
    state.step++;
    render();
  } else {
    $("#followup").scrollIntoView({
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "instant"
        : "smooth",
    });
  }
});
$("#replay-prev").addEventListener("click", () => {
  if (followup && state.replay > 0) {
    state.replay--;
    renderReplay();
  }
});
$("#replay-next").addEventListener("click", () => {
  if (followup) {
    state.replay = (state.replay + 1) % 4;
    renderReplay();
  }
});
$("#copy-command").addEventListener("click", async () => {
  const command = "confirm demo --out outputs/demo";
  try {
    await navigator.clipboard.writeText(command);
    $("#copy-status").textContent = "Command copied.";
  } catch {
    const range = document.createRange();
    range.selectNodeContents($(".command code"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    $("#copy-status").textContent =
      "Command selected. Use your browser’s Copy command.";
  }
});

async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return response.json();
}
try {
  [data, followup] = await Promise.all([
    loadJSON("data/examples.json"),
    loadJSON("data/followup.json"),
  ]);
  if (!data.examples.some((e) => e.id === state.example))
    state.example = "nonreplicating";
  state.step = Math.floor(state.step);
  render();
  renderReplay();
} catch (error) {
  $("#stage-content").innerHTML =
    '<div class="error-box"><h3>The example files could not be loaded.</h3><p>Reload this page, or open the <a href="https://github.com/yuxiangwei0808/CONFIRM/tree/demo/visual-guide/demo">guide source and local preview instructions</a>.</p></div>';
  $("#replay-content").textContent =
    "The recorded follow-up could not be loaded. Please reload the page.";
  $("#previous").disabled = true;
  $("#next").disabled = true;
  $("#replay-prev").disabled = true;
  $("#replay-next").disabled = true;
  console.error(error);
}
