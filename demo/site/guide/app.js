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
  return `<div class="standard-picker"><div class="standards-label">Choose the required checks</div><div class="standards" role="group" aria-label="Reporting standard">${standards.map((s) => `<button type="button" data-standard="${s}" aria-pressed="${state.standard === s}">${names[s]}</button>`).join("")}</div></div>`;
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
      name: "Search history and repeated tests",
      pass: g.search_provenance && g.multiplicity,
      required: true,
      explanation:
        "The plan records how the question was chosen and how many tests were tried. CONFIRM adjusts the significance cutoff for that count.",
      observed: `Tests counted: ${family} · p ${pValue(r.primary.p)} · cutoff: ${(c.gates.multiplicity.alpha / family).toFixed(3)}`,
    },
    {
      name: "Measured confounds",
      pass: g.confound && g.confound_completeness,
      required: true,
      explanation: g.confound_completeness
        ? "A confound is a factor that could explain the apparent association. The check found no measured factor associated with age that was left out of the plan."
        : "Participants differ in age across scanning sites, and the plan leaves out site. The site difference could explain the apparent link with age.",
      observed: `Accounted for: ${c.covariates.map((name) => (name === "eTIV" ? "head size" : name)).join(", ")}${g.confound_completeness ? "" : " · omitted: site"}`,
    },
    {
      name: "Statistical power",
      pass: g.power,
      required: state.standard === "full",
      explanation:
        "Power is the probability of detecting an effect of a specified size with the planned sample. These examples use an effect size of 0.5, chosen before analysis.",
      observed: `Estimated power: ${(r.power.achieved_power * 100).toFixed(1)}% · required: ${(c.gates.power.min_power * 100).toFixed(0)}%`,
    },
    {
      name: "Stability across analysis choices",
      pass: g.multiverse,
      required: state.standard === "full",
      explanation:
        "CONFIRM repeats the analysis with planned changes to the adjustment variables, handling of extreme values, and statistical model. At least 60% of these alternatives must give a consistent result.",
      observed: `Analyses with a consistent result: ${(r.multiverse.fraction_consistent * 100).toFixed(1)}% · required: ${(c.gates.multiverse.min_fraction_consistent * 100).toFixed(0)}%`,
    },
    {
      name: "Independent replication",
      pass: g.replication,
      required: state.standard !== "screen",
      explanation: g.replication
        ? "The association has the expected direction and passes the significance cutoff in a second, independent sample."
        : "The association does not pass the significance test in the second, independent sample.",
      observed: `Replication p ${pValue(r.replication.cohort_results[0].effect.p)} · required: same direction and p < ${c.gates.replication.alpha}`,
    },
  ];
}

function question(example) {
  return `<div class="two-col"><div class="question-copy"><p class="stage-kicker">Start with a scientific question</p><h3>${esc(example.question)}</h3><p>The first sample is used for the initial analysis (discovery). A second, independent sample tests whether the finding repeats (replication). The plan fixes the analysis choices before either result is evaluated.</p><div class="sample-line"><div><b>600</b><span>In the first sample</span></div><div><b>600</b><span>In the second sample</span></div></div><details class="ground-truth"><summary>How this example was constructed</summary><p>${esc(example.setup)}</p></details></div><div class="question-plot">${scatter(example, "DEMO_DISC", "First sample (discovery)")}<p class="plot-description">The colors identify scanning sites. The plan specifies how to test the association with age.</p></div></div>`;
}

function plan(example) {
  const c = example.contract;
  return `<div class="stage-header"><div><p class="stage-kicker">Before the analysis</p><h3>The plan fixes what will be tested.</h3><p>These examples use prepared plans. In an agent run, the model writes the plan and code checks its required fields and format before analysis begins.</p></div></div><div class="two-col"><div class="plan-block"><h4>What will be tested</h4><div class="plan-row"><span class="plan-label">Predictor</span><span>Age</span></div><div class="plan-row"><span class="plan-label">Measurement</span><span>Synthetic imaging measurement</span></div><div class="plan-row"><span class="plan-label">Direction</span><span>${c.estimand.direction === "negative" ? "Decreases" : "Increases"} with age</span></div><div class="plan-row"><span class="plan-label">Adjust for</span><span class="chips">${c.covariates.map((x) => `<span class="chip">${x === "eTIV" ? "Head size" : esc(x[0].toUpperCase() + x.slice(1))}</span>`).join("")}</span></div><div class="plan-row"><span class="plan-label">Samples</span><span>Independent first and second samples</span></div></div><div class="plan-block"><h4>What the checks require</h4><div class="plan-row"><span class="plan-label">Significance</span><span>p &lt; 0.05, adjusted for the number of tests</span></div><div class="plan-row"><span class="plan-label">Power</span><span>At least 80% for a reference effect of 0.5</span></div><div class="plan-row"><span class="plan-label">Stability</span><span>A consistent result in at least 60% of the planned alternatives</span></div><div class="plan-row"><span class="plan-label">Replication</span><span>Same direction and p &lt; 0.05 in new data</span></div><div class="plan-row"><span class="plan-label">Tests planned</span><span>1 pre-specified hypothesis</span></div></div></div><div class="plan-foot"><span>The downloaded plan records these choices.</span><a href="data/${example.id}-plan.json" download class="text-link">Download the full plan ↓</a></div>`;
}

function results(example) {
  const a = example.results.primary,
    b = example.results.replication.cohort_results[0].effect;
  return `<div class="stage-header"><div><p class="stage-kicker">Analysis results</p><h3>Does the finding appear in new data?</h3><p>Both plots use the same scales. Dots show raw measurements; the effect estimates below account for the variables in the plan.</p></div></div><div class="two-col results-grid">${scatter(example, "DEMO_DISC", "Discovery")}${scatter(example, "DEMO_REP", "Replication")}</div><div class="metrics">${[
    ["Discovery", a],
    ["Replication", b],
  ]
    .map(
      ([label, e]) =>
        `<div><div class="metric-line"><span>${label} effect</span><b>${number(e.beta)}</b><span>p ${esc(pValue(e.p))}</span></div><p class="ci-note">95% confidence interval [${number(e.ci_low)}, ${number(e.ci_high)}]</p></div>`,
    )
    .join(
      "",
    )}</div><p class="plot-description">The effect is the estimated change in the measurement for each year of age, using all 600 observations per sample. The confidence interval shows its uncertainty. The p-value is compared with the significance cutoff in the plan.</p>`;
}

function checks(example) {
  return `<div class="stage-header"><div><p class="stage-kicker">Five evidence checks</p><h3>Open a check to see the evidence.</h3><p>Screen requires the search and confound checks. Replicate adds the test in a new sample. Full requires all five.</p></div>${picker()}</div><div class="gate-list">${gates(
    example,
  )
    .map(
      (g, i) =>
        `<details class="gate-row" ${!g.pass ? "open" : ""}><summary><span class="gate-index">0${i + 1}</span><span class="gate-name">${g.name}</span><span class="gate-required">${g.required ? "Required" : "Not required"}</span>${badge(g.pass)}</summary><div class="gate-detail">${g.explanation}<span class="observed">${esc(g.observed)}</span></div></details>`,
    )
    .join(
      "",
    )}</div><p class="optional-note">CONFIRM computes all five checks. It withholds support if any check required by the chosen standard fails.</p>`;
}

function decision(example) {
  const supported = example.support[state.standard],
    failed = gates(example).filter((g) => g.required && !g.pass);
  const why = supported
    ? state.standard === "screen" && !example.support.full
      ? "The initial analysis passes the checks required by Screen. Replicate and Full withhold support because the finding does not pass the test in the second sample."
      : "All checks required by this reporting standard pass."
    : example.id === "site_confounded"
      ? "The plan leaves out scanning site, which is associated with age. It fails the check for measured confounds."
      : "The finding does not pass the replication test in the second sample, which this reporting standard requires.";
  return `<div class="stage-header"><div><p class="stage-kicker">Reporting decision</p><h3>Does this claim meet the requirements?</h3></div>${picker()}</div><div class="decision-layout"><div class="decision-card ${supported ? "" : "withheld"}"><span class="decision-symbol" aria-hidden="true">${supported ? "✓" : "−"}</span><h3>${supported ? "Supported" : "Withheld"}</h3><p>${why}</p><span class="decision-scope">${names[state.standard]} reporting standard${supported && state.standard === "full" ? " · Confirmed" : ""}</span></div><div><p class="quiet">Decisions under each reporting standard</p><table class="comparison"><thead><tr><th scope="col">Standard</th><th scope="col">Reporting decision</th></tr></thead><tbody>${standards.map((s) => `<tr class="${s === state.standard ? "active" : ""}"><td>${names[s]}</td><td>${decisionBadge(example.support[s])}</td></tr>`).join("")}</tbody></table><p class="decision-note">${failed.length ? `Required check that failed: ${failed.map((g) => g.name.toLowerCase()).join(", ")}.` : "Support means the selected requirements passed."} Choose the standard before looking at the results of a new analysis.</p><p class="decision-note">${esc(example.lesson)}</p></div></div>`;
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
    "CONFIRM · Identify the failed check",
    "Agent · Recorded proposals",
    "CONFIRM · Check the proposed changes",
    "CONFIRM · Test the revised plans",
  ][state.replay];
  const panels = [
    `<h3>The plan leaves out scanning site.</h3><p>The original analysis compares how activity in two brain networks varies together (functional connectivity) between two constructed groups. It accounts for age and sex. The groups also differ by scanning site, which the plan leaves out.</p><div class="replay-callout"><strong>Failed check: measured confounds</strong><br>CONFIRM withholds support and tells the agent that the analysis left out site.</div>`,
    `<h3>The agent proposes two revised plans.</h3><div class="proposal"><span>01</span><div><h4>Account for scanning site</h4><p>Repeat the original group comparison while also accounting for scanning site.</p><div class="chips"><span class="chip">Age</span><span class="chip">Sex</span><span class="chip">+ Site</span></div></div></div><div class="proposal"><span>02</span><div><h4>Test related connectivity measurements</h4><p>Compare the same groups across ${c[1].region_count} related connectivity measurements, accounting for site in each test.</p></div></div>`,
    `<h3>Both revised plans are allowed to run.</h3><p>CONFIRM checks the proposed changes. Both plans:</p><ul class="validation-list"><li>Keep the same group comparison and datasets</li><li>Keep age and sex in the analysis and add site</li><li>Keep the same pass criteria for every check</li><li>Use permitted measurements and an analysis the code can run</li></ul><p>The second plan adds site adjustment and tests a set of related measurements. CONFIRM records both changes. The next step tests whether the results pass the evidence checks.</p>`,
    `<h3>Neither revised analysis receives support.</h3><div class="proposal"><span>01</span><div><h4>Original measurement, accounting for site</h4><p>After accounting for site, p ${esc(pValue(c[0].primary.p))} in the first sample. The group difference is no longer statistically significant.</p>${decisionBadge(c[0].supported)}</div></div><div class="proposal"><span>02</span><div><h4>Related connectivity measurements</h4><p>${c[1].significant_regions} of ${c[1].region_count} measurements pass their significance check.</p>${decisionBadge(c[1].supported)}</div></div><p class="replay-callout">The final correction accounts for all ${followup.final_family_size} tests, including the original analysis. The original claim remains withheld.</p>`,
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
