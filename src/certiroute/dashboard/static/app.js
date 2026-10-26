"use strict";
const byId = id => document.getElementById(id);
const percent = (n, digits = 2) => `${(100 * n).toFixed(digits)}%`;
const money = n => `$${n.toFixed(2)}`;
let data;
function text(id, value) { byId(id).textContent = value; }
function option(select, value, label) {
  const element = document.createElement("option"); element.value = value; element.textContent = label; select.append(element);
}
function card(row, title, model, label, featured, strongCost) {
  const element = document.createElement("article"); element.className = `card${featured ? " featured" : ""}`;
  // Structure is static; all result-derived content is assigned via textContent.
  element.innerHTML = '<div class="card-top"><h3></h3><span class="policy-label"></span></div><p class="model"></p><div class="quality"><b></b><span>quality</span></div><div class="small-metrics"><div><strong class="cost"></strong><span>est. cost / 1k requests</span></div><div><strong class="regret"></strong><span>mean clipped regret</span></div></div><div class="cost-track"><div class="cost-fill"></div></div><div class="cost-caption"><span>Cost vs strong-only</span><b></b></div>';
  const set = (selector, value) => { element.querySelector(selector).textContent = value; };
  set("h3", title); set(".policy-label", label); set(".model", model);
  set(".quality b", percent(row.quality)); set(".cost", money(row.cost_per_1000)); set(".regret", percent(row.regret, 3));
  const relative = row.cost_per_1000 / strongCost;
  element.querySelector(".cost-fill").style.width = `${Math.min(100, relative * 100)}%`;
  set(".cost-caption b", `${(relative * 100).toFixed(1)}%`);
  return element;
}
function render() {
  const epsilon = Number(byId("epsilon").value), size = Number(byId("calibration").value);
  const point = data.operating_points.find(r => r.epsilon === epsilon && r.calibration_size === size);
  if (!point) throw new Error("Selected operating point is unavailable.");
  const cheap = data.baselines.find(r => r.policy === "cheap-only");
  const strong = data.baselines.find(r => r.policy === "strong-only");
  text("hero-savings", percent(point.savings));
  byId("cards").replaceChildren(
    card(cheap, "Cheap-only", data.provenance.models.cheap.split("/").pop(), "LOWEST COST", false, strong.cost_per_1000),
    card(strong, "Strong-only", data.provenance.models.strong, "REFERENCE", false, strong.cost_per_1000),
    card(point, "CertiRoute", "Learned score + calibrated threshold", "LEARNED + CRC", true, strong.cost_per_1000),
  );
  text("criterion", `Target ${percent(epsilon, 0)} · Measured regret ${percent(point.regret, 3)} · ${point.non_rejections}/${point.cell_count} training seeds meet non-rejection. This is not proof of validity; ${percent(point.fraction_exceeding_epsilon, 1)} of individual splits exceed the target.`);
  const find = (direction, arm) => data.shift.find(r => r.direction === direction && r.arm === arm && r.epsilon === epsilon);
  const before = find("long-to-short", "shifted"), after = find("long-to-short", "target-recalibrated"), reverse = find("short-to-long", "shifted");
  text("shift-target", percent(epsilon, 0)); text("shift-before", percent(before.mean_regret, 3)); text("shift-after", percent(after.mean_regret, 3));
  text("shift-failure", before.non_rejection ? "Does not reject target" : "Exceeds target · statistically rejected");
  text("shift-recovery", after.mean_within_target ? "Empirical mean within target" : "Empirical mean above target");
  text("shift-note", `Separate study: ${data.shift_study.calibration_size} calibration + ${data.shift_study.test_size} test examples per group, ${data.shift_study.repeats.toLocaleString()} draws, one frozen router. Reverse transfer (short → long): ${percent(reverse.mean_regret, 3)} regret and ${percent(reverse.mean_savings_vs_strong)} savings. Recalibration uses separate labeled target examples.`);
}
async function start() {
  try {
    const response = await fetch("/api/results", { cache: "no-store" });
    data = await response.json(); if (!response.ok) throw new Error(data.error || "Results unavailable.");
    data.epsilons.forEach(e => option(byId("epsilon"), e, percent(e, 0)));
    data.calibration_sizes.forEach(n => option(byId("calibration"), n, n));
    byId("epsilon").value = data.epsilons.includes(.05) ? .05 : data.epsilons[0];
    byId("calibration").value = data.calibration_sizes.includes(500) ? 500 : data.calibration_sizes.at(-1);
    text("example-count", data.provenance.examples.toLocaleString()); text("draw-count", data.provenance.draws.toLocaleString());
    text("seed-count", data.provenance.training_seeds); text("pricing-date", data.provenance.pricing_as_of);
    text("provenance", `Snapshot ${data.provenance.main_sha256.slice(0, 10)} · Shift ${data.provenance.shift_sha256.slice(0, 10)}`);
    render(); byId("content").hidden = false;
    for (const id of ["epsilon", "calibration"]) byId(id).addEventListener("change", () => { try { render(); } catch (error) { fail(error); } });
  } catch (error) { fail(error); }
  finally { byId("loading").hidden = true; }
}
function fail(error) { byId("content").hidden = true; byId("error").hidden = false; text("error", error.message || "Unable to load evaluation results."); }
start();
