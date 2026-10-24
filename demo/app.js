const state = { cases: [], caseId: "invoice", model: "certain" };

const $ = (id) => document.getElementById(id);

function pct(x) {
  if (x === undefined || x === null) return "—";
  return (x * 100).toFixed(1) + "%";
}

function num(x) {
  if (x === undefined || x === null) return "—";
  return Number(x).toFixed(3);
}

async function load() {
  const [cases, board] = await Promise.all([
    fetch("/api/cases").then((r) => r.json()),
    fetch("/api/scoreboard").then((r) => r.json()),
  ]);
  state.cases = cases.cases;
  renderBoard(board);
  renderCaseList();
  selectCase(state.caseId);
}

function renderBoard(board) {
  $("board-meta").textContent = board.mode
    ? `${board.n || 0} held-out problems · ${board.mode}`
    : "no scoreboard yet — run eval/run_audit.py";
  const body = $("board-body");
  body.innerHTML = "";
  const metrics = board.metrics || {};
  const labels = {
    oracle_gold: "Oracle gold certificates",
    constructed_outcome: "Constructed outcome (boxed only)",
    constructed_base: "Constructed weak base",
    certain: "CERTAIN adapter",
    outcome: "Outcome adapter",
    base: "Base model",
  };
  for (const [name, row] of Object.entries(metrics)) {
    const tr = document.createElement("tr");
    const unf = row.unfaithful_correct_rate;
    tr.innerHTML = `
      <td>${labels[name] || name}</td>
      <td>${pct(row.accuracy)}</td>
      <td>${pct(row.faithful_accuracy)}</td>
      <td class="${unf > 0.2 ? "up" : "down"}">${pct(unf)}</td>
      <td>${num(row.mean_certificate_reward)}</td>
      <td>${num(row.mean_outcome_reward)}</td>`;
    body.appendChild(tr);
  }
}

function renderCaseList() {
  const box = $("case-list");
  box.innerHTML = "";
  for (const c of state.cases) {
    const b = document.createElement("button");
    b.textContent = c.title;
    b.className = c.id === state.caseId ? "on" : "";
    b.onclick = () => selectCase(c.id);
    box.appendChild(b);
  }
}

function currentCase() {
  return state.cases.find((c) => c.id === state.caseId);
}

function selectCase(id) {
  state.caseId = id;
  renderCaseList();
  const c = currentCase();
  $("case-title").textContent = c.title;
  $("problem").textContent = c.problem;
  $("narration").textContent = c.narration;
  $("gold").textContent = c.gold;
  $("raw").textContent = "";
  $("lines").innerHTML = "";
  run();
}

async function run() {
  const live = $("live").checked;
  const res = await fetch("/api/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: state.caseId, model: state.model, live }),
  });
  const data = await res.json();
  paint(data);
}

function paint(data) {
  $("raw").textContent = data.text || "";
  $("debits").textContent = data.debits;
  $("credits").textContent = data.credits;
  $("ivalue").textContent = data.value ?? "—";
  $("boxed").textContent = data.boxed_value ?? "—";
  $("creward").textContent = num(data.metrics.certificate_reward);
  $("oreward").textContent = num(data.metrics.outcome_reward);
  $("faithful").textContent = data.metrics.faithful
    ? "yes"
    : data.metrics.unfaithful_correct
      ? "unfaithful-correct"
      : "no";
  const msg = $("books-msg");
  if (data.books_error) {
    msg.textContent = data.books_error;
    msg.className = "books-msg bad";
  } else if (data.books_present && data.books_balanced) {
    msg.textContent = "books balance";
    msg.className = "books-msg ok";
  } else {
    msg.textContent = data.parse_error || (data.undefined || []).map((u) => "undefined: " + u).join(", ");
    msg.className = data.parsed ? "books-msg" : "books-msg bad";
  }

  const ol = $("lines");
  ol.innerHTML = "";
  const rows = (data.lines || []).filter((l) => l.kind !== "blank");
  rows.forEach((line, i) => {
    window.setTimeout(() => {
      const li = document.createElement("li");
      li.className = line.ok ? "ok" : "bad";
      li.innerHTML = `<span>${escapeHtml(line.raw)}</span><span class="err">${escapeHtml(line.error || (line.value ?? ""))}</span>`;
      ol.appendChild(li);
    }, i * 90);
  });
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

$("run").onclick = run;
document.querySelectorAll("#model-pills button").forEach((btn) => {
  btn.onclick = () => {
    state.model = btn.dataset.model;
    document.querySelectorAll("#model-pills button").forEach((b) => b.classList.toggle("on", b === btn));
    run();
  };
});

load();
