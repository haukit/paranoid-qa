const API = "https://paranoid-qa.onrender.com";

// grab the page elements once
const statusEl = document.getElementById("status");
const form = document.getElementById("ask-form");
const questionEl = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const answerEl = document.getElementById("answer");
const historyEl = document.getElementById("history");
const suggestionsEl = document.getElementById("suggestions");
const corpusEl = document.getElementById("corpus");
const docModal = document.getElementById("doc-modal");
const docTitle = document.getElementById("doc-title");
const docText = document.getElementById("doc-text");
const pipelineEl = document.getElementById("pipeline");

let sessionToken = null;

// read the invite code from the URL, e.g. ?k=abc
function inviteFromUrl() {
  return new URLSearchParams(window.location.search).get("k");
}

// on load: trade the invite code for a session
async function startSession() {
  const invite = inviteFromUrl();
  if (!invite) {
    statusEl.textContent = "No invite code in the link.";
    return;
  }
  statusEl.textContent = "Connecting to the demo…";
  try {
    const res = await fetch(`${API}/demo/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: invite }),
    });
    if (!res.ok) {
      statusEl.textContent = res.status === 401 ? "Invalid or expired invite code." : "Could not start a session.";
      return;
    }
    const data = await res.json();
    sessionToken = data.session;
    statusEl.textContent = `Ready! ${data.questions} questions remaining.`;
  } catch {
    statusEl.textContent = "Could not reach the demo backend.";
  }
}

// session-only log of asked questions with their cost, newest first
const history = [];

function fmtTotals(t) {
  if (!t) return "";
  const tokens = (t.tokens_in + t.tokens_out).toLocaleString();
  const cost = "$" + t.cost_usd.toFixed(4);
  const secs = (t.latency_ms / 1000).toFixed(1) + "s";
  return `<i class="ti ti-arrows-left-right"></i> ${tokens} tokens · <i class="ti ti-coin"></i> ${cost} · <i class="ti ti-clock"></i> ${secs}`;
}

// render a history entry as a card: question, a snippet of the answer, verified badge and cost
function addHistory(question, payload) {
  history.unshift({ question, payload });
  historyEl.innerHTML = history
    .map((h) => {
      const p = h.payload || {};
      const cost = p.telemetry ? "$" + p.telemetry.cost_usd.toFixed(4) : "";
      const abstained = p.status === "abstained";
      const badge = abstained
        ? `<span class="badge badge-abstain"><i class="ti ti-hand-stop"></i> abstained</span>`
        : `<span class="badge badge-ok"><i class="ti ti-shield-check"></i> verified</span>`;
      const answerLine = abstained ? "No verified answer in the corpus." : truncate(p.answer || "", 140);
      return `<li class="hist">
        <div class="hist-q">${escapeHtml(h.question)}</div>
        <div class="hist-a">${escapeHtml(answerLine)}</div>
        <div class="hist-meta">${badge}${cost ? ` <span class="badge"><i class="ti ti-coin"></i> ${cost}</span>` : ""}</div>
      </li>`;
    })
    .join("");
}

// truncate a string to n characters with an ellipsis
function truncate(s, n) {
  return s.length > n ? s.slice(0, n).trimEnd() + "…" : s;
}

// one-click examples: a specific-path answer, an aggregate-path answer, and an
// unanswerable question (a flight not in the corpus) that triggers an abstention
const SUGGESTIONS = [
  "What was the probable cause of the Executive Airlines Flight 5401 accident?",
  "What factors recur across these accident reports?",
  "What was the probable cause of the Trans-Global Airlines Flight 88 crash?",
];

function renderSuggestions() {
  suggestionsEl.innerHTML = SUGGESTIONS.map(
    (q) => `<button type="button" class="chip secondary outline">${escapeHtml(q)}</button>`
  ).join("");
  suggestionsEl.querySelectorAll(".chip").forEach((el, i) => {
    el.addEventListener("click", () => {
      questionEl.value = SUGGESTIONS[i];
      if (sessionToken) ask(SUGGESTIONS[i]);
    });
  });
}


// ask a question, stream the graph's progress, then render the answer
async function ask(question) {
  askButton.disabled = true;
  statusEl.textContent = "Thinking…";
  answerEl.hidden = true;
  pipelineEl.innerHTML = "";
  const nodes = [];
  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Demo-Session": sessionToken },
      body: JSON.stringify({ question }),
    });
    if (res.status === 401) return void (statusEl.textContent = "Session expired. Reload the demo link.");
    if (res.status === 429) return void (statusEl.textContent = "Question limit reached for this session.");
    if (!res.ok || !res.body) return void (statusEl.textContent = "Something went wrong.");
    await readSSE(res.body, (event, data) => {
      if (event === "progress") {
        nodes.push(data);
        renderPipeline(nodes, false);
      } else if (event === "answer") {
        renderPipeline(nodes, true);
        renderAnswer(data);
        addHistory(question, data);
      }
    });
    await refreshRemaining();
  } catch {
    statusEl.textContent = "Could not reach the demo backend.";
  } finally {
    askButton.disabled = false;
  }
}

// read a Server-Sent Events stream, invoking cb(event, data) for each frame
async function readSSE(stream, cb) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let i;
    while ((i = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, i);
      buffer = buffer.slice(i + 2);
      let event = "message", data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) cb(event, JSON.parse(data));
    }
  }
}

// user-friendly description of what each graph node is doing
const NODE_LABELS = {
  route: "Deciding whether the question is specific or aggregate",
  retrieve: "Retrieving the most relevant passages",
  grade: "Checking whether the passages are relevant",
  rewrite: "Rephrasing the question for better retrieval",
  generate: "Drafting a grounded answer",
  verify: "Verifying each claim against its source",
  aggregate: "Searching the corpus knowledge graph",
  verify_aggregate: "Verifying the answer against its sources",
  accept: "Answer verified, accepting",
  abstain: "No grounded answer, abstaining",
};

// render the graph nodes as a vertical timeline; show a "running" row until done
function renderPipeline(nodes, done) {
  pipelineEl.innerHTML =
    nodes.map((n) => `<div class="pnode"><strong>${escapeHtml(NODE_LABELS[n.node] || n.node)}</strong> <small>${escapeHtml(nodeSummary(n))}</small>${metricsLine(n.metrics)}</div>`).join("") +
    (done ? "" : `<div class="pnode running">running…</div>`);
}

// per-node model / tokens / cost / time line (only the parts a node actually produced)
function metricsLine(m) {
  if (!m) return "";
  const parts = [];
  if (m.models && m.models.length) parts.push(`<i class="ti ti-cpu"></i> ${escapeHtml([...new Set(m.models.map(cleanModel))].join(", "))}`);
  if (m.tokens_in || m.tokens_out) parts.push(`<i class="ti ti-arrows-left-right"></i> ${m.tokens_in} in · ${m.tokens_out} out`);
  if (m.cost_usd) parts.push(`<i class="ti ti-coin"></i> $${m.cost_usd.toFixed(5)}`);
  if (m.ms != null) parts.push(`<i class="ti ti-clock"></i> ${m.ms} ms`);
  return parts.length ? `<div class="pmetrics">${parts.join(" · ")}</div>` : "";
}

// strip an OpenAI date suffix for display, e.g. gpt-4o-mini-2024-07-18 -> gpt-4o-mini
function cleanModel(name) {
  return name.replace(/-\d{4}-\d{2}-\d{2}$/, "");
}

// short human summary of a node from the fields the progress frame carries
function nodeSummary(n) {
  if (n.route) return `→ ${n.route}`;
  if (n.chunks != null) return `${n.chunks} passages`;
  if (n.verdicts) return n.verdicts.join(", ");
  if (n.claims != null) return `${n.claims} claims`;
  if (n.grade) return n.grade;
  if (n.attempts != null) return `attempt ${n.attempts}`;
  return "";
}

// compose the answer view: badges, cost disclaimer, prose answer, verified-claims panel.
// an abstention renders as a positive "no verified answer" result instead of a prose answer.
function renderAnswer(payload) {
  if (payload.status === "abstained") {
    answerEl.innerHTML = badgesHtml(payload) + abstainedHtml() + costNote();
    answerEl.hidden = false;
    return;
  }
  answerEl.innerHTML =
    badgesHtml(payload) +
    costNote() +
    `<p>${escapeHtml(payload.answer)}</p>` +
    claimsHtml(payload.claims);
  answerEl.hidden = false;
}

function costNote() {
  return `<p class="cost-note"><small>Reported cost excludes query embedding, which is negligible.</small></p>`;
}

// the abstention panel: the system checked and declined rather than emit an unsupported claim
function abstainedHtml() {
  return `<div class="abstain">
    <div class="abstain-title"><i class="ti ti-hand-stop"></i> No verified answer</div>
    <p>The verifier was unable to match the drafted answer to the source documents.</p>
  </div>`;
}

// route, outcome (verified vs abstained), revised count, and the cost/token totals
function badgesHtml(p) {
  const badges = [];
  if (p.route) badges.push(`<span class="badge badge-route"><i class="ti ti-route"></i> ${escapeHtml(p.route)} path</span>`);
  badges.push(p.status === "abstained"
    ? `<span class="badge badge-abstain"><i class="ti ti-hand-stop"></i> abstained</span>`
    : `<span class="badge badge-ok"><i class="ti ti-shield-check"></i> verified</span>`);
  if (p.attempts > 1) badges.push(`<span class="badge"><i class="ti ti-refresh"></i> revised ${p.attempts - 1}×</span>`);
  if (p.telemetry) badges.push(`<span class="badge">${fmtTotals(p.telemetry)}</span>`);
  return `<div class="badges">${badges.join("")}</div>`;
}

// one card per claim: text, quote, citation, and a nested verifier card (verdict + explanation)
function claimsHtml(claims) {
  if (!claims || !claims.length) return "";
  const items = claims.map((c) => {
    const ok = c.verdict === "supported";
    const quote = c.quote ? `<blockquote class="quote">${escapeHtml(c.quote)}</blockquote>` : "";
    const cite = c.citation ? `<div class="cite"><i class="ti ti-file-text"></i> <a href="#"${c.document ? ` data-doc="${escapeHtml(c.document)}"` : ""}>${escapeHtml(c.citation)}</a></div>` : "";
    const badge = `<span class="badge ${ok ? "badge-ok" : "badge-bad"}"><i class="ti ti-${ok ? "check" : "x"}"></i> ${escapeHtml(c.verdict)}</span>`;
    const verifier = `<div class="verifier">
        <div class="verifier-title"><i class="ti ti-gavel"></i> Verifier ${badge}</div>
        ${c.explanation ? `<div class="explain">${escapeHtml(c.explanation)}</div>` : ""}
      </div>`;
    return `<div class="claim"><div>${escapeHtml(c.text)}</div>${quote}${cite}${verifier}</div>`;
  }).join("");
  return `<h4>Verified claims</h4>${items}`;
}

// refresh the remaining-questions count (read-only, no charge)
async function refreshRemaining() {
  try {
    const res = await fetch(`${API}/demo/session`, { headers: { "X-Demo-Session": sessionToken } });
    if (res.ok) statusEl.textContent = `${(await res.json()).remaining} questions left.`;
  } catch {}
}

// neutralize any HTML in model/user text before inserting it
function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = questionEl.value.trim();
  if (q && sessionToken) ask(q);
});

// fetch the corpus document list into the sidebar (public, no session needed)
async function loadCorpus() {
  try {
    const res = await fetch(`${API}/corpus`);
    if (!res.ok) return;
    const docs = (await res.json()).documents || [];
    corpusEl.innerHTML = docs
      .map((d) => `<li><a href="#" data-doc="${escapeHtml(d)}">${escapeHtml(d)}</a></li>`)
      .join("");
  } catch {}
}

// fetch a document's extracted text and show it in the modal
async function viewSource(name) {
  docTitle.textContent = name;
  docText.textContent = "Loading…";
  docModal.showModal();
  try {
    const res = await fetch(`${API}/sources/${encodeURIComponent(name)}`);
    docText.textContent = res.ok ? (await res.json()).text : "Could not load this document.";
  } catch {
    docText.textContent = "Could not load this document.";
  }
}

// any element with data-doc (a corpus item or a claim citation) opens its source
document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-doc]");
  if (el) {
    e.preventDefault();
    viewSource(el.getAttribute("data-doc"));
  }
});
document.getElementById("doc-close").addEventListener("click", () => docModal.close());

startSession();
renderSuggestions();
loadCorpus();
