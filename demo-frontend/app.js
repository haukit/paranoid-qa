const API = "http://localhost:8000" // "https://paranoid-qa.onrender.com";

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
  return `${tokens} tokens · ${cost} · ${secs}`;
}

function addHistory(question, t) {
  history.unshift({ question, t });
  historyEl.innerHTML = history
    .map((h) => `<li>${escapeHtml(h.question)} <small>${h.t ? "$" + h.t.cost_usd.toFixed(4) : ""}</small></li>`)
    .join("");
}

// one-click example questions, one for each path (specific vs aggreagte)
const SUGGESTIONS = [
  "What was the probable cause of the Executive Airlines Flight 5401 accident?",
  "What factors recur across these accident reports?",
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
        addHistory(question, data.telemetry);
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
};

// render the graph nodes as a vertical timeline; show a "running" row until done
function renderPipeline(nodes, done) {
  pipelineEl.innerHTML =
    nodes.map((n) => `<div class="pnode"><strong>${escapeHtml(NODE_LABELS[n.node] || n.node)}</strong> <small>${escapeHtml(nodeSummary(n))}</small></div>`).join("") +
    (done ? "" : `<div class="pnode running">running…</div>`);
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

// compose the answer view
function renderAnswer(payload) {
  answerEl.innerHTML = badgesHtml(payload) + `<p>${escapeHtml(payload.answer)}</p>` + claimsHtml(payload.claims);
  answerEl.hidden = false;
}

function badgesHtml(p) {
  const badges = [];
  if (p.route) badges.push(`<span class="badge">${escapeHtml(p.route)} path</span>`);
  badges.push(p.faithful ? `<span class="badge badge-ok">verified</span>` : `<span class="badge badge-bad">unverified</span>`);
  if (p.attempts > 1) badges.push(`<span class="badge">revised ${p.attempts - 1}×</span>`);
  if (p.telemetry) badges.push(`<span class="badge">${fmtTotals(p.telemetry)}</span>`);
  return `<div class="badges">${badges.join("")}</div>`;
}

function claimsHtml(claims) {
  if (!claims || !claims.length) return "";
  const items = claims.map((c) => {
    const ok = c.verdict === "supported";
    const badge = `<span class="badge ${ok ? "badge-ok" : "badge-bad"}">${escapeHtml(c.verdict)}</span>`;
    const quote = c.quote ? `<blockquote class="quote">${escapeHtml(c.quote)}</blockquote>` : "";
    const cite = c.citation ? `<div class="cite"><a href="#"${c.document ? ` data-doc="${escapeHtml(c.document)}"` : ""}>${escapeHtml(c.citation)}</a></div>` : "";
    const explain = c.explanation ? `<div class="explain">${escapeHtml(c.explanation)}</div>` : "";
    return `<div class="claim"><div class="claim-head"><span>${escapeHtml(c.text)}</span>${badge}</div>${quote}${cite}${explain}</div>`;
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
