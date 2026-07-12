const API = "http://localhost:8000" // "https://paranoid-qa.onrender.com";

// grab the page elements once
const statusEl = document.getElementById("status");
const form = document.getElementById("ask-form");
const questionEl = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const answerEl = document.getElementById("answer");
const historyEl = document.getElementById("history");

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

//
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

// ask a question and render the answer
async function ask(question) {
  askButton.disabled = true;
  statusEl.textContent = "Thinking…";
  answerEl.hidden = true;
  try {
    const res = await fetch(`${API}/ask_json`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Demo-Session": sessionToken },
      body: JSON.stringify({ question }),
    });
    if (res.status === 401) return void (statusEl.textContent = "Session expired. Reload the demo link.");
    if (res.status === 429) return void (statusEl.textContent = "Question limit reached for this session.");
    if (!res.ok) return void (statusEl.textContent = "Something went wrong.");
    const payload = await res.json()
    renderAnswer(payload);
    addHistory(question, payload.telemetry);
    await refreshRemaining();
  } catch {
    statusEl.textContent = "Could not reach the demo backend.";
  } finally {
    askButton.disabled = false;
  }
}

// build the answer HTML from the JSON payload
function renderAnswer(payload) {
  const claims = payload.claims
    .map((c) => `<li>${escapeHtml(c.text)}${c.citation ? ` <small>[${escapeHtml(c.citation)}]</small>` : ""}</li>`)
    .join("");
    answerEl.innerHTML = `
    <p>${escapeHtml(payload.answer)}</p>
    ${claims ? `<ul>${claims}</ul>` : ""}
    <footer>
      <small>${payload.faithful ? "verified against sources" : "could not fully verify"}</small><br />
      <small>${fmtTotals(payload.telemetry)}</small>
    </footer>`;
  answerEl.hidden = false;
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

startSession();
