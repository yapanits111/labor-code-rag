const REPO_URL = "https://github.com/yapanits111/labor-code-rag";  // set to the GitHub repo URL to show a "GitHub" link in the header

const EXAMPLES = [
  { tag: "Pay", q: "I earn ₱610 a day. What's my pay for working on a regular holiday that's also my rest day?" },
  { tag: "13th month", q: "How is 13th-month pay computed?" },
  { tag: "Leave", q: "How many days of maternity leave do I get?" },
  { tag: "Overtime", q: "How much is overtime pay on a rest day?" },
  { tag: "Dismissal", q: "Can my employer fire me without a valid reason?" },
  { tag: "Renumbering", q: "What does Article 279 say?" },
];
const LOADING_STEPS = ["Searching 700+ passages of the law…", "Reading the most relevant articles…", "Writing a cited answer…"];

const $ = (id) => document.getElementById(id);
const input = $("q");
const askBtn = $("ask-btn");
let current = null;       // last successful response
let lastQuestion = "";
let loadingTimer = null;
let lastFocus = null;

// ---------- setup ----------
if (REPO_URL) { $("repo-link").href = REPO_URL; $("repo-link").hidden = false; }

for (const ex of EXAMPLES) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "example";
  card.innerHTML = `<span class="tag"></span><span class="q"></span>`;
  card.querySelector(".tag").textContent = ex.tag;
  card.querySelector(".q").textContent = ex.q;
  card.addEventListener("click", () => { input.value = ex.q; autoGrow(); ask(ex.q); });
  $("examples").append(card);
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}
input.addEventListener("input", autoGrow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
});
$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q.length >= 3) ask(q);
});
$("retry-btn").addEventListener("click", () => lastQuestion && ask(lastQuestion));
$("new-btn").addEventListener("click", () => {
  document.body.classList.remove("has-result");
  show(null);
  input.value = "";
  input.placeholder = "Ask about Philippine labor law…";
  autoGrow();
  history.replaceState(null, "", location.pathname);
  window.scrollTo({ top: 0 });
  input.focus();
});

// ---------- theme ----------
$("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme
    ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* storage unavailable */ }
});

// ---------- asking ----------
function show(section) {
  for (const id of ["loading", "error", "result"]) $(id).hidden = id !== section;
}

function startLoading() {
  let step = 0;
  $("loading-text").textContent = LOADING_STEPS[0];
  clearInterval(loadingTimer);
  loadingTimer = setInterval(() => {
    step = Math.min(step + 1, LOADING_STEPS.length - 1);
    $("loading-text").textContent = LOADING_STEPS[step];
  }, 900);
}

async function ask(question) {
  lastQuestion = question;
  document.body.classList.add("has-result");
  show("loading");
  startLoading();
  askBtn.disabled = true;
  const url = new URL(location.href);
  url.searchParams.set("q", question);
  history.replaceState(null, "", url);  // shareable link to this question
  window.scrollTo({ top: 0, behavior: "smooth" });

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
      throw new Error(detail || `The server returned an error (${res.status}).`);
    }
    current = data;
    render(question, data);
    input.value = "";  // the question is now the page title; the box is for the next one
    input.placeholder = "Ask another question…";
    autoGrow();
  } catch (err) {
    $("error-text").textContent = err.message === "Failed to fetch"
      ? "Can't reach the server. Check your connection and try again."
      : err.message;
    show("error");
  } finally {
    clearInterval(loadingTimer);
    askBtn.disabled = false;
  }
}

// ---------- rendering ----------
function render(question, data) {
  $("result-question").textContent = question;

  const cited = data.passages.filter((p) => p.cited);
  const others = data.passages.filter((p) => !p.cited);
  $("sources").replaceChildren(...cited.map((p) => sourceCard(p)));
  $("sources-label").hidden = $("sources").hidden = cited.length === 0;
  $("others").replaceChildren(...others.map((p) => sourceCard(p)));
  $("others-box").hidden = others.length === 0;

  $("answer").innerHTML = DOMPurify.sanitize(marked.parse(data.answer));
  linkCitations($("answer"), data.passages);

  // off-topic or too vague: nothing in the law matched, so no AI answer was generated
  $("suggest").hidden = !data.declined;
  if (data.declined) {
    $("others-box").hidden = true;
    $("suggest").replaceChildren(...EXAMPLES.slice(1, 5).map((ex) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.textContent = ex.q;
      chip.addEventListener("click", () => ask(ex.q));
      return chip;
    }));
    $("meta").textContent = "No close match in the Labor Code or the Handbook, so no AI model was called";
  } else {
    $("meta").textContent = `${data.seconds}s · ${data.passages.length} passages checked · ` +
      `answered by ${data.provider} · AI-generated, verify with the sources`;
  }
  show("result");
}

function sourceCard(p) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "source";
  card.id = `source-${p.n}`;
  card.innerHTML = `<span class="n"></span><span class="src"></span><span class="ref"></span><span class="head"></span>`;
  card.querySelector(".n").textContent = p.n;
  card.querySelector(".src").textContent = `${p.source} · p. ${p.pages}`;
  card.querySelector(".ref").textContent = p.ref;
  card.querySelector(".head").textContent = p.heading;
  card.title = p.label;
  card.addEventListener("click", () => openPassage(p.n));
  return card;
}

// Turn "[2]" in the answer into a chip that opens source 2.
function linkCitations(root, passages) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const parts = node.textContent.split(/(\[\d+\])/);
    if (parts.length === 1) continue;
    const frag = document.createDocumentFragment();
    for (const part of parts) {
      const m = part.match(/^\[(\d+)\]$/);
      const p = m && passages[Number(m[1]) - 1];
      if (p) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "cite";
        chip.textContent = p.n;
        chip.title = p.label;
        chip.setAttribute("aria-label", `Source ${p.n}: ${p.label}`);
        chip.addEventListener("click", () => openPassage(p.n));
        frag.append(chip);
      } else {
        frag.append(part);
      }
    }
    node.replaceWith(frag);
  }
}

// ---------- passage drawer ----------
function openPassage(n) {
  const p = current?.passages[n - 1];
  if (!p) return;
  lastFocus = document.activeElement;
  $("drawer-kicker").textContent = `Source ${p.n} · ${p.source}`;
  $("drawer-title").textContent = p.heading ? `${p.ref} · ${p.heading}` : p.ref;
  $("drawer-sub").textContent = `Page ${p.pages}${p.cited ? " · cited in the answer" : " · retrieved, not cited"}`;
  $("drawer-text").textContent = p.text;
  $("drawer-link").href = p.url;
  $("drawer").hidden = false;
  document.body.style.overflow = "hidden";
  $("drawer").querySelector(".drawer-head [data-close]").focus();

  const card = $(`source-${n}`);
  if (card) { card.classList.add("flash"); setTimeout(() => card.classList.remove("flash"), 1200); }
}

function closePassage() {
  $("drawer").hidden = true;
  document.body.style.overflow = "";
  lastFocus?.focus();
}
$("drawer").addEventListener("click", (e) => { if (e.target.closest("[data-close]")) closePassage(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("drawer").hidden) closePassage(); });

// ---------- copy / share ----------
function toast(message) {
  const t = $("toast");
  t.textContent = message;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 1800);
}

async function copy(text, message) {
  try {
    await navigator.clipboard.writeText(text);
    toast(message);
  } catch (e) {
    toast("Couldn't copy. Select the text instead.");
  }
}

$("copy-btn").addEventListener("click", () => {
  if (!current) return;
  const sources = current.passages.filter((p) => p.cited).map((p) => `[${p.n}] ${p.label}`).join("\n");
  copy(`${lastQuestion}\n\n${current.answer}\n\nSources:\n${sources}`, "Answer copied");
});
$("share-btn").addEventListener("click", () => copy(location.href, "Link copied"));

// ---------- open a shared link ----------
const initial = new URLSearchParams(location.search).get("q");
if (initial) {
  input.value = initial;
  autoGrow();
  ask(initial);
}
