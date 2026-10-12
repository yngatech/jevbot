// Rocky's logs: one channel's log, its turns and its memories, from rocky_viewer.py's API. Everything said in the chat
// is put on the page as text (textContent), never as HTML.

const POLL_MS = 4000;
const state = {channel: null, tab: "log", page: 0, tags: {}, data: {}};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) node.append(c);
  return node;
}

// A JSON answer, or null when it hasn't changed since the last (the server's ETag)
async function fetchJSON(url) {
  const r = await fetch(url, {headers: state.tags[url] ? {"If-None-Match": state.tags[url]} : {}});
  if (r.status === 304) return null;
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  state.tags[url] = r.headers.get("ETag");
  return r.json();
}

const time = at => new Date(at).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
const day = at => new Date(at).toLocaleDateString([], {weekday: "short", day: "numeric", month: "short", year: "numeric"});
const stamp = at => new Date(at).toLocaleString();
const snippet = (text, n = 60) => !text ? "" : text.length <= n ? text : text.slice(0, n - 1) + "…";
const cents = cost => cost == null ? null : `${(cost * 100).toFixed(3)}¢`;

function args(call) {
  try { return JSON.parse(call.arguments || "{}"); } catch { return {raw: call.arguments}; }
}

// Its thinking as it came back: reasoning_details, in whichever shapes the provider sent
function thinking(details) {
  return (details || []).map(d => d.text || d.summary || (d.type === "reasoning.encrypted" ? "(encrypted)" : ""))
    .filter(Boolean).join("\n\n").trim();
}

function metrics(t) {
  if (!t) return null;
  const items = [
    t.llm_model,
    cents(t.cost),
    t.llm_prompt_tokens != null && `${t.llm_prompt_tokens.toLocaleString()} in` +
      (t.llm_cached_tokens ? ` (${Math.round(100 * t.llm_cached_tokens / Math.max(1, t.llm_prompt_tokens))}% cached)` : ""),
    t.llm_completion_tokens != null && `${t.llm_completion_tokens.toLocaleString()} out`,
    t.harness_requests > 1 && `${t.harness_requests} requests`,
    t.seconds != null && `${t.seconds}s`,
  ].filter(Boolean);
  return el("div", {class: "metrics"}, items.map(i => el("span", {text: i})),
    t.error ? el("span", {class: "error", text: t.error}) : null);
}

// What a call was about: the message it reacted to, the fact it forgot, what it noted
function target(call, a, ctx) {
  if (call.name === "react") {
    const m = a.message != null ? ctx.messages[a.message] : null;
    return a.message != null ? `on #${a.message}${m ? ` ${m.author}: ${snippet(m.text, 40)}` : ""}` : "on the message that woke it";
  }
  if (call.name === "forget") {
    const f = ctx.facts.find(f => f.n === a.number) || ctx.shown.find(f => f.n === a.number);
    return f ? `[${a.number}] ${f.about}: ${f.fact}` : `[${a.number}]`;
  }
  if (call.name === "remember") return `${a.about}: ${a.fact}`;
  if (call.name === "note") return a.text;
  return call.arguments;
}

function rockyTurn(e, ctx, turn, key) {
  const said = thinking(e.reasoning);
  const seeded = (e.calls || []).some(c => c.id.startsWith("seed_"));
  return el("article", {class: "rocky"},
    el("div", {class: "who"},
      el("span", {class: "name", text: ctx.bot}),
      e.at && el("time", {datetime: e.at, title: stamp(e.at), text: time(e.at)}),
      seeded && el("span", {class: "badge", text: "from history"}),
      e.id && el("span", {class: "dim", text: `message ${e.id}`})),
    said && el("details", {class: "thinking", "data-key": `${key}:thinking`},
      el("summary", {text: `Thinking · ${said.split(/\s+/).length} words`}),
      el("pre", {text: said})),
    e.text && el("p", {class: "text", text: e.text}),
    (e.calls || []).map(c => {
      const a = args(c);
      const result = ctx.results[c.id];
      return el("div", {class: `call call-${c.name}`},
        el("span", {class: "tool", text: c.name}),
        c.name === "react" && el("span", {class: "emoji", text: a.emoji}),
        el("span", {class: "about", text: target(c, a, ctx)}),
        result != null && el("span", {class: "result", text: `→ ${result}`}));
    }),
    !e.text && !(e.calls || []).length && el("p", {class: "dim", text: "(said and did nothing)"}),
    metrics(turn));
}

function context(data) {
  const ctx = {bot: "rocky", messages: {}, results: {}, facts: data.facts || [], shown: data.shown || []};
  for (const e of data.events) {
    if (e.type === "message") ctx.messages[e.n] = e;
    if (e.type === "result") ctx.results[e.id] = e.content;
  }
  return ctx;
}

function onWhat(on, youSaid, ctx) {
  if (on === "you") return `${ctx.bot}'s “${snippet(youSaid, 50)}”`;
  const m = ctx.messages[on];
  return m ? `#${on} ${m.author}: “${snippet(m.text, 50)}”` : `#${on}`;
}

function message(e, ctx) {
  const reply = e.reply_to === "you" ? `↪ ${ctx.bot}: “${snippet(e.you_said, 60)}”`
    : e.reply_to ? `↪ #${e.reply_to}${ctx.messages[e.reply_to] ? ` ${ctx.messages[e.reply_to].author}: “${snippet(ctx.messages[e.reply_to].text, 50)}”` : ""}`
    : null;
  return el("article", {class: `message${e.to_bot ? " to-bot" : ""}${e.woke ? " woke" : ""}${e.gone ? " gone" : ""}`, id: `m${e.n}`},
    el("div", {class: "who"},
      el("span", {class: "n", text: `#${e.n}`}),
      el("span", {class: "name", text: e.author}),
      el("time", {datetime: e.at, title: stamp(e.at), text: time(e.at)}),
      e.to_bot && el("span", {class: "badge", text: "to rocky"}),
      e.woke && el("span", {class: "badge strong", text: "woke it"}),
      e.deleted && el("span", {class: "badge warn", text: e.gone ? "deleted · blanked" : "deleted"})),
    reply && el("div", {class: "reply", text: reply}),
    e.gone ? el("p", {class: "text dim", text: "(blanked at the last tidy)"}) : el("p", {class: "text", text: e.text}));
}

function aside(text, cls = "") {
  return el("div", {class: `aside ${cls}`}, text);
}

function memoryList(facts, others, otherLabel) {
  const known = new Set(others.map(f => f.n));
  return el("ol", {class: "facts"}, facts.map(f => el("li", {class: known.has(f.n) ? "" : "differs"},
    el("span", {class: "n", text: `[${f.n}]`}), el("b", {text: f.about}), " ", el("span", {text: f.fact}),
    !known.has(f.n) && el("span", {class: "badge", text: otherLabel}),
    f.at && el("time", {class: "dim", datetime: f.at, title: stamp(f.at), text: day(f.at)}))));
}

function logView(data) {
  const ctx = context(data);
  const out = [];
  out.push(el("details", {class: "memories-block", "data-key": "memories"},
    el("summary", {text: `<memories> — ${data.shown.length} facts, as Rocky sees them until the next tidy`}),
    data.shown.length ? memoryList(data.shown, ctx.facts, "forgotten since") : el("p", {class: "dim", text: "Nothing yet."})));
  let lastDay = null;
  data.events.forEach((e, i) => {
    if (e.at && day(e.at) !== lastDay && e.type === "message") {
      lastDay = day(e.at);
      out.push(el("div", {class: "day", text: lastDay}));
    }
    if (e.type === "message") out.push(message(e, ctx));
    else if (e.type === "rocky") out.push(rockyTurn(e, ctx, data.turns[e.at], `e${i}`));
    else if (e.type === "reaction")
      out.push(aside([el("b", {text: e.author}), e.removed ? " took off " : " reacted ", el("span", {class: "emoji", text: e.emoji}),
        ` on ${onWhat(e.on, e.you_said, ctx)}`], e.removed ? "removed" : ""));
    else if (e.type === "edit")
      out.push(aside([`✎ #${e.on} edited: `, el("span", {class: "quote", text: e.text})]));
    else if (e.type === "deleted")
      out.push(aside(`🗑 ${onWhat(e.on, e.you_said, ctx)} deleted`, "warn"));
  });
  return out;
}

function turnsView(data, log) {
  const ctx = log ? context(log) : {bot: "rocky", messages: {}, results: {}, facts: [], shown: []};
  const pages = Math.max(1, Math.ceil(data.total / data.page_size));
  const go = page => () => { state.page = page; state.scrollToTop = true; refresh(true); };
  const pager = () => el("div", {class: "pager"},
    el("button", {disabled: state.page === 0, onclick: go(state.page - 1), text: "← Newer"}),
    el("span", {text: `${data.total} turns · page ${state.page + 1} of ${pages}`}),
    el("button", {disabled: state.page + 1 >= pages, onclick: go(state.page + 1), text: "Older →"}));
  const list = data.turns.map((t, i) => {
    const events = t.rocky_events || [];
    const results = {...ctx.results};
    for (const e of events) if (e.type === "result") results[e.id] = e.content;
    const turnCtx = {...ctx, results};
    const rocky = events.filter(e => e.type === "rocky");
    return el("section", {class: "turn"},
      el("div", {class: "woke-by"},
        el("time", {datetime: t.at, title: stamp(t.at), text: `${day(t.at)} ${time(t.at)}`}),
        el("b", {text: t.author}), el("span", {class: "quote", text: t.message})),
      rocky.length ? rocky.map((e, k) => rockyTurn({...e, at: e.at && e.at.replace(" ", "T")}, turnCtx, k === 0 ? t : null, `t${t.at}:${k}`))
        // A turn from before its events were logged: what it said or did, and kept
        : el("div", {}, el("p", {class: "dim", text: t.reply ? `said: ${t.reply}` : t.reaction ? `reacted ${t.reaction}` : "(no events recorded for this turn)"}),
          (t.memory_kept || []).map(k => el("div", {class: "kept", text: `${k[0]} [${k[1]}] ${k[2] ?? ""}`})), metrics(t)));
  });
  return [pager(), ...list, data.turns.length > 5 ? pager() : null].filter(Boolean);
}

function memoriesView(data) {
  return [
    el("h2", {text: "Remembered now"}),
    el("p", {class: "dim", text: `Shared by every channel in this server (${data.space}). Ones not yet in Rocky's <memories> it knows from its remember calls in the log.`}),
    data.facts.length ? memoryList(data.facts, data.shown, "not in <memories> yet") : el("p", {class: "dim", text: "Nothing."}),
    el("h2", {text: "In <memories>"}),
    el("p", {class: "dim", text: "The snapshot before the log, rebuilt at a jump or after an hour without a request."}),
    data.shown.length ? memoryList(data.shown, data.facts, "forgotten since") : el("p", {class: "dim", text: "Nothing."}),
  ];
}

// Redraw, keeping open <details> open and the scroll where it was — or at the bottom of the log, if it was there
function draw(nodes) {
  const view = document.getElementById("view");
  const open = new Set([...view.querySelectorAll("details[open]")].map(d => d.dataset.key));
  const atBottom = window.innerHeight + window.scrollY >= document.body.scrollHeight - 40;
  const y = window.scrollY;
  view.replaceChildren(...nodes);
  for (const d of view.querySelectorAll("details")) if (open.has(d.dataset.key)) d.open = true;
  if (state.tab === "log" && (atBottom || state.scrollToEnd)) window.scrollTo(0, document.body.scrollHeight);
  else window.scrollTo(0, state.scrollToTop ? 0 : y);
  state.scrollToEnd = state.scrollToTop = false;
}

async function refresh(force = false) {
  const status = document.getElementById("status");
  try {
    const list = await fetchJSON("/api/channels");
    if (list) fillChannels(list.channels);
    if (!state.channel) {
      document.getElementById("view").replaceChildren(el("p", {class: "dim", text: "No logs yet: rocky_logs/ is empty."}));
      return;
    }
    const base = `/api/channels/${state.channel}`;
    if (force) delete state.tags[base], delete state.tags[`${base}/turns?page=${state.page}`];
    const log = await fetchJSON(base);
    if (log) state.data.log = log;
    let changed = !!log;
    if (state.tab === "turns") {
      const turns = await fetchJSON(`${base}/turns?page=${state.page}`);
      if (turns) state.data.turns = turns;
      changed = changed || !!turns;
    }
    if (changed || force) {
      if (state.tab === "log") draw(logView(state.data.log));
      else if (state.tab === "turns" && state.data.turns) draw(turnsView(state.data.turns, state.data.log));
      else if (state.tab === "memories") draw(memoriesView(state.data.log));
    }
    status.textContent = `updated ${new Date().toLocaleTimeString()}`;
    status.className = "";
  } catch (err) {
    status.textContent = `couldn't update: ${err.message}`;
    status.className = "error";
  }
}

function fillChannels(channels) {
  const select = document.getElementById("channel");
  const wanted = state.channel || new URLSearchParams(location.hash.slice(1)).get("c") || channels[0]?.id;
  select.replaceChildren(...channels.map(c => el("option", {value: c.id, text: `#${c.channel} · ${c.events} events`})));
  if (channels.some(c => c.id === wanted)) {
    select.value = wanted;
    state.channel = wanted;
  } else {
    state.channel = channels[0]?.id || null;
  }
}

function choose(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll("#tabs button")) b.setAttribute("aria-pressed", b.dataset.tab === tab);
  location.hash = new URLSearchParams({c: state.channel || "", tab}).toString();
  state.scrollToEnd = tab === "log";
  state.scrollToTop = tab !== "log";
  refresh(true);
}

document.addEventListener("DOMContentLoaded", () => {
  const hash = new URLSearchParams(location.hash.slice(1));
  if (["log", "turns", "memories"].includes(hash.get("tab"))) state.tab = hash.get("tab");
  for (const b of document.querySelectorAll("#tabs button")) {
    b.setAttribute("aria-pressed", b.dataset.tab === state.tab);
    b.addEventListener("click", () => choose(b.dataset.tab));
  }
  document.getElementById("channel").addEventListener("change", ev => {
    state.channel = ev.target.value;
    state.page = 0;
    state.data = {};
    choose(state.tab);
  });
  state.scrollToEnd = state.tab === "log";
  refresh(true);
  setInterval(() => { if (!document.hidden) refresh(); }, POLL_MS);
});
