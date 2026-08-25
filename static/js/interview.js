// Interview room controller.
// Engine modes:
//   local       , WebSocket audio to the s2s pipeline, arc visualizer
//   local_avatar, s2s brain; LiveAvatar video lip-syncs interviewer replies
//   cloud       , LiveAvatar FULL mode voice chat handles everything
import { AudioEngine } from "/static/js/audio.js";

const cfg = window.INTERVIEW;               // { sessionId, engine }
const el = (id) => document.getElementById(id);
const scriptBody = el("script-body");
const statePill = el("state-pill");
const timerEl = el("timer");
const startBtn = el("start-btn");
const muteBtn = el("mute-btn");
const skipBtn = el("skip-btn");
const endBtn = el("end-btn");
const noticeEl = el("room-notice");

let ws = null;
let audio = null;
let avatarSession = null;
let started = false;
let startedAt = null;
let muted = false;
let lines = [];                              // { role, text, node }
let partialNode = null;

// ── UI helpers ──────────────────────────────────────────────────────

function setState(label, cls) {
  statePill.textContent = label;
  statePill.className = "state-pill" + (cls ? " " + cls : "");
}

function notice(kind, msg) {
  if (!msg) { noticeEl.hidden = true; return; }
  noticeEl.hidden = false;
  noticeEl.className = "notice notice-" + kind;
  noticeEl.textContent = msg;
}

function addLine(role, text) {
  const wrap = document.createElement("div");
  wrap.className = "line " + (role === "assistant" ? "interviewer" : "you");
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "assistant" ? "Interviewer" : "You";
  const said = document.createElement("p");
  said.className = "said";
  said.textContent = text;
  wrap.append(who, said);
  const ph = scriptBody.querySelector(".placeholder");
  if (ph) ph.remove();
  scriptBody.appendChild(wrap);
  scriptBody.scrollTop = scriptBody.scrollHeight;
  lines.push({ role, text });
  return said;
}

function showPartial(text) {
  if (!text) return;
  if (!partialNode) {
    const wrap = document.createElement("div");
    wrap.className = "line you partial";
    wrap.innerHTML = '<span class="who">You</span><p class="said"></p>';
    const ph = scriptBody.querySelector(".placeholder");
    if (ph) ph.remove();
    scriptBody.appendChild(wrap);
    partialNode = wrap;
  }
  partialNode.querySelector(".said").textContent = text;
  scriptBody.scrollTop = scriptBody.scrollHeight;
}

function clearPartial() {
  if (partialNode) { partialNode.remove(); partialNode = null; }
}

// timer
setInterval(() => {
  if (!startedAt) return;
  const s = Math.floor((Date.now() - startedAt) / 1000);
  timerEl.textContent =
    String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}, 500);

// ── Visualizer (concentric arcs breathing with audio) ───────────────

const canvas = el("viz");
let level = { in: 0, out: 0 };
if (canvas) {
  const ctx2d = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = 300 * dpr;
  canvas.height = 300 * dpr;
  ctx2d.scale(dpr, dpr);
  const draw = () => {
    ctx2d.clearRect(0, 0, 300, 300);
    const cx = 150, cy = 150;
    const base = 34;
    const amp = Math.min(1, level.in * 2 + level.out * 2);
    for (let ring = 0; ring < 4; ring++) {
      const r = base + ring * 22 + amp * (10 + ring * 8);
      ctx2d.beginPath();
      ctx2d.arc(cx, cy, r, 0, Math.PI * 2);
      ctx2d.strokeStyle =
        level.out > level.in
          ? `rgba(214, 158, 90, ${0.55 - ring * 0.12})`
          : `rgba(158, 190, 168, ${0.55 - ring * 0.12})`;
      ctx2d.lineWidth = 2;
      ctx2d.stroke();
    }
    ctx2d.beginPath();
    ctx2d.arc(cx, cy, base - 12 + amp * 6, 0, Math.PI * 2);
    ctx2d.fillStyle = "rgba(236, 238, 233, 0.9)";
    ctx2d.fill();
    level.in *= 0.92;
    level.out *= 0.92;
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}

// ── Transcript logging to server (cloud + avatar modes) ─────────────

function logToServer(role, text) {
  fetch(`/api/sessions/${cfg.sessionId}/transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, text }),
  }).catch(() => {});
}

// ── Local engine (s2s over WebSocket) ───────────────────────────────

async function startLocal() {
  setState("connecting", "");
  audio = new AudioEngine({ onLevel: (p, dir) => { level[dir] = Math.max(level[dir], p); } });

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/${cfg.sessionId}`);
  ws.binaryType = "arraybuffer";

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      if (cfg.engine === "local") audio.playChunk(e.data);
      return;
    }
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    switch (ev.type) {
      case "status":
        if (ev.state === "starting") {
          notice("wait", ev.message);
          setState("warming up", "");
        } else if (ev.state === "ready") {
          notice(null);
          setState("listening", "listening");
          startedAt = startedAt || Date.now();
        } else if (ev.state === "error") {
          notice("error", ev.message);
          setState("error", "err");
        }
        break;
      case "partial_transcription":
        showPartial((partialNode?.querySelector(".said")?.textContent || "") + ev.delta);
        break;
      case "transcription_completed":
        clearPartial();
        if (ev.transcript) addLine("user", ev.transcript);
        setState("thinking", "speaking");
        break;
      case "assistant_text":
        if (ev.text) {
          addLine("assistant", ev.text);
          setState("speaking", "speaking");
          if (cfg.engine === "local_avatar" && avatarSession) {
            avatarSession.repeat(ev.text);
          }
          watchForIdle();
        }
        break;
    }
  };
  ws.onclose = () => { if (started) setState("disconnected", "err"); };

  await audio.start((chunk) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(chunk);
  });
}

let idleWatch = null;
function watchForIdle() {
  clearInterval(idleWatch);
  idleWatch = setInterval(() => {
    const avatarBusy = cfg.engine === "local_avatar" && avatarSpeaking;
    if (!avatarBusy && (!audio || !audio.isPlaying())) {
      setState("listening", "listening");
      clearInterval(idleWatch);
    }
  }, 300);
}

// ── LiveAvatar (shared by local_avatar + cloud) ─────────────────────

let avatarSpeaking = false;

async function startAvatar() {
  const resp = await fetch("/api/avatar/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: cfg.sessionId }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || "Couldn't start the avatar session.");

  const SDK = window.LiveAvatarSDK;
  avatarSession = new SDK.LiveAvatarSession(data.session_token, {
    voiceChat: cfg.engine === "cloud",
  });

  avatarSession.on(SDK.SessionEvent.SESSION_STREAM_READY, () => {
    avatarSession.attach(el("avatar-video"));
    el("avatar-video").hidden = false;
    const viz = el("viz-wrap");
    if (viz) viz.hidden = true;
  });
  avatarSession.on(SDK.AgentEventsEnum.AVATAR_SPEAK_STARTED, () => {
    avatarSpeaking = true;
    setState("speaking", "speaking");
  });
  avatarSession.on(SDK.AgentEventsEnum.AVATAR_SPEAK_ENDED, () => {
    avatarSpeaking = false;
    setState("listening", "listening");
  });
  if (cfg.engine === "cloud") {
    avatarSession.on(SDK.AgentEventsEnum.USER_TRANSCRIPTION, (ev) => {
      if (ev.text) { addLine("user", ev.text); logToServer("user", ev.text); }
    });
    avatarSession.on(SDK.AgentEventsEnum.AVATAR_TRANSCRIPTION, (ev) => {
      if (ev.text) { addLine("assistant", ev.text); logToServer("assistant", ev.text); }
    });
  }
  avatarSession.on(SDK.SessionEvent.SESSION_DISCONNECTED, () => {
    if (started) setState("disconnected", "err");
  });

  await avatarSession.start();
  if (cfg.engine === "local_avatar") {
    try { avatarSession.stopListening(); } catch {}
  }
  startedAt = startedAt || Date.now();
  setState("listening", "listening");
}

// ── Lifecycle ───────────────────────────────────────────────────────

startBtn.addEventListener("click", async () => {
  if (started) return;
  started = true;
  startBtn.disabled = true;
  notice(null);
  try {
    if (cfg.engine === "cloud") {
      setState("connecting", "");
      await startAvatar();
    } else if (cfg.engine === "local_avatar") {
      setState("connecting", "");
      await startAvatar();
      await startLocal();
    } else {
      await startLocal();
    }
    startBtn.hidden = true;
    muteBtn.hidden = false;
    endBtn.hidden = false;
    if (cfg.engine === "local") skipBtn.hidden = false;
  } catch (err) {
    started = false;
    startBtn.disabled = false;
    notice("error", err.message || String(err));
    setState("error", "err");
  }
});

skipBtn.addEventListener("click", () => {
  if (audio) audio.stopPlayback();
});

muteBtn.addEventListener("click", () => {
  muted = !muted;
  if (audio) audio.setMuted(muted);
  if (avatarSession && cfg.engine === "cloud") {
    try {
      muted ? avatarSession.voiceChat.mute() : avatarSession.voiceChat.unmute();
    } catch {}
  }
  muteBtn.textContent = muted ? "Unmute mic" : "Mute mic";
});

endBtn.addEventListener("click", async () => {
  endBtn.disabled = true;
  setState("wrapping up", "");
  try { if (ws) ws.close(); } catch {}
  try { if (audio) audio.stop(); } catch {}
  try { if (avatarSession) await avatarSession.stop(); } catch {}
  await fetch(`/api/sessions/${cfg.sessionId}/end`, { method: "POST" }).catch(() => {});
  location.href = `/feedback?session=${cfg.sessionId}`;
});
