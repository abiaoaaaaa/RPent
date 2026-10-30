const $ = (id) => document.getElementById(id);
const state = { history: [], selected: null, imageUrl: null, joint: null };
const format = (value) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(4);

function drawCrops(url) {
  const image = new Image();
  image.onload = () => {
    const crops = { top: [0, 0, image.width, image.height * 2 / 5], left: [0, image.height * 2 / 5, image.width / 2, image.height / 5], right: [image.width / 2, image.height * 2 / 5, image.width / 2, image.height / 5] };
    for (const [id, [sx, sy, sw, sh]] of Object.entries(crops)) {
      const canvas = $(id), scale = canvas.clientWidth / sw;
      canvas.width = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
      canvas.height = Math.max(1, Math.round(canvas.clientWidth * sh / sw * devicePixelRatio));
      const ctx = canvas.getContext("2d"); ctx.imageSmoothingEnabled = true;
      ctx.drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    }
  };
  image.src = url;
}

function redrawChart() {
  const canvas = $("chart"), rect = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr)); canvas.height = Math.round(220 * dpr);
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); const w = rect.width, h = 220;
  ctx.clearRect(0, 0, w, h); ctx.strokeStyle = "#303944"; ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) { const y = i * h / 5; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
  const rows = state.history.map((sample) => sample.joint_rows?.find((row) => row.name === state.joint)?.position).filter((v) => Number.isFinite(Number(v))).map(Number);
  if (!rows.length) return;
  const min = Math.min(...rows), max = Math.max(...rows), span = max - min || 1; ctx.strokeStyle = "#58d6a3"; ctx.lineWidth = 2; ctx.beginPath();
  rows.forEach((value, index) => { const x = rows.length === 1 ? 0 : index * w / (rows.length - 1); const y = h - 12 - (value - min) / span * (h - 28); index ? ctx.lineTo(x,y) : ctx.moveTo(x,y); }); ctx.stroke();
  ctx.fillStyle = "#9ca8b6"; ctx.font = "11px system-ui"; ctx.fillText(`${state.joint}: ${format(rows.at(-1))}`, 8, 16); ctx.fillText(`min ${format(min)}  max ${format(max)}`, 8, h - 5);
}

function render(sample) {
  const rows = sample?.joint_rows || []; if (!rows.length) return;
  $("sampleMeta").textContent = `seq ${sample.sequence} · ${new Date(sample.captured_at * 1000).toLocaleTimeString()}`;
  const select = $("jointSelect"); if (!select.options.length) { rows.forEach((row) => select.add(new Option(row.name, row.name))); state.joint = rows[0].name; select.value = state.joint; }
  if (!state.joint) state.joint = select.value;
  $("jointTable").innerHTML = rows.map((row) => `<tr><td>${row.name}</td><td>${format(row.position)}</td><td>${format(row.velocity)}</td></tr>`).join("");
  $("extra").textContent = JSON.stringify({ eef_left: sample.eef_left, eef_right: sample.eef_right, base_world_pose: sample.base_world_pose, gripper_left: sample.gripper_left, gripper_right: sample.gripper_right, control_mode: sample.control_mode, intervention: sample.intervention }, null, 2);
  redrawChart();
}

function renderTimeline() {
  $("timeline").innerHTML = state.history.map((sample) => `<button class="${sample.sequence === state.selected ? "selected" : ""}" data-seq="${sample.sequence}" title="${new Date(sample.captured_at * 1000).toLocaleTimeString()}">${sample.sequence}</button>`).join("");
  $("timeline").querySelectorAll("button").forEach((button) => button.onclick = () => selectFrame(Number(button.dataset.seq)));
}

function selectFrame(sequence) {
  const sample = state.history.find((item) => item.sequence === sequence) || state.history.at(-1); if (!sample) return;
  state.selected = sample.sequence; render(sample); renderTimeline();
  state.imageUrl = `/api/frame/${sample.sequence}.jpg?cache=${Date.now()}`; drawCrops(state.imageUrl);
  $("frameMeta").textContent = `frame ${sample.sequence} · ${sample.jpeg_size ? Math.round(sample.jpeg_size / 1024) + " KB" : "—"}`;
}

async function poll() {
  try {
    const response = await fetch(`/api/status?now=${Date.now()}`, { cache: "no-store" }), payload = await response.json();
    state.history = payload.history || []; const latest = payload.latest;
    $("status").textContent = payload.error ? `连接异常：${payload.error}` : (latest ? `在线 · ${payload.history_size}/30 帧` : "等待第一帧…");
    $("status").className = `status ${payload.error || !latest ? "stale" : "ok"}`; $("endpoint").textContent = payload.base_url || "—";
    if (latest && (!state.history.some((item) => item.sequence === state.selected) || state.selected === state.history.at(-2)?.sequence)) state.selected = latest.sequence;
    if (latest && state.selected === latest.sequence) selectFrame(latest.sequence); else { renderTimeline(); redrawChart(); }
  } catch (error) { $("status").textContent = `监视器错误：${error}`; $("status").className = "status stale"; }
}

$("jointSelect").onchange = (event) => { state.joint = event.target.value; redrawChart(); };
window.addEventListener("resize", redrawChart);
let polling = false;
async function tick() { if (!polling) { polling = true; await poll(); polling = false; } setTimeout(tick, 33); }
tick();
