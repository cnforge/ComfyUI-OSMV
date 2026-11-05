// cnforge Studio (on the OneShot MV engine): a "Studio" button in ComfyUI that opens the Studio page in a new tab, and the
// Studio's "Open in ComfyUI" links: /?oneshot=<run>[&job=<take>] opens that run's graph (the
// Director's, or one take's job) in a new workflow tab.
import { app } from "../../scripts/app.js";

const open = () => window.open("/oneshot-mv/", "_blank");

async function openRunGraph() {
  const q = new URLSearchParams(location.search), run = q.get("oneshot");
  if (!run) return;
  const job = q.get("job") || "";
  history.replaceState(null, "", location.pathname + location.hash);  // a reload doesn't open it again
  // setup runs before ComfyUI restores its saved workflow tabs, which would replace the graph:
  // wait for the startup spinner to go (30 s at most)
  for (let i = 0; i < 300 && app.extensionManager?.spinner; i++) await new Promise((r) => setTimeout(r, 100));
  try {
    const r = await fetch(`/oneshot-mv/api/graph/${encodeURIComponent(run)}?job=${encodeURIComponent(job)}`);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    await app.loadApiJson(j.graph, j.name);
  } catch (e) {
    app.extensionManager?.toast?.add({ severity: "warn", summary: "cnforge Studio", detail: `Couldn't open the run's graph: ${e.message}`, life: 8000 });
  }
}

app.registerExtension({
  name: "OneShotMV.Studio",
  commands: [{ id: "OneShotMV.OpenStudio", label: "Open cnforge Studio", icon: "pi pi-video", function: open }],
  menuCommands: [{ path: ["cnforge Studio"], commands: ["OneShotMV.OpenStudio"] }],
  actionBarButtons: [{ icon: "icon-[lucide--clapperboard]", label: "cnforge Studio", tooltip: "Open cnforge Studio", onClick: open }],
  setup() { openRunGraph(); },  // not awaited: startup carries on while it waits
});
