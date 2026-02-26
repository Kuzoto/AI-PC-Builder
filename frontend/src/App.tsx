import { useEffect, useMemo, useRef, useState } from "react";
import "./styles.css";

type Mode = "full" | "upgrade";

type PartItem = {
  name: string;
  [key: string]: unknown;
};

type PartCatalog = Record<string, PartItem[]>;

type FormState = {
  cpu: string;
  gpu: string;
  motherboard: string;
  ram: string;
  psu: string;
  storage: string;
  cpuCooler: string;
  monitor: string;
  case: string;
  operatingSystem: string;
  primaryUse: string;
  budget: string;
};

type PartDef = {
  key: keyof FormState;
  label: string;
  file: string;
};

const PRIMARY_USES = [
  "Gaming",
  "Streaming",
  "Video Editing",
  "3D / Rendering",
  "Programming / Dev",
  "School / Office",
  "General Use",
] as const;

const PART_FILES = [
  { key: "cpu", label: "CPU", file: "cpu.json" },
  { key: "gpu", label: "GPU", file: "video-card.json" },
  { key: "motherboard", label: "Motherboard", file: "motherboard.json" },
  { key: "ram", label: "RAM", file: "memory.json" },
  { key: "psu", label: "PSU", file: "power-supply.json" },
  { key: "storage", label: "Storage", file: "internal-hard-drive.json" },
  { key: "cpuCooler", label: "CPU Cooler", file: "cpu-cooler.json" },
  { key: "monitor", label: "Monitor", file: "monitor.json" },
  { key: "case", label: "Case", file: "case.json" },
  { key: "operatingSystem", label: "Operating System", file: "os.json" },
] as const satisfies readonly PartDef[];

type PartKey = (typeof PART_FILES)[number]["key"];

const initialForm: FormState = {
  cpu: "",
  gpu: "",
  motherboard: "",
  ram: "",
  psu: "",
  storage: "",
  cpuCooler: "",
  monitor: "",
  case: "",
  operatingSystem: "",
  primaryUse: PRIMARY_USES[0],
  budget: "",
};

function isPartItem(x: unknown): x is PartItem {
  if (typeof x !== "object" || x === null) return false;
  const rec = x as Record<string, unknown>;
  return typeof rec.name === "string" && rec.name.length > 0;
}

async function loadJsonArray(url: string): Promise<PartItem[]> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  const data: unknown = await res.json();
  if (!Array.isArray(data)) throw new Error(`${url} must be a JSON array`);
  return data.filter(isPartItem);
}

function moneyToNumber(raw: string): number | null {
  const cleaned = raw.replace(/[$, ]/g, "");
  if (cleaned.trim() === "") return null;
  const n = Number(cleaned);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

function normalize(s: string): string {
  return s.trim().toLowerCase();
}

export default function App() {
  const [mode, setMode] = useState<Mode>("full");
  const [catalog, setCatalog] = useState<PartCatalog>({});
  const [form, setForm] = useState<FormState>(initialForm);

  const [openKey, setOpenKey] = useState<PartKey | null>(null);

  // Autocomplete state
  const [query, setQuery] = useState("");
  const [isSuggestOpen, setIsSuggestOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [compatIssues, setCompatIssues] = useState(
    "Compatibility issues will appear here after you run."
  );
  const [aiOutput, setAiOutput] = useState("AI output will appear here after you run.");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadAll() {
      try {
        const entries = await Promise.all(
          PART_FILES.map(async ({ file }) => {
            const items = await loadJsonArray(`/data/${file}`);
            return [file, items] as const;
          })
        );

        if (cancelled) return;

        const next: PartCatalog = {};
        for (const [file, items] of entries) next[file] = items;
        setCatalog(next);
      } catch (err) {
        console.error(err);
        setCompatIssues(
          "Error loading local JSON files. Make sure they exist in /public/data and are valid arrays."
        );
      }
    }

    loadAll();
    return () => {
      cancelled = true;
    };
  }, []);

  const buttonLabel = mode === "full" ? "Generate Build" : "Recommend Upgrade";

  const itemsForOpenKey = useMemo<PartItem[]>(() => {
    if (!openKey) return [];
    const file = PART_FILES.find((p) => p.key === openKey)?.file;
    if (!file) return [];
    return catalog[file] ?? [];
  }, [openKey, catalog]);

  const filteredOptions = useMemo(() => {
    if (!openKey) return [];
    const q = normalize(query);
    if (!q) return [];

    // Filter + cap results so we only render a small list
    const out: Array<{ label: string; value: string }> = [];
    for (const item of itemsForOpenKey) {
      // GPU: search by name OR chipset, and display both
      if (openKey === "gpu") {
        const chipset = typeof item.chipset === "string" ? item.chipset : "";
        const haystack = normalize(`${item.name} ${chipset}`);
        if (haystack.includes(q)) {
          const label = chipset ? `${item.name} — ${chipset}` : item.name;
          out.push({ label, value: item.name });
          if (out.length >= 50) break;
        }
        continue;
      }

      // Default: search by name only
      if (normalize(item.name).includes(q)) {
        out.push({ label: item.name, value: item.name });
        if (out.length >= 50) break; // cap to keep UI snappy
      }
    }
    return out;
  }, [openKey, query, itemsForOpenKey]);

  function update<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function toggleOpen(key: PartKey) {
    setOpenKey((prev) => {
      const next = prev === key ? null : key;
      return next;
    });
  }

  // When opening a new component, reset search UI
  useEffect(() => {
    if (!openKey) {
      setQuery("");
      setIsSuggestOpen(false);
      return;
    }
    // set query to current selection (optional). I prefer blank for searching.
    setQuery("");
    setIsSuggestOpen(false);

    // focus input next tick
    setTimeout(() => inputRef.current?.focus(), 0);
  }, [openKey]);

  function selectOption(value: string, label?: string) {
    if (!openKey) return;
    update(openKey, value);
    setQuery(label ?? value);
    setIsSuggestOpen(false);
  }

  function clearSelection() {
    if (!openKey) return;
    update(openKey, "");
    setQuery("");
    setIsSuggestOpen(false);
    inputRef.current?.focus();
  }

  async function onRun() {
    setIsLoading(true);
    setCompatIssues("Running…");
    setAiOutput("Running…");

    try {
      const budget = moneyToNumber(form.budget);
      if (budget === null) {
        setCompatIssues("Please enter a valid budget (example: 1500).");
        setAiOutput("—");
        return;
      }

      const selected = {
        CPU: form.cpu || "(any)",
        GPU: form.gpu || "(any)",
        Motherboard: form.motherboard || "(any)",
        RAM: form.ram || "(any)",
        PSU: form.psu || "(any)",
        Storage: form.storage || "(any)",
        "CPU Cooler": form.cpuCooler || "(any)",
        Monitor: form.monitor || "(any)",
        Case: form.case || "(any)",
        "Operating System": form.operatingSystem || "(any)",
        "Primary Use": form.primaryUse,
        Budget: `$${budget.toFixed(2)}`,
        Mode: mode === "full" ? "Full PC build" : "Upgrade recommendation",
      };

      setCompatIssues(
        "No compatibility engine connected yet.\n\nSelected:\n" +
          Object.entries(selected)
            .map(([k, v]) => `• ${k}: ${v}`)
            .join("\n")
      );

      setAiOutput(
        mode === "full"
          ? "AI build output will appear here once your backend is hooked up."
          : "AI upgrade recommendations will appear here once your backend is hooked up."
      );
    } finally {
      setIsLoading(false);
    }
  }

  const openLabel = openKey
    ? PART_FILES.find((p) => p.key === openKey)?.label ?? "Component"
    : "";

  const currentSelection = openKey ? form[openKey] : "";

  return (
    <div className="page">
      <header className="header">
        <div className="header__title">AI PC Builder</div>
      </header>

      <main className="content">
        {/* Toggle row */}
        <section className="modeRow">
          <div className="toggle">
            <button
              className={mode === "full" ? "toggle__btn toggle__btn--active" : "toggle__btn"}
              onClick={() => setMode("full")}
              type="button"
            >
              Full PC Build
            </button>
            <button
              className={mode === "upgrade" ? "toggle__btn toggle__btn--active" : "toggle__btn"}
              onClick={() => setMode("upgrade")}
              type="button"
            >
              Upgrade Recommendation
            </button>
          </div>
        </section>

        <section className="grid">
          {/* Left column */}
          <div className="panel">
            <h2 className="panel__title">Selections</h2>

            {/* Component buttons */}
            <div className="componentButtons">
              {PART_FILES.map(({ key, label }) => {
                const selectedName = form[key];

                // For GPU, show "Name — Chipset" on the button (but keep the stored value as the name)
                let displayValue = selectedName;
                if (key === "gpu" && selectedName) {
                  const gpuItems = catalog["video-card.json"] ?? [];
                  const match = gpuItems.find((it) => it.name === selectedName);
                  const chipset = match && typeof match.chipset === "string" ? match.chipset : "";
                  if (chipset) displayValue = `${selectedName} — ${chipset}`;
                }

                return (
                  <button
                    key={key}
                    type="button"
                    className={openKey === key ? "compBtn compBtn--active" : "compBtn"}
                    onClick={() => toggleOpen(key)}
                    title={selectedName ? `Selected: ${displayValue}` : "No selection"}
                  >
                    <span className="compBtn__label">{label}</span>
                    <span className="compBtn__value">{displayValue || "(Any)"}</span>
                  </button>
                );
              })}
            </div>

            {/* Autocomplete panel */}
            {openKey && (
              <div className="dropdownPanel">
                <div className="dropdownPanel__header">
                  <div className="dropdownPanel__title">Select: {openLabel}</div>
                  <div className="dropdownPanel__actions">
                    <button type="button" className="smallBtn" onClick={clearSelection}>
                      Clear
                    </button>
                    <button type="button" className="smallBtn" onClick={() => setOpenKey(null)}>
                      Close
                    </button>
                  </div>
                </div>

                <div className="autoWrap">
                  <input
                    ref={inputRef}
                    className="field__control"
                    placeholder={`Search ${openLabel}...`}
                    value={query}
                    onChange={(e) => {
                      setQuery(e.target.value);
                      setIsSuggestOpen(true);
                    }}
                    onFocus={() => setIsSuggestOpen(true)}
                    onBlur={() => {
                      // allow click selection before closing
                      window.setTimeout(() => setIsSuggestOpen(false), 120);
                    }}
                  />

                  <div className="autoMeta">
                    <span className="muted">
                      Selected: <strong>{currentSelection || "(Any)"}</strong>
                    </span>
                    <span className="muted">
                      {query.trim() ? `Showing up to ${filteredOptions.length} matches` : "Type to search"}
                    </span>
                  </div>

                  {isSuggestOpen && query.trim().length > 0 && (
                    <div className="suggestBox" role="listbox" aria-label="Suggestions">
                      {filteredOptions.length === 0 ? (
                        <div className="suggestEmpty">No matches.</div>
                      ) : (
                        filteredOptions.map((opt) => (
                          <button
                            key={opt.value === opt.label ? opt.value : `${opt.value}__${opt.label}`}
                            type="button"
                            className="suggestItem"
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={() => selectOption(opt.value, opt.label)}
                          >
                            {opt.label}
                          </button>
                        ))
                      )}
                    </div>
                  )}
                </div>

                <div className="dropdownPanel__hint">
                  Tip: results are filtered. Be specific (e.g., “7800X3D”).
                </div>
              </div>
            )}

            {/* Primary use + Budget */}
            <div className="inlineRow">
              <div className="field">
                <label className="field__label" htmlFor="primaryUse">
                  Primary Use
                </label>
                <select
                  id="primaryUse"
                  className="field__control"
                  value={form.primaryUse}
                  onChange={(e) => update("primaryUse", e.target.value)}
                >
                  {PRIMARY_USES.map((u) => (
                    <option key={u} value={u}>
                      {u}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label className="field__label" htmlFor="budget">
                  Budget
                </label>
                <input
                  id="budget"
                  className="field__control"
                  placeholder="e.g. 1500"
                  value={form.budget}
                  onChange={(e) => update("budget", e.target.value)}
                />
              </div>
            </div>

            <div className="actions">
              <button className="primaryBtn" onClick={onRun} disabled={isLoading} type="button">
                {isLoading ? "Working…" : buttonLabel}
              </button>
            </div>
          </div>

          {/* Right column */}
          <div className="panel">
            <h2 className="panel__title">Compatibility Issues</h2>
            <div className="outputBox" role="status" aria-live="polite">
              <pre className="outputBox__pre">{compatIssues}</pre>
            </div>
          </div>
        </section>

        {/* Bottom */}
        <section className="panel">
          <h2 className="panel__title">AI Output</h2>
          <div className="outputBox outputBox--tall">
            <pre className="outputBox__pre">{aiOutput}</pre>
          </div>
        </section>
      </main>
    </div>
  );
}