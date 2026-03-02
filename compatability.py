#!/usr/bin/env python3
"""
PC Build Advisor
----------------
1. Prompts the user for PC component preferences and budget.
2. Searches the local pc-part-dataset (cloned from github.com/docyx/pc-part-dataset)
   to find matching parts and run a rule-based compatibility check.
3. Passes everything to GPT-4o for final recommendations.

Setup:
    pip install openai python-dotenv
    git clone https://github.com/docyx/pc-part-dataset
    # Place this script next to the cloned repo, or set DATASET_DIR env var.

Note on pricing:
    Prices in the dataset were scraped from PCPartPicker at a point in time and
    may be outdated. GPT-4o is instructed to acknowledge this.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
# Override with env var DATASET_DIR if the repo is elsewhere
DATASET_DIR = Path(os.environ.get("DATASET_DIR", _SCRIPT_DIR / "pc-part-dataset")) / "data" / "json"

# Maps user-facing component names to dataset JSON filenames (without .json)
COMPONENT_FILES: dict[str, str] = {
    "CPU":                "cpu",
    "CPU Cooler":         "cpu-cooler",
    "Motherboard":        "motherboard",
    "Memory (RAM)":       "memory",
    "Storage":            "internal-hard-drive",
    "Video Card (GPU)":   "video-card",
    "Case":               "case",
    "Power Supply (PSU)": "power-supply",
    "Operating System":   "os",
    "Monitor":            "monitor",
}

MAX_SEARCH_RESULTS = 5


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PartMatch:
    name:  str
    price: Optional[float]       # None if unlisted in dataset
    data:  dict[str, Any]        # full raw record from JSON


@dataclass
class ComponentSearch:
    category:        str
    user_preference: str
    matches:         list[PartMatch] = field(default_factory=list)
    error:           Optional[str]   = None


ERROR   = "ERROR"
WARNING = "WARNING"
INFO    = "INFO"


@dataclass
class CompatibilityIssue:
    severity:   str          # ERROR | WARNING | INFO
    components: list[str]
    message:    str


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Loads and caches JSON files from the cloned pc-part-dataset repo."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self._cache: dict[str, list[dict]] = {}

    def load(self, filename: str) -> list[dict]:
        if filename in self._cache:
            return self._cache[filename]

        path = self.dataset_dir / f"{filename}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                f"Clone https://github.com/docyx/pc-part-dataset next to this script,\n"
                f"or set the DATASET_DIR environment variable."
            )

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        records: list[dict] = (
            raw if isinstance(raw, list)
            else raw.get("data", raw.get("parts", list(raw.values())[0] if raw else []))
        )
        self._cache[filename] = records
        return records

    def search(self, filename: str, query: str) -> list[PartMatch]:
        """
        Case-insensitive token search over part names.
        Returns up to MAX_SEARCH_RESULTS matches sorted by relevance then price.
        """
        if not query.strip():
            return []

        records    = self.load(filename)
        q_lower    = query.lower().strip()
        tokens     = q_lower.split()
        scored: list[tuple[int, PartMatch]] = []

        for rec in records:
            name       = rec.get("name", "")
            name_lower = name.lower()
            if all(t in name_lower for t in tokens):
                score     = 2 if name_lower.startswith(q_lower) else 1
                price_raw = rec.get("price")
                price     = float(price_raw) if price_raw is not None else None
                scored.append((score, PartMatch(name=name, price=price, data=rec)))

        scored.sort(key=lambda x: (-x[0], x[1].price if x[1].price is not None else 1e9))
        return [m for _, m in scored[:MAX_SEARCH_RESULTS]]

    def top(self, filename: str) -> list[PartMatch]:
        """Return the first MAX_SEARCH_RESULTS records (used when user has no preference)."""
        records = self.load(filename)
        results = []
        for rec in records[:MAX_SEARCH_RESULTS]:
            price_raw = rec.get("price")
            price     = float(price_raw) if price_raw is not None else None
            results.append(PartMatch(name=rec.get("name", "Unknown"), price=price, data=rec))
        return results


# ---------------------------------------------------------------------------
# Compatibility checker
# ---------------------------------------------------------------------------

def _ddr_gen_from_speed(speed_field: Any) -> Optional[int]:
    """
    Extract DDR generation from the dataset speed field.
    The pc-part-dataset stores memory speed as [ddr_version, mhz], e.g. [4, 3200].
    Also handles plain strings like "DDR4-3200" or "DDR5".
    """
    if isinstance(speed_field, (list, tuple)) and len(speed_field) >= 1:
        try:
            return int(speed_field[0])
        except (TypeError, ValueError):
            pass
    # Fallback: parse "DDR4", "DDR5", "DDR3-1600", etc. from a string
    m = re.search(r"ddr\s*(\d)", str(speed_field).lower())
    return int(m.group(1)) if m else None

def _ddr_gen_from_str(text: str) -> Optional[int]:
    """Extract DDR generation from a freeform string (names, user preferences)."""
    m = re.search(r"ddr\s*(\d)", str(text).lower())
    return int(m.group(1)) if m else None

def _norm(s: str) -> str:
    return re.sub(r"[\s\-_]", "", str(s).lower())

def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class CompatibilityChecker:
    """
    Rule-based, offline compatibility checker.
    Uses spec fields from the pc-part-dataset records.
    """

    _PSU_OVERHEAD = 1.20    # 20 % headroom above component draw
    _PLATFORM_W   = 75      # baseline watts for MB + RAM + storage + fans

    def __init__(self, searches: list[ComponentSearch]):
        self._searches = searches   # kept for preference string fallback in checks
        self._best: dict[str, Optional[PartMatch]] = {
            cs.category: (cs.matches[0] if cs.matches else None)
            for cs in searches
        }
        self.issues: list[CompatibilityIssue] = []

    def run_all(self) -> list[CompatibilityIssue]:
        self.issues = []
        self._cpu_mb_socket()
        self._cpu_mb_chipset()
        self._ram_type()
        self._ram_slots()
        self._case_form_factor()
        self._psu_wattage()
        self._cooler_socket()
        self._cooler_tdp()
        self._storage_m2()
        return self.issues

    # -- helpers --------------------------------------------------------------

    def _f(self, cat: str, *keys: str) -> Optional[Any]:
        p = self._best.get(cat)
        if not p:
            return None
        for k in keys:
            v = p.data.get(k)
            if v is not None and v != "":
                return v
        return None

    def _add(self, sev: str, comps: list[str], msg: str) -> None:
        self.issues.append(CompatibilityIssue(sev, comps, msg))

    # -- checks ---------------------------------------------------------------

    def _cpu_mb_socket(self) -> None:
        cpu_s = self._f("CPU", "socket")
        mb_s  = self._f("Motherboard", "socket")
        if not cpu_s or not mb_s:
            if cpu_s or mb_s:
                self._add(INFO, ["CPU", "Motherboard"],
                          "Could not verify socket — one part is missing socket data. Check manually.")
            return
        cpu_n  = _norm(str(cpu_s))
        mb_set = {_norm(s) for s in re.split(r"[,/\s]+", _norm(str(mb_s))) if s}
        if cpu_n in mb_set or _norm(str(mb_s)) in cpu_n:
            self._add(INFO, ["CPU", "Motherboard"],
                      f"Socket OK — CPU ({cpu_s}) matches Motherboard ({mb_s}).")
        else:
            self._add(ERROR, ["CPU", "Motherboard"],
                      f"Socket MISMATCH — CPU is {cpu_s}, Motherboard is {mb_s}. Incompatible.")

    def _cpu_mb_chipset(self) -> None:
        cpu_p = self._best.get("CPU")
        mb_p  = self._best.get("Motherboard")
        if not cpu_p or not mb_p:
            return
        cn, mn = cpu_p.name.lower(), mb_p.name.lower()
        if re.search(r"i[3-9]-\d{4,5}k\b", cn) and not re.search(r"\bz\d{3}\b", mn):
            self._add(WARNING, ["CPU", "Motherboard"],
                      "Intel K-series CPU with non-Z chipset: overclocking unavailable.")
        if re.search(r"ryzen\s+\d\s+\d{3,4}x\b", cn) and re.search(r"\ba[35]20\b", mn):
            self._add(WARNING, ["CPU", "Motherboard"],
                      "AMD Ryzen X-series CPU on A320/A520: overclocking not supported.")

    def _ram_type(self) -> None:
        """
        Check DDR generation compatibility between RAM and Motherboard.

        Dataset field layout (from API.md):
          memory.speed  = [ddr_version, speed_mhz]  e.g. [4, 3200] for DDR4-3200
          motherboard   = no explicit ddr_type field; infer from name or memory_slots label

        We also check the user's raw preference string so that typing "DDR3" is
        caught even if a stray match was found in the dataset.
        """
        ram_cs = next((cs for cs in self._searches if cs.category == "Memory (RAM)"), None)
        mb_cs  = next((cs for cs in self._searches if cs.category == "Motherboard"),  None)
        ram_p  = self._best.get("Memory (RAM)")
        mb_p   = self._best.get("Motherboard")

        # --- Determine RAM DDR generation ---
        # Priority: dataset speed field (most reliable) > name > user preference string
        ram_ddr: Optional[int] = None
        if ram_p:
            ram_ddr = (_ddr_gen_from_speed(ram_p.data.get("speed"))
                       or _ddr_gen_from_str(ram_p.name))
        # If no match found in dataset, fall back to what the user typed
        if ram_ddr is None and ram_cs and ram_cs.user_preference:
            ram_ddr = _ddr_gen_from_str(ram_cs.user_preference)

        # --- Determine Motherboard DDR support ---
        # The dataset motherboard records don't have an explicit ddr_type field,
        # so we infer from the board name (e.g. "DDR5" often appears in the name).
        mb_ddr: Optional[int] = None
        if mb_p:
            mb_ddr = (_ddr_gen_from_str(mb_p.name)
                      or _ddr_gen_from_str(str(mb_p.data.get("memory_type", ""))))
        if mb_ddr is None and mb_cs and mb_cs.user_preference:
            mb_ddr = _ddr_gen_from_str(mb_cs.user_preference)

        # --- Compare ---
        if ram_ddr and mb_ddr:
            if ram_ddr == mb_ddr:
                self._add(INFO, ["Memory (RAM)", "Motherboard"],
                          f"RAM type OK — both are DDR{ram_ddr}.")
            else:
                self._add(ERROR, ["Memory (RAM)", "Motherboard"],
                          f"RAM type MISMATCH — selected RAM is DDR{ram_ddr} "
                          f"but Motherboard supports DDR{mb_ddr}. "
                          f"DDR{ram_ddr} modules are physically incompatible with a "
                          f"DDR{mb_ddr} motherboard and will not fit or function.")
        elif ram_ddr and not mb_ddr:
            self._add(WARNING, ["Memory (RAM)", "Motherboard"],
                      f"RAM is DDR{ram_ddr} — could not confirm motherboard DDR support. "
                      f"Verify the board explicitly supports DDR{ram_ddr} before purchasing.")
        elif mb_ddr and not ram_ddr:
            self._add(WARNING, ["Memory (RAM)", "Motherboard"],
                      f"Motherboard supports DDR{mb_ddr} — could not confirm RAM DDR generation. "
                      f"Ensure your RAM kit is DDR{mb_ddr}.")
        else:
            self._add(INFO, ["Memory (RAM)", "Motherboard"],
                      "Could not determine DDR generation for either part — "
                      "verify DDR4 vs DDR5 compatibility manually.")

    def _ram_slots(self) -> None:
        mods  = _num(self._f("Memory (RAM)", "modules", "module_count"))
        slots = _num(self._f("Motherboard", "memory_slots"))
        if mods and slots:
            if mods > slots:
                self._add(ERROR, ["Memory (RAM)", "Motherboard"],
                          f"RAM has {int(mods)} sticks but board has only {int(slots)} DIMM slots.")
            else:
                self._add(INFO, ["Memory (RAM)", "Motherboard"],
                          f"DIMM slots OK — {int(mods)}-stick kit fits in {int(slots)}-slot board.")

    def _case_form_factor(self) -> None:
        mb_ff   = self._f("Motherboard", "form_factor")
        case_ff = self._f("Case", "type")
        if not mb_ff or not case_ff:
            self._add(INFO, ["Motherboard", "Case"],
                      "Could not verify form factor — check ATX/mATX/ITX fit manually.")
            return
        SIZE = {"miniitx": 1, "microatx": 2, "matx": 2, "atx": 3, "eatx": 4}
        mb_r   = next((v for k, v in SIZE.items() if k in _norm(str(mb_ff))), None)
        case_r = max((v for k, v in SIZE.items() if k in _norm(str(case_ff))), default=None)
        if mb_r and case_r:
            if mb_r > case_r:
                self._add(ERROR, ["Motherboard", "Case"],
                          f"Motherboard ({mb_ff}) too large for Case ({case_ff}).")
            else:
                self._add(INFO, ["Motherboard", "Case"],
                          f"Form factor OK — {mb_ff} fits in {case_ff} case.")

    def _psu_wattage(self) -> None:
        psu_w   = _num(self._f("Power Supply (PSU)", "wattage"))
        cpu_tdp = _num(self._f("CPU", "tdp"))
        gpu_tdp = _num(self._f("Video Card (GPU)", "tdp"))
        if not psu_w:
            self._add(INFO, ["Power Supply (PSU)"],
                      "PSU wattage not in dataset — verify manually.")
            return
        draw = (cpu_tdp or 0) + (gpu_tdp or 0)
        if not draw:
            self._add(INFO, ["Power Supply (PSU)"],
                      f"PSU is {int(psu_w)}W — no TDP data to verify against.")
            return
        total = draw + self._PLATFORM_W
        rec   = int(total * self._PSU_OVERHEAD)
        parts = []
        if cpu_tdp: parts.append(f"CPU ~{int(cpu_tdp)}W")
        if gpu_tdp: parts.append(f"GPU ~{int(gpu_tdp)}W")
        parts.append(f"overhead ~{self._PLATFORM_W}W")
        summary = " + ".join(parts) + f" = ~{int(total)}W"
        if psu_w < total:
            self._add(ERROR, ["CPU", "Video Card (GPU)", "Power Supply (PSU)"],
                      f"PSU UNDERSIZED. Draw: {summary}. PSU: {int(psu_w)}W. Need >= {rec}W.")
        elif psu_w < rec:
            self._add(WARNING, ["CPU", "Video Card (GPU)", "Power Supply (PSU)"],
                      f"PSU tight. Draw: {summary}. PSU: {int(psu_w)}W. Recommend {rec}W.")
        else:
            self._add(INFO, ["CPU", "Video Card (GPU)", "Power Supply (PSU)"],
                      f"PSU OK — {int(psu_w)}W covers ~{int(total)}W draw.")

    def _cooler_socket(self) -> None:
        cpu_s    = self._f("CPU", "socket")
        cooler_s = self._f("CPU Cooler", "socket", "sockets", "socket_compatibility")
        if not cpu_s or not cooler_s:
            self._add(INFO, ["CPU", "CPU Cooler"],
                      "Could not verify cooler socket — check manually.")
            return
        if _norm(str(cpu_s)) in _norm(str(cooler_s)):
            self._add(INFO, ["CPU", "CPU Cooler"], f"Cooler socket OK — supports {cpu_s}.")
        else:
            self._add(WARNING, ["CPU", "CPU Cooler"],
                      f"Cooler may not support {cpu_s}. Listed: {cooler_s}. Verify first.")

    def _cooler_tdp(self) -> None:
        cpu_tdp    = _num(self._f("CPU", "tdp"))
        cooler_tdp = _num(self._f("CPU Cooler", "tdp", "max_tdp", "cooling_capacity"))
        if not cpu_tdp or not cooler_tdp:
            return
        if cooler_tdp < cpu_tdp:
            self._add(WARNING, ["CPU", "CPU Cooler"],
                      f"Cooler rated {int(cooler_tdp)}W but CPU TDP is {int(cpu_tdp)}W. May struggle.")
        else:
            self._add(INFO, ["CPU", "CPU Cooler"],
                      f"Cooler TDP OK — {int(cooler_tdp)}W handles {int(cpu_tdp)}W CPU.")

    def _storage_m2(self) -> None:
        storage = self._best.get("Storage")
        if not storage:
            return
        iface = str(storage.data.get("interface", "")).lower()
        form  = str(storage.data.get("form_factor", "")).lower()
        if "m.2" not in form and "nvme" not in iface and "pcie" not in iface:
            return
        m2 = _num(self._f("Motherboard", "m2_slots", "m.2_slots"))
        if m2 is not None:
            if m2 == 0:
                self._add(ERROR, ["Storage", "Motherboard"],
                          "M.2/NVMe storage selected but Motherboard has 0 M.2 slots.")
            else:
                self._add(INFO, ["Storage", "Motherboard"],
                          f"M.2 OK — Motherboard has {int(m2)} M.2 slot(s).")
        else:
            self._add(INFO, ["Storage", "Motherboard"],
                      "M.2/NVMe storage selected — verify board has an available M.2 slot.")


# ---------------------------------------------------------------------------
# Dataset search
# ---------------------------------------------------------------------------

def search_dataset(loader: DatasetLoader, preferences: dict[str, str]) -> list[ComponentSearch]:
    results: list[ComponentSearch] = []

    print("\n" + "=" * 60)
    print("  Searching local pc-part-dataset...")
    print("=" * 60)

    for component, preference in preferences.items():
        filename = COMPONENT_FILES.get(component, "")
        cs = ComponentSearch(category=component, user_preference=preference)

        print(f"\n  [{component}]  query: \"{preference or '(any)'}\"")

        if not filename:
            cs.error = "No dataset file mapped."
            results.append(cs)
            continue

        try:
            matches = loader.search(filename, preference) if preference else loader.top(filename)
            if not matches:
                print("    No matches found.")
                cs.error = "No matches found in dataset."
            else:
                for m in matches:
                    ps = f"${m.price:,.2f}" if m.price else "Price N/A"
                    print(f"    - {m.name[:55]:55s}  {ps}")
                cs.matches = matches
        except FileNotFoundError as exc:
            cs.error = str(exc)
            print(f"    ERROR: {exc}")

        results.append(cs)

    return results


# ---------------------------------------------------------------------------
# Compatibility report
# ---------------------------------------------------------------------------

def run_compatibility_check(searches: list[ComponentSearch]) -> list[CompatibilityIssue]:
    checker = CompatibilityChecker(searches)
    issues  = checker.run_all()

    print("\n" + "=" * 60)
    print("  COMPATIBILITY REPORT")
    print("=" * 60)

    if not issues:
        print("  No issues detected.")
        return issues

    icons = {ERROR: "ERROR", WARNING: "WARNING", INFO: "INFO"}
    for sev, label in [(ERROR, "ERRORS"), (WARNING, "WARNINGS"), (INFO, "NOTES")]:
        group = [i for i in issues if i.severity == sev]
        if not group:
            continue
        print(f"\n  -- {label} --")
        for issue in group:
            print(f"\n  [{issue.severity}] ({', '.join(issue.components)})")
            print(f"    {issue.message}")

    errors   = sum(1 for i in issues if i.severity == ERROR)
    warnings = sum(1 for i in issues if i.severity == WARNING)
    infos    = sum(1 for i in issues if i.severity == INFO)
    print(f"\n  Summary: {errors} error(s), {warnings} warning(s), {infos} note(s)")
    print("=" * 60)
    return issues


def _compat_for_gpt(issues: list[CompatibilityIssue]) -> str:
    if not issues:
        return "No compatibility issues detected."
    return "\n".join(
        f"[{i.severity}] ({', '.join(i.components)}) {i.message}"
        for i in issues
    )


# ---------------------------------------------------------------------------
# User input  (based on prototype.py)
# ---------------------------------------------------------------------------

def prompt_user_inputs() -> tuple[dict[str, str], float]:
    print("\n" + "=" * 60)
    print("        PC BUILD ADVISOR")
    print("=" * 60)
    print("Let's gather your preferences for each component.")
    print("Leave a field blank if you have no preference.\n")

    preferences: dict[str, str] = {}
    for component in COMPONENT_FILES:
        value = input(f"  {component}: ").strip()
        preferences[component] = value if value else ""

    print()
    while True:
        budget_str = input("What is your total budget (USD)? $").strip()
        try:
            budget = float(budget_str.replace(",", ""))
            if budget <= 0:
                raise ValueError
            break
        except ValueError:
            print("   Please enter a valid positive number.")

    print()
    use_case = input(
        "What will you primarily use this PC for?\n"
        "   (e.g., gaming, video editing, 3D rendering, general use): "
    ).strip()
    if not use_case:
        use_case = "General use / gaming"

    preferences["_use_case"] = use_case
    return preferences, budget


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        key = input("Enter your OpenAI API key: ").strip()
    return key


# ---------------------------------------------------------------------------
# GPT-4o prompt + call  (based on prototype.py)
# ---------------------------------------------------------------------------

def _dataset_block(searches: list[ComponentSearch]) -> str:
    lines: list[str] = []
    for cs in searches:
        lines.append(f"\n### {cs.category}")
        lines.append(f"  User preference: {cs.user_preference or 'None (any)'}")
        if cs.error:
            lines.append(f"  Data: UNAVAILABLE — {cs.error}")
            continue
        if not cs.matches:
            lines.append("  Data: No matches found.")
            continue
        lines.append(f"  Dataset matches ({len(cs.matches)}):")
        for i, m in enumerate(cs.matches, 1):
            ps = f"${m.price:,.2f}" if m.price else "N/A"
            lines.append(f"    {i}. {m.name}  |  Price (may be outdated): {ps}")
            specs = [k for k in m.data if k not in ("name", "price") and m.data[k] is not None]
            preview = ", ".join(f"{k}: {m.data[k]}" for k in specs[:8])
            if preview:
                lines.append(f"       Specs: {preview}")
    return "\n".join(lines)


def build_prompt(preferences: dict[str, str], budget: float) -> str:
    use_case = preferences.pop("_use_case", "General use")
    # These are filled in by main() after dataset search and compat check;
    # the function signature matches prototype.py for easy integration.
    # Actual dataset + compat data is injected via build_full_prompt().
    raise NotImplementedError("Use build_full_prompt() instead.")


def build_full_prompt(
    preferences: dict[str, str],
    budget: float,
    use_case: str,
    dataset_block: str,
    compat_block: str,
) -> str:
    return f"""You are a senior PC hardware expert with deep knowledge of component \
compatibility, current market prices, and performance benchmarking. A user is planning \
a PC build and needs your expert guidance.

You have been given two inputs to work from:
  1. A rule-based compatibility pre-check (automated, may miss edge cases or be wrong)
  2. Dataset search results showing parts that matched the user's preferences

Your job is NOT to simply repeat this data back. You are the expert — use the pre-check \
as a starting point, then apply your own knowledge to verify, correct, and expand on it.

===========================================================
BUILD CONTEXT
===========================================================
Budget      : ${budget:,.2f} USD (all-inclusive)
Use Case    : {use_case}

===========================================================
AUTOMATED COMPATIBILITY PRE-CHECK  (verify and expand on this)
===========================================================
{compat_block}

The pre-check above is rule-based and has limitations. You must:
- Confirm or correct each finding with your own expertise
- Catch anything the automated check missed (e.g. PCIe lane conflicts, power connector
  requirements, cooler clearance, RAM XMP support, NVMe gen mismatches, etc.)
- Add your own compatibility observations even if the automated check found nothing

===========================================================
DATASET MATCHES  (user preferences + closest parts found)
===========================================================
{dataset_block}

NOTE: Dataset prices are from a historical PCPartPicker snapshot and are likely outdated.
Treat them as rough ballpark figures only.

===========================================================
YOUR RECOMMENDATIONS
===========================================================
The user's stated preferences are a STARTING POINT, not a constraint. If a better part
exists within budget — better performance, better value, better compatibility — recommend
it instead and briefly explain why it's the superior choice.

For each component provide a section with:
  • Your recommended part (name + model)
  • Estimated current market price (your knowledge, not the dataset)
  • Why this is the best choice for the use case and budget
  • Any compatibility notes specific to this part

Then provide:

**Compatibility Assessment**
Your independent compatibility verdict for the full build. Address every issue the
automated pre-check flagged, confirm or correct each one, and add anything it missed.
Be direct: if parts are incompatible, say so clearly and explain exactly why and what
to replace.

**Total Estimated Cost**
Sum of your recommended parts at current market prices. State whether it fits the budget.
If over budget, suggest which components to downgrade and by how much.

**Performance Notes**
Expected real-world performance for {use_case}. Be specific — frame rates, render times,
workload suitability — not generic statements.

**Price Disclaimer**
Remind the user that your prices are estimates and the dataset prices are historical.
Always verify on PCPartPicker or a retailer before purchasing."""


def get_recommendations(client: OpenAI, prompt: str) -> str:
    print("\nConsulting GPT-4o for your personalized build recommendations...\n")
    print("=" * 60)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior PC hardware expert. "
                    "You have deep knowledge of CPU socket compatibility, chipset features, "
                    "RAM DDR generations, PSU sizing, case form factors, GPU power requirements, "
                    "NVMe generations, and current market pricing as of early 2025. "
                    "You independently verify compatibility rather than just repeating automated checks. "
                    "You prioritize recommending the best parts for the user's use case and budget "
                    "over sticking to the user's stated preferences — if a better option exists, "
                    "you recommend it and explain why. "
                    "You are direct and specific: you name exact incompatibilities, explain the "
                    "physical or electrical reason they fail, and always suggest a concrete fix."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2500,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main  (based on prototype.py structure)
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    # Verify dataset directory exists
    if not DATASET_DIR.exists():
        print(f"\nERROR: Dataset directory not found: {DATASET_DIR}")
        print("Clone the dataset repo first:")
        print("  git clone https://github.com/docyx/pc-part-dataset")
        print("Then run this script from the same directory, or set:")
        print("  export DATASET_DIR=/path/to/pc-part-dataset")
        sys.exit(1)

    api_key = get_api_key()
    client  = OpenAI(api_key=api_key)

    # Step 1: gather user inputs
    preferences, budget = prompt_user_inputs()
    use_case = preferences.pop("_use_case", "General use / gaming")

    # Step 2: search local dataset for matching parts
    loader   = DatasetLoader(DATASET_DIR)
    searches = search_dataset(loader, preferences)

    # Step 3: run compatibility checks (zero extra API calls)
    issues       = run_compatibility_check(searches)
    compat_block = _compat_for_gpt(issues)
    dataset_blk  = _dataset_block(searches)

    # Step 4: build prompt and call GPT-4o (single call)
    prompt          = build_full_prompt(preferences, budget, use_case, dataset_blk, compat_block)
    recommendations = get_recommendations(client, prompt)

    print(recommendations)
    print("=" * 60)

    # Step 5: optionally save full report
    save = input("\nWould you like to save these recommendations to a file? (y/n): ").strip().lower()
    if save == "y":
        filename = "pc_build_recommendations.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("PC BUILD ADVISOR — FULL REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("--- COMPATIBILITY PRE-CHECK ---\n\n")
            for issue in issues:
                f.write(f"  [{issue.severity}] ({', '.join(issue.components)})\n")
                f.write(f"    {issue.message}\n\n")

            f.write("\n--- DATASET SEARCH RESULTS ---\n\n")
            for cs in searches:
                f.write(f"[{cs.category}]  Preference: {cs.user_preference or 'None'}\n")
                if cs.error:
                    f.write(f"  Error: {cs.error}\n\n")
                    continue
                for m in cs.matches:
                    ps = f"${m.price:,.2f}" if m.price else "N/A"
                    f.write(f"  - {m.name}  |  {ps}\n")
                    specs = [k for k in m.data if k not in ("name", "price") and m.data[k] is not None]
                    for k in specs[:8]:
                        f.write(f"    {k}: {m.data[k]}\n")
                f.write("\n")

            f.write("\n--- GPT-4o RECOMMENDATIONS ---\n\n")
            f.write(recommendations)
            f.write("\n\n" + "=" * 60 + "\n")

        print(f"Recommendations saved to '{filename}'")

    print("\nHappy building!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.\n")
        sys.exit(0)

# ---------------------------------------------------------------------------
# FastAPI server
# Run with:  uvicorn compatability:app --reload --port 8000
# ---------------------------------------------------------------------------

if not _FASTAPI_AVAILABLE:
    raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

app = FastAPI(title="PC Build Advisor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
        ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    # The exact `selected` object built in App.tsx onRun():
    #   keys: component names + "_use_case" + "Budget" + "Mode"
    selected: dict[str, str]
    openai_api_key: str


def _build_searches(selected: dict[str, str]) -> tuple[list, float, str, str]:
    """
    Parse the `selected` dict from the frontend into the structures
    that the existing compatibility / prompt functions expect.
    Returns: (searches, budget, use_case, mode)
    """
    use_case   = selected.get("_use_case", "General use / gaming")
    mode       = selected.get("Mode", "Full PC build")
    budget_raw = selected.get("Budget", "0").replace("$", "").replace(",", "").strip()
    try:
        budget = float(budget_raw)
    except ValueError:
        budget = 0.0

    # Build preferences — blank out "(any)" placeholders the frontend sends
    preferences: dict[str, str] = {}
    for component in COMPONENT_FILES:
        val = selected.get(component, "")
        preferences[component] = "" if val in ("", "(any)") else val

    loader   = DatasetLoader(DATASET_DIR)
    searches = search_dataset(loader, preferences)
    return searches, budget, use_case, mode


def _fmt_compat(issues: list) -> str:
    """Format compat issues for the frontend Compatibility Issues panel."""
    if not issues:
        return "No compatibility issues detected."

    lines: list[str] = []
    for sev, label in [(ERROR, "ERRORS"), (WARNING, "WARNINGS"), (INFO, "NOTES")]:
        group = [i for i in issues if i.severity == sev]
        if not group:
            continue
        lines.append(f"── {label} ──")
        for issue in group:
            icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}.get(issue.severity, "•")
            lines.append(f"{icon}  [{', '.join(issue.components)}]")
            lines.append(f"   {issue.message}")
            lines.append("")

    errors   = sum(1 for i in issues if i.severity == ERROR)
    warnings = sum(1 for i in issues if i.severity == WARNING)
    infos    = sum(1 for i in issues if i.severity == INFO)
    lines.append(f"Summary: {errors} error(s), {warnings} warning(s), {infos} note(s)")
    return "\n".join(lines)


@app.post("/run")
def run_endpoint(req: RunRequest):
    """
    Single-call endpoint — returns both compat issues and full AI output at once.
    Frontend sets setCompatIssues and setAiOutput from the response.
    """
    searches, budget, use_case, mode = _build_searches(req.selected)

    issues       = run_compatibility_check(searches)
    compat_text  = _fmt_compat(issues)
    compat_block = _compat_for_gpt(issues)
    dataset_blk  = _dataset_block(searches)

    preferences = {c: (req.selected.get(c) or "") for c in COMPONENT_FILES}
    prompt      = build_full_prompt(preferences, budget, use_case, dataset_blk, compat_block)

    client    = OpenAI(api_key=req.openai_api_key)
    ai_output = get_recommendations(client, prompt)

    return {"compat_issues": compat_text, "ai_output": ai_output}


@app.post("/stream")
def stream_endpoint(req: RunRequest):
    """
    Streaming endpoint (SSE).
    Sends compat issues first, then streams AI tokens.
    Event format:  data: {"type": "compat"|"token"|"done", "text": "..."}
    """
    searches, budget, use_case, mode = _build_searches(req.selected)

    issues       = run_compatibility_check(searches)
    compat_text  = _fmt_compat(issues)
    compat_block = _compat_for_gpt(issues)
    dataset_blk  = _dataset_block(searches)

    preferences = {c: (req.selected.get(c) or "") for c in COMPONENT_FILES}
    prompt      = build_full_prompt(preferences, budget, use_case, dataset_blk, compat_block)
    client      = OpenAI(api_key=req.openai_api_key)

    def generate():
        # Send compat results immediately so the UI updates before AI starts
        yield f"data: {json.dumps({'type': 'compat', 'text': compat_text})}\n\n"

        with client.chat.completions.stream(
            model="gpt-4o",
            max_tokens=2500,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior PC hardware expert. "
                        "You have deep knowledge of CPU socket compatibility, chipset features, "
                        "RAM DDR generations, PSU sizing, case form factors, GPU power requirements, "
                        "NVMe generations, and current market pricing as of early 2025. "
                        "You independently verify compatibility rather than just repeating automated checks. "
                        "You prioritize recommending the best parts for the user's use case and budget. "
                        "You are direct and specific: name exact incompatibilities and always suggest a fix."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        ) as stream:
            for event in stream:
                print(event)
                if event.type == "response.output_text.delta":
                    yield f"data: {json.dumps({'type': 'token', 'text': event.delta})}\n\n"
                elif event.type == "content.delta":
                    yield f"data: {json.dumps({'type': 'token', 'text': event.delta})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )