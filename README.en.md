# OBI — Okikamuro Bite Index

**A fisherman in Yamaguchi, Japan built his own location-specific fish-bite index in a day, with Claude Code.**

[日本語版 README はこちら](./README.md)

---

## Live Demo

**https://sakaeruman.github.io/obi-tide-bite-index/**

Auto-updates every day at **05:00 JST** via GitHub Actions. The page shows tomorrow's hour-by-hour bite forecast (★0–5) for four target species off Okikamuro Island in the Seto Inland Sea.

---

## What & Why

OBI (**O**kikamuro **B**ite **I**ndex) is an open-source bite-time forecaster tuned to a single location: **Okikamuro Island** (沖家室島, 33°55′N 132°15′E), a small fishing island off the southern coast of Suo-Oshima, facing the **Iyo-nada Sea** in western Japan.

Generic fishing apps (Solunar, tide tables, "big tide / small tide" labels) are too coarse for narrow-channel inshore fishing where bite quality is dominated by **local current acceleration** rather than absolute tide height. So I built one that:

- Takes JMA (Japan Meteorological Agency) hourly tide predictions at Tokuyama station (~30 km away)
- Computes `dh/dt` (tide-height time derivative) as a proxy for current speed
- Layers astronomical twilight, moon illuminance, and species-specific seasonality
- Outputs a per-hour × per-species ★0–5 score, plus an HTML page and a heatmap PNG

Target species (v1): **red sea bream (madai)**, **horse mackerel (aji)**, **largehead hairtail (tachiuo)**, **Japanese Spanish mackerel (sawara)**.

**The author is not a researcher.** I'm a hijiki seaweed fisherman who left Tokyo for an island of 130 people in 2018. This whole thing — formula, data pipeline, deploy — was built in one day in collaboration with Claude Code, by intentionally treating the AI as a **research assistant**, not a stenographer.

---

## Architecture

```
JMA Tide Table (Tokuyama, station=QA, annual .txt)
                │
                ▼
    src/fetch_tide.py  ──► data/tide_cache/
                │
                ▼
    src/astro.py  (Skyfield + de421.bsp)
        ├── sunrise / sunset
        ├── civil twilight window
        ├── moon phase / illuminance
        └── moon altitude (for night correction)
                │
                ▼
    src/obi.py  ── computes per-hour scores for each species
                │   weights = w1·f_v + w2·f_dh + w3·U + w4·B_twilight
                │           + w5·M_moon + w6·S_season
                │   then multiplied by P_temp · P_pressure
                │
                ▼
    src/render.py  ──► out/obi_YYYYMMDD.{html, md, png}
                │
                ▼
    GitHub Actions (daily @ 05:00 JST)
                │
                ▼
    docs/index.html  ──►  GitHub Pages
```

- **Language**: Python 3.11+
- **Key deps**: `skyfield` (ephemeris), `numpy`, `pandas`, `matplotlib`, `pyyaml`, `requests`
- **No paid APIs**. Tide data is JMA open data. Ephemeris is the public `de421.bsp` (~16 MB).

---

## OBI Formula

Full form:

```
OBI(t) = [ w1·f_v + w2·f_dh + w3·U + w4·B_twilight + w5·M_moon + w6·S_season ]
         × P_temp × P_pressure
```

In v1, the current-speed term `f_v` can't be computed directly — Japan Coast Guard MSIL current vectors aren't trivially scrapable yet — so `f_dh` (the normalized magnitude of the tide-height derivative) carries its weight as a proxy.

| Symbol | Meaning | v1 weight | Notes |
|---|---|---|---|
| `f_v` | `|v| / v_ref` (current speed, normalized) | `w1 = 0.00` | Disabled in v1. Re-enabled in v2 once MSIL integration lands. |
| `f_dh` | `|dh/dt| / dh_ref` (dh_ref = 80 cm/h) | `w2 = 0.50` | v1 primary signal. Expanded to absorb the disabled `f_v` weight. |
| `U` | Upwelling index | `w3 = 0.15` | Hard-coded 0.5 in v1 (purely terrain-dependent). v2 will compute dynamically from current vectors + bathymetric gradient. |
| `B_twilight` | Triangular window over sunrise/sunset ±1.5 h | `w4 = 0.15` | Captures the classic *mazume* dawn/dusk feeding window. |
| `M_moon` | Moon illuminance × (night-only correction when altitude > 0) | `w5 = 0.10` | Moonlight raises nocturnal predator activity. |
| `S_season` | Per-month × per-species seasonality coefficient (0..1) | `w6 = 0.10` | From a hand-curated fishing-season calendar specific to the Iyo-nada Sea. |
| `P_temp` | Water-temperature optimality factor | (multiplicative) | Fixed at 1.0 in v1. |
| `P_pressure` | Pressure-change-rate factor | (multiplicative) | Fixed at 1.0 in v1. |

Raw scores are min-max normalized **per day, per species**, then rounded to ★0–5 for display. Per-species normalization matters: a top-bite hour for hairtail should always read as ★5 even on a low-tide-range day, because anglers compare *within* a species, not across.

### Explicitly rejected

These were considered and removed during research:

- **Solunar theory** — not supported by the peer-reviewed literature; perpetuated by app marketing.
- **"Big tide / medium tide / small tide" (大潮/中潮/小潮) categorical labels** — the underlying tide-range number is shown alongside the forecast, but the label itself adds no predictive signal beyond `|dh/dt|`.

### Known v1 simplifications

- **Standing-wave vs. progressive-wave phase correction**: hard-coded `config.tide_station.phase_shift_hours: 1.5`. v2 will calibrate this against on-site observations over ~1 year.

---

## Discoveries while building

The most interesting output of this project wasn't the code. It was finding that several pieces of "common knowledge" repeated for decades in Japanese sport-fishing media are **not supported by the underlying biology or fluid mechanics** — and that the actual fishing-tackle industry has already quietly built products around the corrected understanding.

### 1. Red sea bream don't actually "see red"

**Folklore:** "Use a red jig head for red sea bream — they're attracted to the color of their own kind." Every tackle aisle in Japan is half-red because of this.

**Reality:** Red sea bream (*Pagrus major*) have a **pseudogenized LWS opsin** — the long-wavelength-sensitive cone gene that vertebrates use to perceive red light is non-functional in this species (Wang et al., 2009, *Comp. Biochem. Physiol.*). At the depths and turbidity where they actually feed, red wavelengths are also attenuated fastest. So "the red lure works because they see red" is a literal physical impossibility. The red lures still catch fish — but because of silhouette, action, and reflectivity contrast, not hue.

**Why it matters for OBI:** Color-based lure recommendations are out of scope. The model focuses on *timing* (when the fish will eat), because lure-color "advice" downstream of bad science is noise.

### 2. "Spanish mackerel attack from below" — unsupported

**Folklore:** Sawara (Japanese Spanish mackerel) supposedly ambush bait from underneath, so you should retrieve your jig **higher than the bait ball**.

**Reality:** Sawara have **no swim bladder**. Negatively buoyant pelagic predators are constrained in vertical mobility — they can't hold position above prey for long, and a fast vertical attack from below is metabolically expensive. The actual feeding behavior, supported by gut-content and stomach-orientation studies, is horizontal pursuit. The "from below" framing appears to be a translation/import of striped-bass folklore, applied to a species it doesn't fit.

### 3. Mackerel burst-swim 18 body-lengths per second — Shimano's "land them one at a time" is right

**Source:** Wardle (1988), *Fish Swimming* (Chapman & Hall) reports Atlantic mackerel achieving sustained burst speeds of ~18 BL/s. For a 30 cm mackerel that's ~5.4 m/s.

**What I noticed:** Shimano's official Light Game guidance says "**land mackerel one at a time, don't try to multi-hook them**." For years anglers have grumbled about this as overly cautious marketing. It isn't. At 18 BL/s burst speeds with two fish on a sabiki rig pulling perpendicular to each other, leader breakage isn't bad luck — it's the load math. Shimano's recommendation is scientifically validated; the manufacturer just doesn't bother explaining *why*.

### 4. The leader trade-off was already solved — by Owner

**Problem:** Hairtail (tachiuo) and sawara have teeth that shear monofilament instantly, but full-wire leaders kill bite quality (stiffness telegraphs the lure as fake, fluorocarbon is invisible underwater).

**Industry's actual solution:** Composite-metal leader **only on the top 20–30 cm** adjacent to the hook — long enough to defeat the teeth, short enough that the rest of the leader is still fluoro. **Owner F-6330** productizes exactly this geometry. Most sport-fishing articles online still treat this as an "either/or" choice; the answer has been sitting on tackle-shop pegboards for years.

---

These four findings came out of running **three parallel research axes** simultaneously (see next section). None of them are in OBI's code directly — OBI is a timing model, not a tackle recommender — but they shaped what *not* to include, and they're the kind of result you only get when you let the AI fan out across academic literature, manufacturer documentation, and product specifications in parallel and then **make them argue with each other**.

---

## How AI was used (Claude Code case study)

This project is a worked example of using Claude Code as a **research-first** tool, not as autocomplete.

### Four parallel Ultracode workflows

For each target species I ran **four Ultracode sessions in parallel**, each with a different research mandate:

1. **Academic literature** — peer-reviewed papers on the species' visual system, swim mechanics, feeding ecology, distribution in the Inland Sea.
2. **Manufacturer knowledge** — Shimano, Daiwa, Owner, Gamakatsu technical documents, lure-action whitepapers, official rigging guides.
3. **Product specifications** — actual SKUs on the market, leader composition, hook geometry, what the industry has *already* productized as a solution to known problems.
4. **Folklore audit** — Japanese-language fishing blogs, YouTube channels, tackle-shop conventional wisdom, then cross-checked against axes 1–3.

This 3-axis (+ folklore audit) structure is what surfaced the LWS-opsin finding, the swim-bladder argument, and the Owner F-6330 trade-off resolution. Each individual session would have produced a plausible-but-mediocre answer; the **adversarial cross-checking** between them is where the real signal came from.

### Parallel agent verification

Whenever a finding "felt too clean" — i.e. it neatly explained a piece of folklore but I hadn't seen it stated elsewhere — I spawned a fresh, context-free agent to try to **falsify** the claim, with explicit instructions to look for counter-evidence rather than confirmation. The LWS-opsin claim survived this. Several other promising-sounding claims did not, and got cut.

### Why this matters

The temptation with an LLM is to treat it as a stenographer: ask a question, accept the answer, move on. The output is fast and feels authoritative — and is wrong often enough to be dangerous, especially in a domain like fishing where wrong-but-plausible has been the industry standard for a century.

Treating Claude Code as a research assistant — fan out, verify, falsify, synthesize — takes longer per question, but the answers are durable. The four findings above aren't going to change next year. The conventional fishing advice they replace has been wrong since before the internet existed.

**Total wall-clock time from "no project" to "deployed, auto-updating GitHub Pages":** one day.

---

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/sakaeruman/obi-tide-bite-index.git
cd obi-tide-bite-index
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first run, Skyfield will auto-download `de421.bsp` (~16 MB) into `data/`. For offline environments, place it there manually beforehand.

### Run

```bash
# Compute tomorrow's OBI (JST), open browser
python -m src.daily

# Specific date
python -m src.daily --date 2026-06-19

# Subset of species
python -m src.daily --species madai,aji

# Headless (for cron / CI)
python -m src.daily --no-open
```

Outputs land in `out/`:

- `out/obi_YYYYMMDD.html` — the daily forecast page
- `out/obi_YYYYMMDD.md` — same data, plain Markdown table
- `out/obi_YYYYMMDD.png` — 24h × species heatmap (viridis)
- `out/latest.html` — always the most recent run
- `logs/daily_YYYYMMDD.log` — per-step progress and tracebacks

---

## Customize for other locations

OBI is geographically hard-coded to Okikamuro / Iyo-nada, but the model itself is portable. To re-tune it:

1. **`config.yaml`** — change `location.lat / lon`, set `tide_station` to your nearest JMA station code, and re-survey `phase_shift_hours` against local observations.
2. **`config.season`** — replace the per-species monthly coefficients with your region's actual season calendar. The defaults are calibrated to the Seto Inland Sea and will mislead you anywhere else.
3. **Species list** — `src/obi.py` species table. Add/remove targets and their per-species parameters.
4. **Validate** — log a few weeks of catches and run the `校正方法` (calibration) procedure in the Japanese README. Don't trust the default weights for your water.

PRs that add other regions are welcome, but please keep them as **separate config presets** rather than overwriting the Okikamuro defaults.

---

## Roadmap

- [ ] **MSIL current-speed integration** (`f_v` enabled, `w1 = 0.35`, `w2` shrunk back to 0.15)
- [ ] **Phase-shift calibration** against on-site observations (`phase_calib.py`, 1-year Tokuyama-vs-Okikamuro fit)
- [ ] **Weather integration** (pressure + water temperature from JMA Open Data / OpenMeteo) — escape `P_temp = P_pressure = 1.0`
- [ ] **Bayesian weight update from catch logs** — once the log reaches ~20 trips, swap manual tuning for a posterior-based update with per-species independent weights
- [ ] **Dynamic upwelling `U`** from MSIL current vectors and bathymetric gradient
- [ ] **Discord / LINE daily push** — already partly wired; needs Bot token handling

---

## License

MIT. See [LICENSE](./LICENSE).

The fishing-folklore corrections above are findings, not legal claims — cite the underlying primary sources (Wang et al. 2009, Wardle 1988, etc.) directly if you reuse them.

---

## Author

**Sakaeru** (栄/さかえる) — hijiki seaweed fisherman, Okikamuro Island, Yamaguchi, Japan.
55 t/year of hijiki, an EC shop, an island village chairman seat, and apparently a fish-bite forecaster now.

- Site: https://sakaeru.online
- Built with [Claude Code](https://www.anthropic.com/claude-code) (Anthropic)

If you find this useful, the highest-value thing you can do is **log your catches against the forecast and open a PR with the calibration delta**. The model is only as good as the next round of real-water feedback.
