from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


# ========================================================================================
# DATA: Gazetteer and ecological lookup tables. All figures are rounded, order-of-magnitude values.
# ========================================================================================
@dataclass
class Place:
    key: str
    name: str
    kind: str                      # country | region | city | ocean | world
    area: float = 0.0              # km2
    pop: float = 0.0               # people (metro pop for cities)
    biomes: list = field(default_factory=list)
    forest_frac: float = 0.0
    city: str | None = None        # key of primary target city
    density: float = 0.0           # people/km2 in urban core (cities only)
    aliases: list = field(default_factory=list)


PLACES: dict[str, Place] = {}


def _add(p: Place):
    PLACES[p.key] = p


# key, name, area km2, pop (M), biomes, forest fraction, primary city, extra aliases
_COUNTRIES = [
    ("mexico", "Mexico", 1_964_375, 128, "desert shrubland tropical_dry temperate_forest rainforest", .34, "mexico city", []),
    ("usa", "United States", 9_834_000, 335, "temperate_forest grassland desert boreal shrubland", .34, "new york", ["united states", "united states of america", "america", "usa", "u.s.", "u.s.a."]),
    ("canada", "Canada", 9_985_000, 40, "boreal tundra temperate_forest grassland", .38, "toronto", []),
    ("brazil", "Brazil", 8_516_000, 216, "rainforest savanna wetland temperate_forest", .59, "sao paulo", []),
    ("russia", "Russia", 17_098_000, 144, "boreal tundra grassland temperate_forest", .50, "moscow", []),
    ("china", "China", 9_597_000, 1410, "temperate_forest grassland desert montane", .23, "beijing", []),
    ("india", "India", 3_287_000, 1430, "tropical_dry savanna montane rainforest desert", .24, "delhi", []),
    ("pakistan", "Pakistan", 881_900, 240, "desert grassland montane", .05, "karachi", []),
    ("australia", "Australia", 7_692_000, 26, "desert savanna shrubland temperate_forest", .17, "sydney", []),
    ("uk", "United Kingdom", 243_600, 67, "temperate_forest grassland wetland", .13, "london", ["united kingdom", "britain", "great britain", "england", "uk"]),
    ("france", "France", 551_700, 68, "temperate_forest grassland montane", .31, "paris", []),
    ("germany", "Germany", 357_600, 84, "temperate_forest grassland", .32, "berlin", []),
    ("japan", "Japan", 377_900, 124, "temperate_forest montane", .68, "tokyo", []),
    ("indonesia", "Indonesia", 1_905_000, 277, "rainforest mangrove coral", .49, "jakarta", []),
    ("egypt", "Egypt", 1_001_000, 112, "desert wetland", .0, "cairo", []),
    ("nigeria", "Nigeria", 923_800, 223, "savanna rainforest wetland", .23, "lagos", []),
    ("south africa", "South Africa", 1_221_000, 60, "savanna shrubland grassland", .07, "johannesburg", []),
    ("argentina", "Argentina", 2_780_000, 46, "grassland shrubland montane", .10, "buenos aires", []),
    ("chile", "Chile", 756_100, 19.5, "desert temperate_forest montane", .24, "santiago", []),
    ("peru", "Peru", 1_285_000, 34, "rainforest montane desert", .57, "lima", []),
    ("colombia", "Colombia", 1_142_000, 52, "rainforest montane savanna", .53, "bogota", []),
    ("iceland", "Iceland", 103_000, .38, "tundra", .005, "reykjavik", []),
    ("italy", "Italy", 301_300, 59, "temperate_forest shrubland montane", .32, "rome", []),
    ("spain", "Spain", 506_000, 48, "shrubland temperate_forest grassland", .37, "madrid", []),
    ("ukraine", "Ukraine", 603_500, 38, "grassland temperate_forest", .17, "kyiv", []),
    ("iran", "Iran", 1_648_000, 89, "desert montane shrubland", .07, "tehran", []),
    ("saudi arabia", "Saudi Arabia", 2_150_000, 36, "desert", .005, "riyadh", []),
    ("kenya", "Kenya", 580_400, 55, "savanna montane", .07, "nairobi", []),
    ("drc", "DR Congo", 2_345_000, 102, "rainforest savanna wetland", .55, "kinshasa", ["dr congo", "democratic republic of the congo", "congo"]),
    ("madagascar", "Madagascar", 587_000, 30, "rainforest tropical_dry", .21, "antananarivo", []),
    ("new zealand", "New Zealand", 268_000, 5.2, "temperate_forest montane", .38, "auckland", []),
    ("costa rica", "Costa Rica", 51_100, 5.2, "rainforest tropical_dry", .59, "san jose", []),
    ("north korea", "North Korea", 120_500, 26, "temperate_forest montane", .46, "pyongyang", []),
    ("israel", "Israel", 22_100, 9.8, "shrubland desert", .08, "tel aviv", []),
    ("turkey", "Turkey", 783_600, 85, "shrubland temperate_forest montane", .29, "istanbul", []),
    ("scotland", "Scotland", 77_900, 5.5, "montane temperate_forest wetland", .19, None, []),
    ("new mexico", "New Mexico", 314_900, 2.1, "desert shrubland montane", .25, "albuquerque", []),
    ("california", "California", 423_970, 39, "shrubland temperate_forest desert", .33, "los angeles", []),
    ("florida", "Florida", 170_300, 22, "wetland temperate_forest mangrove", .50, "miami", []),
    ("texas", "Texas", 695_600, 30, "grassland shrubland temperate_forest desert", .24, "houston", []),
    ("alaska", "Alaska", 1_723_000, .73, "boreal tundra temperate_forest", .37, "anchorage", []),
]
for k, n, a, p, b, f, c, al in _COUNTRIES:
    _add(Place(k, n, "country" if k not in ("scotland", "california", "florida", "texas", "alaska", "new mexico") else "region",
               a, p * 1e6, b.split(), f, c, 0, al))

# key, name, area km2, biomes, forest fraction, pop (M), aliases
_REGIONS = [
    ("amazon", "the Amazon basin", 5_500_000, "rainforest wetland", .85, 30, ["amazon", "amazon rainforest", "amazon basin"]),
    ("great barrier reef", "the Great Barrier Reef", 344_400, "coral", 0, 0, ["great barrier reef"]),
    ("borneo", "Borneo", 743_330, "rainforest mangrove montane", .55, 21, []),
    ("siberia", "Siberia", 13_100_000, "boreal tundra", .5, 37, []),
    ("yellowstone", "Yellowstone", 8_991, "montane temperate_forest", .80, 0.01, ["yellowstone national park"]),
    ("europe", "Europe", 10_180_000, "temperate_forest grassland boreal", .35, 745, []),
    ("africa", "Africa", 30_370_000, "savanna desert rainforest", .21, 1400, []),
    ("antarctica", "Antarctica", 14_200_000, "polar", 0, 0.001, []),
    ("arctic", "the Arctic", 14_000_000, "polar tundra", 0, 4, []),
    ("sahara", "the Sahara", 9_200_000, "desert", 0, 4, []),
    ("gulf of mexico", "the Gulf of Mexico", 1_550_000, "ocean wetland coral mangrove", 0, 0, []),
    ("mediterranean", "the Mediterranean Sea", 2_500_000, "ocean", 0, 0, ["mediterranean sea"]),
    ("pacific", "the Pacific Ocean", 165_000_000, "ocean", 0, 0, ["pacific ocean"]),
    ("atlantic", "the Atlantic Ocean", 85_000_000, "ocean", 0, 0, ["atlantic ocean"]),
    ("world", "the whole world", 510_000_000, "mixed", .31 * 0.29, 8100, ["the world", "world", "globally", "worldwide", "earth", "the planet", "planet", "globe"]),
]
for k, n, a, b, f, p, al in _REGIONS:
    kind = "ocean" if b.startswith("ocean") or k in ("pacific", "atlantic", "mediterranean") else ("world" if k == "world" else "region")
    _add(Place(k, n, kind, a, p * 1e6, b.split(), f, None, 0, al))

# key, name, metro pop (M), core density (people/km2)
_CITIES = [
    ("mexico city", "Mexico City", 21.8, 9000), ("new york", "New York City", 19, 10000),
    ("los angeles", "Los Angeles", 12.5, 3200), ("toronto", "Toronto", 6.2, 4300),
    ("sao paulo", "Sao Paulo", 22, 8000), ("moscow", "Moscow", 12.6, 5000),
    ("beijing", "Beijing", 21, 7000), ("delhi", "Delhi", 32, 11000), ("karachi", "Karachi", 16, 24000),
    ("sydney", "Sydney", 5.3, 2000), ("london", "London", 9.6, 5500), ("paris", "Paris", 11, 20000),
    ("berlin", "Berlin", 3.7, 4100), ("tokyo", "Tokyo", 37, 15000), ("jakarta", "Jakarta", 11, 15000),
    ("cairo", "Cairo", 21, 19000), ("lagos", "Lagos", 15, 7000), ("johannesburg", "Johannesburg", 6, 2900),
    ("buenos aires", "Buenos Aires", 15, 14000), ("santiago", "Santiago", 6.8, 8000), ("lima", "Lima", 11, 3500),
    ("bogota", "Bogota", 11, 4500), ("reykjavik", "Reykjavik", .24, 1700), ("rome", "Rome", 4.3, 2200),
    ("madrid", "Madrid", 6.7, 5400), ("kyiv", "Kyiv", 3, 3300), ("tehran", "Tehran", 9.5, 11000),
    ("riyadh", "Riyadh", 7.6, 4000), ("nairobi", "Nairobi", 5, 4500), ("kinshasa", "Kinshasa", 17, 5500),
    ("antananarivo", "Antananarivo", 3.5, 7000), ("auckland", "Auckland", 1.7, 1300), ("san jose", "San Jose", 1.4, 4500),
    ("pyongyang", "Pyongyang", 3, 7000), ("tel aviv", "Tel Aviv", 4, 8000), ("istanbul", "Istanbul", 15.8, 2900),
    ("albuquerque", "Albuquerque", .9, 1100), ("miami", "Miami", 6.1, 4800), ("houston", "Houston", 7.3, 1400),
    ("anchorage", "Anchorage", .29, 70), ("chicago", "Chicago", 9.4, 4600), ("washington", "Washington DC", 6.4, 4000),
    ("shanghai", "Shanghai", 28, 8000), ("mumbai", "Mumbai", 21, 20000), ("san francisco", "San Francisco", 4.6, 7200),
    ("seattle", "Seattle", 4, 3500), ("vancouver", "Vancouver", 2.6, 5500), ("rio de janeiro", "Rio de Janeiro", 13.7, 5400),
]
_CITY_ALIASES = {"new york": ["nyc", "new york city"], "washington": ["washington dc", "washington d.c."],
                 "sao paulo": ["são paulo"], "kyiv": ["kiev"], "bogota": ["bogotá"], "rio de janeiro": ["rio"]}
for k, n, p, d in _CITIES:
    _add(Place(k, n, "city", 0, p * 1e6, ["urban"], 0, None, d, _CITY_ALIASES.get(k, [])))


def alias_index() -> list[tuple[str, Place]]:
    """(alias, place) pairs, longest alias first so 'mexico city' wins over 'mexico'."""
    out = []
    for p in PLACES.values():
        for a in {p.key, p.name.lower().removeprefix("the "), *p.aliases}:
            out.append((a.lower(), p))
    return sorted(out, key=lambda t: -len(t[0]))


# ---- biome tables ---------------------------------------------------------------------------
# fuel consumed per hectare in a fire (t carbon), full-recovery time, forest carbon stock (t C/ha)
BIOMES = {
    "rainforest":       dict(label="tropical rainforest", fire_c=60, recover="100-300+ years (often never, if it tips to savanna)", stock=150),
    "tropical_dry":     dict(label="tropical dry forest", fire_c=25, recover="30-80 years", stock=70),
    "temperate_forest": dict(label="temperate forest", fire_c=30, recover="50-150 years", stock=100),
    "boreal":           dict(label="boreal forest/taiga", fire_c=25, recover="60-150 years (peat/permafrost carbon may burn for much longer)", stock=90),
    "savanna":          dict(label="savanna", fire_c=3, recover="1-5 years (fire-adapted)", stock=20),
    "grassland":        dict(label="grassland", fire_c=2, recover="1-3 years", stock=10),
    "shrubland":        dict(label="shrubland/chaparral", fire_c=8, recover="5-30 years (type-conversion to invasive grass is a risk)", stock=25),
    "desert":           dict(label="desert", fire_c=1, recover="decades (slow-growing, fire-intolerant plants)", stock=5),
    "tundra":           dict(label="tundra", fire_c=10, recover="decades to centuries", stock=30),
    "montane":          dict(label="montane forest", fire_c=30, recover="50-150 years", stock=90),
    "wetland":          dict(label="wetland", fire_c=15, recover="5-50 years (peat is slow)", stock=60),
    "mangrove":         dict(label="mangrove", fire_c=40, recover="20-50 years", stock=200),
    "coral":            dict(label="coral reef", fire_c=0, recover="10-50+ years", stock=0),
    "ocean":            dict(label="open ocean", fire_c=0, recover="varies", stock=0),
    "polar":            dict(label="polar ice/ice sheet", fire_c=0, recover="n/a", stock=0),
    "urban":            dict(label="urban", fire_c=10, recover="n/a", stock=0),
    "mixed":            dict(label="mixed biomes", fire_c=15, recover="varies", stock=60),
}

FOREST_BIOMES = ("rainforest", "tropical_dry", "temperate_forest", "boreal", "montane", "mangrove")


def biome_info(place: Place | None, prefer_forest=False) -> tuple[str, dict]:
    """Dominant biome key + info for a place (falls back to mixed)."""
    if place:
        bs = [b for b in place.biomes if b in BIOMES]
        if prefer_forest:
            bs = [b for b in bs if b in FOREST_BIOMES] or bs
        if bs:
            return bs[0], BIOMES[bs[0]]
    return "mixed", BIOMES["mixed"]


# ---- species (keystone/functional groups) --------------------------------------------------
# key -> aliases, role, loss effects (text, severity), addition effects
SPECIES = {
    "bees": dict(aliases=["bee", "bees", "honeybee", "honeybees", "honey bee", "honey bees", "pollinators", "pollinator"],
                 role="pollinators",
                 loss=[("~75% of leading food crops and ~35% of global crop volume depend at least partly on animal pollination; fruits, nuts, vegetables, oilseeds hit hardest (staple grains like wheat/rice/maize are wind-pollinated)", 4, "1-3 yrs"),
                       ("Wild plants dependent on insect pollination (~85% of flowering plants) decline in seed set, reshuffling plant communities", 4, "years-decades"),
                       ("Birds and mammals that eat fruit/seed lose food sources; cascading loss of dependent species", 3, "years-decades"),
                       ("Hand-pollination or substitute pollinators (flies, moths, birds, bats) only partly compensate", 2, "ongoing")],
                 add=[("Improved pollination raises fruit/seed set; watch for competition with native bees and pathogen spillover", 2, "years")]),
    "wolves": dict(aliases=["wolf", "wolves", "apex predator", "apex predators", "predators"],
                   role="apex predator",
                   loss=[("Ungulate (deer/elk) populations rise without predation, overbrowse saplings and willow/aspen", 3, "5-20 yrs"),
                         ("Loss of riparian vegetation: stream bank erosion, fewer songbirds/beavers", 3, "10-30 yrs"),
                         ("Mesopredators (coyotes, foxes) increase and suppress small mammals and ground-nesting birds", 2, "5-15 yrs")],
                   add=[("Trophic cascade: elk/deer numbers and behaviour change, browse-damaged woodland and river banks begin to recover (Yellowstone after 1995)", 3, "5-30 yrs"),
                        ("Carrion supports scavengers (ravens, eagles, bears, beetles); livestock conflict is the main human-side cost", 2, "years")]),
    "sharks": dict(aliases=["shark", "sharks"], role="marine apex predator",
                   loss=[("Mesopredator release (rays, small sharks, mid-size fish) - overgrazing of prey such as scallops and seagrass", 3, "5-20 yrs"),
                         ("Reef and seagrass structure degrades; fisheries for some species collapse from cascades", 3, "10-30 yrs"),
                         ("Carcass removal and disease control by sharks lost; predators no longer shape prey behaviour, so habitats are over-used", 2, "years")],
                   add=[("Predation pressure restores prey behaviour and diversity; recovery is slow as sharks reproduce late", 2, "decades")]),
    "whales": dict(aliases=["whale", "whales"], role="ocean nutrient engineers",
                   loss=[("Reduced 'whale pump' of iron and nitrogen in surface waters lowers phytoplankton productivity in places", 3, "decades"),
                         ("Whale falls feed a specialist deep-sea community that loses its food source", 2, "decades"),
                         ("Less carbon sequestered by living whales and sinking carcasses", 2, "decades")],
                   add=[("Whale recovery boosts nutrient cycling and supports krill/fish productivity", 2, "decades")]),
    "elephants": dict(aliases=["elephant", "elephants"], role="mega-herbivore/seed disperser",
                      loss=[("Forest elephants disperse large-seeded trees; their loss shifts forests to smaller-seeded, lower-carbon species (~7% drop in carbon storage estimated)", 3, "decades-centuries"),
                            ("Savanna woody cover increases without browsing, reducing grazing habitat and changing fire regimes", 3, "decades"),
                            ("Waterholes and trails they create disappear, affecting many other species", 2, "years")],
                      add=[("Seed dispersal and gap creation increase forest/savanna heterogeneity", 2, "decades")]),
    "otters": dict(aliases=["otter", "otters", "sea otter", "sea otters"], role="kelp-forest keystone",
                   loss=[("Sea urchins boom and graze kelp to bare 'urchin barrens'", 4, "5-15 yrs"),
                         ("Kelp-dependent fish and invertebrates collapse; coastal carbon storage and wave buffering fall", 3, "10+ yrs")],
                   add=[("Urchin numbers drop and kelp forests rebound, raising fish production and carbon storage", 3, "5-20 yrs")]),
    "beavers": dict(aliases=["beaver", "beavers"], role="ecosystem engineer",
                    loss=[("Ponds and wetlands drain; habitat for amphibians, fish, waterfowl shrinks", 3, "5-20 yrs"),
                          ("Flood and drought buffering falls; streams become flashier", 3, "years")],
                    add=[("Dams create wetlands, raise biodiversity, slow floods and filter sediment", 3, "2-10 yrs")]),
    "bats": dict(aliases=["bat", "bats"], role="insect predators/pollinators",
                 loss=[("Insect pests surge; studies link white-nose bat declines to more pesticide use and higher infant mortality (~$ billions/yr in avoided pest costs)", 3, "1-5 yrs"),
                       ("Tropical plants pollinated/dispersed by bats (agave, durian, figs) decline", 3, "years-decades"),
                       ("Cave guano ecosystems collapse", 2, "years")],
                 add=[("More nocturnal insect control and seed dispersal", 2, "years")]),
    "mosquitoes": dict(aliases=["mosquito", "mosquitoes", "mosquitos"], role="blood-feeding insects",
                       loss=[("Malaria, dengue, Zika etc. largely vanish: ~600,000+ deaths/year avoided", 1, "immediate"),
                             ("Ecological impact is likely modest: only ~a dozen of ~3,500 species bite people, and other insects fill most pollinator/prey roles", 1, "1-5 yrs"),
                             ("Some fish, bats, and birds lose a prey item; arctic tundra midge/mosquito pulses matter for migratory birds", 2, "years")],
                       add=[("Disease burden rises; local food webs see a new prey base", 3, "years")]),
    "ants": dict(aliases=["ant", "ants"], role="soil engineers/seed dispersers",
                 loss=[("Soil aeration, nutrient turnover and seed dispersal drop sharply; arthropod and plant diversity fall", 4, "years-decades"),
                       ("Ant-plant mutualisms and many predator-prey chains unravel", 4, "years-decades")],
                 add=[("Invasive ants (fire, Argentine) typically cause biodiversity loss; native-range ants boost soil health", 2, "years")]),
    "earthworms": dict(aliases=["earthworm", "earthworms", "worms"], role="soil engineers",
                       loss=[("Litter decomposition and soil structure degrade, water infiltration and crop yields drop", 3, "years")],
                       add=[("In previously worm-free northern forests, earthworms strip the leaf-litter layer and harm native understorey plants", 3, "years")]),
    "trees": dict(aliases=["tree", "trees"], role="primary producers",
                  loss=[("Terrestrial carbon stocks (~450 Gt C in plants) released or lost; biosphere and climate collapse", 5, "years")],
                  add=[("Carbon uptake and habitat restoration where species and water are suitable", 2, "decades")]),
    "phytoplankton": dict(aliases=["phytoplankton", "plankton", "algae"], role="base of ocean food web, ~50% of oxygen",
                          loss=[("Marine food webs collapse within weeks; atmospheric oxygen falls over millennia (large reservoir)", 5, "weeks-years"),
                                ("Ocean carbon pump stops, amplifying warming", 5, "years")],
                          add=[("More algae can mean blooms; in iron-limited seas productivity rises but fixes carbon only briefly", 2, "years")]),
    "insects": dict(aliases=["insect", "insects", "bugs"], role="food-web base",
                    loss=[("Most birds, amphibians, reptiles, and many fish lose their main food; ~80% of wild plants lose pollinators", 5, "years"),
                          ("Decomposition stalls: dead matter and dung accumulate", 5, "years")],
                    add=[("n/a: insects are already everywhere", 1, "n/a")]),
    "fungi": dict(aliases=["fungi", "mushrooms", "mycorrhiza", "mycorrhizae"], role="decomposers/root partners",
                  loss=[("~90% of plants depend on mycorrhizal fungi; forests die back and litter stops decomposing", 5, "years")],
                  add=[("Soil fungal inoculation speeds restoration in degraded land", 2, "years")]),
    "coral": dict(aliases=["corals", "coral reefs", "coral reef", "reefs"], role="reef builders",
                  loss=[("~25% of marine species rely on reefs; fish stocks feeding ~1 billion people decline", 4, "years-decades"),
                        ("Coastlines lose a natural wave barrier: erosion and flood damage increase", 3, "decades")],
                  add=[("Reef restoration raises fish biomass but needs cool, clear, low-acid water", 2, "decades")]),
    "humans": dict(aliases=[], role="", loss=[], add=[]),
}


# ========================================================================================
# PARSER: Turn a free-text 'what if ...' question into a structured Spec.
# ========================================================================================
NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12, "twenty": 20,
             "fifty": 50, "hundred": 100, "thousand": 1000}
MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mil": 1e6, "billion": 1e9, "b": 1e9}

# first match wins, so order is priority order
EVENTS = [
    ("human_absence", r"\b(humans?|people|humanity|mankind)\b.{0,30}\b(disappear|vanish|gone|extinct|die|left|leave|absent)|without (humans|people|humanity)|\bno (more )?(humans|people)\b|\bhumans? (went|go) extinct"),
    ("nuclear", r"\bnukes?\b|nuclear|atomic|\ba-?bomb|\bh-?bomb|hydrogen bomb|thermonuclear|warhead|\bicbm|\bww ?(3|iii)\b|world war (3|iii|three)|tsar bomba|hiroshima|nagasaki"),
    ("asteroid", r"asteroid|meteor|comet|space rock|bolide|chicxulub"),
    ("volcano", r"volcan|erupt|caldera|krakatoa|pinatubo|tambora|st\.? helens"),
    ("oil_spill", r"oil spill|oil leak|oil tanker|tanker (spill|sink|sank|crash)|pipeline (burst|leak|rupture)|crude oil|deepwater|oil rig"),
    ("wildfire", r"(acres|hectares|km2|square (miles|kilometers)).{0,15}(burn|burned|burnt|on fire)|wild ?fires?|forest fires?|bush ?fires?|brush fires?|fire (swept|burn|hit|tore|ravage)|burn(s|ed|ing|t)?.{0,25}(acres|hectares|forest|down|through|km)"),
    ("deforestation", r"deforest|clear[- ]?cut|\blogg(ed|ing)\b|cut down|chop(ped)? down|bulldoz|cleared|lose .{0,20}forest|destroy(ed)? .{0,20}forest"),
    ("drought", r"drought|dry spell|no rain|water shortage|stop raining|stopped raining"),
    ("species", None),  # resolved by species lookup
    ("warming", r"warm|heat ?wave|climate change|hotter|temperature (rise|increase|goes)|\d\s*(°|degrees?)|global heating"),
]
LOSS_RE = r"extinct|disappear|die|died|dying|vanish|wiped|eradicat|gone|no more|without|killed|all dead|lost|remove|collapse"
ADD_RE = r"reintroduc|re-introduc|introduc|release|bring back|brought back|return(ed)?\b|restor|rewild|add(ed)?\b"


@dataclass
class Spec:
    text: str
    event: str | None = None
    places: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _num(s: str) -> float:
    s = s.replace(",", "")
    return float(s) if re.fullmatch(r"\d*\.?\d+", s) else float(NUM_WORDS.get(s, 1))


def find_places(text: str) -> list[Place]:
    found, taken = [], []
    for alias, place in alias_index():
        for m in re.finditer(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), place))
    seen, out = set(), []
    for _, p in sorted(found, key=lambda t: t[0]):
        if p.key not in seen:
            seen.add(p.key)
            out.append(p)
    return out


def find_species(text: str) -> str | None:
    for key, sp in SPECIES.items():
        for a in sp["aliases"]:
            if re.search(r"(?<![\w])" + re.escape(a) + r"(?![\w])", text):
                return key
    return None


_NUM = r"(\d[\d,]*\.?\d*|\.\d+)"
_MULT = r"(?:\s*(thousand|million|billion|mil|k|m|b)\b)?"


def _with_mult(m_num, m_mult) -> float:
    v = _num(m_num)
    return v * MULT.get((m_mult or "").lower(), 1)


def parse(text: str) -> Spec:
    t = text.lower().strip().rstrip("?.! ")
    t = re.sub(r"^(what (would|will|might|could) happen )?(if|when)\s+", "", t)
    spec = Spec(text=text)
    spec.places = find_places(t)

    # ---- event -------------------------------------------------------------------------
    sp_key = find_species(t)
    for name, pat in EVENTS:
        if name == "species":
            if sp_key and sp_key != "humans":
                spec.event = "species_add" if re.search(ADD_RE, t) and not re.search(LOSS_RE, t) else "species_loss"
                spec.params["species"] = sp_key
                break
        elif re.search(pat, t):
            spec.event = name
            break
    if spec.event is None and re.search(LOSS_RE, t) and sp_key:
        spec.event = "species_loss"
        spec.params["species"] = sp_key

    p = spec.params
    # ---- generic quantities --------------------------------------------------------------
    m = re.search(_NUM + r"\s*(%|percent|per cent)", t)
    if m:
        p["percent"] = float(m.group(1).replace(",", ""))

    m = re.search(_NUM + _MULT + r"\s*(hectares?|ha|acres?|km2|km²|sq\.? ?km|square kilomet(?:er|re)s?|sq\.? ?mi(?:les?)?|square miles?)\b", t)
    if m:
        v = _with_mult(m.group(1), m.group(2))
        u = m.group(3)
        p["area_km2"] = v * (0.01 if u.startswith(("hect", "ha")) else 0.004047 if u.startswith("acre") else 2.59 if "mi" in u else 1)

    m = re.search(_NUM + _MULT + r"\s*(barrels?|bbl|gallons?|litres?|liters?|tonnes?|tons?)\b", t)
    if m:
        v = _with_mult(m.group(1), m.group(2))
        u = m.group(3)
        p["oil_bbl"] = v * (1 if u.startswith(("barrel", "bbl")) else 1 / 42 if u.startswith("gal") else 1 / 159 if u[0] == "l" else 7.33)

    m = re.search(_NUM + r"\s*[- ]?(years?|yrs?|months?|decades?)\b", t)
    if m:
        v = float(m.group(1).replace(",", ""))
        p["years"] = v * (1 / 12 if m.group(2).startswith("mo") else 10 if m.group(2).startswith("dec") else 1)

    # ---- event-specific ------------------------------------------------------------------
    if spec.event == "nuclear":
        if "tsar bomba" in t:
            p["yield_kt"] = 50_000
        elif re.search(r"hiroshima|little boy", t):
            p["yield_kt"] = 15
        elif "nagasaki" in t or "fat man" in t:
            p["yield_kt"] = 21
        m = re.search(_NUM + r"\s*[- ]?(kt|kilotons?|mt|megatons?|gigatons?)\b", t)
        if m:
            v = float(m.group(1).replace(",", ""))
            u = m.group(2)
            p["yield_kt"] = v * (1 if u.startswith(("kt", "kilo")) else 1e3 if u.startswith(("mt", "mega")) else 1e6)
        m = re.search(r"\b(\d[\d,]*|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve|dozen|twenty|fifty|hundred|thousand)s?\s+(?:\w+\s+){0,2}?(nukes?|nuclear|warheads?|bombs?|missiles?|weapons?|icbms?)", t)
        if m:
            p["count"] = int(_num(m.group(1)))
        elif re.search(r"\b(nukes|warheads|bombs|missiles)\b", t):
            p["count"] = 10
            spec.notes.append('Plural "nukes" with no number: assuming 10.')
        if re.search(r"nuclear war|nuclear exchange|nuclear conflict|world war (3|iii|three)|\bww ?(3|iii)\b|all[- ]out|full[- ]scale|mutually assured", t):
            p["war"] = True
        if re.search(r"ground burst|surface burst|ground[- ]level|groundburst", t):
            p["ground"] = True
    elif spec.event == "asteroid":
        m = re.search(_NUM + r"\s*[- ]?(km|kilomet(?:er|re)s?|meters?|metres?|m|miles?|mi|feet|ft|foot)\b", t)
        if m:
            v = float(m.group(1).replace(",", ""))
            u = m.group(2)
            p["diameter_m"] = v * (1000 if u.startswith(("km", "kilo")) else 1609 if u.startswith("mi") else 0.3048 if u.startswith(("f")) else 1)
        elif "chicxulub" in t or "dinosaur" in t:
            p["diameter_m"] = 10_000
        m = re.search(_NUM + r"\s*(?:km/s|kps)", t)
        if m:
            p["speed_kms"] = float(m.group(1))
    elif spec.event == "volcano":
        m = re.search(r"vei\s*[-:]?\s*(\d)", t)
        if m:
            p["vei"] = int(m.group(1))
        elif re.search(r"super ?volcano|yellowstone|toba|campi flegrei", t):
            p["vei"] = 8
        else:
            for name, v in (("tambora", 7), ("krakatoa", 6), ("pinatubo", 6), ("st. helens", 5), ("st helens", 5)):
                if name in t:
                    p["vei"] = v
    elif spec.event == "warming":
        m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:°\s*|degrees?\s*|deg\s*)(celsius|fahrenheit|c|f)?(?![a-z])", t)
        if m:
            v = float(m.group(1))
            p["delta_c"] = v * 5 / 9 if (m.group(2) or "").startswith("f") else v
        if re.search(r"\b(from today|from now|above today|more than now|additional|another)\b", t):
            p["from_today"] = True
    elif spec.event in ("deforestation", "drought", "wildfire"):
        if re.search(r"\b(all|entire|whole|every)\b", t) and spec.event == "deforestation":
            p.setdefault("percent", 100.0)
    return spec


# ========================================================================================
# LLM: llama.cpp-backed scenario parser.
# ========================================================================================
# Reuse the Vulkan build + model already on this machine unless overridden.
DEFAULT_SERVER = r"C:\dev\machiavelli\vendor\llama.cpp\llama-server.exe"
DEFAULT_MODEL = r"C:\dev\machiavelli\models\Qwen3-8B-Q4_K_M.gguf"

EVENT_NAMES = ["nuclear", "asteroid", "volcano", "oil_spill", "wildfire", "deforestation", "drought",
          "warming", "species_loss", "species_add", "human_absence", "unknown"]

SCHEMA = {
    "type": "object",
    "properties": {
        "event": {"type": "string", "enum": EVENT_NAMES},
        "locations": {"type": "array", "items": {"type": "string"}},
        "species": {"type": ["string", "null"]},
        "nuclear_war": {"type": "boolean"},
        "ground_burst": {"type": "boolean"},
        "warhead_count": {"type": ["integer", "null"]},
        "yield_kilotons": {"type": ["number", "null"]},
        "diameter_meters": {"type": ["number", "null"]},
        "volcano_vei": {"type": ["integer", "null"]},
        "warming_celsius": {"type": ["number", "null"]},
        "percent": {"type": ["number", "null"]},
        "area_km2": {"type": ["number", "null"]},
        "oil_barrels": {"type": ["number", "null"]},
        "duration_years": {"type": ["number", "null"]},
    },
    "required": ["event", "locations"],
}

SYSTEM = (
    "You convert a user's 'what if' environmental scenario into JSON for an ecological simulator. "
    "Extract only what the user stated; use null when a quantity is not given - never invent numbers. "
    "Convert units: acres/hectares/square miles -> area_km2; gallons/tonnes -> oil_barrels; "
    "megatons -> kilotons; miles/feet/km -> meters; Fahrenheit deltas -> Celsius. "
    "event is one of: nuclear (nukes, bombs, nuclear war), asteroid (asteroid/comet/meteor impact), "
    "volcano (eruption), oil_spill, wildfire, deforestation (logging/clearing forest), drought, "
    "warming (temperature rise/climate change), species_loss (a species or group dies out or is removed), "
    "species_add (reintroduction/introduction), human_absence (humans vanish), unknown. "
    "locations: place names exactly as a reader would recognise them (countries, cities, regions, oceans); "
    "use 'world' for global scenarios. species: a plural common name like bees, wolves, sharks, whales, "
    "elephants, otters, beavers, bats, mosquitoes, ants, earthworms, trees, phytoplankton, insects, fungi, coral. "
    "nuclear_war is true for all-out/regional nuclear war or exchange; ground_burst only if stated. "
    "Reply with JSON only."
)


class LLMUnavailable(RuntimeError):
    pass


class LlamaParser:
    def __init__(self, url: str | None = None, server: str | None = None, model: str | None = None,
                 port: int = 8088, n_ctx: int = 4096, startup_timeout: float = 240.0):
        self.url = (url or os.environ.get("ECOFORECAST_LLM_URL") or "").rstrip("/")
        self.server = server or os.environ.get("ECOFORECAST_SERVER") or DEFAULT_SERVER
        self.model = model or os.environ.get("ECOFORECAST_MODEL") or DEFAULT_MODEL
        self.port, self.n_ctx, self.startup_timeout = port, n_ctx, startup_timeout
        self.proc: subprocess.Popen | None = None
        self.external = bool(self.url)
        if not self.url:
            self.url = "http://127.0.0.1:%d" % port

    # -- lifecycle ----------------------------------------------------------------------
    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def start(self) -> None:
        if self._healthy():
            return                        # something (maybe a previous run) already serves this port
        if self.external:
            raise LLMUnavailable("no llama-server answering at %s" % self.url)
        if not Path(self.server).exists():
            raise LLMUnavailable("llama-server not found at %s (set ECOFORECAST_SERVER)" % self.server)
        if not Path(self.model).exists():
            raise LLMUnavailable("model not found at %s (set ECOFORECAST_MODEL)" % self.model)
        log = Path(__file__).resolve().parent / "llama-server.log"
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        args = [self.server, "-m", self.model, "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(self.n_ctx), "-ngl", "999", "--no-webui", "--log-disable"]
        self.proc = subprocess.Popen(args, stdout=log.open("wb"), stderr=subprocess.STDOUT, creationflags=flags)
        atexit.register(self.stop)
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise LLMUnavailable("llama-server exited with code %s (see %s)" % (self.proc.returncode, log))
            if self._healthy():
                return
            time.sleep(1)
        self.stop()
        raise LLMUnavailable("llama-server did not become ready")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    # -- parsing ------------------------------------------------------------------------
    def _chat(self, text: str) -> dict:
        body = {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text + " /no_think"}],
            "temperature": 0, "max_tokens": 400,
            "response_format": {"type": "json_object", "schema": SCHEMA},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
        except (urllib.error.URLError, OSError) as exc:
            raise LLMUnavailable("llama-server request failed: %s" % exc) from exc
        content = out["choices"][0]["message"]["content"].strip()
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable("model returned invalid JSON: %r" % content[:200]) from exc

    def parse(self, text: str) -> Spec:
        self.start()
        data = self._chat(text)
        base = parse(text)          # regex result fills anything the model left out
        spec = Spec(text=text, event=base.event, places=base.places, params=dict(base.params),
                    notes=["Parsed by llama.cpp (%s)." % Path(self.model).name])

        ev = data.get("event")
        if ev in EVENT_NAMES and ev != "unknown":
            spec.event = ev
        places = []
        for loc in data.get("locations") or []:
            places += find_places(str(loc).lower())
        if places:
            spec.places = list({p.key: p for p in places}.values())
        elif data.get("locations"):
            spec.notes.append("Model saw location(s) %s but they aren't in the gazetteer." % data["locations"])

        p = spec.params
        sp = data.get("species")
        if sp:
            key = find_species(str(sp).lower())
            if key:
                p["species"] = key
        if spec.event in ("species_loss", "species_add") and "species" not in p:
            spec.event = base.event if base.event in ("species_loss", "species_add") else spec.event
        mapping = {"warhead_count": "count", "yield_kilotons": "yield_kt", "diameter_meters": "diameter_m",
                   "volcano_vei": "vei", "warming_celsius": "delta_c", "percent": "percent",
                   "area_km2": "area_km2", "oil_barrels": "oil_bbl", "duration_years": "years"}
        # Exact regex-extracted quantities win (the model slips on unit maths and can invent
        # numbers); the model only fills gaps, and never adds an area alongside a percentage.
        has_digits = bool(re.search(r"\d|(one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundred|thousand|million|billion)", text.lower()))
        for src, dst in mapping.items():
            if data.get(src) is None or dst in p or not has_digits:
                continue
            if dst == "area_km2" and "percent" in p:
                continue
            p[dst] = data[src]
        # the model over-eagerly flags "war"; trust it only when it isn't a handful of warheads
        if data.get("nuclear_war") and (p.get("count") or 99) >= 5:
            p["war"] = True
        elif "war" in p and (p.get("count") or 99) < 5:
            del p["war"]
        if data.get("ground_burst"):
            p["ground"] = True
        return spec


# ========================================================================================
# MODELS: Deterministic scenario models. Coarse, literature-anchored estimates - not precision forecasts.
# ========================================================================================
SEV = {1: "MINOR", 2: "MODERATE", 3: "MAJOR", 4: "SEVERE", 5: "CATASTROPHIC"}


@dataclass
class Report:
    title: str
    assumptions: list = field(default_factory=list)
    numbers: list = field(default_factory=list)      # (label, value)
    effects: list = field(default_factory=list)      # (severity, when, text)
    pop: list = field(default_factory=list)          # population-update entries (dicts)
    caveats: list = field(default_factory=list)

    def fx(self, sev, when, text):
        self.effects.append((sev, when, text))

    def num(self, label, value):
        self.numbers.append((label, value))

    def people(self, region, before, lo, hi, affected=0.0, label="injured / displaced / affected",
               per_year=False, note=""):
        """Record a population update. lo/hi = deaths range (negative = lives saved)."""
        self.pop.append(dict(region=region, before=before, lo=lo, hi=hi, affected=affected,
                             label=label, per_year=per_year, note=note))


WORLD_POP = 8.1e9


def dens(p) -> float:
    """People per km2: urban-core density for cities, else regional average."""
    if p is None or p.kind == "world":
        return 54.0
    if p.kind == "city":
        return p.density
    return p.pop / p.area if p.area and p.pop else 54.0


def _pop(p):
    return p.pop if p and p.pop else None


def interp(table, x):
    """Piecewise-linear interpolation, clamped at the ends."""
    if x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def fmt(v, unit=""):
    if v >= 1e9:
        s = "%.3g billion" % (v / 1e9)
    elif v >= 1e6:
        s = "%.3g million" % (v / 1e6)
    elif v >= 1e4:
        s = "{:,.0f}".format(round(v, -int(math.log10(v)) + 2))
    elif v >= 100:
        s = "{:,.0f}".format(v)
    elif v >= 1:
        s = "%.1f" % v
    else:
        s = "%.2g" % v
    return (s + " " + unit).strip()


def _place(spec: Spec, *kinds) -> Place | None:
    for p in spec.places:
        if not kinds or p.kind in kinds:
            return p
    return spec.places[0] if spec.places else None


def _where(p):
    return p.name if p else "an unspecified location"


# =====================================================================================
# lo, hi (deaths; |hi| <= 1 means fraction of population), affected, label, per_year, note
SPECIES_POP = {
    "bees": (0.25e6, 0.6e6, 1e9, "diets worsened (less fruit, veg, nuts)", True, "Smith et al. 2022 estimate ~0.43 M excess deaths/yr from lost pollination; staple grains largely unaffected."),
    "mosquitoes": (-0.8e6, -0.6e6, 0, "", True, "Negative = lives saved per year (malaria ~0.6 M plus dengue etc.)."),
    "coral": (0, 0, 5e8, "lose fish, tourism or shore protection", False, "~0.5-1 billion people draw food or coastal protection from reefs."),
    "bats": (0, 0, 2e7, "hit by pest/pesticide effects", False, "Linked to ~8% higher infant mortality in counties that lost bats."),
    "phytoplankton": (.9, .99, 0, "", False, "Food web and ~half of oxygen production collapse; near-total human die-off over years."),
    "insects": (.8, .98, 0, "", False, "Crop pollination and decomposition fail; civilisation-ending famine."),
    "fungi": (.8, .98, 0, "", False, "Plant growth and decomposition fail; civilisation-ending famine."),
    "trees": (.6, .95, 0, "", False, "Oxygen, rainfall, food and building materials lost; most people die over years."),
}


def nuclear(spec: Spec) -> Report:
    P = spec.params
    places = spec.places
    Y = P.get("yield_kt")
    n = P.get("count")
    war = P.get("war", False) or (n or 0) >= 20
    r = Report("Nuclear " + ("war" if war else "detonation") + " - " + (", ".join(p.name for p in places) or "unspecified location"))

    if war and n is None:
        if places and {p.key for p in places} <= {"india", "pakistan"}:
            n, Y = 100, Y or 15
            r.assumptions.append("Regional India-Pakistan style exchange: 100 x 15 kt weapons on cities (the scenario studied by Toon/Coupe/Xia).")
        else:
            n, Y = 3000, Y or 300
            r.assumptions.append("Full-scale exchange: ~3,000 warheads of ~300 kt detonated on cities/industry (order of a US-Russia war).")
    if Y is None:
        Y = 300
        r.assumptions.append("No yield given: assumed 300 kt, a typical modern strategic warhead (Hiroshima was 15 kt).")
    n = n or 1
    ground = P.get("ground", False)
    r.assumptions.append("%s burst. %d warhead(s) of %s kt." % ("Surface" if ground else "Optimal-height air", n, fmt(Y)))

    # target
    target = _place(spec)
    city = None
    if target and target.kind == "city":
        city = target
    elif target and target.city:
        city = PLACES.get(target.city)
        if not war:
            r.assumptions.append("'%s' is a whole region, so the strike is assumed to hit its largest city, %s." % (target.name, city.name))
    if city is None:
        r.assumptions.append("No target city recognised: assumed a city of 5,000 people/km2.")
    density = city.density if city else 5000
    cpop = city.pop if city else 5e6

    # single-weapon physics (Glasstone-Dolan cube-root scaling, rounded)
    k = 0.8 if ground else 1.0
    r_fire = 0.05 * Y ** 0.4 * (1.3 if ground else 1.0)
    r5 = 0.69 * Y ** (1 / 3) * k
    r_th = 0.9 * Y ** 0.41 * (0.7 if ground else 1.0)
    a_fire, a5, a_th = (math.pi * x * x for x in (r_fire, r5, r_th))
    r.num("Fireball / crater radius", "%.2f km (%s km2)" % (r_fire, fmt(a_fire)))
    r.num("Heavy blast radius (5 psi, trees flattened)", "%.1f km (%s km2)" % (r5, fmt(a5)))
    r.num("Thermal burn/ignition radius (3rd degree)", "%.1f km (%s km2)" % (r_th, fmt(a_th)))

    if not war or n <= 20:
        tot5 = a5 * n
        if target and target.area:
            r.num("Share of %s inside blast zone" % target.name, "%.4f%%" % (100 * min(1, tot5 / target.area)))
        exposed = min(cpop, a5 * density) * n
        if target and target.pop:
            exposed = min(exposed, 0.6 * target.pop)
        r.num("People inside the 5 psi ring (very rough)", fmt(exposed))
        if n > 1 and not war:
            r.assumptions.append("Each warhead is assumed to hit a separate city like the target city (capped at 60%% of the region's population).")

    # local ecology
    bkey, binfo = biome_info(target if target and target.kind != "city" else (PLACES.get("mexico") if False else None))
    eco = ", ".join(BIOMES[b]["label"] for b in (target.biomes if target else []) if b in BIOMES and b != "urban")
    if eco:
        r.assumptions.append("Ecosystems in the wider region: %s." % eco)
    r.fx(5, "seconds", "Everything within %.2f km is vaporised or excavated%s; soil is fused, nothing survives." % (r_fire, " (a crater of ~%.0f m)" % (r_fire * 1000 * 0.9) if ground else ""))
    r.fx(4, "seconds-hours", "Trees and structures are flattened out to ~%.1f km (%s km2 per weapon); vegetation and soil fauna in the zone are destroyed or buried under debris." % (r5, fmt(a5)))
    r.fx(4, "minutes-days", "Thermal pulse ignites dry fuel out to ~%.1f km; in cities and dry forest this can merge into a firestorm, burning up to ~%s km2 and emitting smoke and black carbon." % (r_th, fmt(a_th * 0.5)))
    if ground:
        L = 4.5 * Y ** 0.45
        area = math.pi / 4 * L * (0.12 * L)
        r.fx(4, "hours-weeks", "Surface burst lofts tons of irradiated soil: a lethal fallout plume (>500 rem unsheltered) extends ~%.0f km downwind over ~%s km2 per weapon." % (L, fmt(area)))
    else:
        r.fx(2, "hours-weeks", "Air burst means little local fallout; fine radioactive particles still disperse widely and fall out in rain.")
    r.fx(3, "weeks-years", "Radiocaesium-137 (30-yr half-life) and strontium-90 (29 yr) enter soils, water and food webs; iodine-131 (8 days) dosing is short but intense, mostly through milk and leafy vegetables.")
    r.fx(3, "months", "Conifers are the most radiosensitive plants (Chernobyl's 'Red Forest' pines died at tens of Gy); mammals die at ~5-8 Gy, insects and many plants tolerate far more - so community composition shifts toward resilient species.")
    r.fx(1, "years-decades", "Lessons from Chernobyl, Bikini and Hiroshima: vegetation regrows within weeks-months and, with people excluded, wildlife populations can rebound strongly despite chronic radiation - though mutation and cancer rates stay elevated locally.")

    # global: soot + climate
    if Y >= 1:
        soot_per = 0.05 * min(1.0, (Y / 15) ** 0.5)
        if not war and city is None:
            soot_per *= 0.3
        if ground:
            soot_per *= 0.8
        soot = soot_per * n
        r.num("Soot lofted to stratosphere (total)", "%.2f Tg" % soot)
        if soot >= 1:
            cool = interp([(0, 0), (5, 1.0), (16, 2.5), (27, 4), (37, 5), (47, 6), (150, 8.5)], soot)
            cal = interp([(0, 0), (5, 7), (16, 14), (27, 22), (37, 30), (47, 38), (150, 90)], soot)
            ozone = interp([(0, 0), (5, 8), (47, 25), (150, 50)], soot)
            sev = 5 if soot >= 27 else 4 if soot >= 5 else 3
            r.num("Global cooling (peak, 1-3 yrs)", "~%.1f C" % cool)
            r.num("Global crop calories (years 3-5)", "-%.0f%%" % cal)
            r.fx(sev, "1-10 yrs", "Stratospheric smoke blocks sunlight: global mean temperature drops by ~%.1f C (far more over continental interiors) for several years, rainfall falls, growing seasons shorten." % cool)
            r.fx(sev, "1-5 yrs", "Crop production falls ~%.0f%% on average (Xia et al. 2022 range); most nations' reserves last months, so famine risk is global even in countries never targeted." % cal)
            r.fx(sev, "1-5 yrs", "Marine primary productivity falls as the surface cools and dims; fisheries are hit after a lag, and ocean acidity and mixing change for decades.")
            r.fx(min(5, sev), "2-10 yrs", "Smoke heating destroys stratospheric ozone (~%.0f%% globally), so UV-B surges as the smoke clears, stressing plants, phytoplankton and amphibians." % ozone)
            if soot >= 27:
                r.fx(5, "years-millennia", "Nuclear-winter scale: mass die-offs of plants and large animals, collapse of many food webs, global biodiversity crash; recovery takes centuries to millennia.")
        else:
            r.fx(1, "global", "Soot and dust from this few warheads is too small for global climate effects (<1 Tg); impacts stay regional (possible transient regional cooling, acid/particulate pollution).")

    # population update
    scope = places if places else [PLACES["world"]]
    before = WORLD_POP if (war and not places) else (sum(_pop(x) or 0 for x in scope) or cpop)
    per5 = min(cpop, a5 * density)
    perth = min(cpop, a_th * density)
    ring = max(0.0, perth - per5)
    lo = n * (0.4 * per5 + 0.05 * ring)
    hi = n * (0.9 * per5 + 0.3 * ring) * (1.15 if ground else 1.0)
    inj = n * (0.3 * per5 + 0.5 * ring)
    cap = (0.4 if war and n >= 20 else 0.6) * before
    lo, hi = min(lo, cap), min(hi, cap)
    r.people(", ".join(x.name for x in scope), before, lo, hi, max(0.0, min(inj, before - hi)),
             "injured (burns, blast, radiation)",
             note="Deaths from blast, fire and fallout in the first weeks; people assumed unsheltered and unwarned.")
    soot_v = soot if Y >= 1 else 0
    if soot_v >= 1:
        fam = interp([(0, 0), (5, .044), (16, .11), (27, .19), (37, .26), (47, .32), (150, .65)], soot_v)
        r.people("World (famine, years 2-5)", WORLD_POP, 0.6 * fam * WORLD_POP, min(0.95, 1.4 * fam) * WORLD_POP,
                 note="On top of the direct deaths above: crop-failure deaths after nuclear winter, in countries never targeted (Xia 2022 scale; no adaptation or trade assumed).")
    r.caveats += ["Blast/thermal radii use rounded cube-root scaling; real results depend on height of burst, weather, terrain and fuels.",
                  "Nuclear winter soot, cooling and crop numbers are interpolated from published model runs (Coupe 2019, Xia 2022) and carry large uncertainty."]
    return r


# =====================================================================================
def asteroid(spec: Spec) -> Report:
    P = spec.params
    t = _place(spec)
    d = P.get("diameter_m")
    v = P.get("speed_kms", 20)
    r = Report("Asteroid impact - " + _where(t))
    if d is None:
        d = 100
        r.assumptions.append("No size given: assumed a 100 m rocky asteroid.")
    r.assumptions.append("Stony (3000 kg/m3), %.0f km/s, 45-degree entry." % v)
    rho = 3000
    E = 0.5 * rho * math.pi / 6 * d ** 3 * (v * 1000) ** 2
    Mt = E / 4.184e15
    ocean = bool(t and (t.kind == "ocean" or "ocean" in t.biomes))
    r.num("Impact energy", "%s megatons TNT (~%s Hiroshima bombs)" % (fmt(Mt), fmt(Mt * 1000 / 15)))
    if d < 50:
        r.num("Type", "airburst, no crater")
        a = 2150 * (Mt / 12) ** (2 / 3)
        r.num("Forest flattened / area damaged", "~%s km2" % fmt(a))
        r.fx(3, "seconds", "Airburst (Chelyabinsk/Tunguska-class): blast and thermal flash damage ~%s km2, flattening trees and igniting fuel under the burst." % fmt(a))
        r.fx(1, "years", "No lasting global effect; forest regrows, leaving a local scar.")
    else:
        Dtc = 1.161 * (rho / 2500) ** (1 / 3) * d ** 0.78 * (v * 1000) ** 0.44 * 9.81 ** -0.22 * math.sin(math.radians(45)) ** (1 / 3)
        Dc = 1.25 * Dtc
        r.num("Crater diameter" if not ocean else "Transient cavity (ocean)", "%.1f km" % (Dc / 1000))
        a_dev = math.pi * (Dc / 2000 * (4 + 3 * math.log10(max(Mt, 1)))) ** 2
        r.fx(5, "seconds", "Vaporisation and ejecta blanket within ~%.0f km of ground zero; %s km2 suffers near-total ecological destruction." % (Dc / 1000 * 2, fmt(a_dev)))
        r.fx(4, "minutes-hours", "Airblast, thermal flash and seismic shaking (magnitude ~%.1f) ignite regional forests and scatter hot ejecta." % min(9.5, 4 + 0.6 * math.log10(max(E, 1e9)) - 3.6))
        if ocean:
            r.fx(4, "hours", "Ocean impact raises a tsunami; coastlines within hundreds to thousands of km (depending on size) are flooded, salinating soils and wiping out coastal ecosystems.")
        if d >= 200:
            r.fx(3 if d < 1000 else 4, "months-years", "Stratospheric dust, sulphate and soot cool the planet by a few tenths to a few degrees and dim sunlight.")
        if d >= 1000:
            cool = interp([(1000, 1), (3000, 4), (10000, 15)], d)
            r.fx(5 if d >= 5000 else 4, "years-decades", "Global impact winter: ~%.0f C cooling, crop failure worldwide, photosynthesis collapse for months to years." % cool)
            r.fx(5 if d >= 5000 else 3, "decades-millennia", "Mass extinction risk: land and ocean food webs collapse%s." % (" - a Chicxulub-class (10 km) impact wiped out ~75% of species" if d >= 5000 else ""))
        if d >= 10000:
            r.fx(5, "hours-days", "Re-entering ejecta heats the sky and sets fire to much of the world's forest; most large land animals die in the first days.")
        if d < 200:
            r.fx(2, "years", "Effects stay regional (hundreds of km); no lasting global climate signal.")
    Aloc = 2150 * (Mt / 12) ** (2 / 3) if d < 50 else math.pi * (Dc / 2000 * (4 + 3 * math.log10(max(Mt, 1)))) ** 2
    if ocean:
        fr = interp([(50, 0), (100, .0002), (200, .001), (500, .005), (1000, .02)], d)
        if fr:
            r.people("Coastal populations (tsunami)", WORLD_POP, 0.5 * fr * WORLD_POP, 2 * fr * WORLD_POP,
                     note="Basin-wide tsunami exposure; scales steeply with size and distance to shore.")
    else:
        b = _pop(t)
        exp = min(b or 1e18, Aloc * dens(t))
        big = d >= 50
        r.people(t.name if t else "Impact zone (average world density)", b or exp,
                 (0.3 if big else 0.05) * exp, (0.9 if big else 0.3) * exp, 0.5 * exp,
                 "injured (blast, burns, collapse)" if big else "injured (glass, burns)",
                 note="People inside the destruction zone at %.0f people/km2." % dens(t))
    if d >= 1000:
        fam = interp([(1000, .02), (3000, .3), (10000, .9)], d)
        r.people("World (impact-winter famine)", WORLD_POP, 0.6 * fam * WORLD_POP, min(1, 1.4 * fam) * WORLD_POP,
                 note="Crop failure under dust and soot for years; very uncertain.")
    r.caveats.append("Crater scaling from Collins et al. (2005), simplified; damage radii are approximate and ignore target geology.")
    return r


# =====================================================================================
def volcano(spec: Spec) -> Report:
    P = spec.params
    t = _place(spec)
    vei = int(P.get("vei", 4))
    r = Report("Volcanic eruption (VEI %d) - %s" % (vei, _where(t)))
    if "vei" not in P:
        r.assumptions.append("No size given: assumed VEI 4 (Eyjafjallajokull-class).")
    V = 10 ** (vei - 4) * 0.1 if vei >= 4 else 10 ** (vei - 4) * 0.1
    ash_area = 1500 * max(V, .01) ** 0.9
    r.num("Erupted volume", "~%s km3" % fmt(V))
    r.num("Area with >10 cm ash (very rough)", "~%s km2" % fmt(ash_area))
    sev = min(5, max(1, vei - 2))
    r.fx(sev, "hours-days", "Pyroclastic flows, lahars and ashfall bury ecosystems near the vent; heavy ash collapses forest canopy, smothers crops and clogs streams across ~%s km2." % fmt(ash_area))
    r.fx(max(1, sev - 1), "weeks-years", "Ash-fertilised soils and lakes eventually bloom (iron and phosphorus), but acid rain and fluoride poison grazing animals and fish in the short term.")
    if vei >= 5:
        cool = interp([(5, 0.2), (6, 0.5), (7, 0.7), (8, 3.5)], vei)
        r.num("Global cooling", "~%.1f C for 1-3 yrs" % cool)
        r.fx(sev, "1-3 yrs", "Sulphate aerosols cool the planet ~%.1f C and shift monsoons and rainfall; crop failures and famine are likely (Tambora 1815 'year without a summer')." % cool)
    if vei >= 7:
        r.fx(4, "years", "Ozone depletion and multi-year harvest shortfalls on several continents; marine productivity dips.")
    if vei >= 8:
        r.fx(5, "years-decades", "Supervolcano: continental-scale ashfall (metres near vent, tens of cm across thousands of km), volcanic winter with multi-degree cooling, collapse of grassland food webs, global famine.")
        r.fx(4, "millennia", "Biosphere recovers over centuries; Toba-type events may have bottlenecked populations.")
    rz = 10 * (max(V, .01) / 0.1) ** 0.25
    b = _pop(t)
    inside = min(b or 1e18, math.pi * rz * rz * dens(t))
    ash_pop = min(b or 1e18, ash_area * dens(t))
    r.people(t.name if t else "Area around the vent (average world density)", b or inside, 0.1 * inside, 0.4 * inside,
             max(ash_pop - inside, 0), "displaced / affected by ashfall",
             note="Deaths in the ~%.0f km pyroclastic and lahar zone; ashfall adds respiratory harm, roof collapse and evacuation." % rz)
    if vei >= 6:
        fam = interp([(6, .0005), (7, .01), (8, .15)], vei)
        r.people("World (volcanic-winter famine)", WORLD_POP, 0.6 * fam * WORLD_POP, 1.4 * fam * WORLD_POP,
                 note="Harvest failures from aerosol cooling; poorly constrained.")
    r.caveats.append("Ash dispersal depends strongly on wind and eruption column height; aerosol impact scales with sulphur, not just volume.")
    return r


# =====================================================================================
def wildfire(spec: Spec) -> Report:
    P = spec.params
    t = _place(spec)
    a = P.get("area_km2")
    r = Report("Wildfire - " + _where(t))
    if a is None:
        a = 1000
        r.assumptions.append("No area given: assumed 1,000 km2 (100,000 ha), a large but typical megafire.")
    bkey, b = biome_info(t, prefer_forest=True)
    r.assumptions.append("Dominant burned biome: %s." % b["label"])
    c = a * 100 * b["fire_c"]          # tonnes C
    co2 = c * 3.67
    r.num("Area burned", "%s km2" % fmt(a))
    if t and t.area and t.kind != "city":
        r.num("Share of %s" % t.name, "%.2f%%" % (100 * a / t.area))
    r.num("CO2 released", "~%s tonnes (%s cars for a year)" % (fmt(co2), fmt(co2 / 4.6)))
    sev = 4 if a > 5000 or bkey in ("rainforest", "boreal") else 3 if a > 500 else 2
    r.fx(sev, "days-weeks", "Canopy and understorey burn; slow-moving animals (reptiles, amphibians, small mammals, nestlings) die in large numbers, larger ones flee and face crowding and starvation.")
    r.fx(3, "weeks-months", "Smoke raises PM2.5 over hundreds-thousands of km; burned slopes lose soil cover and heavy rain brings ash, sediment and nutrients into rivers (fish kills, drinking-water problems).")
    r.fx(2, "years", "Recovery of %s: %s." % (b["label"], b["recover"]))
    if bkey == "rainforest":
        r.fx(4, "decades", "Tropical rainforests are not fire-adapted: a burn makes the stand drier and more flammable, risking permanent shift to degraded savanna/scrub.")
    if bkey in ("boreal", "tundra", "wetland"):
        r.fx(4, "years-decades", "Peat and permafrost carbon can smoulder for months or overwinter ('zombie fires'), adding far more CO2 than the canopy alone.")
    if bkey in ("savanna", "grassland"):
        r.fx(1, "weeks", "These biomes are fire-adapted; burned patches green up within weeks and many species benefit.")
    b = _pop(t)
    dis = min(b or 1e18, a * dens(t)) * 0.6
    r.people(t.name if t else "Burn area (average world density)", b or dis, 0.0005 * dis, 0.004 * dis, dis,
             "evacuated / displaced",
             note="Direct deaths are rare; smoke kills far more people than flames (PM2.5 deaths not included).")
    r.caveats.append("Carbon figures use average fuel consumption per biome; actual fire severity varies widely.")
    return r


# =====================================================================================
def oil_spill(spec: Spec) -> Report:
    P = spec.params
    t = _place(spec)
    bbl = P.get("oil_bbl")
    r = Report("Oil spill - " + _where(t))
    if bbl is None:
        bbl = 500_000
        r.assumptions.append("No volume given: assumed 500,000 barrels (~2x Exxon Valdez).")
    coast = 2100 * (bbl / 257_000) ** 0.6
    r.num("Volume", "%s barrels (%s tonnes)" % (fmt(bbl), fmt(bbl * 0.136)))
    r.num("Shoreline oiled (Exxon Valdez-scaled)", "~%s km" % fmt(coast))
    lo, hi = bbl * 0.1, bbl * 1.0
    r.num("Seabirds killed (range)", "~%s - %s" % (fmt(lo), fmt(hi)))
    r.fx(4 if bbl > 200_000 else 3, "days-weeks", "Oiled seabirds, sea otters, seals and turtles die of hypothermia, poisoning and ingestion; fish eggs and larvae are killed by toxic PAHs.")
    r.fx(3, "weeks-months", "Plankton, filter-feeders and intertidal life are smothered; the fishery and tourism economy closes.")
    sensitive = t and any(b in t.biomes for b in ("mangrove", "wetland", "coral"))
    if sensitive:
        r.fx(4, "years-decades", "Mangrove, marsh and reef sediments trap oil; it can persist for decades and re-oil on storms, stunting recovery (Exxon Valdez oil is still present in beach sediments).")
    else:
        r.fx(2, "years-decades", "Open rocky shores clean faster through wave action, but buried oil pockets persist (Exxon Valdez herring and some otter groups took decades to recover).")
    if bbl > 1_000_000:
        r.fx(3, "months", "Deepwater-scale release forms subsurface plumes and oxygen-depleted zones, harming deep corals, bluefin tuna spawning and dolphins.")
    r.fx(2, "1-10 yrs", "Oil-eating microbes degrade much of the light fraction within months; heavier fractions and tar mats last years.")
    r.people(t.name if t else "Affected coast", _pop(t) or 0, 0, 0.0001 * coast * 150, coast * 150,
             "livelihoods hit (fishing, tourism, cleanup exposure)",
             note="About 150 coastal residents per km of shoreline assumed; direct deaths are rare.")
    r.caveats.append("Bird and shoreline numbers are scaled from Exxon Valdez (1989) and Deepwater Horizon (2010); oil type, season, and response matter enormously.")
    return r


# =====================================================================================
def deforestation(spec: Spec) -> Report:
    P = spec.params
    t = _place(spec)
    r = Report("Deforestation - " + _where(t))
    if t and t.forest_frac and t.area:
        forest = t.area * t.forest_frac
    else:
        forest, t_note = 40_000_000, True
        r.assumptions.append("No forested region recognised: using the world's ~40 million km2 of forest.")
    bkey, b = biome_info(t, prefer_forest=True)
    if "area_km2" in P:
        lost = min(P["area_km2"], forest)
        frac = lost / forest
    else:
        frac = min(1.0, P.get("percent", 30) / 100)
        if "percent" not in P:
            r.assumptions.append("No amount given: assumed 30% of the forest is cleared.")
        lost = forest * frac
    r.num("Forest in scope", "%s km2" % fmt(forest))
    r.num("Forest cleared", "%s km2 (%.0f%%)" % (fmt(lost), frac * 100))
    carbon = lost * 100 * (b["stock"] or 60)
    r.num("Carbon released", "~%s tonnes CO2" % fmt(carbon * 3.67))
    sp = 1 - (1 - frac) ** 0.25
    r.num("Eventual species loss in that habitat (species-area, z=0.25)", "~%.0f%%" % (sp * 100))
    r.fx(4 if frac > 0.3 else 3, "years", "Habitat loss eliminates canopy specialists first; fragmentation isolates populations and raises edge effects.")
    r.fx(3 if frac > 0.2 else 2, "years", "Less evapotranspiration cuts regional rainfall and raises local temperature; soil erodes and rivers silt up.")
    r.fx(4 if sp > 0.3 else 3, "decades", "Extinction debt: ~%.0f%% of the habitat's species are expected to disappear over decades, even if clearing stops." % (sp * 100))
    if t and t.key in ("amazon", "brazil") or (bkey == "rainforest" and frac >= .2):
        r.fx(5 if frac >= .25 else 3, "decades", "Amazon tipping point: beyond ~20-25% total deforestation (already ~17-20%) the basin may flip to savanna, dieback accelerating itself.")
    b = _pop(t) or WORLD_POP
    r.people(t.name if t else "World", b, 0, 0, b * frac * 0.25, "livelihood, water or climate dependent",
             note="A quarter of residents assumed forest-adjacent, scaled by the share cleared; indirect heat, smoke and disease effects not counted.")
    r.caveats.append("Carbon stocks are biome averages (aboveground + roots); species-area scaling is a rough guide.")
    return r


# =====================================================================================
def drought(spec: Spec) -> Report:
    P = spec.params
    t = _place(spec)
    yrs = P.get("years", 3)
    r = Report("Drought - " + _where(t))
    if "years" not in P:
        r.assumptions.append("No duration given: assumed a 3-year severe drought.")
    bkey, b = biome_info(t)
    sev = 2 if yrs < 1 else 3 if yrs < 3 else 4 if yrs < 8 else 5
    r.num("Duration", "%.1f years" % yrs)
    r.num("Crop yield loss per dry year (typical severe)", "~10-30%")
    r.fx(sev, "months", "Rivers, lakes and wetlands shrink; fish, amphibians and waterfowl concentrate, then die as oxygen drops and temperatures rise.")
    r.fx(sev, "1-3 yrs", "Drought-stressed trees lose defence against bark beetles and fire; mass tree mortality (as in the SW US and Australia) rises with each year.")
    if yrs >= 3:
        r.fx(4, "years", "Aquifers and soil carbon drop; wildfire frequency and severity rise sharply.")
    r.fx(2, "years", "Main ecosystem at risk here: %s; recovery: %s once rains return." % (b["label"], b["recover"]))
    if yrs >= 8:
        r.fx(5, "decades", "Megadrought-scale: shrubland and desert replace forests and grasslands; human migration follows water.")
    b = _pop(t) or WORLD_POP
    share = min(1, 0.3 + 0.1 * yrs)
    aff = b * share
    r.people(t.name if t else "World", b, aff * yrs * 0.0005, aff * yrs * 0.005, aff * 0.05 * yrs, "displaced / food insecure",
             note="Mortality 0.05-0.5%% of affected people per drought-year (Horn of Africa 2011 was ~2.6%% in the worst zones); ~%.0f%% of residents assumed affected." % (100 * share))
    r.caveats.append("Impacts depend heavily on baseline aridity and water management; this is a generic scaling.")
    return r


# =====================================================================================
def warming(spec: Spec) -> Report:
    P = spec.params
    t = _place(spec)
    dT = P.get("delta_c")
    r = Report("Global warming scenario - " + (_where(t) if t else "world"))
    if dT is None:
        dT = 2.0
        r.assumptions.append("No amount given: assumed +2.0 C.")
    if P.get("from_today"):
        dT += 1.3
        r.assumptions.append("Interpreted as added to today's ~1.3 C, so the total is %.1f C above pre-industrial." % dT)
    else:
        r.assumptions.append("Temperature is relative to pre-industrial (the planet is already ~1.3 C above it).")
    r.num("Warming (vs pre-industrial)", "+%.1f C" % dT)
    ext = interp([(1.0, 2.8), (2.0, 5.2), (3.0, 8.5), (4.3, 16), (6, 26)], dT)
    coral = interp([(1.0, 10), (1.5, 80), (2.0, 99), (2.1, 99)], dT)
    r.num("Species at high extinction risk (Urban 2015 meta-analysis)", "~%.0f%%" % ext)
    r.num("Tropical coral reef loss", "~%.0f%%" % coral)
    r.num("Shift of climate zones", "~17 km/decade poleward (per degree, roughly)")
    sev = 2 if dT < 1.5 else 3 if dT < 2.5 else 4 if dT < 4 else 5
    r.fx(sev, "decades", "Species track their climate niches poleward and uphill; mountain-top and Arctic specialists run out of room.")
    r.fx(4 if coral > 70 else 3, "years-decades", "Mass bleaching: ~%.0f%% of warm-water reefs are lost, removing habitat for about a quarter of marine species." % coral)
    r.fx(sev, "decades", "Heat extremes and wildfire seasons lengthen; tree mortality from heat and drought rises, weakening forest carbon sinks.")
    r.fx(sev, "decades", "Crop yields fall roughly 3-7%% per degree for wheat, rice, maize and soy without adaptation (~%.0f%% total)." % (5 * dT))
    if dT >= 2:
        r.fx(4, "decades", "Arctic summers become ice-free about once per decade; polar bears, ringed seals and ice algae lose habitat.")
    if dT >= 1.5:
        r.fx(sev, "centuries", "Committed sea-level rise of roughly 2-3 m per degree over centuries to millennia drowns coastal wetlands, mangroves and deltas unless they can migrate.")
    if dT >= 3:
        r.fx(5, "decades", "Risk of tipping points: Amazon dieback, permafrost carbon release, ice-sheet collapse and AMOC slowdown.")
    if t and any(b in t.biomes for b in ("tundra", "polar")):
        r.fx(4, "decades", "High latitudes warm 2-4x faster than the global mean; permafrost thaws, turning tundra into a net carbon source.")
    share = (_pop(t) / WORLD_POP) if _pop(t) else 1.0
    extra = max(dT - 1.3, 0)
    r.people(t.name if t else "World", _pop(t) or WORLD_POP, 0.1e6 * extra * share, 0.6e6 * extra * share * max(1, dT / 2),
             100e6 * extra * share, "displaced by end of century (heat, sea level, crop failure)", per_year=True,
             note="Extra heat and disease deaths per year beyond today's level (WHO ~250k/yr scale); displacement is cumulative.")
    r.caveats.append("Threshold values are summarised from IPCC AR6 and Urban (2015); extinction values beyond +4.3 C are extrapolated.")
    return r


# =====================================================================================
def species(spec: Spec, add=False) -> Report:
    key = spec.params.get("species")
    t = _place(spec)
    sp = SPECIES.get(key)
    if not sp:
        raise ValueError("unknown species")
    r = Report("%s %s - %s" % ("Reintroducing" if add else "Loss of", key, _where(t) if t else "worldwide"))
    r.assumptions.append("Role: %s." % sp["role"])
    r.assumptions.append("Scope: %s." % ("the named region" if t else "global") + (" All individuals removed." if not add else ""))
    rows = sp["add"] if add else sp["loss"]
    for text, sev, when in rows:
        r.fx(sev, when, text)
    if not add and t and t.kind not in ("world", "ocean"):
        r.assumptions.append("Regional removal is partially offset by recolonisation from neighbouring areas over time.")
    if not add:
        b = WORLD_POP if (not t or t.kind == "world") else (_pop(t) or WORLD_POP)
        k = b / WORLD_POP
        if key in SPECIES_POP:
            lo, hi, aff, lab, per_year, note = SPECIES_POP[key]
            if abs(hi) <= 1:               # given as a fraction of the population
                lo, hi = lo * b, hi * b
            else:
                lo, hi = lo * k, hi * k
            r.people(t.name if t and t.kind != "world" else "World", b, lo, hi, aff * k, lab or "affected", per_year, note)
        else:
            r.people(t.name if t and t.kind != "world" else "World", b, 0, 0, 0,
                     note="No direct human mortality documented; only indirect food, fisheries and disease effects.")
    r.caveats.append("Qualitative: based on documented cases of keystone losses/returns, not a food-web simulation.")
    return r


# =====================================================================================
def human_absence(spec: Spec) -> Report:
    r = Report("A world without humans")
    r.fx(5, "days-weeks", "Unattended nuclear plants (~400+ reactors) lose cooling as grids fail: many meltdowns and local contamination, plus fires and chemical releases from unattended industry.")
    r.fx(2, "days-months", "Livestock and pets die, escape, or go feral; cropland is invaded by weeds; street lights go out.")
    r.fx(1, "1-10 yrs", "Fields go to scrub; urban weeds crack pavement; wolf, deer and coyote populations expand across former suburbia; invasive species' spread slows as trade stops.")
    r.fx(1, "10-100 yrs", "Forest returns across most temperate land; bridges and buildings start to collapse; rivers reclaim their floodplains; large mammals and fisheries rebound (see the Korean DMZ and Chernobyl).")
    r.fx(2, "100-1000 yrs", "Plastics, PCBs, radionuclides and metals persist; many species remain lost forever, but megafauna populations rebuild. Atmospheric CO2 falls slowly as oceans and plants absorb it.")
    r.fx(2, "thousands of yrs", "Most visible structures erode away; the geological record keeps the 'Anthropocene' layer of plastics, concrete and isotopes.")
    r.people("World", WORLD_POP, WORLD_POP, WORLD_POP, note="By definition; how they vanish is left open.")
    r.caveats.append("Follows the standard 'world without us' reasoning (e.g. Weisman 2007); timescales are illustrative.")
    return r


MODELS = {
    "nuclear": nuclear, "asteroid": asteroid, "volcano": volcano, "wildfire": wildfire,
    "oil_spill": oil_spill, "deforestation": deforestation, "drought": drought, "warming": warming,
    "species_loss": lambda s: species(s, False), "species_add": lambda s: species(s, True),
    "human_absence": human_absence,
}


def run(spec: Spec) -> Report:
    if spec.event not in MODELS:
        raise ValueError(spec.event)
    return MODELS[spec.event](spec)


# ========================================================================================
# CLI: Command line: `python -m ecoforecast "what if a nuke hit mexico"` or run with no args for a REPL.
# ========================================================================================
HELP_EXAMPLES = [
    "what if a nuke hit mexico",
    "what if 100 nukes hit india and pakistan",
    "what if a 1 km asteroid hits the atlantic ocean",
    "what happens if yellowstone erupts",
    "what if 500000 acres burned in california",
    "what if all the bees died",
    "what if wolves were reintroduced to scotland",
    "what if the world warms 3 degrees",
    "what if 50% of the amazon is cut down",
    "what if humans disappeared",
]


def render(spec: Spec, rep: Report, width: int = 100) -> str:
    wrap = lambda s, ind="  ", sub="    ": textwrap.fill(s, width, initial_indent=ind, subsequent_indent=sub)
    out = ["", "=" * width, rep.title.upper(), "=" * width, 'Question: "%s"' % spec.text]
    for n in spec.notes:
        out.append(wrap(n, "  * ", "    "))
    if rep.assumptions:
        out += ["", "Assumptions"] + [wrap(a, "  - ") for a in rep.assumptions]
    if rep.numbers:
        out += ["", "Key numbers"] + ["  %-58s %s" % (k + ":", v) for k, v in rep.numbers]
    if rep.pop:
        out += ["", "Population update (approximate, 2024 baseline)"]
        for e in rep.pop:
            out.append("  " + e["region"] + (" - per year" if e["per_year"] else ""))
            b, lo, hi = e["before"], e["lo"], e["hi"]
            if b:
                out.append("    Population before:   ~%s" % fmt(b))
            if hi < 0:
                out.append("    Lives saved:         ~%s - %s" % (fmt(-hi), fmt(-lo)))
            elif hi > 0:
                out.append("    Deaths:              ~%s - %s%s" % (fmt(lo) if lo else "0", fmt(hi),
                           "  (%.1f%% - %.1f%%)" % (100 * lo / b, 100 * hi / b) if b and not e["per_year"] else ""))
            else:
                out.append("    Deaths:              ~0 (no direct deaths modelled)")
            if e["affected"]:
                out.append("    Affected:            ~%s (%s)" % (fmt(e["affected"]), e["label"]))
            if b and hi > 0 and not e["per_year"]:
                out.append("    Population after:    ~%s - %s" % (fmt(max(0, b - hi)), fmt(max(0, b - lo))))
            if e["note"]:
                out.append(textwrap.fill(e["note"], width, initial_indent="    ~ ", subsequent_indent="      "))
    out += ["", "Ecological effects (most severe first)"]
    for sev, when, text in sorted(rep.effects, key=lambda e: -e[0]):
        out.append(wrap("[%s | %s] %s" % (SEV[sev], when, text), "  ", "      "))
    if rep.caveats:
        out += ["", "Caveats"] + [wrap(c, "  ~ ", "    ") for c in rep.caveats]
    return "\n".join(out + [""])


def to_json(spec: Spec, rep: Report) -> str:
    return json.dumps({
        "question": spec.text, "event": spec.event, "places": [p.key for p in spec.places],
        "params": spec.params, "title": rep.title, "assumptions": rep.assumptions,
        "numbers": dict(rep.numbers), "population": rep.pop,"numbers": dict(rep.numbers),
        "effects": [{"severity": SEV[s], "when": w, "text": t} for s, w, t in rep.effects],
        "caveats": rep.caveats}, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ecoforecast", description="Predict ecological outcomes of 'what if' scenarios.")
    ap.add_argument("question", nargs="*", help="scenario in plain English (omit for interactive mode)")
    ap.add_argument("--no-llm", action="store_true", help="skip llama.cpp, use the built-in regex parser")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--server", help="path to llama-server(.exe)")
    ap.add_argument("--model", help="path to a .gguf model")
    ap.add_argument("--url", help="use an already-running llama-server, e.g. http://127.0.0.1:8080")
    args = ap.parse_args(argv)

    llm = None
    if not args.no_llm:
        llm = LlamaParser(url=args.url, server=args.server, model=args.model)

    def handle(q: str) -> int:
        nonlocal llm
        spec = None
        if llm:
            try:
                spec = llm.parse(q)
            except LLMUnavailable as exc:
                print("[llm unavailable: %s]\n[falling back to regex parser]" % exc, file=sys.stderr)
                llm = None
        if spec is None:
            spec = parse(q)
            spec.notes.append("Parsed by built-in regex parser.")
        if spec.event is None:
            print("I couldn't tell what kind of event that is. Try e.g.:\n  " + "\n  ".join(HELP_EXAMPLES))
            return 1
        try:
            rep = run(spec)
        except ValueError as exc:
            print("Couldn't model that (%s). Try e.g.:\n  %s" % (exc, "\n  ".join(HELP_EXAMPLES)))
            return 1
        print(to_json(spec, rep) if args.json else render(spec, rep))
        return 0

    if args.question:
        return handle(" ".join(args.question))
    print("EcoForecast - describe a scenario (blank line or 'quit' to exit). Examples:\n  " + "\n  ".join(HELP_EXAMPLES))
    while True:
        try:
            q = input("\nwhat if> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("", "q", "quit", "exit"):
            break
        handle(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
