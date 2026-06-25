from pathlib import Path

from paranoid_qa.models import make_llm

OUT = Path("data/corpus/sample")
OUT.mkdir(exist_ok=True)
SCENARIOS = [
    "a single-engine trainer that lost power shortly after takeoff",
    "a twin-engine charter flight that encountered severe icing in cruise",
    "a helicopter air-ambulance with a tail-rotor control failure",
    "a regional turboprop runway excursion during a gusting crosswind",
    "a light sport aircraft fuel-exhaustion forced landing",
    "a business jet pressurization fault at high altitude",
]

llm = make_llm(temperature=0.7)
SYS = (
    "You write short, self-contained fictional aviation incident reports for a demo corpus. "
    "200-300 words. Include a fictional registration, date, location, sequence of events, "
    "findings, and a one-line probable cause. Entirely fictional; do not reference real events."
)
for i, s in enumerate(SCENARIOS, 1):
    text = llm.invoke([("system", SYS), ("human", f"Incident: {s}.")]).content
    (OUT / f"report_{i:02d}.txt").write_text(str(text).strip() + "\n")
    print("wrote", OUT / f"report_{i:02d}.txt")
