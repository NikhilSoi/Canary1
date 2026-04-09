"""
canary_api.py
-------------
Canary assessment endpoint — local proof of concept.
Loads ofs.json directly from disk, passes to Claude, returns citation-first flags.

Requirements:
    pip3 install fastapi uvicorn anthropic

Usage:
    python3 canary_api.py

Then POST to:
    http://localhost:8000/assess

Example with curl:
    curl -X POST http://localhost:8000/assess \
      -H "Content-Type: application/json" \
      -d @test_institution.json
"""

import json
import os
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    import anthropic
except ImportError:
    import sys
    sys.exit("Run: pip3 install fastapi uvicorn anthropic")


# ── Config ────────────────────────────────────────────────────────────────────

STANDARDS_DIR = Path("./data/standards")
app = FastAPI(title="Canary API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class InstitutionData(BaseModel):
    institution_name: str
    evidence: dict  # free-form evidence keyed by theme


class RiskFlag(BaseModel):
    condition_id: str
    condition_title: str
    risk_level: str          # green | amber | red
    finding: str
    evidence_cited: str
    recommendation: str


class AssessmentResponse(BaseModel):
    institution: str
    standards_version: str
    flag_count: dict         # {green: n, amber: n, red: n}
    flags: list


# ── Standards loader ──────────────────────────────────────────────────────────

def load_standards() -> dict:
    all_conditions = []
    for f in STANDARDS_DIR.glob("*.json"):
        data = json.load(open(f))
        all_conditions.extend(data.get("conditions", []))
    return {
        "body": "OfS",
        "document": "Full UK Regulatory Framework",
        "version_date": "2024",
        "conditions": all_conditions,
        "condition_count": len(all_conditions)
    }


# ── Assessment prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Canary, a regulatory risk assessment agent for UK higher education.

You assess institutions against OfS (Office for Students) Conditions of Registration.

RULES:
- Every flag MUST cite a specific condition ID (e.g. B1, B3) and a specific criterion
- Risk levels: green (compliant), amber (watch), red (breach risk)
- Be specific — reference actual evidence provided, not generic statements
- Return ONLY valid JSON, no markdown, no commentary
- If evidence is insufficient to assess a condition, flag it amber with finding "Insufficient evidence to assess"
- Prioritise red flags — these represent genuine regulatory risk

OUTPUT FORMAT:
{
  "flags": [
    {
      "condition_id": "B1",
      "condition_title": "Academic experience",
      "risk_level": "red|amber|green",
      "finding": "Specific finding based on evidence provided",
      "evidence_cited": "The specific piece of evidence that led to this finding",
      "recommendation": "Concrete action to address this"
    }
  ]
}"""


def build_prompt(standards: dict, institution: InstitutionData) -> str:
    # Trim standards to essential fields to stay within context
    conditions_summary = []
    for c in standards.get("conditions", []):
        conditions_summary.append({
            "condition_id": c["condition_id"],
            "title": c.get("title", ""),
            "criteria": [
                {
                    "ref": cr.get("ref", ""),
                    "text": cr.get("text", "")[:300],
                    "risk_indicators": cr.get("risk_indicators", [])
                }
                for cr in c.get("criteria", [])[:5]
            ]
        })

    return f"""
REGULATORY STANDARDS (OfS Conditions of Registration):
{json.dumps(conditions_summary, indent=2)}

INSTITUTION: {institution.institution_name}

EVIDENCE SUBMITTED:
{json.dumps(institution.evidence, indent=2)}

Assess this institution against every condition above.
Return a flag for each condition — green, amber, or red.
Be specific. Cite the evidence. Every flag needs a recommendation.
"""


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/assess", response_model=AssessmentResponse)
async def assess(institution: InstitutionData):

    # Load standards
    try:
        standards = load_standards()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Call Claude
    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=6000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": build_prompt(standards, institution)
            }]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {e}")

    # Parse response
    raw = response.content[0].text.strip()
    import re
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Could not parse Claude response: {raw[:200]}")

    flags = result.get("flags", [])

    # Count by risk level
    flag_count = {"green": 0, "amber": 0, "red": 0}
    for f in flags:
        level = f.get("risk_level", "amber").lower()
        if level in flag_count:
            flag_count[level] += 1

    return AssessmentResponse(
        institution=institution.institution_name,
        standards_version=standards.get("version_date", "2024"),
        flag_count=flag_count,
        flags=flags
    )


@app.get("/standards")
async def get_standards():
    """Returns the loaded standards metadata — useful for debugging."""
    standards = load_standards()
    return {
        "body": standards.get("body"),
        "document": standards.get("document"),
        "condition_count": standards.get("condition_count"),
        "conditions": [
            {
                "condition_id": c["condition_id"],
                "title": c.get("title", ""),
                "criteria_count": len(c.get("criteria", []))
            }
            for c in standards.get("conditions", [])
        ]
    }


@app.get("/health")
async def health():
    standards_ok = STANDARDS_PATH.exists()
    return {
        "status": "ok" if standards_ok else "degraded",
        "standards_loaded": standards_ok,
        "standards_path": str(STANDARDS_PATH)
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Canary API ──────────────────────────────")
    print(f"   Standards path : {STANDARDS_PATH}")
    print(f"   Standards exist: {STANDARDS_PATH.exists()}")
    print(f"\n   Endpoints:")
    print(f"   POST http://localhost:8000/assess")
    print(f"   GET  http://localhost:8000/standards")
    print(f"   GET  http://localhost:8000/health")
    print(f"\n   Docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
