"""
ofs_parser.py
-------------
Parses the OfS Conditions of Registration PDF into structured JSON.
Output: ofs.json — the first file in Canary's /data/standards/ mount.

Requirements:
    pip install pdfplumber anthropic

OfS PDF source:
    https://www.officeforstudents.org.uk/media/ors5djqt/conditions-of-registration.pdf

Usage:
    python ofs_parser.py --pdf conditions-of-registration.pdf
    python ofs_parser.py --pdf conditions-of-registration.pdf --output ./data/standards/ofs.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    sys.exit("Missing dependency: pip install pdfplumber")

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic")


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = {
    "body": "string — regulatory body name",
    "document": "string — document title",
    "version_date": "string — last updated date if visible, else empty string",
    "market": "string — UK",
    "level": "string — HE",
    "conditions": [
        {
            "condition_id": "string — e.g. B3, C1, E2",
            "title": "string — full condition title",
            "category": "string — one of: Quality | Standards | Governance | Student | Finance | Access",
            "criteria": [
                {
                    "ref": "string — numbered ref e.g. B3.1, B3.2 (infer if not explicit)",
                    "text": "string — full criterion text verbatim",
                    "threshold": "string — what must be demonstrated or achieved, brief",
                    "risk_indicators": ["string — observable signs of non-compliance"]
                }
            ],
            "notes": "string — any context, definitions, or scope notes from the document"
        }
    ]
}


# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """Extract all text from the PDF, joining pages with clear separators."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[PAGE {i+1}]\n{text.strip()}")
    return "\n\n".join(pages)


def find_condition_pages(text: str) -> list[str]:
    """
    Split the full document text into per-condition chunks.
    OfS conditions follow a pattern: capital letter + number (B1, B2 ... E2).
    This keeps each chunk manageable for the API call.
    """
    # Pattern matches OfS condition headers like "Condition B1", "Condition B2" etc.
    pattern = r"(?=Condition\s+[A-E]\d+)"
    chunks = re.split(pattern, text, flags=re.IGNORECASE)
    # Remove preamble/intro sections (before first real condition)
    condition_chunks = [c for c in chunks if re.match(r"Condition\s+[A-E]\d+", c.strip(), re.IGNORECASE)]
    return condition_chunks if condition_chunks else [text]


# ── Claude parsing ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a regulatory document parser for UK higher education.
Your job is to extract structured data from OfS (Office for Students) Conditions of Registration.

Rules:
- Return ONLY valid JSON. No markdown fences, no preamble, no commentary.
- Extract verbatim text for criterion fields — do not paraphrase.
- If a condition has no explicit sub-numbering, infer refs as B3.1, B3.2 etc. in sequence.
- For risk_indicators: generate 2–4 observable signs that would indicate an institution
  is at risk of failing this criterion. These are Canary's detection signals.
- category must be one of: Quality | Standards | Governance | Student | Finance | Access
"""

CONDITION_PROMPT = """Parse this OfS condition into the following JSON schema.
Return the single condition object only (not wrapped in an array).

Schema:
{schema}

Document text:
{text}
"""

FULL_DOC_PROMPT = """Parse this OfS Conditions of Registration document into the following JSON schema.
The document may contain multiple conditions — extract all of them.

Schema:
{schema}

Document text:
{text}
"""


def parse_with_claude(text: str, client: anthropic.Anthropic, mode: str = "full"):
    """Send text to Claude and return parsed JSON."""

    if mode == "condition":
        prompt = CONDITION_PROMPT.format(
            schema=json.dumps(SCHEMA["conditions"][0], indent=2),
            text=text[:8000]
        )
    else:
        prompt = FULL_DOC_PROMPT.format(
            schema=json.dumps(SCHEMA, indent=2),
            text=text[:15000]
        )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    return json.loads(raw)


# ── Assembly ───────────────────────────────────────────────────────────────────

def build_output(pdf_path: str, client: anthropic.Anthropic) -> dict:
    """Full pipeline: extract → chunk → parse → assemble."""

    print(f"Extracting text from: {pdf_path}")
    full_text = extract_text(pdf_path)
    print(f"Extracted {len(full_text):,} characters across all pages")

    chunks = find_condition_pages(full_text)
    print(f"Found {len(chunks)} condition chunk(s)")

    conditions = []

    if len(chunks) <= 1:
        # Single-pass for short or unparseable structure
        print("Parsing full document in single pass...")
        result = parse_with_claude(full_text, client, mode="full")
        if isinstance(result, dict) and "conditions" in result:
            conditions = result["conditions"]
        elif isinstance(result, list):
            conditions = result
    else:
        # Per-condition chunks — more accurate, stays within token limits
        for i, chunk in enumerate(chunks):
            condition_id = re.match(r"Condition\s+([A-E]\d+)", chunk.strip(), re.IGNORECASE)
            label = condition_id.group(1) if condition_id else f"chunk_{i+1}"
            print(f"  Parsing condition {label} ({len(chunk):,} chars)...")
            try:
                condition = parse_with_claude(chunk, client, mode="condition")
                if isinstance(condition, dict):
                    conditions.append(condition)
                elif isinstance(condition, list):
                    conditions.extend(condition)
            except json.JSONDecodeError as e:
                print(f"  WARNING: JSON parse failed for {label}: {e}")
                conditions.append({
                    "condition_id": label,
                    "title": "Parse error — review manually",
                    "category": "Unknown",
                    "criteria": [],
                    "notes": chunk[:500]
                })

    return {
        "body": "OfS",
        "document": "Conditions of Registration",
        "version_date": "",
        "market": "UK",
        "level": "HE",
        "source_pdf": str(Path(pdf_path).name),
        "condition_count": len(conditions),
        "conditions": conditions
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_output(data: dict) -> list[str]:
    """Basic sanity checks on the output. Returns list of warnings."""
    warnings = []
    if not data.get("conditions"):
        warnings.append("No conditions extracted")
        return warnings
    for c in data["conditions"]:
        if not c.get("condition_id"):
            warnings.append(f"Missing condition_id in: {c.get('title', '?')}")
        if not c.get("criteria"):
            warnings.append(f"No criteria for condition {c.get('condition_id', '?')}")
        for crit in c.get("criteria", []):
            if not crit.get("risk_indicators"):
                warnings.append(f"No risk_indicators for {c.get('condition_id')}.{crit.get('ref', '?')}")
    return warnings


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse OfS Conditions of Registration PDF into JSON")
    parser.add_argument("--pdf", required=True, help="Path to the OfS conditions PDF")
    parser.add_argument("--output", default="ofs.json", help="Output JSON file path (default: ofs.json)")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        sys.exit(f"PDF not found: {args.pdf}")

    client_kwargs = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    client = anthropic.Anthropic(**client_kwargs)

    print("\n── OfS Parser ──────────────────────────────")
    data = build_output(args.pdf, client)

    warnings = validate_output(data)
    if warnings:
        print(f"\n── Validation warnings ({len(warnings)}) ──")
        for w in warnings:
            print(f"  ⚠  {w}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n── Done ──────────────────────────────────────")
    print(f"   Conditions extracted : {data['condition_count']}")
    print(f"   Output written to    : {output_path}")
    print(f"\nNext: add qaa_parser.py, khda_parser.py using the same pattern")
    print(f"      then load ofs.json into /data/standards/ in Supabase Storage\n")


if __name__ == "__main__":
    main()
