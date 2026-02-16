# Research Engineering Practices

## Problem
Research teams often struggle with engineering practices:
- Product engineering is overkill (enterprise patterns don't fit research)
- Pure research code is unmaintainable (notebooks, no tests, no structure)
- No clear guidance on "minimum viable engineering" for research

## Solution
A curated guide that bridges research and product engineering:
- Literature review of industry labs (Google, Meta, DeepMind)
- Practical recommendations (schemas, unit tests, type hints)
- Team rituals adapted for research (async-first, protect deep work)
- Clear priorities: reusability over polish

## Contents
- READING_LIST.md - authoritative sources
- PRACTICES.md - actionable engineering practices (3 tiers)
- RITUALS.md - team ritual templates

## Key Insight
Modern Python (3.10+) + Pydantic + basic pytest = enough for most research code.

See PROJECT_SPEC.md for full details.
