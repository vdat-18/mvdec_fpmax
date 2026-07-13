@RTK.md

## Code Quality Principles

Every code change must satisfy:

- **SOC** (Separation of Concerns) — each module/function handles one concern; don't mix data access, business logic, and presentation.
- **DRY** (Don't Repeat Yourself) — no duplicated logic; one authoritative place per rule/behavior.
- **KISS** (Keep It Simple, Stupid) — simplest solution that fully solves the problem; no unnecessary complexity.
- **YAGNI** (You Aren't Gonna Need It) — don't build for hypothetical future needs; only what's actually required now.
- **Maintainability** — code should be easy for another developer to understand, modify, and extend safely.
- **Security** — validate inputs at trust boundaries, avoid known vulnerability patterns (injection, unsafe deserialization, etc.).
- **Cleanliness** — consistent style, clear naming, no dead code or leftover debug artifacts.
