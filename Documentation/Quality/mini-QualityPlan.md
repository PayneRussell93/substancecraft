# SubstanceCraft Mini Quality Plan

- **Project:** SubstanceCraft
- **Version:** 0.1.0
- **Scope:** Initial Rust project model and the first mesh-to-texture-painting vertical slice.
- **Profile:** Lightweight ISO/IEC 29110-style records for a small project.
- **Quality objectives:** Keep domain logic UI-independent; reject malformed project data without panicking; verify model invariants with automated tests; preserve a clean-room implementation.
- **Verification:** Run formatting, workspace tests, and Clippy with warnings denied for each change. Add focused tests for each new validation or user-visible behavior. Record any unavailable checks in the change summary.
- **Exit criteria:** The changed slice passes its focused tests and static checks; project data is versioned and validated; no known critical failures remain open.
- **Review:** Review milestones and active risks when the roadmap changes materially.

This starter plan is process guidance, not a claim of formal ISO certification or completed audit evidence. Requirement and risk records are maintained in the adjacent CSV files.