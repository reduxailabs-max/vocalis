# Task 8 report: docs and changelog

Status: DONE (docs only; no source/test changes; tests not run, as instructed).

Updated: packaging/installer/README.md, docs/user-docs.md, docs/architecture.md (new "Stage 8: installer wizard" section; stale "hard gate" lines at the flow diagram fixed), docs/file-purpose.md (new section + preflight row fixed), docs/folder-structure.md, docs/workflow.md. Created .claude/2026-10-30_0141_CHANGELOG.md.

Facts verified in code: phases/flags, default flow, exit codes 0/1/3 and fallback for 3/126/127/>=128, STEP/NOTE parsing, process-group terminate (SIGTERM then SIGKILL after 3 s), 4096 MB threshold, 5 wizard pages, Settings 3 tries.

Observations / concerns:
- install.log is written only by the wizard (wizard_pages._append_log); a terminal install does not write it. Docs say so.
- Wizard "Try again" re-runs only the failed phase; a closed/stopped wizard needs the installer re-run. Docs match.
- Design spec was not re-read line by line; docs follow the built code. No contradiction found.
- The old Stage 3 workflow.md section ("Installing Vocalis (Stage 3)") still describes the pre-Stage-8 linear flow and the "GPU driver" preflight; left as history, the new Stage 8 section follows it.
- Real-PC checks are listed in the changelog and architecture doc.
