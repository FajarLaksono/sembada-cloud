---
name: root-cause-analysis
description: Systematic methodology for debugging, issue investigation, and root cause analysis of software defects and production incidents.
license: MIT
compatibility: opencode
metadata:
  audience: support-engineers
  workflow: investigation
---

## What I Do

I provide a structured approach for investigating and debugging issues:

1. **Reproduce** — Determine exact steps, environment, and conditions to reliably trigger the issue
2. **Isolate** — Narrow down the faulty component using binary search, logging, or divide-and-conquer
3. **Analyze** — Examine logs, stack traces, metrics, and code paths to identify the failure point
4. **Diagnose** — Apply techniques like:
   - **Five Whys** — Iterative questioning to trace symptoms to root cause
   - **Fishbone (Ishikawa)** — Categorize potential causes (people, process, code, config, environment)
   - **Fault Tree Analysis** — Top-down logical deduction of failure paths
   - **Time Correlation** — Cross-reference logs, metrics, and events on a timeline
5. **Document** — Write a clear RCA report covering:
   - Summary of the issue and impact
   - Timeline of events
   - Root cause(s) identified
   - Contributing factors
   - Action items and prevention plan

## When to Use Me

- Production incidents and outages
- Flaky tests or intermittent failures
- Performance regressions
- Unexplained errors or crashes
- Logic defects and edge cases
- Security vulnerability analysis

## Methodology

Always follow this order:
1. Gather data first — logs, traces, metrics, screenshots, error messages
2. Formulate hypotheses — list possible explanations
3. Test each hypothesis — the simplest or most likely first
4. Confirm root cause — ensure you can reproduce the fix
5. Propose corrective actions — short-term mitigation + long-term prevention
