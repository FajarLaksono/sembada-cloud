# AI Engineering Specification

> Defines how AI-assisted development (via opencode) is configured, governed, and
> used in the Sembada Cloud project.

| Status | Last Updated | Author |
|--------|-------------|--------|
| Draft  | 2026-05-14  | Engineering |

---

## 1. Purpose

Standardize the AI engineering toolchain so that every contributor gets
consistent, predictable behaviour from opencode — regardless of their
experience level. This document covers:

- Agent architecture and when to use each agent
- Permission model (what AI can and cannot do without approval)
- Custom commands for common tasks
- Skills (loadable instruction sets for domain-specific work)
- Configuration reference

---

## 2. Design Principles

1. **Least privilege** — AI agents start restricted; permissions are granted
   explicitly per operation.
2. **Transparency** — All AI actions are logged and reversible (`/undo`).
3. **Domain awareness** — Skills encode project-specific knowledge so the AI
   doesn't guess.
4. **Progressive complexity** — Built-in agents cover 90% of work; custom
   agents are added only when a clear gap exists.

---

## 3. Agent Architecture

### 3.1 Primary Agents (cycle with `Tab`)

| Agent | Mode | When to use |
|-------|------|-------------|
| `build` | primary (default) | Writing code, running tests, making changes — full access |
| `plan` | primary | "How should I implement X?" — read-only analysis, no edits |

### 3.2 Sub-agents (invoke with `@name`)

| Agent | Mode | When to use |
|-------|------|-------------|
| `explore` | subagent (built-in) | "Find where X is defined" — fast read-only search |
| `general` | subagent (built-in) | Complex multi-step tasks, parallel research |
| `scout` | subagent (built-in) | Research external dependencies, inspect library source |
| `SupportEngineer` | subagent (custom) | Debugging, root cause analysis, incident investigation |

### 3.3 Decision: Why only one custom agent

The original setup defined 7 custom agents (EngineeringManager, DataEngineer,
DataAnalyst, CloudExpert, SoftwareEngineer, CodeReviewer, SupportEngineer).
Most duplicated functionality already provided by built-in agents:

| Custom agent (removed) | Replaced by |
|------------------------|-------------|
| EngineeringManager | `plan` + `general` |
| DataEngineer | `build` (data pipeline work is just code) |
| DataAnalyst | `build` + `general` |
| CloudExpert | `general` + web research |
| SoftwareEngineer | `build` (identical role) |
| CodeReviewer | `plan` (code review without edits) |

Keeping only `SupportEngineer` avoids agent menu clutter while preserving a
specialist for the one workflow that differs significantly from general
development.

---

## 4. Permission Model

### 4.1 Global defaults (`opencode.json`)

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "read": {
      "*": "allow",
      "*.env": "deny",
      "*.env.*": "deny",
      "*.env.example": "allow"
    },
    "bash": {
      "*": "ask",
      "git *": "allow",
      "pytest *": "allow",
      "black *": "allow",
      "python -m app.src.*": "ask",
      "python -m app.src.qa_report": "allow"
    },
    "edit": "ask",
    "webfetch": "ask",
    "websearch": "ask",
    "external_directory": "deny"
  }
}
```

### 4.2 Per-agent overrides

Global rules apply to all agents. Built-in agents additionally re-assert the
read deny patterns at the agent level (defence in depth). The per-agent
overrides are:

| Agent | `read` (sensitive files) | `edit` | `bash` | `external_directory` | `task` |
|-------|-------------------------|--------|--------|----------------------|--------|
| `build` | deny `.env`, `*credential*`, `*secret*`, `*.pem`, `*.key`, `*password*`, `*token*` | `ask` | as above | `deny` | `ask` |
| `plan` | same as `build` | `deny` | `deny` | `deny` | `allow` |
| `explore` | same as `build` | `deny` | `deny` | `deny` | `deny` |
| `general` | same as `build` | `ask` | as above | `deny` | `ask` |
| `scout` | same as `build` | `deny` | `ask` | `deny` | `deny` |
| `SupportEngineer` | same as `build` | `deny` | granular (see agent file) | `deny` | `ask` |

The `plan` agent is deliberately locked down — it can analyse and suggest but
never modify files. Switch to `build` when ready to implement.

All agents also have a **system prompt directive** that explicitly instructs
the AI: *"Never read .env, credential, secret, password, token, .pem, or .key
files. Never access files or directories outside the project worktree. Never
output or log sensitive values in tool calls or responses."* This creates a
three-layer security model:

1. **Global config** — broad pattern-based denials in `opencode.json`
2. **Agent config** — per-agent re-assertion of sensitive-file denials
3. **System prompt** — behavioural directive the AI must follow

---

## 5. Custom Commands

Defined in `opencode.json` under the `command` key:

| Command | Action | Safe for CI? |
|---------|--------|-------------|
| `/test` | `pytest app/tests/ -v` | Yes |
| `/test-file {file}` | `pytest app/tests/{file} -v` | Yes |
| `/lint` | `black --check app/src/ app/tests/ && flake8 app/src/` | Yes |
| `/format` | `black app/src/ app/tests/` | Yes |
| `/qa` | `python -m app.src.qa_report` | Yes |
| `/notebook-03a` | `papermill notebooks/03a_feature_engineering.ipynb NUL --log-output --progress-bar --execution-timeout 600` | No (100GB data) |
| `/notebook-03b` | `papermill notebooks/03b_tabular_models.ipynb NUL --log-output --progress-bar --execution-timeout 600` | No |
| `/notebook-03c` | `papermill notebooks/03c_timeseries_forecasting.ipynb NUL --log-output --progress-bar --execution-timeout 600` | No |
| `/ci-local` | `pytest app/tests/ -v && python -m app.src.qa_report` | Yes |

Notebook commands are excluded from CI because they require the full Azure
dataset (100GB+ CPU readings) that cannot fit in CI runners. They are
designated local-only.

---

## 6. Skills

Skills are loadable instruction sets that an agent pulls in on demand via the
`skill` tool. They are NOT in the context permanently — only when the agent
determines they are relevant.

### 6.1 `root-cause-analysis` (existing)

- **Location**: `.opencode/skills/root-cause-analysis/SKILL.md`
- **Triggered by**: SupportEngineer agent (via `skill: allow`)
- **Content**: 5-step RCA methodology (Reproduce Isolate Analyse
  Diagnose Document), techniques (Five Whys, Fishbone, Fault Tree), report
  template.

### 6.2 `feature-engineering` (new)

- **Location**: `.opencode/skills/feature-engineering/SKILL.md`
- **Purpose**: Prevents agents from making mistakes when modifying `features.py`
  or `03a_feature_engineering.ipynb`.
- **Content**:
  - Bucket parsing rules (core: `>24` -> 48, memory: `>64` -> 128, defaults)
  - Target variable definitions (is_idle < 5% CPU, waste_tier Low/Medium/High)
  - Temporal feature encoding (sin/cos for hour and day-of-week)
  - Leakage rules: never use target-related columns as features
  - Pricing lookup logic (core_bucket mem_bucket -> rate_per_hour -> vm_cost)

### 6.3 `model-evaluation` (new)

- **Location**: `.opencode/skills/model-evaluation/SKILL.md`
- **Purpose**: Ensures models meet the project's Quality Insurance gates.
- **Content**:
  - Acceptance thresholds: MAPE <= 15%, R >= 0.70, F1 >= 0.85
  - Evaluation methodology: chronological train/test split (no leakage),
    regression + classification + clustering + timeseries tasks
  - Model comparison: comparison_table() with best-value highlighting
  - QA report generation: `python -m app.src.qa_report`

---

## 7. Configuration Files

### 7.1 `opencode.json` (project root)

Primary configuration. Managed in version control. See Appendix A for full
contents.

Includes three layers of security:
- **Global permissions** deny `.env`, credential, secret, and key file reads
- **Agent-level permissions** re-assert the same denials for every built-in
  agent (build, plan, explore, general, scout)
- **System prompts** instruct the AI to never access sensitive files or paths
  outside the project worktree

External directory access is denied globally — agents cannot read files outside
the project root without approval.

### 7.2 `AGENTS.md` (project root)

Project context that opencode reads at startup. Contains:
- Project description and tech stack
- Code standards (PEP 8, type hints, docstrings)
- Development commands reference
- CI/CD workflow documentation

Updated whenever the toolchain changes.

### 7.3 `.opencode/agents/` (per-project)

Custom agent definitions (currently only `SupportEngineer.md`). One file per
agent in YAML frontmatter + Markdown body format. See opencode documentation
for schema.

### 7.4 `.opencode/skills/` (per-project)

Loadable skill definitions. Each skill is a directory containing `SKILL.md`
with YAML frontmatter (`name`, `description`) and Markdown body.

---

## 8. Workflow Guide

### 8.1 Daily development

```
$ opencode                    # Opens TUI in build mode (default)

# Make changes interactively
"Add a Ridge regression variant to models.py"

# Run tests
/test

# Format if linter fails
/format

# Check model quality
/qa
```

### 8.2 Planning a feature

```
# Tab -> switch to plan mode
"Design a new clustering evaluation metric"

# Review the plan, then Tab -> back to build
"Implement the plan above"

/test
```

### 8.3 Debugging a failure

```
@SupportEngineer "The test_model.py::test_evaluate is failing with
KeyError for 'roc_auc'. Investigate."
```

The SupportEngineer will load the `root-cause-analysis` skill and follow the
structured debug methodology autonomously.

### 8.4 Working on feature engineering

```
"Add a new feature for peak-to-average ratio by core bucket"

# The agent auto-detects this touches features.py and loads
# the feature-engineering skill for context.
```

---

## 9. Maintenance

| Task | Frequency | Who |
|------|-----------|-----|
| Update `AGENTS.md` with new commands | When toolchain changes | Any contributor |
| Add new skills | When project knowledge becomes non-obvious | Tech lead |
| Review permissions | Quarterly | Tech lead |
| Prune unused agents/skills | Quarterly | Tech lead |

---

## Appendix A: `opencode.json` (full)

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "share": "disabled",
  "snapshot": false,
  "autoupdate": true,
  "server": {
    "port": 4096,
    "hostname": "0.0.0.0",
    "mdns": true,
    "mdnsDomain": "myproject.local",
    "cors": ["http://localhost:5173"]
  },
  "lsp": true,
  "mcp": {
    "context7": {
      "type": "remote",
      "url": "https://mcp.context7.com/mcp",
      "enabled": true,
      "timeout": 10000
    }
  },
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": 10000
  },
  "watcher": {
    "ignore": ["node_modules/**", "dist/**", ".git/**"]
  },
  "permission": {
    "read": {
      "*": "allow",
      "*.env": "deny",
      "*.env.*": "deny",
      "*.env.example": "allow"
    },
    "bash": {
      "*": "ask",
      "git *": "allow",
      "pytest *": "allow",
      "black *": "allow",
      "python -m app.src.*": "ask",
      "python -m app.src.qa_report": "allow"
    },
    "edit": "ask",
    "webfetch": "ask",
    "websearch": "ask",
    "external_directory": "deny"
  },
  "agent": {
    "build": {
      "permission": {
        "read": {
          "*": "allow",
          "*.env": "deny",
          "*.env.*": "deny",
          "*.env.example": "allow",
          "**/*credential*": "deny",
          "**/*secret*": "deny",
          "**/*.pem": "deny",
          "**/*.key": "deny",
          "**/*password*": "deny",
          "**/*token*": "deny"
        },
        "external_directory": "deny"
      }
    },
    "plan": {
      "permission": {
        "read": {
          "*": "allow",
          "*.env": "deny",
          "*.env.*": "deny",
          "*.env.example": "allow",
          "**/*credential*": "deny",
          "**/*secret*": "deny",
          "**/*.pem": "deny",
          "**/*.key": "deny",
          "**/*password*": "deny",
          "**/*token*": "deny"
        },
        "external_directory": "deny"
      }
    },
    "explore": {
      "permission": {
        "read": {
          "*": "allow",
          "*.env": "deny",
          "*.env.*": "deny",
          "*.env.example": "allow",
          "**/*credential*": "deny",
          "**/*secret*": "deny",
          "**/*.pem": "deny",
          "**/*.key": "deny",
          "**/*password*": "deny",
          "**/*token*": "deny"
        },
        "external_directory": "deny"
      }
    },
    "general": {
      "permission": {
        "read": {
          "*": "allow",
          "*.env": "deny",
          "*.env.*": "deny",
          "*.env.example": "allow",
          "**/*credential*": "deny",
          "**/*secret*": "deny",
          "**/*.pem": "deny",
          "**/*.key": "deny",
          "**/*password*": "deny",
          "**/*token*": "deny"
        },
        "external_directory": "deny"
      }
    },
    "scout": {
      "permission": {
        "read": {
          "*": "allow",
          "*.env": "deny",
          "*.env.*": "deny",
          "*.env.example": "allow",
          "**/*credential*": "deny",
          "**/*secret*": "deny",
          "**/*.pem": "deny",
          "**/*.key": "deny",
          "**/*password*": "deny",
          "**/*token*": "deny"
        },
        "external_directory": "deny"
      }
    }
  },
  "command": [
    {
      "name": "test",
      "description": "Run all unit tests",
      "prompt": "Running all unit tests...",
      "command": "pytest app/tests/ -v"
    },
    {
      "name": "test-file",
      "description": "Run a specific test file",
      "prompt": "Running tests in {file}...",
      "command": "pytest app/tests/{file} -v"
    },
    {
      "name": "lint",
      "description": "Check code style",
      "prompt": "Checking code style...",
      "command": "black --check app/src/ app/tests/ && flake8 app/src/"
    },
    {
      "name": "format",
      "description": "Format code with black",
      "prompt": "Formatting code...",
      "command": "black app/src/ app/tests/"
    },
    {
      "name": "qa",
      "description": "Generate QA compliance report",
      "prompt": "Generating QA report...",
      "command": "python -m app.src.qa_report"
    },
    {
      "name": "notebook-03a",
      "description": "Execute feature engineering notebook (local only)",
      "prompt": "Executing 03a_feature_engineering...",
      "command": "papermill notebooks/03a_feature_engineering.ipynb NUL --log-output --progress-bar --execution-timeout 600"
    },
    {
      "name": "notebook-03b",
      "description": "Execute tabular models notebook (local only)",
      "prompt": "Executing 03b_tabular_models...",
      "command": "papermill notebooks/03b_tabular_models.ipynb NUL --log-output --progress-bar --execution-timeout 600"
    },
    {
      "name": "notebook-03c",
      "description": "Execute timeseries notebook (local only)",
      "prompt": "Executing 03c_timeseries_forecasting...",
      "command": "papermill notebooks/03c_timeseries_forecasting.ipynb NUL --log-output --progress-bar --execution-timeout 600"
    },
    {
      "name": "ci-local",
      "description": "Run full CI locally (tests + QA)",
      "prompt": "Running local CI pipeline...",
      "command": "pytest app/tests/ -v && python -m app.src.qa_report"
    }
  ]
}
```

---

## Appendix B: File Manifest

| Path | Type | Description |
|------|------|-------------|
| `opencode.json` | config | Main opencode configuration |
| `AGENTS.md` | docs | Project context read by opencode at startup |
| `.opencode/.gitignore` | config | Ignores only `node_modules/` |
| `.opencode/agents/SupportEngineer.md` | agent | Debugging and RCA specialist |
| `.opencode/skills/root-cause-analysis/SKILL.md` | skill | RCA methodology |
| `.opencode/skills/feature-engineering/SKILL.md` | skill | Feature engineering conventions |
| `.opencode/skills/model-evaluation/SKILL.md` | skill | Model QA criteria |
| `docs/AI_ENGINEERING_SPEC.md` | docs | This document |

---

## Appendix C: Migration from Current State

| From | To | Reason |
|------|----|--------|
| `.opencode/package.json` + `node_modules/` | Delete | No plugins = no dependency needed |
| `.opencode/agents/EngineeringManager.md` | Delete | Replaced by built-in `plan` + `general` |
| `.opencode/agents/DataEngineer.md` | Delete | Replaced by built-in `build` |
| `.opencode/agents/DataAnalyst.md` | Delete | Replaced by built-in `build` + `general` |
| `.opencode/agents/CloudExpert.md` | Delete | Replaced by built-in `general` + web search |
| `.opencode/agents/SoftwareEngineer.md` | Delete | Replaced by built-in `build` (identical) |
| `.opencode/agents/CodeReviewer.md` | Delete | Replaced by built-in `plan` |
| `.opencode/agents/SupportEngineer.md` | Keep | Well-crafted, fills a genuine gap |
| `.opencode/skills/root-cause-analysis/` | Keep | Already well-formed |
| `.opencode/plans/ci-migration-plan.md` | Move to `docs/` | Retrospective doc, not a plan template |
| `.opencode/.gitignore` | Fix | Stop ignoring `package.json` |
| No `opencode.json` | Create | Central config needed |
| No feature-engineering skill | Create | Domain knowledge encoding |
| No model-evaluation skill | Create | QA criteria encoding |
