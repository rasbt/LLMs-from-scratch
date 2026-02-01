# Headless CLI Assembly for AI Coding Agents

## Google AI Search Summary
(Source: Google Search "headless cli assembly in detail as useful for an AI coding agent")

"Headless CLI Assembly" refers to the programmatic orchestration of AI coding agents (like Claude Code, Cline, or Aider) via terminal commands without manual user intervention. For an AI coding agent, this setup is the difference between being a "chatbot in a terminal" and an "autonomous CI/CD engine."

### 1. Architectural Blueprint
To assemble a robust headless agent, treat the CLI as a versioned API surface area rather than just a UI.
- **The Kernel Loop:** Use a simple bash or Python "while" loop to call the agent until a defined stopping criterion (e.g., all tests pass) is met.
- **The SDK vs. CLI:** Prefer programmatic SDKs (Python/TypeScript) when they exist (e.g., Claude Code Agent SDK) for superior context management and error handling.
- **Environment Isolation:** Deploy the agent within Docker containers with git worktrees as volume mounts. This allows for safe `--dangerously-skip-permissions` execution without risking the host machine.

### 2. Context Engineering (The Secret Sauce)
The primary failure point for headless agents is context rot—the accumulation of stale or irrelevant information.
- **Fresh-Start Strategy:** Instead of one long conversation, treat each task iteration as a fresh context window. Use a "spec" and "implementation plan" as the persistent source of truth.
- **LSP Integration:** Force your agent to use the Language Server Protocol (LSP). This allows it to perform hover for documentation, documentSymbol for file structure, and workspaceSymbol for cross-project searches without reading every file into the prompt.
- **Project-Specific Rules:** Create a `.claudecode/rules` or `.cursorrules` file. These provide "always-on" context, defining your testing standards, preferred libraries, and architectural patterns.

### 3. Tool-Call Assembly
A headless agent is only as good as the tools it can execute.

| Tool Category | Recommended CLI Tools | Purpose for Agent |
| :--- | :--- | :--- |
| Git Operations | `gh` CLI | Opening PRs, reading issue comments. |
| Infrastructure | `terraform` / `ansible` | Environment provisioning with `--check` dry runs. |
| Verification | `pytest` / `npm test` | Providing verifiable "exit code" signals. |
| Analysis | `grep`, `fd`, `ripgrep` | Efficient filesystem traversal. |

### 4. Operational Guardrails
For autonomous workflows, implement these safety and efficiency checks:
- **Structured Output:** Use flags (like `--output json`) or libraries (like Pydantic-AI) to force the agent to return machine-readable data for easier chaining.
- **Dry Runs & Diffs:** Always require the agent to run a `git diff` before committing. This provides a clear "reviewable chunk" of work.
- **Spend Limits:** Set hard API spend limits and use telemetry to identify high-token usage patterns.

### 5. Advanced Workflow: Spec-Driven Development
Shift from "Build X" to a structured loop:
1. **Spec:** AI generates a technical requirement document.
2. **Test:** AI writes failing tests based on the spec.
3. **Code:** AI writes implementation until tests pass in the headless loop.
4. **Audit:** A secondary "auditor" agent reviews the diff for security and style.

---
## Scraped Results from Search Links

### 1. Building your own CLI Coding Agent with Pydantic-AI (Martin Fowler)
*Source: https://martinfowler.com/articles/build-own-coding-agent.html*

#### The wavefront of CLI Coding Agents
CLI coding agents are fundamentally different from chatbots or autocomplete tools - they're agents that can read code, run tests, and update a codebase.

#### Key Architectural Components
* **Core AI Model:** Claude 3.5 Sonnet (via AWS Bedrock)
* **Framework:** Pydantic-AI
* **MCP Servers:** Pluggable capabilities via Model Context Protocol
* **CLI Interface:** User interaction

#### Critical Capabilities Added:
1. **Testing Loop:** The agent can run `pytest`, identify failures, and fix the implementation (not the tests).
2. **MCP Pluggable Tools:**
   - **Sandboxed Python:** For reliable calculations and prototyping.
   - **Context7:** For up-to-date documentation access.
   - **AWS Labs MCP:** For cloud-native development and debugging.
   - **Desktop Commander:** Surgical code editing, file system operations, and terminal management.

#### Lessons Learned:
- **Context is King:** The agent needs to maintain state across different tools (e.g., matching test failures to code changes).
- **Specialization Matters:** Agents are most effective when they understand *your* project's eccentricities and standards.

---

### 2. Run Claude Code programmatically (Claude Code Docs)
*Source: https://code.claude.com/docs/en/headless*

#### Agent SDK and CLI Options
Claude Code can be run non-interactively using the `-p` (or `--print`) flag.

#### Key Commands:
- `claude -p "prompt"`: Run a single prompt non-interactively.
- `--allowedTools "Read,Edit,Bash"`: Auto-approve specific tools for autonomous execution.
- `--output-format json`: Get structured outputs for CI/CD integration.
- `--continue` / `--resume`: Maintain state across multiple script calls.

#### Usage Patterns:
- **Auto-Fixing:** `claude -p "Run the test suite and fix any failures" --allowedTools "Bash,Read,Edit"`
- **Commit Generation:** Reviewing staged changes and creating commits autonomously.
---

### 3. Best Practices for Coding with Agents (Cursor Team)
*Source: https://cursor.com/blog/agent-best-practices*

#### The Agent Harness
An agent's effectiveness is built on three pillars:
1. **Instructions:** System prompts and rules (like `.cursorrules`).
2. **Tools:** File editing, terminal execution, code search.
3. **User Messages:** The direct prompts guiding the work.

#### Key Strategies:
- **Plan Mode:** Use a research-first approach. Let the agent create a markdown plan, review/edit it, and *then* execute.
- **Context Management:** Don't tag every file. Let the agent use semantic search/grep to pull context on demand. Mention `@Branch` to orient the agent to current changes.
- **Iteration over Fixing:** If an agent gets confused, it's often better to revert and start fresh with a more specific prompt than to try and "patch" a confused thread.
- **TDD Loop:** Pair agents with tests. The agent writes tests, you verify them, then the agent writes code until the tests pass.

#### Specialized Workflows:
- **Git Worktrees:** Running multiple agents in parallel on the same repo using isolated worktrees to prevent file conflicts.
- **Skills & Hooks:** Customizable scripts that run before/after agent actions (e.g., a hook that keeps the agent running until `npm test` passes).

---

### 4. Claude Code: Professional Headless Workflows
*Source: https://code.claude.com/docs/en/overview*

#### Unix Philosophy Integration
Claude Code is designed to be composable. Example patterns:
- `tail -f app.log | claude -p "Slack me if you see anomalies"`
- `gh pr diff | claude -p "Review for security vulnerabilities" --append-system-prompt "..."`

#### Core Capabilities:
- **Autonomous Action:** Directly edits files, runs commands, and creates commits.
- **MCP Integration:** Pulls from external sources (Google Drive, Slack, etc.) to inform code changes.
- **CI/CD Native:** Ideal for automated translation, documentation updates, and linting fixes.

---

## Operationalizing Headless CLI Assembly

Based on the research above, here is the "Ultimate Blueprint" for an AI Coding Agent:

### 1. The Kernel (The Loop)
```bash
#!/bin/bash
while true; do
  # Run the agent in headless mode
  claude -p "Implement the next feature in the spec and fix failing tests" --allowedTools "Read,Edit,Bash"
  # Check if the stopping condition is met (e.g. exit code 0 and tests passing)
  if pytest && git diff --quiet; then
    echo "Task Complete!"
    break
  fi
done
```

### 2. The Context Layer
- **Persistent Truth:** Keep a `SPEC.md` or `IMPLEMENTATION_PLAN.md` file.
- **Rule Enforcement:** Use `.claudecode/rules` or `.cursorrules` to define architectural boundaries.
- **Symbolic Search:** Ensure the agent has tools for `ripgrep`, `fd`, and `LSP` traversal to avoid token-bloat.

### 3. The Guardrails
- **Sandboxing:** Always run in Docker or a dedicated VM.
- **Reviewable Diffs:** Force the agent to produce a `git diff` for human or "Auditor Agent" verification.
- **Budgeting:** Set token/spend limits per task to prevent runaway costs.

### 4. Verifiable Success
Implement **Test-Driven Development (TDD)** as the primary feedback mechanism. An agent is "done" when the code it generates causes the test runner to return Exit Code 0.

---
*End of Scraped Content*
