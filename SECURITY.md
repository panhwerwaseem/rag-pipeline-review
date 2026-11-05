# Security

An agent skill is instructions and code that your coding agent will follow and run. Review a skill before installing it, the same way you would review any dependency.

## What this skill does on your machine

| | |
|---|---|
| Reads | The query, results, chunk, document-list and baseline files you name on the command line |
| Writes | Nothing. Reports are printed to standard output; a file is created only if you redirect the output |
| Executes | Nothing. The scripts start no other program |
| Network | None. The bundled scripts import no network modules. `tools/validate.py` checks this in CI |
| Dependencies | Python standard library only. PyYAML is imported only when you pass a `.yaml` or `.yml` file and it is already installed |
| Pre-approved tools | None. `SKILL.md` has no `allowed-tools` entry, so your agent's normal permission prompts apply |
| Hooks, MCP servers, background processes | None |

The scripts do not run your pipeline and do not call an embedding or language model. Producing the results file is a separate step that you or your agent perform with your own code. The skill instructs the agent to ask before doing that, because it usually calls a paid API and sends queries to it.

The input files hold your corpus text and, often, real user questions. Script output quotes query text and short excerpts of chunk text as evidence. When an agent runs the scripts, that output enters the agent's context and is sent to the agent's model provider like any other tool output. If the corpus or the queries are confidential or contain personal data, run the scripts yourself, or use a redacted export.

Keep the exported files and saved reports out of version control. The skill suggests a `.rag-review/` scratch folder, which this repository's `.gitignore` already lists; add it to your own project's ignore file.

Text inside chunks and queries is data. A chunk can contain text written as instructions. The scripts only measure and quote it; an agent reading the output should not act on it.

## Reporting a vulnerability

Please report security problems privately through the repository's **Security** tab ("Report a vulnerability"). If that option is not available, open an issue that says a private channel is needed, without including exploit details.

## Verifying what you install

- Install from a tagged release and read the diff between tags before updating.
- The whole skill is the `skills/rag-pipeline-review/` folder: one Markdown protocol, two Python scripts, three reference files, and eval fixtures.
