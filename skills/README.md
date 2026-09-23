# Skills

Four agent skills that encode the working practice behind limbic. Each is a
folder with a `SKILL.md`; Claude Code loads it when the task matches the
skill's description. `drive` also ships an `agents/openai.yaml` for Codex.

| Skill | What it is for | Needs limbic? |
|---|---|---|
| [`packet-worker`](packet-worker/SKILL.md) | Apply a codebook, extract or classify across N documents as stateless packet calls with budgets, a yield probe and exact-quote checks, instead of a tool-using agent per document | Yes — `cerebellum.packet`, `hippocampus.resolve` |
| [`thin-worker-brief`](thin-worker-brief/SKILL.md) | Before delegating to a subagent: decide whether to delegate at all, then write a brief small enough that the worker does not spend ~50K tokens rediscovering the project | No |
| [`coordinator-hygiene`](coordinator-hygiene/SKILL.md) | When one thread runs a long multi-step or multi-agent job: when to hand off, how to wait, when to run gates, and checks before `rm -rf`, `git reset`, `stash` or a replan | No |
| [`drive`](drive/SKILL.md) | Turn a long, vague request ("research this", "improve this") into one bounded pilot plan grounded in nearby evidence. Planning only; it dispatches nothing | Optional — `python -m limbic.drive validate` checks the plan |

## Install

Claude Code reads personal skills from `~/.claude/skills/<name>/SKILL.md`, and
project skills from `.claude/skills/<name>/SKILL.md` inside a repository.

```bash
git clone https://github.com/houshuang/limbic.git
mkdir -p ~/.claude/skills
cp -r limbic/skills/packet-worker ~/.claude/skills/     # or any of the four
```

Use `ln -s "$PWD/limbic/skills/packet-worker" ~/.claude/skills/` instead of
`cp -r` to pick up updates with `git pull`. Start a new Claude Code session
afterwards; the skill can be invoked by name
(`/packet-worker`) or picked up from its description.

For the skills that call limbic, install the library in the environment your
project uses:

```bash
pip install git+https://github.com/houshuang/limbic.git
```
