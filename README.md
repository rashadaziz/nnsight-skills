# Skills for the NDIF Ecosystem

Agent skills for neural network interpretability with [NNsight](https://nnsight.net/).

Compatible with both **Claude Code** and **OpenAI Codex** via the [Agent Skills Specification](https://agentskills.io/).

## Installation

### Claude Code

```bash
# Open Claude Code terminal
claude

# Add the marketplace (one time)
/plugin marketplace add https://github.com/ndif-team/skills.git

# Install all skills
/plugin install nnsight@skills
```

### OpenAI Codex

```bash
# Open OpenAI Codex terminal
codex

# Install skills
skill-installer install https://github.com/ndif-team/skills.git
```

## Included Skills

| Skill | Use When... |
| ----- | ----------- |
| **nnsight-basics** | Setting up models, tracing activations, saving values, basic interventions |
| **logit-lens** | Analyzing layer-wise predictions, understanding information flow |
| **activation-patching** | Finding causally important layers, heads, or positions |
| **attribution-patching** | Scaling circuit analysis with gradient approximations |
| **causal-tracing** | Investigating information flow and mediation |
| **model-steering** | Controlling outputs with steering vectors and persistent edits |

## Example Prompts

Once installed, just ask naturally:

- "Help me implement logit lens to see what GPT-2 predicts at each layer"
- "Find which attention heads are important for this task using activation patching"
- "Create a steering vector to make the model more positive"
- "Trace where the model stores factual information about the Eiffel Tower"

The agent will automatically apply the relevant skills.

## Structure

```text
skills/
├── .claude-plugin/
│   └── marketplace.json          # Claude Code marketplace
├── .codex/
│   └── skills/                   # Codex skills (symlinks)
│       ├── nnsight-basics -> ...
│       ├── logit-lens -> ...
│       └── ...
└── plugins/
    └── nnsight/
        ├── .claude-plugin/
        │   └── plugin.json
        └── skills/               # Actual skill files
            ├── nnsight-basics/
            │   └── SKILL.md
            ├── logit-lens/
            │   └── SKILL.md
            └── ...
```

## Resources

- [NNsight Documentation](https://nnsight.net/documentation/)
- [NNsight Tutorials](https://nnsight.net/tutorials/tutorials/)
- [NDIF Platform](https://ndif.us/) - Remote access to large models
- [Agent Skills Specification](https://agentskills.io/)
