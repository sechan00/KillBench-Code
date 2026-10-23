# Data sources and third-party notices

## SafeArena tasks and derivatives

The SafeArena task definitions originate from
[McGill-NLP/safearena on Hugging Face](https://huggingface.co/datasets/McGill-NLP/safearena).
The upstream project is described at [SafeArena](https://safearena.github.io/)
and [its code repository](https://github.com/McGill-NLP/safearena).

The original task files are retained in `KillBench/data/harm.json`,
`KillBench/harm.json`, and `KillBench/safearena_data/safe.json`.
`KillBench/bypassed_prompts.json`, `KillBench/safearena_data/harm.json`,
`KillBench/safearena_data/_killswitch_mapping.json`, and experiment results
containing their intent text are derivatives. KillBench wraps selected intents
in jailbreak templates, renumbers task IDs, and retains task metadata. The
generation code is in `KillBench/build_bypassed_prompts.py` and
`KillBench/build_safearena_tasks.py`.

SafeArena is access-gated upstream. Obtain any additional upstream material
through its official access process. The dataset card's terms, reproduced in
[licenses/SafeArena-TERMS.md](licenses/SafeArena-TERMS.md), apply to the task data
and its derivatives. They do not provide an unconditional license for all
third-party source material. Users must assess the applicability of research
fair use and any additional third-party terms in their jurisdiction.

## Chart.js and bundled color library

`scripts/vendor/chart.umd.min.js` includes:

| Component | Version | License notice |
|---|---|---|
| [Chart.js](https://github.com/chartjs/Chart.js) | 4.4.1 | [MIT](licenses/Chart.js-LICENSE.txt) |
| [@kurkle/color](https://github.com/kurkle/color) | 0.3.2 | [MIT](licenses/kurkle-color-LICENSE.txt) |

The original copyright headers remain in the bundle. Some visualization pages
load Chart.js 4.4.0 from jsDelivr rather than bundling that version locally.

## Local benchmark fixtures and outputs

The project website describes `data/website/` as purpose-made sandbox fixtures:
synthetic faculty records, fictitious news articles, and a faux electronics
store. Person-like records and example credentials in these fixtures are test
inputs, not credentials for real services. News fixtures are benchmark content
and should not be presented as verified reporting.

Aggregated results, generated prompt libraries, screenshots, and GCG artifacts
are research outputs. They may reproduce task or fixture text; the applicable
source terms continue to apply. This notice does not relicense third-party
content or assert that every external right has been independently verified.

Dependencies installed separately remain governed by their own licenses.
