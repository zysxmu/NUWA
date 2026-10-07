# NUWA-Agent

NUWA-Agent is the agentic mRNA-design component built on top of the NUWA
foundation model. Given a target protein sequence and an expression host, it
selects a domain-specific NUWA generator, produces synonymous coding sequences,
evaluates them with fixed surrogate predictors and sequence-level constraints,
and maintains a Pareto front during iterative search.

The optimization results should be interpreted as a proof of concept using
available surrogate predictors. Host awareness applies to species resolution,
domain-model selection, generator conditioning, codon usage, and configured
sequence constraints. The translation-efficiency, stability, and expression
scores are fixed computational ranking objectives and are not universally
validated host-specific measurements.

## Repository layout

```text
nuwa_agent/
├── main.py                  # Main command-line entry point
├── run_demo.py              # Small smoke-test example
├── batch_run.py             # Batch runner for prepared input sets
├── config.py                # Environment-based configuration
├── species_resolver.py      # Host name -> domain and species class ID
├── orchestrator.py          # Initial model/species orchestration
├── model_registry.py        # NUWA model loading and sequence generation
├── evaluator.py             # Surrogate scoring and sequence metrics
├── constraint_checker.py    # Feasibility checks
├── pareto_selector.py       # Non-dominated sorting and hypervolume
├── round_deliberation.py    # Six-stage round-level LLM deliberation
├── iteration_controller.py  # Iterative search and controller guardrails
├── feedback_analyzer.py     # Correlation helper and legacy feedback code
├── docs/
│   └── LLM_PROMPTS.md       # Prompt templates and message flow
├── reproducibility/
│   ├── README.md            # English index of released records
│   ├── nuwa_output_full_20260923.zip
│   └── nuwa_EPO_NanoLuc_20260924.zip
├── tests/
└── requirements.txt
```

Model weights are not included in the repository. The released checkpoints
occupy several gigabytes and must be downloaded or transferred separately.

## Requirements

- Linux is recommended for formal runs.
- Python 3.10
- A CUDA-capable GPU is recommended for NUWA generation and scoring.
- ViennaRNA Python bindings (`RNA`)
- `cai2`
- Access to the configured OpenAI-compatible LLM endpoint when LLM scheduling
  is enabled

Create an environment and install the Python dependencies:

```bash
conda create -n nuwa-agent python=3.10 -y
conda activate nuwa-agent

cd /path/to/NUWA/nuwa_agent
pip install -r requirements.txt
conda install -c bioconda viennarna -y
pip install cai2
```

Formal evaluation requires the external biological tools. Heuristic fallbacks
may be useful for software debugging but must not be used as formal experimental
results.

## Model and data layout

By default, `config.py` resolves `NUWA_BASE_DIR` to the parent directory of
`nuwa_agent/`. The expected layout is:

```text
$NUWA_BASE_DIR/
├── nuwa_agent/
└── nuwa_weights/
    ├── domain_models/
    │   ├── bacteria/checkpoint-*
    │   ├── eukaryote/checkpoint-*
    │   └── archaea/checkpoint-*
    ├── finetuned/
    │   ├── te/checkpoint-1000
    │   ├── stability/checkpoint-1000
    │   └── expression/checkpoint-1000
    ├── species_maps/
    │   ├── bacteria_species_mapping.json
    │   ├── eukaryote_species_mapping.json
    │   └── archaea_species_mapping.json
    └── codon_tables/
        └── *.json
```

If the weights are stored elsewhere, set:

```bash
export NUWA_BASE_DIR=/absolute/path/to/the/directory/containing/nuwa_weights
```

## API credentials

The repository does not contain an API key. Supply the credential through an
environment variable:

```bash
export NUWA_API_KEY="YOUR_API_KEY"
```

Do not write a credential into `config.py`, commit it to Git, or include it in
a reproducibility archive. The generated audit artifacts do not contain
`NUWA_API_KEY`.

## Running NUWA-Agent

### Smoke test

```bash
cd /path/to/NUWA/nuwa_agent
python run_demo.py
```

The smoke test verifies installation and model loading. It is not a formal
experiment.

### Formal single-target run

Formal runs require an explicit random seed and a target-specific MFE/nt
interval:

```bash
cd /path/to/NUWA/nuwa_agent

export NUWA_API_KEY="YOUR_API_KEY"
export NUWA_RUN_SEED=20260918
export NUWA_MFE_PER_NT_MIN=-0.35
export NUWA_MFE_PER_NT_MAX=-0.20
export NUWA_AUDIT_HASH_MODELS=1

python main.py \
  --protein-file /path/to/one_protein.fasta \
  --protein-id NP_000000.1 \
  --host "Escherichia coli" \
  --gc-min 0.30 \
  --gc-max 0.70
```

The interactive form is also available:

```bash
python main.py
```

### Rule-based scheduling control

The LLM deliberation can be disabled while retaining the same generator,
evaluators, constraints, Pareto selection, and controller guardrails:

```bash
export NUWA_MULTIAGENT_ENABLED=0
python main.py
```

## LLM prompt workflow

The active LLM workflow contains:

1. One initial orchestration call for domain-model selection, species-class
   selection, and constraint configuration.
2. Two independent round-level proposals: generation/exploration and
   synonymous editing/exploitation.
3. A biology and constraint review that receives both proposals.
4. A generation revision.
5. An editing revision that also receives the generation revision.
6. A central next-round decision that receives the complete validated
   discussion.

The fixed prompt templates, output fields, and inter-call message flow are
documented in [`docs/LLM_PROMPTS.md`](docs/LLM_PROMPTS.md).

`feedback_analyzer.py` contains a deprecated legacy feedback-generation method.
It is not the active LLM decision path for the released multi-agent runs. The
current controller uses `FeedbackAnalyzer._compute_score_correlations()` only
to construct numerical round evidence; next-round LLM decisions are produced
by `RoundDeliberation` in `round_deliberation.py`.

## Optimization workflow

For each evaluated round, NUWA-Agent:

1. generates new candidates and, after the first round, synonymous mutants of
   selected parents;
2. optionally applies the deterministic CAI post-processing schedule;
3. evaluates the three fixed surrogate scores;
4. calculates CAI, GC content, length-normalized MFE, maximum stem length, and
   maximum homopolymer length;
5. removes candidates that violate an active hard constraint;
6. performs non-dominated sorting and calculates hypervolume;
7. records the numerical evidence used by the next-round scheduler; and
8. validates and bounds any proposed search parameters before execution.

The audit record distinguishes:

- `central_requested_decision`: the raw central LLM proposal;
- `guardrail_adjusted_decision`: the validated and bounded proposal for the
  next round; and
- `applied_decision`: the parameters actually used to generate the current
  round.

The decision produced after Round `r` applies to Round `r+1`; it must not be
interpreted as a description of a round that has already been evaluated.

## Outputs

Each completed run writes three files to `output/`:

```text
nuwa_agent_<timestamp>.json
nuwa_agent_chain_<timestamp>.md
nuwa_agent_reviewer_<timestamp>.md
```

- The JSON file is the authoritative machine-readable record.
- The chain report is a readable rendering of the prompt/response transcript.
- The reviewer report is a compact run summary.

The JSON record includes the input sequence and digest, host, random seed,
software and dependency versions, source-code fingerprints, codon-table
fingerprints, model-artifact manifests, candidate histories, constraint
outcomes, Pareto fronts, hypervolume values, LLM messages, and returned
sequences.

Formal records should satisfy:

```text
artifact_validation.passed = true
```

If validation fails, the console prints an `AUDIT WARNING`; that run should not
be used as a formal result.

## Reproducibility notes

- A fixed run seed controls the local Python, NumPy, and PyTorch random states.
- Remote LLM responses and some GPU operations may still vary.
- Complete prompts and responses are retained in the audit record.
- Raw historical records are preserved unchanged. English documentation is
  provided separately so that executed prompts and responses are not silently
  rewritten.
- Candidates within one run are not independent biological replicates.
- Surrogate scores are computational ranking objectives rather than measured
  host-specific expression, half-life, or folding stability.

See [`reproducibility/README.md`](reproducibility/README.md) for an English
index of the released archives.

## Running tests

From the repository root:

```bash
PYTHONPATH="$(pwd)" python -m pytest nuwa_agent/tests -q -p no:cacheprovider
```

## Troubleshooting

| Problem | Check |
|---|---|
| Missing `NUWA_API_KEY` | Export the environment variable before enabling LLM scheduling. |
| Model `FileNotFoundError` | Verify `NUWA_BASE_DIR` and the `nuwa_weights/` layout. |
| `No module named RNA` | Install ViennaRNA in the active environment. |
| `cai2` unavailable | Install `cai2` and rerun the formal backend check. |
| LLM endpoint unavailable | Check network access and the configured base URL. |
| A deliberation phase fails | Inspect `discussion_fallback_used`, `discussion_error`, and the transcript. |
| CUDA out of memory | Reduce the candidate batch size or use a device with more memory. |

## Citation and scope

If you use NUWA-Agent, cite the associated NUWA/NUWA-Agent manuscript and the
released code version or commit. Report the exact model checkpoints, run seed,
constraint configuration, candidate-evaluation budget, LLM model, and whether
LLM or rule-based scheduling was enabled.
