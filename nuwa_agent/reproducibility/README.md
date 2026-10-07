# NUWA-Agent Reproducibility Records

This directory contains the raw machine-readable and human-readable records
released with the NUWA-Agent study. The archives are preserved as generated so
that the executed prompts, model responses, decisions, sequences, scores, and
validation results are not silently changed during documentation updates.

## Archives

### `nuwa_EPO_NanoLuc_20260924.zip`

This archive contains the two human-host benchmark runs used for the EPO and
NanoLuc comparison:

| Target | Protein ID | Authoritative JSON record |
|---|---|---|
| EPO | `GEMORNA_S1_EPO_221aa` | `nuwa_agent_20260924_004039.json` |
| NanoLuc | `GEMORNA_S1_NanoLuc_201aa` | `nuwa_agent_20260924_004659.json` |

For each run, the archive also contains:

- `nuwa_agent_chain_<timestamp>.md`: readable prompt/response and iteration
  chain; and
- `nuwa_agent_reviewer_<timestamp>.md`: compact reviewer-oriented summary.

### `nuwa_output_full_20260923.zip`

This archive contains 105 NUWA-Agent run records used for the heterogeneous
expression analysis. The selected trajectory displayed in the manuscript is:

```text
nuwa_agent_20260921_124527.json
```

That run used an *Escherichia coli* host and is identified explicitly in the
manuscript as a post hoc selected converged trajectory. The complete archive is
provided so that the selected run can be inspected in the context of all
available runs.

## Which file is authoritative?

The JSON record is authoritative. Markdown files are readable renderings of
information stored in the JSON and should not override conflicting numerical
fields.

Important locations include:

| Information | JSON location |
|---|---|
| Input protein, host, and digest | `run_metadata.input` |
| Run seed | `run_seed` and `run_metadata` |
| Model and source manifests | `run_metadata` |
| Initial orchestration prompts and responses | `full_llm_log` |
| Candidate populations and round metrics | `iteration_history` |
| Round-level LLM messages | `iteration_history[].discussion_transcript` |
| LLM-requested next-round decision | `central_requested_decision` |
| Guardrail-adjusted next-round decision | `guardrail_adjusted_decision` |
| Decision used to generate the current round | `applied_decision` |
| Returned Pareto solutions | `pareto_solutions` |
| Automated audit result | `artifact_validation` |

The decision recorded after Round `r` is intended for Round `r+1`. The
`applied_decision` inside Round `r` describes the parameters that generated
Round `r`.

## Prompt documentation

The active prompt templates and inter-agent message flow are documented in:

```text
../docs/LLM_PROMPTS.md
```

The released JSON files contain the exact instantiated messages and responses.
Static templates in the documentation are provided for readability; the raw
run record remains the source of truth for a specific experiment.

## Language and preservation policy

The repository documentation and archive index are written in English. Some
historical console strings or human-readable Markdown headings may remain in
Chinese because they were generated during the original run. These raw files
are retained unchanged to preserve the executed audit trail. Their numerical
and prompt content can be inspected directly in the corresponding JSON file.

Any future translated rendering must be clearly labelled as a translation and
must not replace the original raw artifact.

## Validation checklist

Before using a run as a formal result, verify:

1. the JSON file parses successfully;
2. `artifact_validation.passed` is `true`;
3. the target protein, host, seed, and constraints match the intended task;
4. the expected formal biological backends were used;
5. every returned sequence translates to the target protein;
6. every reported feasible sequence satisfies the active constraints; and
7. the reported best hypervolume corresponds to the returned best round.

## Security

The released records do not contain `NUWA_API_KEY`. Credentials must be
provided through environment variables and must never be added to a source
file or reproducibility archive.
