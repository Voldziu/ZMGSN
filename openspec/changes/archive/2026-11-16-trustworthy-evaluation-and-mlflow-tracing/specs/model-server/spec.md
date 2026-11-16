# Spec Delta

## MODIFIED Requirements

### Requirement: Multi-preset evaluation command
The project SHALL provide one command that, for each preset in a given list:
1. starts the preset's server on the current platform;
2. waits until the server is ready;
3. runs the evaluation with a given number of repeats;
4. stops the server.

The command SHALL stop the active server even when interrupted. It SHALL write one results file per preset, and it SHALL resume finished runs when it is run again with the same configuration. It SHALL keep each preset's run workspaces in a directory of that preset's own, so that one preset's runs never delete another preset's.

#### Scenario: Several presets in a row
- **WHEN** a user runs `make eval-presets PRESETS="llama gemma" REPEATS=3` with no server running
- **THEN** `llama` and then `gemma` are each served, evaluated three times and stopped, results go to `results/presets-llama.jsonl` and `results/presets-gemma.jsonl`, and a per-level summary is printed for each

#### Scenario: Workspaces kept per preset
- **WHEN** `llama` and then `gemma` have been evaluated on the same scenarios
- **THEN** `llama`'s reviews are still in `workspace/preset-runs/llama/` next to `gemma`'s in `workspace/preset-runs/gemma/`

#### Scenario: Interrupted run leaves no server behind
- **WHEN** the user presses Ctrl-C while a preset is being evaluated
- **THEN** the active server is stopped before the command exits

#### Scenario: Server already running
- **WHEN** a server already answers on port 8000
- **THEN** the command refuses to start and tells the user to stop it first

#### Scenario: Unknown preset
- **WHEN** `PRESETS` contains a name that is not a preset
- **THEN** the command exits before starting any server and lists the valid presets
