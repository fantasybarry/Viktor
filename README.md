# Viktor

Viktor is a multimodal large language model for embodied intelligence and robotics. It connects perception, language, and action to help agents understand their surroundings, plan toward goals, and execute behavior in the physical world.

## Capabilities

Viktor is being developed for tasks including:

- Path planning in complex environments
- Trajectory prediction and following
- Vision-language-action (VLA) modeling
- Multimodal scene and context understanding
- Goal-conditioned decision-making
- Real-world robot control and navigation

## Vision

The goal of Viktor is to provide a general foundation for translating real-world observations and natural-language instructions into grounded plans and actions. By learning across multiple modalities, Viktor aims to support adaptable agents that can reason about dynamic environments and interact with them safely and effectively.

## Project status

Viktor is under active development. Architecture details, setup instructions, model checkpoints, and examples will be added as the project evolves.

## Experiment: LLM-guided A* (geometric benchmark)

`viktor/llm_astar.py` implements the bounded guided A* of `docs/math/llm_astar_pseudocode.md` (Section 7A) on a static point-lattice benchmark. `viktor/llm_astar_experiment.py` runs the Section 7C baseline-vs-guided comparison. No trained Viktor model exists yet: the default guide provider is a labeled synthetic demo, and timings are CPython wall times, not vehicle or 20-30 ms guarantees. Python 3 standard library only.

```bash
python -m unittest discover -s tests -v              # tests
python -m viktor.llm_astar_experiment --help         # options
python -m viktor.llm_astar_experiment --budget-ms 20 --records records.json
```
