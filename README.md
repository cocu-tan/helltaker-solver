# Helltaker Puzzle Solver

## What is helltaker
- Free puzzle game in [Steam](https://store.steampowered.com/app/1289310/Helltaker/).
- Play it (just a few hours).

## Technical components
- SAT Solver ([pycosat](https://github.com/ContinuumIO/pycosat) / [picosat](http://fmv.jku.at/picosat/))
- CNF Converter

## Known Issue
**Some rules are missing. This solver outputs invalid solutions for some tricky stages other than in-game stages.**


## How to use
- `pip install pycosat`
- `python helltaker_solver.py  example/stage01.txt`


## How to test
- `pytest helltaker_solver_tests.py`


## Reference
- Introduction Video in NicoNico (Jp)
    - TODO add link here