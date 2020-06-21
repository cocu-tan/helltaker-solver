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

## Stage text spec
See `CellType` class

```python
{
      '.': cls.EMPTY,
      '*': cls.STONE,
      '$': cls.STONE_SPIKE,
      'E': cls.ENEMY,
      'W': cls.WALL,
      'K': cls.KEY,
      'L': cls.LOCK,
      'G': cls.GOAL,
      'S': cls.START,
      '|': cls.SPIKE,
      'A': cls.SPIKE_TOGGLE_ACTIVE,
      '_': cls.SPIKE_TOGGLE_INACTIVE,
      '2': cls.STONE_SPIKE_TOGGLE_INACTIVE,
}
```


## Reference
- Introduction Video in NicoNico (Jp)
    - TODO add link here