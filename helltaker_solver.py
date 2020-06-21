from __future__ import annotations

import abc
import enum
import fileinput
import itertools
import sys
from dataclasses import dataclass
from typing import Tuple, List, Set, Dict, Optional, Union, Generator
import pycosat
from collections import Counter


########################################################################################################################
# Sat Solver
class ILiteral(abc.ABC):
  @abc.abstractmethod
  def pos(self) -> Literal:
    pass

  @abc.abstractmethod
  def neg(self) -> Literal:
    pass


@dataclass(frozen=True)
class Variable(ILiteral):
  name: str

  def __repr__(self):
    return repr(f'V[{self.name}]')

  def __str__(self):
    return self.name

  def __hash__(self):
    return hash((self.name))

  def pos(self) -> Literal:
    return Literal(variable=self, sign=True)

  def neg(self) -> Literal:
    return Literal(variable=self, sign=False)

  def __invert__(self):
    return self.neg()

  def __lt__(self, other):
    if isinstance(other, Variable):
      return self.name < other.name


@dataclass(frozen=True)
class Literal(ILiteral):
  variable: Variable
  sign: bool

  def __str__(self):
    sign_str = '' if self.sign else '-'
    return sign_str + str(self.variable)

  def __hash__(self):
    return hash((self.variable, self.sign))

  def __invert__(self):
    return Literal(variable=self.variable, sign=not self.sign)

  def pos(self) -> Literal:
    return self

  def neg(self) -> Literal:
    return Literal(variable=self.variable, sign=not self.sign)


@dataclass(frozen=True)
class Clause:
  literals: List[Literal]

  def __repr__(self):
    return str(self)

  def __str__(self):
    return ' '.join(map(str, self.literals))

  def __iter__(self):
    return iter(self.literals)

  def extend(self, rest):
    return Clause(literals=rest + self.literals)


class SatProblem:
  variables: Set[Variable]
  clauses: List[Clause]

  def __init__(self):
    self.variables = set()
    self.clauses = []

  def add_variable(self, name: str) -> Variable:
    var = Variable(name=name)
    if var in self.variables:
      raise ValueError(f'duplicated name: {name}')

    self.variables.add(var)
    return var

  def add_clauses(self, clauses: List[Clause]):
    self.clauses.extend(clauses)

  def solve(self) -> Union[str, Dict[Variable, bool]]:
    variables = enumerate(sorted(self.variables, key=str), start=1)
    variable2id = {
      v: _id
      for _id, v in variables
    }
    id2variable = {v: k for k, v in variable2id.items()}

    cnf = self._convert_to_cnf(variable2id)
    solution = pycosat.solve(cnf)
    if solution == 'UNSAT':
      return 'UNSAT'
    return {
      id2variable[abs(s)]: s > 0
      for s in solution
    }

  def _convert_to_cnf(self, variable2id: Dict[Variable, int]):
    return [
      [
        (1 if sv.sign else -1) *
        variable2id.get(sv.variable)
        for sv in c
      ]
      for c in self.clauses
    ]


########################################################################################################################
# Helltaker Stage Representation

class CellType(enum.Enum):
  EMPTY = enum.auto()
  STONE = enum.auto()
  ENEMY = enum.auto()
  WALL = enum.auto()
  KEY = enum.auto()
  LOCK = enum.auto()
  STONE_SPIKE = enum.auto()
  SPIKE = enum.auto()
  SPIKE_TOGGLE_ACTIVE = enum.auto()
  SPIKE_TOGGLE_INACTIVE = enum.auto()
  STONE_SPIKE_TOGGLE_INACTIVE = enum.auto()
  START = enum.auto()
  GOAL = enum.auto()

  @classmethod
  def from_string(cls, char) -> CellType:
    cell = {
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
    }.get(char)
    assert cell is not None, f'unexpected char: {char}'
    return cell

  def is_spike(self):
    return self in [
      self.SPIKE,
      self.STONE_SPIKE,
      self.SPIKE_TOGGLE_ACTIVE,
      self.SPIKE_TOGGLE_INACTIVE,
      self.STONE_SPIKE_TOGGLE_INACTIVE,
    ]

  def is_spike_toggle(self):
    return self in [
      self.SPIKE_TOGGLE_ACTIVE,
      self.SPIKE_TOGGLE_INACTIVE,
      self.STONE_SPIKE_TOGGLE_INACTIVE,
    ]

  def is_spike_toggle_inactive(self):
    return self in [
      self.SPIKE_TOGGLE_INACTIVE,
      self.STONE_SPIKE_TOGGLE_INACTIVE,
    ]

  def is_constant_spike(self):
    return self in [
      self.SPIKE,
      self.STONE_SPIKE,
    ]

  def is_stone(self):
    return self in [
      self.STONE,
      self.STONE_SPIKE,
      self.STONE_SPIKE_TOGGLE_INACTIVE,
    ]


class StageDataValidator:
  @staticmethod
  def validate(data: StageData) -> None:
    """
    :raises ValueError: if invalid
    """
    same_all_line_length = len(set(len(line) for line in data)) == 1
    assert same_all_line_length, "Data shape should be rectangle"

    # for simplicity, this solver restricts the number of keys,
    # To mitigate this restriction, change HelltakerSatProblemGenerator to support multiple keys.
    # To do that, needs to hold the number of keys/locks somehow.
    cell_type2num = Counter(cell for line in data for cell in line)
    assert cell_type2num.get(CellType.KEY, 0) <= 1, 'number of keys must be up to 1.'
    assert cell_type2num.get(CellType.LOCK, 0) <= 1, 'number of locks must be up to 1.'


@dataclass(frozen=True)
class StageData:
  MapData = Tuple[Tuple[CellType, ...], ...]
  map: MapData
  width: int
  height: int
  steps: int

  @classmethod
  def from_string(cls, body: str):
    lines = body.splitlines()
    steps = int(lines[0])
    map_lines = lines[1:]
    data = tuple([
      tuple(
        [
          CellType.from_string(char)
          for char in line
        ]
      )
      for line in map_lines
    ])
    instance = cls(
      steps=steps,
      map=data,
      width=len(data[0]),
      height=len(data),
    )
    StageDataValidator.validate(instance)
    return instance

  def get_cell(self, x: int, y: int) -> CellType:
    assert 0 <= x < self.width and 0 <= y < self.height
    return self.map[y][x]

  def __iter__(self):
    return iter(self.map)

  def items(self) -> Generator[Tuple[Tuple[XCoord, YCoord], CellType], None, None]:
    return (
      ((x, y), cell)
      for y, y_data in enumerate(self.map)
      for x, cell in enumerate(y_data)
    )


########################################################################################################################
# Helltaker Solver 


Step = int
XCoord = int
YCoord = int
StepPos = Tuple[Step, XCoord, YCoord]

GenericVariable = Union[Literal, Variable]


def gen_any_rules(variables: List[ILiteral]) -> List[Clause]:
  return [
    Clause([v.pos() for v in variables])  # not all false
  ]


def gen_no_two_true_rules(variables: List[ILiteral]) -> List[Clause]:
  return [
    Clause([v1.neg(), v2.neg()])
    for (v1, v2) in itertools.product(variables, repeat=2)
    if v1 != v2
  ]


def gen_unique_true_rules(variables: List[ILiteral]) -> List[Clause]:
  is_not_all_false = [Clause([v.pos() for v in variables])]
  no_two_trues = gen_no_two_true_rules(variables)
  return is_not_all_false + no_two_trues


def gen_exists_then_true_rule(conditions: List[ILiteral], then: ILiteral):
  return [
           Clause([cond.neg(), then.pos()])
           for cond in conditions
         ] + [
           Clause([c.pos() for c in conditions] + [then.neg()])
         ]


def all_then(conditions: List[ILiteral],
    then: Union[ILiteral, List[ILiteral]]):
  if not isinstance(then, list):
    then = [then]
  return [
    Clause(
      [
        cond.neg()
        for cond in conditions
      ] + [
        _then.pos()
      ]
    )
    for _then in then
  ]


def gen_single_false_and_then_true_rule(conditions: List[ILiteral], then: ILiteral):
  return [
    Clause(
      [
        c.neg()
        for c in conditions
        if c != focus_cond
      ] + [
        focus_cond.pos()
      ] + [
        then.pos()
      ]
    )
    for focus_cond in conditions
  ]


def gen_all_and_then_no_two_true_rules(
    conditions: List[ILiteral],
    then_uniques: List[ILiteral],
) -> List[Clause]:
  return [
    c.extend([
      v.neg()
      for v in conditions
    ])
    for c in gen_no_two_true_rules(then_uniques)
  ]


@dataclass(frozen=True)
class CellState:
  steppos: StepPos

  is_empty: Variable
  is_stone: Variable
  is_enemy: Variable
  is_player: Variable
  is_wall: Variable
  is_goal: Variable
  is_lock: Variable

  is_lock_next: Variable

  is_spike: Variable
  is_spike_active: Variable
  is_enemy_pre: Variable

  # Util variables
  is_blocker: Variable
  is_not_kickable: Variable

  @classmethod
  def create(cls, steppos: StepPos, problem: SatProblem) -> CellState:
    step, x, y = steppos
    prefix = f'S{step:02d}:S:X{x:02d}Y{y:02d}'
    return cls(
      steppos=steppos,
      is_empty=problem.add_variable(f'{prefix}:empty'),
      is_stone=problem.add_variable(f'{prefix}:stone'),
      is_enemy=problem.add_variable(f'{prefix}:enemy'),
      is_player=problem.add_variable(f'{prefix}:player'),
      is_wall=problem.add_variable(f'{prefix}:wall'),
      is_goal=problem.add_variable(f'{prefix}:goal'),
      is_lock=problem.add_variable(f'{prefix}:lock'),
      is_lock_next=problem.add_variable(f'{prefix}:lock_next'),

      is_blocker=problem.add_variable(f'{prefix}:blocker'),
      is_not_kickable=problem.add_variable(f'{prefix}:notkickable'),

      is_spike=problem.add_variable(f'{prefix}:spike'),
      is_spike_active=problem.add_variable(f'{prefix}:spike_a'),
      is_enemy_pre=problem.add_variable(f'{prefix}:enemy_pre'),
    )

  def gen_rules(self) -> List[Clause]:
    return gen_unique_true_rules([
      self.is_empty,
      self.is_stone,
      self.is_enemy,
      self.is_player,
      self.is_wall,
      self.is_goal,
      self.is_lock_next,
    ]) + gen_exists_then_true_rule(
      conditions=[self.is_wall, self.is_enemy, self.is_stone, self.is_goal, self.is_lock_next],
      then=self.is_blocker,
    ) + self.gen_kickable_rules() + self.gen_enemy_pre_rules()

  def gen_kickable_rules(self):
    return \
      gen_exists_then_true_rule(
        conditions=[self.is_wall, self.is_goal],
        then=self.is_not_kickable,
      )

  def gen_spike_rules(self):
    return \
      all_then(
        conditions=[self.is_spike.neg()],
        then=self.is_spike_active.neg(),
      )

  def gen_enemy_pre_rules(self):
    return \
      all_then(
        conditions=[self.is_spike, self.is_spike_active, self.is_enemy_pre],
        then=self.is_enemy.neg(),
      ) + all_then(
        conditions=[self.is_spike, self.is_spike_active.neg(), self.is_enemy_pre],
        then=self.is_enemy.pos(),
      ) + all_then(
        conditions=[self.is_spike.neg(), self.is_enemy_pre.neg()],
        then=self.is_enemy.neg(),
      ) + all_then(
        conditions=[self.is_spike.neg(), self.is_enemy_pre.pos()],
        then=self.is_enemy.pos(),
      )


@dataclass(frozen=True)
class UserInput:
  step: Step

  is_left: Variable
  is_up: Variable
  is_right: Variable
  is_down: Variable
  skip: Variable

  def gen_rules(self) -> List[Clause]:
    return gen_unique_true_rules([
      self.is_left,
      self.is_up,
      self.is_right,
      self.is_down,
      self.skip,
    ])

  @classmethod
  def create(cls, step: Step, problem: SatProblem):
    return cls(
      step=step,
      is_down=problem.add_variable(f'S{step:02d}:C:down'),
      is_left=problem.add_variable(f'S{step:02d}:C:left'),
      is_right=problem.add_variable(f'S{step:02d}:C:right'),
      is_up=problem.add_variable(f'S{step:02d}:C:up'),
      skip=problem.add_variable(f'S{step:02d}:C:skip'),
    )


StageStates: Dict[StepPos, CellState]
ControlStates: Dict[Step, UserInput]


@dataclass(frozen=True)
class AuxStates:
  has_key: Dict[Step, Variable]

  @classmethod
  def create(cls, steps: int, problem: SatProblem):
    return cls(
      has_key={
        step: problem.add_variable(f'S{step:02d}:Aux:has_key')
        for step in range(steps + 1)
      },
    )


class HelltakerProblem:
  stage_data: StageData
  stage_states: StageStates
  control_states: ControlStates
  aux_states: AuxStates
  problem: SatProblem
  solution: Optional[Union[Dict[Variable, bool], str]]

  def __init__(
      self,
      stage_data: StageData,
      stage_states: StageStates,
      control_states: ControlStates,
      aux_states: AuxStates,
      problem: SatProblem):
    self.stage_data = stage_data
    self.stage_states = stage_states
    self.control_states = control_states
    self.aux_states = aux_states
    self.problem = problem
    self.solution = None

  def solve(self):
    self.solution = self.problem.solve()

  def visualize(self):
    for step in range(self.stage_data.steps, -1, -1):
      print(f'Step: {step:02d}, {self._format_control(step):<6s} {self._format_aux_state(step)}')
      print(self._format_step(step=step))
      print()

  def show_steps(self):
    steps_range = range(self.stage_data.steps, -1, -1)
    controls = [['', 0]]
    for control in (f'{self._format_control(step)}' for step in steps_range):
      if control == 'Skip':
        continue
      if controls[-1][0] == control:
        controls[-1][1] += 1
      else:
        controls.append([control, 1])

    # omit fake initial value
    controls = controls[1:]
    print('\n'.join(f'{control:<5s} x{num}' for control, num in controls))

  def get_solution(self) -> Union[str, List[Tuple[str, str]]]:
    assert self.solution is not None, 'call solve'
    if self.solution == 'UNSAT':
      return 'UNSAT'
    return [
      (
        self._format_control(step),
        self._format_step(step=step)
      )
      for step in range(self.stage_data.steps, -1, -1)
    ]

  def _format_control(self, step: Step) -> str:
    if step == 0:
      return 'N/A'
    control_state: UserInput = self.control_states[step]
    if self.solution[control_state.is_up]:
      return 'Up'
    if self.solution[control_state.is_down]:
      return 'Down'
    if self.solution[control_state.is_left]:
      return 'Left'
    if self.solution[control_state.is_right]:
      return 'Right'
    if self.solution[control_state.skip]:
      return 'Skip'
    return '-----'

  def _format_step(self, step: Step) -> str:
    return '\n'.join(
      ''.join(
        self._format_cell(steppos=(step, x, y))
          for x in range(self.stage_data.width)
      )
        for y in range(self.stage_data.height)
    )

  def _format_cell(self, steppos: StepPos) -> str:
    cell_state: CellState = self.stage_states[steppos]
    step, x, y = steppos
    if self.solution[cell_state.is_stone]:
      return '*'
    if self.solution[cell_state.is_wall]:
      return 'W'
    if self.solution[cell_state.is_enemy]:
      return 'E'
    if self.solution[cell_state.is_player]:
      return 'P'
    if self.solution[cell_state.is_lock]:
      return 'L'
    if self.stage_data.get_cell(x=x, y=y) == CellType.KEY:
      return 'K'
    if self.solution[cell_state.is_spike_active]:
      return 'A'
    if self.solution[cell_state.is_spike]:
      return '_'
    if self.stage_data.get_cell(x=x, y=y) == CellType.START:
      return 'S'
    if self.stage_data.get_cell(x=x, y=y) == CellType.GOAL:
      return 'G'
    if self.solution[cell_state.is_lock]:
      return 'L'
    if self.solution[cell_state.is_empty]:
      return '.'
    return '?'

  def _format_aux_state(self, step):
    has_key = self.aux_states.has_key[step]
    if self.solution[has_key]:
      return 'Key'
    return ''


def _next_steppos(steppos: StepPos) -> StepPos:
  step, x, y = steppos
  return (step - 1, x, y)


def _prev_steppos(steppos: StepPos) -> StepPos:
  step, x, y = steppos
  return (step + 1, x, y)


class HelltakerProblemFactory:
  _stage_data: StageData

  def __init__(self, stage_data: StageData):
    self._stage_data = stage_data

  def generate_problem(self) -> HelltakerProblem:
    sat_problem = SatProblem()
    helltaker_problem = HelltakerProblem(
      stage_data=self._stage_data,
      stage_states=self._create_stage_states(sat_problem),
      control_states=self._create_control_states(sat_problem),
      aux_states=self._create_aux_states(sat_problem),
      problem=sat_problem,
    )

    self._setup_init_state(helltaker_problem=helltaker_problem)
    self._setup_final_state(helltaker_problem=helltaker_problem)
    self._setup_movement_rules(helltaker_problem=helltaker_problem)
    self._setup_kickable_rules(helltaker_problem=helltaker_problem)
    self._setup_skip_rules(helltaker_problem=helltaker_problem)
    self._setup_spike_state(helltaker_problem=helltaker_problem)
    self._setup_stone_available_rules(helltaker_problem=helltaker_problem)
    self._setup_stone_rules(helltaker_problem=helltaker_problem)

    self._setup_has_key_rules(helltaker_problem=helltaker_problem)
    self._setup_lock_rules(helltaker_problem=helltaker_problem)

    self._setup_enemy_rules(helltaker_problem=helltaker_problem)

    return helltaker_problem

  def _create_stage_states(self, problem: SatProblem) -> StageStates:
    stage_states: StageStates = {}
    height = self._stage_data.height
    width = self._stage_data.width
    for step in range(self._stage_data.steps + 1):
      for y in range(height):
        for x in range(width):
          steppos = (step, x, y)
          state = CellState.create(steppos=steppos, problem=problem)
          stage_states[steppos] = state

          problem.add_clauses(state.gen_rules())

    for step in range(self._stage_data.steps + 1):
      # single player state
      rules = gen_unique_true_rules([
        stage_states[(step, x, y)].is_player
        for y in range(height)
        for x in range(width)
      ])
      problem.add_clauses(rules)

    return stage_states

  def _create_control_states(self, problem: SatProblem) -> ControlStates:
    control_states: ControlStates = {}
    for step in range(1, self._stage_data.steps + 1):
      state = UserInput.create(step=step, problem=problem)
      control_states[step] = state

      problem.add_clauses(state.gen_rules())

    return control_states

  def _create_aux_states(self, sat_problem: SatProblem):
    return AuxStates.create(self._stage_data.steps, sat_problem)

  def _setup_movement_rules(self, helltaker_problem: HelltakerProblem):
    problem: SatProblem = helltaker_problem.problem
    control_states: ControlStates = helltaker_problem.control_states
    stage_states: StageStates = helltaker_problem.stage_states
    stage_data: StageData = helltaker_problem.stage_data

    clauses = []
    for steppos, state in stage_states.items():
      (step, x, y) = steppos
      if step <= 0:
        continue

      control: UserInput = control_states[step]
      y_minus = max(y - 1, 0)
      y_plus = min(y + 1, stage_data.height - 1)
      x_minus = max(x - 1, 0)
      x_plus = min(x + 1, stage_data.width - 1)

      pos_up = (step, x, y_minus)
      pos_down = (step, x, y_plus)
      pos_right = (step, x_plus, y)
      pos_left = (step, x_minus, y)

      clauses.extend([
        # up no move
        all_then(
          conditions=[stage_states.get(pos_up).is_blocker, stage_states.get(steppos).is_player, control.is_up],
          then=stage_states.get(_next_steppos(steppos)).is_player,
        ),
        # up move
        all_then(
          conditions=[~stage_states.get(pos_up).is_blocker, stage_states.get(steppos).is_player, control.is_up],
          then=[
            stage_states.get(_next_steppos(steppos)).is_empty,
            stage_states.get(_next_steppos(pos_up)).is_player,
          ],
        ),
        # down no move
        all_then(
          conditions=[stage_states.get(pos_down).is_blocker, stage_states.get(steppos).is_player, control.is_down],
          then=stage_states.get(_next_steppos(steppos)).is_player,
        ),
        # down move
        all_then(
          conditions=[~stage_states.get(pos_down).is_blocker, stage_states.get(steppos).is_player, control.is_down],
          then=[
            stage_states.get(_next_steppos(steppos)).is_empty,
            stage_states.get(_next_steppos(pos_down)).is_player,
          ],
        ),
        # left no move
        all_then(
          conditions=[stage_states.get(pos_left).is_blocker, stage_states.get(steppos).is_player, control.is_left],
          then=stage_states.get(_next_steppos(steppos)).is_player,
        ),
        # left move
        all_then(
          conditions=[~stage_states.get(pos_left).is_blocker, stage_states.get(steppos).is_player, control.is_left],
          then=[
            stage_states.get(_next_steppos(steppos)).is_empty,
            stage_states.get(_next_steppos(pos_left)).is_player,
          ],
        ),
        # right no move
        all_then(
          conditions=[stage_states.get(pos_right).is_blocker, stage_states.get(steppos).is_player, control.is_right],
          then=stage_states.get(_next_steppos(steppos)).is_player,
        ),
        # right move
        all_then(
          conditions=[~stage_states.get(pos_right).is_blocker, stage_states.get(steppos).is_player, control.is_right],
          then=[
            stage_states.get(_next_steppos(steppos)).is_empty,
            stage_states.get(_next_steppos(pos_right)).is_player,
          ],
        ),
        # skip
        all_then(
          conditions=[stage_states.get(steppos).is_player, control.skip],
          then=stage_states.get(_next_steppos(steppos)).is_player,
        )
      ])
    problem.add_clauses(sum(clauses, []))

  def _setup_movement_available_rules(self, helltaker_problem: HelltakerProblem):
    problem: SatProblem = helltaker_problem.problem
    stage_states: StageStates = helltaker_problem.stage_states
    stage_data: StageData = helltaker_problem.stage_data

    clauses = []
    for steppos, state in stage_states.items():
      (step, x, y) = steppos
      if stage_data.steps > step:
        clauses.extend(
          all_then(
            conditions=[stage_states.get(c).is_player.neg() for c in self._adjacent_cells(step + 1, x, y)],
            then=stage_states.get(steppos).is_player.neg(),
          )
        )
    problem.add_clauses(clauses)

  def _setup_stone_available_rules(self, helltaker_problem: HelltakerProblem):
    problem: SatProblem = helltaker_problem.problem
    stage_states: StageStates = helltaker_problem.stage_states
    stage_data: StageData = helltaker_problem.stage_data

    clauses = []
    for steppos, state in stage_states.items():
      (step, x, y) = steppos
      if stage_data.steps > step:
        clauses.extend(
          all_then(
            conditions=[stage_states.get(c).is_stone.neg() for c in self._adjacent_cells(step + 1, x, y)],
            then=stage_states.get(steppos).is_stone.neg(),
          )
        )
    problem.add_clauses(clauses)

  def _setup_enemy_rules(self, helltaker_problem: HelltakerProblem):
    problem: SatProblem = helltaker_problem.problem
    control_states: ControlStates = helltaker_problem.control_states
    stage_states: StageStates = helltaker_problem.stage_states

    def _gen_rule(ctrl: Variable, steppos_from: StepPos, steppos_player: StepPos, steppos_to: StepPos):
      clauses = []

      next_steppos_from = _next_steppos(steppos_from)
      next_steppos_player = _next_steppos(steppos_player)
      next_steppos_to = _next_steppos(steppos_to)
      if not all(map(self._is_in_stage, [steppos_from, steppos_player, next_steppos_from, next_steppos_player])):
        return []

      # Kill/move enemy when kicked.
      move_conditions: List[Literal] = [
        ctrl.pos(),
        stage_states.get(steppos_from).is_player.pos(),
        stage_states.get(steppos_player).is_enemy.pos()
      ]
      clauses.extend(all_then(
        conditions=move_conditions,
        then=stage_states.get(next_steppos_player).is_enemy_pre.neg(),
      ))
      # Move enemy if `to` is empty.
      if self._is_in_stage(steppos_to):
        clauses.extend(all_then(
          conditions=move_conditions + [stage_states.get(steppos_to).is_empty.pos()],
          then=stage_states.get(next_steppos_to).is_enemy_pre.pos(),
        ))

      # Stay if control is not applicable
      clauses.extend(all_then(
        conditions=[
          ctrl.neg(),
          stage_states.get(steppos_from).is_player.pos(),
          stage_states.get(steppos_player).is_enemy.pos(),
        ],
        then=stage_states.get(next_steppos_player).is_enemy_pre.pos(),
      ))
      clauses.extend(all_then(
        conditions=[
          ctrl.neg(),
          stage_states.get(steppos_from).is_player.pos(),
          stage_states.get(steppos_player).is_enemy.neg(),
        ],
        then=stage_states.get(next_steppos_player).is_enemy_pre.neg(),
      ))

      # Keep empty in player cell if control is not applicable
      clauses.extend(all_then(
        conditions=[
          ctrl.neg(),
          stage_states.get(steppos_from).is_player.pos(),
          stage_states.get(steppos_player).is_enemy.neg(),
        ],
        then=stage_states.get(next_steppos_player).is_enemy_pre.neg(),
      ))
      clauses.extend(all_then(
        conditions=[
          ctrl.neg(),
          stage_states.get(steppos_from).is_player.pos(),
          stage_states.get(steppos_player).is_enemy.pos(),
        ],
        then=stage_states.get(steppos_player).is_enemy_pre.pos(),
      ))
      if self._is_in_stage(steppos_to):
        # keep `to` state if not applicable
        clauses.extend(all_then(
          conditions=[
            ctrl.neg(),
            stage_states.get(steppos_from).is_player.pos(),
            stage_states.get(steppos_to).is_enemy.neg(),
          ],
          then=stage_states.get(next_steppos_to).is_enemy_pre.neg(),
        ))
        clauses.extend(all_then(
          conditions=[
            ctrl.pos(),
            stage_states.get(steppos_from).is_player.pos(),
            stage_states.get(steppos_player).is_enemy.neg(),
            stage_states.get(steppos_to).is_enemy.neg(),
          ],
          then=stage_states.get(next_steppos_to).is_enemy_pre.neg(),
        ))

      return clauses

    clauses = []
    for steppos, state in stage_states.items():
      (step, x, y) = steppos
      next_steppos = (step - 1, x, y)
      if step == 0:
        continue

      # Enemy do not pop up from void
      clauses.extend(
        all_then(
          conditions=[stage_states.get(c).is_enemy.neg() for c in self._adjacent_cells(*steppos)],
          then=[
            stage_states.get(next_steppos).is_enemy_pre.neg(),
            stage_states.get(next_steppos).is_enemy.neg(),
          ]
        )
      )

      # Enemy stey keep if player is not in adj 
      clauses.extend(
        all_then(
          conditions=[
                       stage_states.get(c).is_player.neg() for c in self._adjacent_cells(*steppos)
                     ] + [stage_states.get(steppos).is_enemy],
          then=stage_states.get(next_steppos).is_enemy_pre,
        )
      )
      clauses.extend(
        all_then(
          conditions=[
                       stage_states.get(c).is_player.neg() for c in self._adjacent_cells(*steppos, delta=2)
                     ] + [stage_states.get(steppos).is_enemy.neg()],
          then=stage_states.get(next_steppos).is_enemy_pre.neg(),
        )
      )

      control = control_states[step]

      steppos_left = (step, x - 1, y)
      steppos_right = (step, x + 1, y)
      steppos_up = (step, x, y - 1)
      steppos_down = (step, x, y + 1)

      clauses.extend(_gen_rule(control.is_right, steppos_left, steppos, steppos_right))
      clauses.extend(_gen_rule(control.is_left, steppos_right, steppos, steppos_left))
      clauses.extend(_gen_rule(control.is_down, steppos_up, steppos, steppos_down))
      clauses.extend(_gen_rule(control.is_up, steppos_down, steppos, steppos_up))

    problem.add_clauses(clauses)

  def _setup_stone_rules(self, helltaker_problem: HelltakerProblem):
    problem: SatProblem = helltaker_problem.problem
    control_states: ControlStates = helltaker_problem.control_states
    stage_states: StageStates = helltaker_problem.stage_states

    def _gen_rule(ctrl: Variable, steppos_from: StepPos, steppos_player: StepPos, steppos_to: StepPos):
      clauses = []

      next_steppos_from = _next_steppos(steppos_from)
      next_steppos_player = _next_steppos(steppos_player)
      next_steppos_to = _next_steppos(steppos_to)
      if not all(map(self._is_in_stage, [steppos_from, steppos_player, next_steppos_from, next_steppos_player])):
        return []

      # Move/stay stone when kicked.
      move_conditions: List[Literal] = [
        ctrl.pos(),
        stage_states.get(steppos_from).is_player.pos(),
        stage_states.get(steppos_player).is_stone.pos()
      ]
      if self._is_in_stage(steppos_to):
        # move if empty
        clauses.extend(all_then(
          conditions=move_conditions + [stage_states.get(steppos_to).is_empty.pos()],
          then=stage_states.get(next_steppos_to).is_stone.pos(),
        ))
        # stay if not empty
        clauses.extend(all_then(
          conditions=move_conditions + [stage_states.get(steppos_to).is_empty.neg()],
          then=stage_states.get(next_steppos_player).is_stone.pos(),
        ))
      else:
        # stay
        clauses.extend(all_then(
          conditions=move_conditions,
          then=stage_states.get(next_steppos_player).is_stone.pos(),
        ))

      # Stay if control is not applicable
      clauses.extend(all_then(
        conditions=[
          ctrl.neg(),
          stage_states.get(steppos_from).is_player.pos(),
          stage_states.get(steppos_player).is_stone.pos(),
        ],
        then=stage_states.get(next_steppos_player).is_stone.pos(),
      ))
      clauses.extend(all_then(
        conditions=[
          ctrl.neg(),
          stage_states.get(steppos_from).is_player.pos(),
          stage_states.get(steppos_player).is_stone.neg(),
        ],
        then=stage_states.get(next_steppos_player).is_stone.neg(),
      ))

      # Keep empty in player cell if control is not applicable
      clauses.extend(all_then(
        conditions=[
          ctrl.neg(),
          stage_states.get(steppos_from).is_player.pos(),
          stage_states.get(steppos_player).is_stone.neg(),
        ],
        then=stage_states.get(next_steppos_player).is_stone.neg(),
      ))
      clauses.extend(all_then(
        conditions=[
          ctrl.neg(),
          stage_states.get(steppos_from).is_player.pos(),
          stage_states.get(steppos_player).is_stone.pos(),
        ],
        then=stage_states.get(steppos_player).is_stone.pos(),
      ))
      if self._is_in_stage(steppos_to):
        # keep `to` state if control is not applicable 
        clauses.extend(all_then(
          conditions=[
            ctrl.neg(),
            stage_states.get(steppos_from).is_player.pos(),
            stage_states.get(steppos_to).is_stone.neg(),
          ],
          then=stage_states.get(next_steppos_to).is_stone.neg(),
        ))
        clauses.extend(all_then(
          conditions=[
            ctrl.pos(),
            stage_states.get(steppos_from).is_player.pos(),
            stage_states.get(steppos_player).is_stone.neg(),
            stage_states.get(steppos_to).is_stone.neg(),
          ],
          then=stage_states.get(next_steppos_to).is_stone.neg(),
        ))

      return clauses

    clauses = []
    for steppos, state in stage_states.items():
      (step, x, y) = steppos
      next_steppos = (step - 1, x, y)
      if step == 0:
        continue

      # Stone do not pop up from void if adj cells don't have stone
      clauses.extend(
        all_then(
          conditions=[stage_states.get(c).is_stone.neg() for c in self._adjacent_cells(*steppos)],
          then=stage_states.get(next_steppos).is_stone.neg(),
        )
      )

      # Stone stay keep if player is not in adj 
      clauses.extend(
        all_then(
          conditions=[
                       stage_states.get(c).is_player.neg() for c in self._adjacent_cells(*steppos)
                     ] + [stage_states.get(steppos).is_stone],
          then=stage_states.get(next_steppos).is_stone,
        )
      )
      clauses.extend(
        all_then(
          conditions=[
                       stage_states.get(c).is_player.neg() for c in self._adjacent_cells(*steppos, delta=2)
                     ] + [stage_states.get(steppos).is_stone.neg()],
          then=stage_states.get(next_steppos).is_stone.neg(),
        )
      )

      control = control_states[step]

      steppos_left = (step, x - 1, y)
      steppos_right = (step, x + 1, y)
      steppos_up = (step, x, y - 1)
      steppos_down = (step, x, y + 1)

      clauses.extend(_gen_rule(control.is_right, steppos_left, steppos, steppos_right))
      clauses.extend(_gen_rule(control.is_left, steppos_right, steppos, steppos_left))
      clauses.extend(_gen_rule(control.is_down, steppos_up, steppos, steppos_down))
      clauses.extend(_gen_rule(control.is_up, steppos_down, steppos, steppos_up))

    problem.add_clauses(clauses)

  def _setup_init_state(self, helltaker_problem: HelltakerProblem):
    problem: SatProblem = helltaker_problem.problem
    stage_data: StageData = helltaker_problem.stage_data
    stage_states: StageStates = helltaker_problem.stage_states
    initial_step = stage_data.steps

    clauses = []
    for y, y_data in enumerate(stage_data):
      for x, cell in enumerate(y_data):
        state: CellState = stage_states[(initial_step, x, y)]
        assert isinstance(cell, CellType)

        if cell == CellType.START:
          clauses.append(Clause([state.is_player.pos()]))
        if cell.is_stone():
          clauses.append(Clause([state.is_stone.pos()]))
        else:
          clauses.append(Clause([state.is_stone.neg()]))
        if cell == CellType.ENEMY:
          clauses.append(Clause([state.is_enemy.pos()]))
        else:
          clauses.append(Clause([state.is_enemy.neg()]))
        if cell == CellType.LOCK:
          clauses.append(Clause([state.is_lock.pos()]))

        if cell == CellType.SPIKE_TOGGLE_ACTIVE:
          clauses.append(Clause([state.is_spike.pos()]))
          clauses.append(Clause([state.is_spike_active.pos()]))
        elif cell.is_spike_toggle_inactive():
          clauses.append(Clause([state.is_spike.pos()]))
          clauses.append(Clause([state.is_spike_active.neg()]))

        # permanent state
        for step in range(stage_data.steps + 1):
          state: CellState = stage_states[(step, x, y)]
          if cell == CellType.WALL:
            clauses.append(Clause([state.is_wall.pos()]))
          else:
            clauses.append(Clause([state.is_wall.neg()]))
          if cell == CellType.GOAL:
            clauses.append(Clause([state.is_goal.pos()]))
          else:
            clauses.append(Clause([state.is_goal.neg()]))
          if cell != CellType.LOCK:
            clauses.append(Clause([state.is_lock.neg()]))

          if cell.is_spike():
            clauses.append(Clause([state.is_spike.pos()]))
            if cell == CellType.SPIKE:
              clauses.append(Clause([state.is_spike_active.pos()]))
          else:
            clauses.append(Clause([state.is_spike.neg()]))
            clauses.append(Clause([state.is_spike_active.neg()]))

    problem.add_clauses(clauses)

  def _setup_spike_state(self, helltaker_problem: HelltakerProblem):
    problem: SatProblem = helltaker_problem.problem
    stage_data: StageData = helltaker_problem.stage_data
    stage_states: StageStates = helltaker_problem.stage_states
    control_states: ControlStates = helltaker_problem.control_states

    for steppos, cell_state in stage_states.items():
      step, x, y = steppos
      cell_type = stage_data.get_cell(x, y)

      if cell_type.is_constant_spike():
        problem.add_clauses([
          Clause([cell_state.is_spike.pos()]),
          Clause([cell_state.is_spike_active.pos()]),
        ])
        continue
      elif cell_type.is_spike_toggle():
        problem.add_clauses([Clause([cell_state.is_spike.pos()])])
        if step == 0:
          continue
        # has next step and control
        control: UserInput = control_states[step]
        problem.add_clauses(all_then(
          conditions=[cell_state.is_spike_active.pos(), control.skip.pos()],
          then=[stage_states[_next_steppos(steppos)].is_spike_active.pos()],
        ))
        problem.add_clauses(all_then(
          conditions=[cell_state.is_spike_active.neg(), control.skip.pos()],
          then=[stage_states[_next_steppos(steppos)].is_spike_active.neg()],
        ))
        problem.add_clauses(all_then(
          conditions=[cell_state.is_spike_active.pos(), control.skip.neg()],
          then=[stage_states[_next_steppos(steppos)].is_spike_active.neg()],
        ))
        problem.add_clauses(all_then(
          conditions=[cell_state.is_spike_active.neg(), control.skip.neg()],
          then=[stage_states[_next_steppos(steppos)].is_spike_active.pos()],
        ))
        continue
      else:
        # not spike cell 
        problem.add_clauses([Clause([cell_state.is_spike.neg()])])
        problem.add_clauses([Clause([cell_state.is_spike_active.neg()])])
        continue

  def _setup_final_state(self, helltaker_problem: HelltakerProblem):
    problem: SatProblem = helltaker_problem.problem
    stage_data: StageData = helltaker_problem.stage_data
    stage_states: StageStates = helltaker_problem.stage_states

    for y in range(stage_data.height):
      for x in range(stage_data.width):
        cell = stage_data.get_cell(x, y)

        if cell == CellType.GOAL:
          problem.add_clauses(gen_any_rules([
            stage_states[steppos].is_player
            for steppos in self._adjacent_cells(0, x, y, include_itself=False)
          ]))

  def _setup_constant_states(self, helltaker_problem: HelltakerProblem):
    stage_data: StageData = helltaker_problem.stage_data
    stage_states: StageStates = helltaker_problem.stage_states

    clauses = []
    for steppos, cell_state in stage_states.items():
      step, x, y = steppos

      if stage_data.get_cell(x, y) == CellType.WALL:
        clauses.append(Clause([cell_state.is_wall.pos()]))
      else:
        clauses.append(Clause([cell_state.is_wall.neg()]))
    helltaker_problem.problem.add_clauses(clauses)

  def _setup_skip_rules(self, helltaker_problem: HelltakerProblem):
    stage_states: StageStates = helltaker_problem.stage_states
    control_states: ControlStates = helltaker_problem.control_states

    clauses = []
    for steppos, cell_state in stage_states.items():
      step, x, y = steppos
      control = control_states.get(step)
      prev_control = control_states.get(step + 1)
      if control is None:
        continue
      clauses.extend(all_then(
        conditions=[cell_state.is_player, cell_state.is_spike_active.neg()],
        then=control.skip.neg(),
      ))
      if prev_control is not None:
        clauses.extend(gen_no_two_true_rules([prev_control.skip, control.skip]))
        clauses.extend(all_then(
          conditions=[prev_control.skip.pos(), cell_state.is_player, cell_state.is_spike_active],
          then=control.skip.neg(),
        ))
        clauses.extend(all_then(
          conditions=[prev_control.skip.neg(), cell_state.is_player, cell_state.is_spike_active],
          then=control.skip.pos(),
        ))
    helltaker_problem.problem.add_clauses(clauses)

  def _setup_has_key_rules(self, helltaker_problem: HelltakerProblem):
    stage_data: StageData = helltaker_problem.stage_data
    stage_states: StageStates = helltaker_problem.stage_states
    aux_states: AuxStates = helltaker_problem.aux_states
    problem: SatProblem = helltaker_problem.problem

    for (x, y), cell_type in stage_data.items():
      if cell_type != CellType.KEY:
        continue
      key_x, key_y = x, y
      del x, y

      problem.add_clauses([Clause([aux_states.has_key[stage_data.steps].neg()])])  # initial state
      for target_step in range(stage_data.steps):
        key_cell_state: CellState = stage_states[(target_step, key_x, key_y)]
        has_key_state: Variable = aux_states.has_key[target_step]
        has_key_state_prev: Variable = aux_states.has_key[target_step + 1]
        problem.add_clauses(all_then(
          conditions=[key_cell_state.is_player],
          then=has_key_state.pos(),
        ))
        problem.add_clauses(all_then(
          conditions=[has_key_state_prev],
          then=has_key_state.pos(),
        ))
        problem.add_clauses(all_then(
          conditions=[has_key_state_prev.neg(), key_cell_state.is_player.neg()],
          then=has_key_state.neg(),
        ))

  def _setup_lock_rules(self, helltaker_problem: HelltakerProblem):
    stage_states: StageStates = helltaker_problem.stage_states
    stage_data: StageData = helltaker_problem.stage_data
    control_states: ControlStates = helltaker_problem.control_states
    aux_states: AuxStates = helltaker_problem.aux_states
    problem: SatProblem = helltaker_problem.problem

    def gen_unlock_rule(
        control: Variable, lock_steppos: StepPos, player_steppos: StepPos,
    ) -> List[Clause]:
      step, _, _ = lock_steppos
      assert player_steppos[0] == step
      has_key = aux_states.has_key[step]
      if not self._is_in_stage(player_steppos):
        return []
      unlock_condition = all_then(
        conditions=[has_key.pos(), stage_states[player_steppos].is_player.pos(), control.pos()],
        then=stage_states[_next_steppos(lock_steppos)].is_lock.neg(),
      )
      keep_conditions = all_then(
        conditions=[
          has_key.pos(), stage_states[player_steppos].is_player.pos(),
          control.neg(), stage_states[lock_steppos].is_lock.pos(),
        ],
        then=stage_states[_next_steppos(lock_steppos)].is_lock.pos(),
      ) + all_then(
        conditions=[
          has_key.pos(), stage_states[player_steppos].is_player.neg(),
          control.pos(), stage_states[lock_steppos].is_lock.pos(),
        ],
        then=stage_states[_next_steppos(lock_steppos)].is_lock.pos(),
      )
      return unlock_condition + keep_conditions

    for steppos, _ in stage_states.items():
      next_steppos = _next_steppos(steppos)
      if not self._is_in_stage(next_steppos):
        next_steppos = steppos

      # propagate is_lock_next and is_lock
      problem.add_clauses(all_then(
        conditions=[stage_states[next_steppos].is_lock.pos()],
        then=[stage_states[steppos].is_lock_next.pos()],
      ))
      problem.add_clauses(all_then(
        conditions=[stage_states[next_steppos].is_lock.neg()],
        then=[stage_states[steppos].is_lock_next.neg()],
      ))

    for (x, y) in self._lock_pos_list():
      for step in range(1, stage_data.steps + 1):
        steppos = (step, x, y)
        # no key then lock
        problem.add_clauses(all_then(
          conditions=[aux_states.has_key[step].neg()],
          then=[
            stage_states.get(steppos).is_lock.pos(),
            stage_states.get(_next_steppos(steppos)).is_lock.pos(),
          ],
        ))
        # unlock then unlock
        problem.add_clauses(all_then(
          conditions=[stage_states[steppos].is_lock.neg()],
          then=stage_states.get(_next_steppos(steppos)).is_lock.neg(),
        ))
        # no player in adj cells then lock
        problem.add_clauses(all_then(
          conditions=[stage_states[steppos].is_player.neg() for steppos in self._adjacent_cells(step, x, y)] + \
                     [stage_states[steppos].is_lock.pos()],
          then=stage_states.get(_next_steppos(steppos)).is_lock.pos(),
        ))

        control = control_states[step]
        steppos_left = (step, x - 1, y)
        steppos_right = (step, x + 1, y)
        steppos_up = (step, x, y - 1)
        steppos_down = (step, x, y + 1)

        problem.add_clauses(gen_unlock_rule(control.is_right, steppos, steppos_left))
        problem.add_clauses(gen_unlock_rule(control.is_left, steppos, steppos_right))
        problem.add_clauses(gen_unlock_rule(control.is_down, steppos, steppos_up))
        problem.add_clauses(gen_unlock_rule(control.is_up, steppos, steppos_down))

  def _setup_kickable_rules(self, helltaker_problem: HelltakerProblem):
    stage_states: StageStates = helltaker_problem.stage_states
    control_states: ControlStates = helltaker_problem.control_states
    problem: SatProblem = helltaker_problem.problem

    def gen_rules(steppos: StepPos, control: Variable, adj_steppos: StepPos) -> List[Clause]:
      if self._is_in_stage(adj_steppos):
        return \
          all_then(
            conditions=[stage_states.get(steppos).is_player, stage_states.get(adj_steppos).is_not_kickable],
            then=control.neg(),
          )
      else:
        return \
          all_then(
            conditions=[stage_states.get(steppos).is_player],
            then=control.neg(),
          )

    for steppos, cell_state in stage_states.items():
      step, x, y = steppos
      control = control_states.get(step)
      if control is None:
        continue

      steppos_left = (step, x - 1, y)
      steppos_right = (step, x + 1, y)
      steppos_up = (step, x, y - 1)
      steppos_down = (step, x, y + 1)

      problem.add_clauses(gen_rules(steppos, control.is_left, steppos_left))
      problem.add_clauses(gen_rules(steppos, control.is_right, steppos_right))
      problem.add_clauses(gen_rules(steppos, control.is_up, steppos_up))
      problem.add_clauses(gen_rules(steppos, control.is_down, steppos_down))

  def _lock_pos_list(self):
    return [
      pos
      for pos, cell_type in self._stage_data.items()
      if cell_type == CellType.LOCK
    ]

  def _adjacent_cells(self, step: int, base_x: int, base_y: int,
      delta: int = 1, include_itself: bool = True) -> List[StepPos]:
    pos_list = [
      (base_x - delta, base_y),
      (base_x, base_y - delta),
      (base_x + delta, base_y),
      (base_x, base_y + delta),
    ]
    if include_itself:
      pos_list.append((base_x, base_y))
    candidate_cells = [
      (step, x, y)
      for x, y in pos_list
    ]
    return list(filter(self._is_in_stage, candidate_cells))

  def _is_in_stage(self, steppos: StepPos) -> bool:
    step, x, y = steppos
    return (0 <= step <= self._stage_data.steps) \
           and (0 <= x < self._stage_data.width) \
           and (0 <= y < self._stage_data.height)
    pass


def main():
  if len(sys.argv) == 1:
    stage_data = ''.join(fileinput.input())
  else:
    stage_data = open(sys.argv[1]).read()

  stage = StageData.from_string(stage_data)
  generator = HelltakerProblemFactory(stage_data=stage)
  problem = generator.generate_problem()

  problem.solve()
  if problem.solution == 'UNSAT':
    print('UNSAT')
    return
  problem.visualize()
  print()
  print(f'#Variables: {len(problem.problem.variables)}')
  print(f'#Clauses  : {len(problem.problem.clauses)}')
  print()
  problem.show_steps()


if __name__ == '__main__':
  main()
