import unittest
from typing import Optional, List, Tuple

import helltaker_solver


class Controls:
  R = 'Right'
  L = 'Left'
  U = 'Up'
  D = 'Down'
  S = 'Skip'
  N = 'N/A'


def solve(steps: int, stage_str: str) -> Optional[List[Tuple[str, str]]]:
  stage = helltaker_solver.StageData.from_string(f'{steps}\n{stage_str}')
  problem = helltaker_solver.HelltakerProblemFactory(stage_data=stage).generate_problem()

  problem.solve()
  return problem.get_solution()


def get_key_controls(solution: Tuple[Tuple[str, str]]):
  assert solution != 'UNSAT'
  return [
    control
    for control in [c for c, stage_state in solution]
    if control not in [
      Controls.S,
      Controls.N,
    ]
  ]


def _stage(*args):
  return '\n'.join(args)


class TestMovement(unittest.TestCase):
  def test_one_move_right(self):
    self.assertEqual(
      solve(1, 'S.G'),
      [
        (Controls.R, 'P.G'),
        (Controls.N, 'SPG'),
      ]
    )

  def test_one_move_left(self):
    self.assertEqual(
      solve(1, 'G.S'),
      [
        (Controls.L, 'G.P'),
        (Controls.N, 'GPS'),
      ]
    )

  def test_one_move_up(self):
    self.assertEqual(
      solve(1, _stage('G', '.', 'S')),
      [
        (Controls.U, _stage('G', '.', 'P')),
        (Controls.N, _stage('G', 'P', 'S')),
      ]
    )

  def test_one_move_down(self):
    self.assertEqual(
      solve(1, _stage('S', '.', 'G')),
      [
        (Controls.D, _stage('P', '.', 'G')),
        (Controls.N, _stage('S', 'P', 'G')),
      ]
    )

  def test_two_move(self):
    self.assertEqual(
      solve(2, 'S..G'),
      [
        (Controls.R, 'P..G'),
        (Controls.R, 'SP.G'),
        (Controls.N, 'S.PG'),
      ]
    )

  def test_two_move_rev(self):
    self.assertEqual(
      solve(2, 'G..S'),
      [
        (Controls.L, 'G..P'),
        (Controls.L, 'G.PS'),
        (Controls.N, 'GP.S'),
      ]
    )

  def test_one_move__solvable(self):
    self.assertNotEqual(
      solve(1, _stage('S.', '.G')),
      'UNSAT'
    )

  def test_two_move__not_solvable(self):
    self.assertEqual(
      solve(2, _stage('S...G')),
      'UNSAT'
    )

  def test_move_on_spike(self):
    self.assertEqual(
      solve(3, _stage('S|.G')),
      [
        (Controls.R, 'PA.G'),
        (Controls.S, 'SP.G'),
        (Controls.R, 'SP.G'),
        (Controls.N, 'SAPG'),
      ]
    )

  def test_move_two_spikes(self):
    self.assertEqual(
      solve(5, _stage('S||.G')),
      [
        (Controls.R, 'PAA.G'),
        (Controls.S, 'SPA.G'),
        (Controls.R, 'SPA.G'),
        (Controls.S, 'SAP.G'),
        (Controls.R, 'SAP.G'),
        (Controls.N, 'SAAPG'),
      ]
    )

  def test_move_spikes_case2(self):
    self.assertEqual(
      solve(3, _stage('AS|.G')),
      [
        (Controls.R, 'APA.G'),
        (Controls.S, '_SP.G'),
        (Controls.R, '_SP.G'),
        (Controls.N, 'ASAPG'),
      ]
    )

  def test_move_two_toggle_spikes(self):
    self.assertEqual(
      solve(3, _stage('SA_.G')),
      [
        (Controls.R, 'PA_.G'),
        (Controls.R, 'SPA.G'),
        (Controls.R, 'SAP.G'),
        (Controls.N, 'S_APG'),
      ]
    )


class Spike(unittest.TestCase):
  def test_constant_spike(self):
    self.assertEqual(
      solve(3, '|S...G'),
      [
        (Controls.R, 'AP...G'),
        (Controls.R, 'ASP..G'),
        (Controls.R, 'AS.P.G'),
        (Controls.N, 'AS..PG'),
      ]
    )

  def test_toggle_spike(self):
    self.assertEqual(
      solve(3, 'A_S...G'),
      [
        (Controls.R, 'A_P...G'),
        (Controls.R, '_ASP..G'),
        (Controls.R, 'A_S.P.G'),
        (Controls.N, '_AS..PG'),
      ]
    )

  def test_toggle_spike_skip(self):
    self.assertEqual(
      solve(5, 'A_S||.G'),
      [
        (Controls.R, 'A_PAA.G'),
        (Controls.S, '_ASPA.G'),
        (Controls.R, '_ASPA.G'),
        (Controls.S, 'A_SAP.G'),
        (Controls.R, 'A_SAP.G'),
        (Controls.N, '_ASAAPG'),
      ]
    )


class Wall(unittest.TestCase):
  def test_unreachable(self):
    self.assertEqual(
      solve(2, 'SWG'),
      'UNSAT'
    )

  def test_can_not_wait_by_kick(self):
    self.assertEqual(
      solve(3, _stage('S..G', '.W..')),
      'UNSAT'
    )

  def test_can_not_wait_by_kick_2(self):
    self.assertEqual(
      solve(3, _stage('S..G')),
      'UNSAT'
    )


class Enemy(unittest.TestCase):
  def test_kick(self):
    self.assertEqual(
      solve(3, _stage('SE.', 'W.G')),
      [
        (Controls.R, _stage('PE.', 'W.G')),
        (Controls.R, _stage('P.E', 'W.G')),
        (Controls.D, _stage('SPE', 'W.G')),
        (Controls.N, _stage('S.E', 'WPG')),
      ]
    )

  def test_kill(self):
    self.assertEqual(
      solve(2, 'SEG'),
      [
        (Controls.R, 'PEG'),
        (Controls.R, 'P.G'),
        (Controls.N, 'SPG'),
      ]
    )

  def test_double_kill(self):
    self.assertEqual(
      solve(6, 'SEE.G'),
      [
        (Controls.R, 'PEE.G'),
        (Controls.R, 'P.E.G'),
        (Controls.R, 'SPE.G'),
        (Controls.R, 'SP.EG'),
        (Controls.R, 'S.PEG'),
        (Controls.R, 'S.P.G'),
        (Controls.N, 'S..PG'),
      ]
    )

  def test_kill_by_spike(self):
    self.assertEqual(
      solve(5, 'SE|.G'),
      [
        (Controls.R, 'PEA.G'),
        (Controls.R, 'P.A.G'),
        (Controls.R, 'SPA.G'),
        (Controls.S, 'S.P.G'),
        (Controls.R, 'S.P.G'),
        (Controls.N, 'S.APG'),
      ]
    )

  def test_kill_by_spike2(self):
    self.assertEqual(
      solve(4, 'SEA.G'),
      [
        (Controls.R, 'PEA.G'),
        (Controls.R, 'P.E.G'),
        (Controls.R, 'SPA.G'),
        (Controls.R, 'S.P.G'),
        (Controls.N, 'S.APG'),
      ]
    )


class Stone(unittest.TestCase):
  def test_kick(self):
    self.assertEqual(
      solve(3, _stage('S*.', 'W.G')),
      [
        (Controls.R, _stage('P*.', 'W.G')),
        (Controls.R, _stage('P.*', 'W.G')),
        (Controls.D, _stage('SP*', 'W.G')),
        (Controls.N, _stage('S.*', 'WPG')),
      ]
    )

  def test_kick_unsat1(self):
    self.assertEqual(
      solve(5, 'S**G.'),
      'UNSAT'
    )

  def test_kick_unsat2(self):
    self.assertEqual(
      solve(6, 'S*.G*'),
      'UNSAT'
    )

  def test_kick_unsat3(self):
    self.assertEqual(
      solve(4, 'S*G.'),
      'UNSAT'
    )


class Lock(unittest.TestCase):
  def test_normal(self):
    self.assertEqual(
      solve(3, _stage('SKL.G')),
      [
        (Controls.R, _stage('PKL.G')),
        (Controls.R, _stage('SPL.G')),
        (Controls.R, _stage('SKP.G')),
        (Controls.N, _stage('SK.PG')),
      ]
    )


class Helltaker(unittest.TestCase):
  def test_stage1(self):
    stage_data = _stage(
      'WWWW.SW',
      'W..E..W',
      'W.E.EWW',
      '..WWWWW',
      '.*..*.W',
      '.*.*..G',
    )
    result = solve(23, stage_data)
    self.assertNotEqual(result, 'UNSAT')

  def test_stage2(self):
    stage_data = _stage(
      'W....WW',
      'WEW||..',
      '.|WW$$*',
      '..WW.|.',
      'S.WW.E.',
      'WWWWG.E',
    )
    result = solve(24, stage_data)
    self.assertNotEqual(result, 'UNSAT')
