# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Forward step functions."""
import torch
import functools
from typing import Optional, Sequence

import torch as jax
# from jax import numpy as jp
import torch as jp
import mujoco
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import constraint
from mujoco.mjx._src import math
from mujoco.mjx._src import passive
from mujoco.mjx._src import scan
from mujoco.mjx._src import smooth
from mujoco.mjx._src import solver
from mujoco.mjx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import BiasType
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import DynType
from mujoco.mjx._src.types import GainType
from mujoco.mjx._src.types import IntegratorType
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np

# RK4 tableau
_RK4_A = np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 1.0],
])
_RK4_B = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])


def named_scope(fn, name: str = ''):
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    # with jax.named_scope(name or getattr(fn, '__name__')):
    res = fn(*args, **kwargs)
    return res

  return wrapper


@named_scope
def _position(m: Model, d: Data) -> Data:
  """Position-dependent computations."""
  # TODO(robotics-simulation): tendon
  d = smooth.kinematics(m, d)
  d = smooth.com_pos(m, d)
  d = smooth.crb(m, d)
  d = smooth.factor_m(m, d, d.qM)
  d = collision_driver.collision(m, d)
  d = constraint.make_constraint(m, d)
  d = smooth.transmission(m, d)
  return d


@named_scope
def _velocity(m: Model, d: Data) -> Data:
  """Velocity-dependent computations."""
  d = d.replace(actuator_velocity=d.actuator_moment @ d.qvel)
  d = smooth.com_vel(m, d)
  d = passive.passive(m, d)
  d = smooth.rne(m, d)
  return d


@named_scope
def _actuation(m: Model, d: Data) -> Data:
  """Actuation-dependent computations."""
  if not m.nu or m.opt.disableflags & DisableBit.ACTUATION:
    return d.replace(
        act_dot=jp.zeros((m.na,)),
        qfrc_actuator=jp.zeros((m.nv,)),
    )

  ctrl = d.ctrl
  if not m.opt.disableflags & DisableBit.CLAMPCTRL:
    ctrlrange = jp.where(
        m.actuator_ctrllimited[:, None],
        m.actuator_ctrlrange,
        jp.tensor([-jp.inf, jp.inf]),
    )
    ctrl = jp.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

  # act_dot for stateful actuators
  def get_act_dot(dyn_typ, dyn_prm, ctrl, act):
    if dyn_typ == DynType.NONE:
      act_dot = jp.tensor(0.0)
    elif dyn_typ == DynType.INTEGRATOR:
      act_dot = ctrl
    elif dyn_typ == DynType.FILTER:
      act_dot = (ctrl - act) / jp.clip(dyn_prm[0], mujoco.mjMINVAL)
    else:
      raise NotImplementedError(f'dyntype {dyn_typ.name} not implemented.')
    return act_dot

  act_dot = jp.zeros((m.na,))
  if m.na:
    act_dot = scan.flat(
        m,
        get_act_dot,
        'uuua',
        'a',
        m.actuator_dyntype,
        m.actuator_dynprm,
        ctrl,
        d.act,
        group_by='u',
    )

  ctrl_act = ctrl
  if m.na:
    act_last_dim = d.act[m.actuator_actadr + m.actuator_actnum - 1]
    ctrl_act = jp.where(m.actuator_actadr == -1, ctrl, act_last_dim)

  def get_force(*args):
    gain_t, gain_p, bias_t, bias_p, len_, vel, ctrl_act = args

    typ, prm = GainType(gain_t), gain_p
    if typ == GainType.FIXED:
      gain = prm[0]
    elif typ == GainType.AFFINE:
      gain = prm[0] + prm[1] * len_ + prm[2] * vel
    else:
      raise RuntimeError(f'unrecognized gaintype {typ.name}.')

    typ, prm = BiasType(bias_t), bias_p
    bias = jp.tensor(0.0)
    if typ == BiasType.AFFINE:
      bias = prm[0] + prm[1] * len_ + prm[2] * vel

    return gain * ctrl_act + bias

  force = scan.flat(
      m,
      get_force,
      'uuuuuuu',
      'u',
      m.actuator_gaintype,
      m.actuator_gainprm,
      m.actuator_biastype,
      m.actuator_biasprm,
      d.actuator_length,
      d.actuator_velocity,
      ctrl_act,
      group_by='u',
  )
  forcerange = jp.where(
      m.actuator_forcelimited[:, None],
      m.actuator_forcerange,
      jp.tensor([-jp.inf, jp.inf]),
  )
  force = jp.clip(force, forcerange[:, 0], forcerange[:, 1])

  qfrc_actuator = d.actuator_moment.T @ force

  # clamp qfrc_actuator
  actfrcrange = jp.where(
      m.jnt_actfrclimited[:, None],
      m.jnt_actfrcrange,
      jp.tensor([-jp.inf, jp.inf]),
  )
  ids = sum(
      ([i] * JointType(j).dof_width() for i, j in enumerate(m.jnt_type)), []
  )
  actfrcrange = jp.take(actfrcrange, jp.tensor(ids), axis=0)
  qfrc_actuator = jp.clip(qfrc_actuator, actfrcrange[:, 0], actfrcrange[:, 1])

  d = d.replace(act_dot=act_dot, qfrc_actuator=qfrc_actuator)
  return d


@named_scope
def _acceleration(m: Model, d: Data) -> Data:
  """Add up all non-constraint forces, compute qacc_smooth."""
  qfrc_applied = d.qfrc_applied + support.xfrc_accumulate(m, d)
  qfrc_smooth = d.qfrc_passive - d.qfrc_bias + d.qfrc_actuator + qfrc_applied
  qacc_smooth = smooth.solve_m(m, d, qfrc_smooth)
  d = d.replace(qfrc_smooth=qfrc_smooth, qacc_smooth=qacc_smooth)
  return d


@named_scope
def _integrate_pos(
    jnt_typs: Sequence[str], qpos: jax.Tensor, qvel: jax.Tensor, dt: jax.Tensor
) -> jax.Tensor:
  """Integrate position given velocity."""
  qs, qi, vi = [], 0, 0

  for jnt_typ in jnt_typs:
    if jnt_typ == JointType.FREE:
      pos = qpos[qi : qi + 3] + dt * qvel[vi : vi + 3]
      quat = math.quat_integrate(
          qpos[qi + 3 : qi + 7], qvel[vi + 3 : vi + 6], dt
      )
      qs.append(jp.concatenate([pos, quat]))
      qi, vi = qi + 7, vi + 6
    elif jnt_typ == JointType.BALL:
      quat = math.quat_integrate(qpos[qi : qi + 4], qvel[vi : vi + 3], dt)
      qs.append(quat)
      qi, vi = qi + 4, vi + 3
    elif jnt_typ in (JointType.HINGE, JointType.SLIDE):
      pos = qpos[qi] + dt * qvel[vi]
      qs.append(pos[None])
      qi, vi = qi + 1, vi + 1
    else:
      raise RuntimeError(f'unrecognized joint type: {jnt_typ}')

  return jp.concatenate(qs) if qs else jp.empty((0,))


@named_scope
def _advance(
    m: Model,
    d: Data,
    act_dot: jax.Tensor,
    qacc: jax.Tensor,
    qvel: Optional[jax.Tensor] = None,
) -> Data:
  """Advance state and time given activation derivatives and acceleration."""
  act = d.act
  if m.na:
    act = d.act + act_dot * m.opt.timestep
    actrange = jp.where(
        m.actuator_actlimited[:, None],
        m.actuator_actrange,
        jp.tensor([-jp.inf, jp.inf]),
    )
    fn = lambda act, actrange: jp.clip(act, actrange[0], actrange[1])
    act = scan.flat(m, fn, 'au', 'a', act, actrange, group_by='u')

  # advance velocities
  d = d.replace(qvel=d.qvel + qacc * m.opt.timestep)

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  qvel = d.qvel if qvel is None else qvel
  integrate_fn = lambda *args: _integrate_pos(*args, dt=m.opt.timestep)
  qpos = scan.flat(m, integrate_fn, 'jqv', 'q', m.jnt_type, d.qpos, qvel)

  # advance time
  time = d.time + m.opt.timestep

  return d.replace(act=act, qpos=qpos, time=time)


@named_scope
def _euler(m: Model, d: Data) -> Data:
  """Euler integrator, semi-implicit in velocity."""
  # integrate damping implicitly
  qacc = d.qacc
  if not m.opt.disableflags & DisableBit.EULERDAMP:
    # TODO(robotics-simulation): can this be done with a smaller perf hit
    mh = torch.scatter_add(d.qM, dim=0, index=torch.tensor(m.dof_Madr, dtype=torch.long), src=m.opt.timestep * m.dof_damping)
    dh = smooth.factor_m(m, d, mh)
    qfrc = d.qfrc_smooth + d.qfrc_constraint
    qacc = smooth.solve_m(m, dh, qfrc)
  return _advance(m, d, d.act_dot, qacc)


@named_scope
def _rungekutta4(m: Model, d: Data) -> Data:
  """Runge-Kutta explicit order 4 integrator."""
  d_t0 = d
  # pylint: disable=invalid-name
  A, B = _RK4_A, _RK4_B
  C = jp.tril(A).sum(axis=0)  # C(i) = sum_j A(i,j)
  T = d.time + C * m.opt.timestep
  # pylint: enable=invalid-name

  kqvel = d.qvel  # intermediate RK solution
  # RK solutions sum
  qvel, qacc, act_dot = jax.tree_map(
      lambda k: B[0] * k, (kqvel, d.qacc, d.act_dot)
  )
  integrate_fn = lambda *args: _integrate_pos(*args, dt=m.opt.timestep)

  def f(carry, x):
    qvel, qacc, act_dot, kqvel, d = carry
    a, b, t = x  # tableau numbers
    dqvel, dqacc, dact_dot = jax.tree_map(
        lambda k: a * k, (kqvel, d.qacc, d.act_dot)
    )
    # get intermediate RK solutions
    kqpos = scan.flat(m, integrate_fn, 'jqv', 'q', m.jnt_type, d_t0.qpos, dqvel)
    kact = d_t0.act + dact_dot * m.opt.timestep
    kqvel = d_t0.qvel + dqacc * m.opt.timestep
    d = d.replace(qpos=kqpos, qvel=kqvel, act=kact, time=t)
    d = forward(m, d)

    qvel += b * kqvel
    qacc += b * d.qacc
    act_dot += b * d.act_dot

    return (qvel, qacc, act_dot, kqvel, d), None

  abt = jp.vstack([jp.diag(A), B[1:4], T]).T
  out, _ = jax.lax.scan(f, (qvel, qacc, act_dot, kqvel, d), abt, unroll=3)
  qvel, qacc, act_dot, *_ = out

  d = _advance(m, d_t0, act_dot, qacc, qvel)
  return d


@named_scope
def forward(m: Model, d: Data) -> Data:
  """Forward dynamics."""
  d = _position(m, d)
  d = _velocity(m, d)
  d = _actuation(m, d)
  d = _acceleration(m, d)

  if d.efc_J.size == 0:
    d = d.replace(qacc=d.qacc_smooth)
    return d

  d = named_scope(solver.solve)(m, d)

  return d


@named_scope
def step(m: Model, d: Data) -> Data:
  """Advance simulation."""
  d = forward(m, d)

  if m.opt.integrator == IntegratorType.EULER:
    d = _euler(m, d)
  elif m.opt.integrator == IntegratorType.RK4:
    d = _rungekutta4(m, d)
  else:
    raise NotImplementedError(f'integrator {m.opt.integrator} not implemented.')

  return d
