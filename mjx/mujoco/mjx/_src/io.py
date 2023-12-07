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
"""Functions to initialize, load, or save data."""

# from jax import numpy as jp
import torch as jp
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import constraint
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import Contact
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np

DEFAULT_DTYPE = None

def make_data(m: Model) -> Data:
  """Allocate and initialize Data."""

  # create first d to get num contacts and nc
  d = Data(
      # solver_niter=jp.tensor(0, dtype=jp.int32),
      solver_niter=jp.tensor(0, dtype=jp.int32),
      ne=0,
      nf=0,
      nl=0,
      nefc=0,
      ncon=0,
      time=jp.zeros((), dtype=DEFAULT_DTYPE),
      qpos=m.qpos0,
      qvel=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      act=jp.zeros(m.na, dtype=DEFAULT_DTYPE),
      qacc_warmstart=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      ctrl=jp.zeros(m.nu, dtype=DEFAULT_DTYPE),
      qfrc_applied=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      xfrc_applied=jp.zeros((m.nbody, 6), dtype=DEFAULT_DTYPE),
      eq_active=jp.zeros(m.neq, dtype=jp.int32),
      qacc=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      act_dot=jp.zeros(m.na, dtype=DEFAULT_DTYPE),
      xpos=jp.zeros((m.nbody, 3), dtype=DEFAULT_DTYPE),
      xquat=jp.zeros((m.nbody, 4), dtype=DEFAULT_DTYPE),
      xmat=jp.zeros((m.nbody, 3, 3), dtype=DEFAULT_DTYPE),
      xipos=jp.zeros((m.nbody, 3), dtype=DEFAULT_DTYPE),
      ximat=jp.zeros((m.nbody, 3, 3), dtype=DEFAULT_DTYPE),
      xanchor=jp.zeros((m.njnt, 3), dtype=DEFAULT_DTYPE),
      xaxis=jp.zeros((m.njnt, 3), dtype=DEFAULT_DTYPE),
      geom_xpos=jp.zeros((m.ngeom, 3), dtype=DEFAULT_DTYPE),
      geom_xmat=jp.zeros((m.ngeom, 3, 3), dtype=DEFAULT_DTYPE),
      subtree_com=jp.zeros((m.nbody, 3), dtype=DEFAULT_DTYPE),
      cdof=jp.zeros((m.nv, 6), dtype=DEFAULT_DTYPE),
      cinert=jp.zeros((m.nbody, 10), dtype=DEFAULT_DTYPE),
      actuator_length=jp.zeros(m.nu, dtype=DEFAULT_DTYPE),
      actuator_moment=jp.zeros((m.nu, m.nv), dtype=DEFAULT_DTYPE),
      crb=jp.zeros((m.nbody, 10), dtype=DEFAULT_DTYPE),
      qM=jp.zeros(m.nM, dtype=DEFAULT_DTYPE),
      qLD=jp.zeros(m.nM, dtype=DEFAULT_DTYPE),
      qLDiagInv=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qLDiagSqrtInv=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      contact=Contact.zero(),
      efc_J=jp.zeros((), dtype=DEFAULT_DTYPE),
      efc_frictionloss=jp.zeros((), dtype=DEFAULT_DTYPE),
      efc_D=jp.zeros((), dtype=DEFAULT_DTYPE),
      actuator_velocity=jp.zeros(m.nu, dtype=DEFAULT_DTYPE),
      cvel=jp.zeros((m.nbody, 6), dtype=DEFAULT_DTYPE),
      cdof_dot=jp.zeros((m.nv, 6), dtype=DEFAULT_DTYPE),
      qfrc_bias=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qfrc_passive=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      efc_aref=jp.zeros((), dtype=DEFAULT_DTYPE),
      actuator_force=jp.zeros(m.nu, dtype=DEFAULT_DTYPE),
      qfrc_actuator=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qfrc_smooth=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qacc_smooth=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qfrc_constraint=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qfrc_inverse=jp.zeros(m.nv, dtype=DEFAULT_DTYPE),
      efc_force=jp.zeros((), dtype=DEFAULT_DTYPE),
  )

  # get contact data with correct shapes
  ncon = collision_driver.ncon(m)
  d = d.replace(contact=Contact.zero((ncon,)), ncon=ncon)
  d = d.tree_replace({'contact.dim': 3 * np.ones(ncon)})

  ne, nf, nl, nc = constraint.count_constraints(m, d)
  d = d.replace(ne=ne, nf=nf, nl=nl, nefc=ne + nf + nl + nc)
  ns = ne + nf + nl
  d = d.tree_replace({'contact.efc_address': np.arange(ns, ns + ncon * 4, 4)})
  d = d.replace(
      efc_J=jp.zeros((d.nefc, m.nv), dtype=DEFAULT_DTYPE),
      efc_frictionloss=jp.zeros(d.nefc, dtype=DEFAULT_DTYPE),
      efc_D=jp.zeros(d.nefc, dtype=DEFAULT_DTYPE),
      efc_aref=jp.zeros(d.nefc, dtype=DEFAULT_DTYPE),
      efc_force=jp.zeros(d.nefc, dtype=DEFAULT_DTYPE),
  )

  return d
