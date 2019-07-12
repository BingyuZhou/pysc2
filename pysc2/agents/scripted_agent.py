# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy 

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS


def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))


class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not beacon:
        return FUNCTIONS.no_op()
      beacon_center = numpy.mean(beacon, axis=0).round()
      return FUNCTIONS.Move_screen("now", beacon_center)
    else:
      return FUNCTIONS.select_army("select")


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()  # Average location.
      distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
      closest_mineral_xy = minerals[numpy.argmin(distances)]
      return FUNCTIONS.Move_screen("now", closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")


class CollectMineralShardsFeatureUnits(base_agent.BaseAgent):
  """An agent for solving the CollectMineralShards map with feature units.

  Controls the two marines independently:
  - select marine
  - move to nearest mineral shard that wasn't the previous target
  - swap marine and repeat
  """

  def setup(self, obs_spec, action_spec):
    super(CollectMineralShardsFeatureUnits, self).setup(obs_spec, action_spec)
    if "feature_units" not in obs_spec:
      raise Exception("This agent requires the feature_units observation.")

  def reset(self):
    super(CollectMineralShardsFeatureUnits, self).reset()
    self._marine_selected = False
    self._previous_mineral_xy = [-1, -1]

  def step(self, obs):
    super(CollectMineralShardsFeatureUnits, self).step(obs)
    marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]
    if not marines:
      return FUNCTIONS.no_op()
    marine_unit = next((m for m in marines
                        if m.is_selected == self._marine_selected), marines[0])
    marine_xy = [marine_unit.x, marine_unit.y]

    if not marine_unit.is_selected:
      # Nothing selected or the wrong marine is selected.
      self._marine_selected = True
      return FUNCTIONS.select_point("select", marine_xy)

    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      # Find and move to the nearest mineral.
      minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                  if unit.alliance == _PLAYER_NEUTRAL]

      if self._previous_mineral_xy in minerals:
        # Don't go for the same mineral shard as other marine.
        minerals.remove(self._previous_mineral_xy)

      if minerals:
        # Find the closest.
        distances = numpy.linalg.norm(
            numpy.array(minerals) - numpy.array(marine_xy), axis=1)
        closest_mineral_xy = minerals[numpy.argmin(distances)]

        # Swap to the other marine.
        self._marine_selected = False
        self._previous_mineral_xy = closest_mineral_xy
        return FUNCTIONS.Move_screen("now", closest_mineral_xy)

    return FUNCTIONS.no_op()


class DefeatRoaches(base_agent.BaseAgent):
  """An agent specifically for solving the DefeatRoaches map."""

  def step(self, obs):
    super(DefeatRoaches, self).step(obs)
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      roaches = _xy_locs(player_relative == _PLAYER_ENEMY)
      if not roaches:
        return FUNCTIONS.no_op()

      # Find the roach with max y coord.
      target = roaches[numpy.argmax(numpy.array(roaches)[:, 1])]
      return FUNCTIONS.Attack_screen("now", target)

    if FUNCTIONS.select_army.id in obs.observation.available_actions:
      return FUNCTIONS.select_army("select")

    return FUNCTIONS.no_op()

class CollectMineralsAndGas(base_agent.BaseAgent):

  def reset(self):
    super().reset()
    self.select_scv = False
    self.gas_list = []

  def find_supplydepot(self, minerals):
    pass


  def step(self, obs):
    super().step(obs)

    minerals_xy = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.unit_type == units.Neutral.MineralField]
    gas_xy = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.unit_type == units.Neutral.VespeneGeyser]
  
    scv = [unit for unit in obs.observation.feature_units if unit.unit_type==units.Terran.SCV and unit.is_selected==self.select_scv]

    refinery_count = 0 # Count the refineries
    for unit in obs.observation.feature_units:
      if unit.unit_type == units.Terran.Refinery:
        refinery_count += 1

    if refinery_count > 0:
      if (obs.observation.player.food_used < obs.observation.player.food_cap):
        if (FUNCTIONS.Train_SCV_quick.id in obs.observation.available_actions):
          return FUNCTIONS.Train_SCV_quick("now")
        else:
          if numpy.random.rand(1) <0.5:
            cd_center_xy = [ [unit.x, unit.y] for unit in obs.observation.feature_units if unit.unit_type == units.Terran.CommandCenter]
            return FUNCTIONS.select_point("select", cd_center_xy[0])
      else:
        supply_xy = self.find_supplydepot(minerals_xy)
        return FUNCTIONS.Build_SupplyDepot_screen("queued", supply_xy)

    if FUNCTIONS.Build_Refinery_screen.id in obs.observation.available_actions:
      # Select scv
      if not (self.select_scv and scv):
        if obs.observation.player.idle_worker_count !=0:
          print("select idle")
          self.select_scv = True
          return FUNCTIONS.select_idle_worker('select')
        
        if scv:
          cluster = scv
        else:
          cluster = obs.observation.feature_units
          
        for unit in cluster:
          if unit.unit_type == units.Terran.SCV:
            print("select scv mineral")
            self.select_scv = True
            return FUNCTIONS.select_point("select", [unit.x,unit.y])
      
      
      # When refineries are full
      if refinery_count == len(gas_xy):
        # Send SCV to refinery
        print("harvest gas")
        self.select_scv = False
        return FUNCTIONS.Harvest_Gather_screen("queued", gas_xy[0])
          

      # Refineries are not fully utilized
      for gas in gas_xy:
        if gas not in self.gas_list:
          self.gas_list.append(gas)
          print("build refinery")
          self.select_scv = False
          return FUNCTIONS.Build_Refinery_screen("queued", gas)
    
    if not self.select_scv:
      if FUNCTIONS.select_idle_worker.id in obs.observation.available_actions:
        self.select_scv = True
        return FUNCTIONS.select_idle_worker("select")
      else:
        return FUNCTIONS.no_op()
   
    if FUNCTIONS.Harvest_Gather_screen.id in obs.observation.available_actions:
      self.select_scv = False
      print("harvest mineral")
      return FUNCTIONS.Harvest_Gather_screen("now", minerals_xy[0])

    print("no op")
    return FUNCTIONS.no_op()
