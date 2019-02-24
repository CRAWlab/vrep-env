#! /usr/bin/env python

###############################################################################
# singleLeg_vrep_env.py
#
# File to do Reinforcement Learning of the single leg jupming experimental setup
# within V-REP. Load the single_leg_complete_flexible.ttt scene V-REP prior to 
# running this script. This also requires the vrep_env module, the CRAWLAB
# fork of which can be found at:
#  https://github.com/CRAWlab/vrep-env
# 
# This wraps the V-REP interaction to conform to the OpenAI gym protocol, 
# making it easy to use from most Reinforcement Learning modules.
#  https://gym.openai.com
#  https://github.com/openai/gym
#
# Also make sure to change the simulation timestep to custom in V-REP. The 
# pulldown to do so is next to the "play" button in V-REP.
#
# Make sure to have the server side running in V-REP: 
# in a child script of a V-REP scene, add following command
# to be executed just once, at simulation start:
#
# simExtRemoteApiStart(19997)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!
#
# NOTE: Any plotting is set up for output, not viewing on screen.
#       So, it will likely be ugly on screen. The saved PDFs should look
#       better.
#
# Created: 02/23/19
#   - Joshua Vaughan
#   - joshua.vaughan@louisiana.edu
#   - http://www.ucs.louisiana.edu/~jev9637
#
# Modified:
#   * 
#
# TODO:
#   * 
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

# Import the vrep_env
from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent

import os

# You can also set this path manually to the full path where your V-REP scenes
# are located on your computer
vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces

import time

PRINT_STATES = False # Set true to print out the states as the simulation runs
PLOT_RESULTS = False # Set true to plot the results agter the simulation


class SingleLegVrepEnv(vrep_env.VrepEnv):
    metadata = {'render.modes': [],}

    def __init__(self,
                 server_addr='127.0.0.1',
                 server_port=19997,
                 scene_path=vrep_scenes_path+'/single_leg_complete_highRes_flexible.ttt',
                 ):
                 
        # Pass the server information to V-REP
        vrep_env.VrepEnv.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        # Get the handles for the objects in the scene we want to interact with. The 
        # names here should match the names in the scene in V-REP
        # Get the handles for the two motors
        servo_joints = ['hip_joint', 'knee_joint']
        self.servo_handles =  list(map(self.get_object_handle, servo_joints))

        # Get the handle for the links
        links = ['bearing_high', 'femur_high', 'tibia_high']
        self.link_handles = list(map(self.get_object_handle, links))
    
        # One action per joint
        dim_act = len(self.servo_handles)
        
        # Multiple dimensions per link
        dim_obs = 9 # x,y,z for each of the 3 links
        
        # Set up the limits on the actuation and states
        max_act = np.deg2rad(180) * np.ones([dim_act])
        max_obs = np.inf * np.ones([dim_obs])
        
        self.action_space      = gym.spaces.Box(-max_act, max_act)
        self.observation_space = gym.spaces.Box(-max_obs, max_obs)
        self.observation_array = np.empty((3,3))
        
        # counter for number of steps
        # Used to limit episode length
        self.counter = 0
        self.MAX_STEPS = 100
        
        print('SingleLegVrepEnv initialized')


    def _make_observation(self):
        self.observation_array = np.empty((3,3))
        self.joint_angles = np.empty((2,1))
        
        # Include link positions in observation
        # TODO: 02/23/19 - JEV - Should we use actuator angles instead, to match
        #                        what we'd be able to measure in experiment?
        for index, handle in enumerate(self.link_handles):
            self.observation_array[index,:] = self.obj_get_position(handle)
            
        for index, handle in enumerate(self.servo_handles):
            self.joint_angles[index,:] = self.obj_get_joint_angle(handle)
        
        self.observation = self.observation_array.flatten()


    def _make_action(self, action):
        for index, joint_vel in enumerate(action):
            self.obj_set_velocity(self.servo_handles[index], joint_vel)

        # TODO: 02/23/19 - JEV - add clipping


    def _step(self, action):
        # Clip xor Assert
        #actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        # Actuate
        self._make_action(action)
        
        # Step
        self.step_simulation()
        
        # Observe
        self._make_observation()
        
        # Reward
        # TODO: 02/23/19 - JEV - determine what this should be
        
        reward = np.sum(self.observation_array[:,2])
        
        # TODO: 02/23/19 - JEV - refine early stopping conditions
        if (self.observation_array[0,2] < 5e-2 or 
            self.counter > self.MAX_STEPS or
            np.max(np.abs(self.joint_angles)) > np.deg2rad(45)):
            done = True
        else:
            self.counter = self.counter + 1
            done = False
        
        return self.observation, reward, done, {}

    def _reset(self):
        """ Reset the simulation """
        # TODO: 02/23/19 - JEV - set up randomized initialization?
        if self.sim_running:
            self.stop_simulation()
        
        # reset the counter
        self.counter = 0
        
        self.start_simulation()
        
        self._make_observation()
        
        return self.observation
    
    def _render(self, mode='human', close=False):
        pass
    
    def _seed(self, seed=None):
        return []


if __name__ == '__main__':
    env = SingleLegVrepEnv()

    try:
        for episode in range(16):
            observation = env.reset()
            total_reward = 0
        
            for t in range(256):
                action = env.action_space.sample()
            
                observation, reward, done, _ = env.step(action)
            
                total_reward += reward
            
                if done:
                    break
                    
            print("Episode finished after {:3d} timesteps.\tTotal reward: {:6.2f}".format(t+1, total_reward))
    
    finally:    
        env.close()
