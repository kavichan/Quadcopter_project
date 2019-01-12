import numpy as np
from physics_sim import PhysicsSim

class Landing():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.runtime = runtime
        
        # Goal
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # ideally zero velocity
        self.last_timestamp = 0
        self.last_position = np.array([0.0, 0.0, 0.0]) 
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
    
    
    def istargetzone(self):
        """Checks whether the copter has landed in target zone or not"""
        flag = False
        cntr=0
        position = self.sim.pose[:3] 
        
        #Set upper bound and lower bound for target zone
        target_bounds = 40 
        lower_bounds = np.array([-target_bounds / 2, -target_bounds / 2, 0])
        upper_bounds = np.array([ target_bounds / 2, target_bounds / 2, target_bounds])
        
        #Set boundary conditions
        lower_pos = (self.target_pos + lower_bounds)
        upper_pos = (self.target_pos + upper_bounds)
        
                  
        #Check whether the copter has landed with the boundaries of target zone
        for j in range(3):  
            
            #Check for the boundary conditions
            if (lower_pos[j] <= position[j] and position[j] < upper_pos[j]):
                cntr = cntr + 1 
                
        #Check if all 3 conditions have been satisfied
        if cntr==3:
            flag = True
            
        return flag
    
    def get_reward(self):
        """Uses current pose of sim to return reward."""
            
        #Calculate distance between current position and target position
        distance = np.linalg.norm((self.sim.pose[:3] - self.target_pos))
        distance_max = np.linalg.norm(self.sim.upper_bounds)                                  
           
        #Calculate velocity
        velocity = np.linalg.norm((self.sim.v - self.target_velocity))
        
        # Calculate distance factor and velocity factor
        distance_factor = 1 / max(distance,0.1)
        vel_discount = (1.0 - max(velocity,0.1) ** (distance_factor))

        reward=0
        
        # Penalize agent running out of time
        if self.sim.time >= self.runtime:  
            reward = -10.0 
            self.sim.done=True          
        else :            
            # Agent has touched the ground surface (i.e. z=0)
            if (self.sim.pose[2] == self.target_pos[2]):                 
               
                # If velocity is less than the specified threshold
                # it implies that the agent has landed successfulyy
                if (self.sim.v[2]<=1): 
                    
                    if (self.istargetzone()==True):
                        #Landed safely. Give bonus points for landing in the target zone 
                        landing_reward= 100.0
                        print('Agent has landed in the target zone')
                        
                    else: 
                        reward =-100.0    #Landed outside target zone   
                        print('outside')
                       
                else:    
                    #Penalize agent for crashing
                    reward=-100 # Crashed          
                    self.sim.done=True
                    
            else:
                if(np.isnan(self.sim.v[2])==False):
                    # Depending upon the distance of the copter from the target position a normal penalty has been applied
                    distance_reward =   0.2 - (distance/distance_max)**0.1       
                    reward = vel_discount * distance_reward 
                else:
                    #Penalize agent for crashing
                    reward=-100 # Crashed          
                    self.sim.done=True
                    
        #Apply tanh to avoid instability in training due to exploding gradients            
        reward = np.tanh(reward)
        
        return reward

    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0       
       
        pose_all = []
        for _ in range(self.action_repeat):
            #print(rotor_speeds)
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
           
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.last_timestamp = 0
        self.last_position = np.array([0.0, 0.0, 0.0]) 
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
