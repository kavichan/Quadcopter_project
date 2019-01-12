import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def plot_single_multiple_run(results):
    
    plt.subplots(figsize=(15,15))
    plt.subplot(3,3,1)
    plt.title('Position of Quadcopter')
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.title("Position (z --> 0)")
    plt.xlabel('Time, seconds')
    plt.ylabel('Position')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,3,2)
    plt.title('Velocity of Quadcopter')
    plt.plot(results['time'], results['x_velocity'], label='x_hat')
    plt.plot(results['time'], results['y_velocity'], label='y_hat')
    plt.plot(results['time'], results['z_velocity'], label='z_hat')
    plt.title("Velocities (|v| --> small)")
    plt.xlabel('Time, seconds')
    plt.ylabel('Velocity')
    plt.grid(True)
    plt.legend()

    plt.subplot(3,3,3)
    plt.title('Euler angles')
    plt.plot(results['time'], normalize_angle(results['phi']), label='phi')
    plt.plot(results['time'], normalize_angle(results['theta']), label='theta')
    plt.plot(results['time'], normalize_angle(results['psi']), label='psi')
    plt.title("Orientation ")
    plt.xlabel('Time, seconds')
    plt.grid(True)
    plt.legend()

    plt.subplot(3,3,4)
    plt.title('Angular Velocity')
    plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    plt.xlabel('Time, seconds')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 3, 5)
    plt.title('Rotor Speed')
    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4')
    plt.xlabel('Time, seconds')
    plt.ylabel('Rotor Speed, revolutions / second')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.title('Reward')
    plt.plot(results['time'], results['reward'], label='Reward')
    plt.xlabel('Time, seconds')
    plt.ylabel('Reward')
    plt.show()
    return plt

def plot_lastdata(results,numofepisode):

    plt.figure(figsize=(15,10))
    plt.subplot(2,2,1)
    plt.plot(results['episode'], results['reward'])
    plt.xlabel('No of episodes')
    plt.ylabel('Rewards ')
    plt.xlim(xmin=0,xmax=np.max(results['episode'])+2)
    plt.title("Rewards per episode")

    plt.subplot(2,2,2)
    plt.plot(results['episode'][-numofepisode:], results['reward'][-numofepisode:])
    plt.xlabel('No of episodes')
    plt.ylabel('Rewards ')
    plt.xlim(xmin=np.max(results['episode'])-20,xmax=np.max(results['episode'])+2)
    plt.ylim(ymin=0,ymax=np.max(results['reward'])+2)
    plt.title("Reward for last 10 episodes")
    plt.show()

    return plt

def normalize_angle(angles):
    # Adjust angles to range -pi to pi
    norm_angles = np.copy(angles)
    for i in range(len(norm_angles)):
        while norm_angles[i] > np.pi:
            norm_angles[i] -= 2 * np.pi
    return norm_angles
