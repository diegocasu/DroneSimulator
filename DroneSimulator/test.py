from simulator.DroneSimulator import DroneSimulator
import numpy as np
import time

start_time = time.time()

drone_simulator = DroneSimulator(
    bitmap = './maps/test-with-2-levels.bmp',
    batch_size = 1,
    observation_range = 5,
    drone_size = 0,
    amount_of_drones = 3,
    stigmergy_evaporation_speed = np.array([0, 0, 0]),
    stigmergy_colours = np.array([[255,64,0],[255,128,0],[255,255,0]]),
    inertia = 0,
    collision_detection = np.array([True, False]),
    max_steps = 1000,
    rendering_allowed = True
)

shape = (1, 3, 2)
stig_actions = np.array([[[0,1],[1,2],[2,3]]])
actions = np.array([[[1,0],[1,0],[1,0]]])

for i in range(2):
    obs, rew, done, info = drone_simulator.step(actions, stig_actions)
    drone_simulator.render()
    print("--- %s seconds ---" % (time.time() - start_time))

print("--- %s seconds ---" % (time.time() - start_time))
