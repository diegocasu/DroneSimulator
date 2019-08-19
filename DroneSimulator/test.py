from simulator.DroneSimulator import DroneSimulator
import numpy as np
import time

start_time = time.time()

drone_simulator = DroneSimulator(
    bitmap = './maps/test-with-2-levels.bmp',
    batch_size = 3000,
    observation_range = 5,
    drone_size = 1,
    amount_of_drones = 12,
    stigmergy_evaporation_speed = np.array([10, 20, 30]),
    stigmergy_colours = np.array([[255,64,0],[255,128,0],[255,255,0]]),
    inertia = 0.4,
    collision_detection = np.array([True, True]),
    max_steps = 100000,
    render_allowed = False
)

shape = (3000, 12, 2)
actions = np.ones(shape)
stig_actions = np.ones(shape)

for i in range(1000):
    drone_simulator.render()
    obs, rew, done, info = drone_simulator.step(actions, stig_actions)
    print("--- %s seconds ---" % (time.time() - start_time))

print("--- %s seconds ---" % (time.time() - start_time))