from simulator.DroneSimulator import DroneSimulator
import numpy as np

drone_simulator = DroneSimulator(
    bitmap = './test-with-2-levels.bmp',
    batch_size = 1,
    observation_range = 5,
    drone_size = 1,
    amount_of_drones = 3,
    stigmergy_evaporation_speed = np.array([10, 20, 30]),
    stigmergy_colours = np.array([[255,64,0],[255,128,0],[255,255,0]]),
    inertia = 0,
    collision_detection = np.array([True, True]),
    max_steps = 12,
    render_allowed = True
)

# drones_actions : ndarray(shape=(batch_size, amount_of_drones, 2))
# stigmergy _actions(shape=(batch_size, amount_of_drones, 2)

drone_simulator.render()


shape = (1, 3, 2)

actions = np.zeros(shape)
actions[0,0] = [1, 1]
actions[0,1] = [1, 1]
actions[0,2] = [1, 1]

stig_actions = np.ones(shape)

drone_simulator.step(actions, stig_actions)
drone_simulator.render()


for i in range(3):
    drone_simulator.step(actions, stig_actions)
    drone_simulator.render()
    print("---FINE STEP")