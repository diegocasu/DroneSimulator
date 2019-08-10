from simulator.DroneSimulator import DroneSimulator
import numpy as np

drone_simulator = DroneSimulator(
    bitmap = './test-with-2-levels.bmp',
    batch_size = 1,
    observation_range = 2,
    drone_size = 1,
    amount_of_drones = 3,
    stigmation_evaporation_speed = np.array([1, 2, 3]),
    inertia = 0.3,
    collision_detection = np.array([True, True]),
    max_steps = 6,
    render_allowed = True
)

drone_simulator.render()

# first batch dimension, drone_dimension, action_dimension
#drone_simulator.step(np.ones((1, 1, 4)))