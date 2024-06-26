from Environment import Environment
from Utils import Utils
from DecisionMaking import DecisionMaking
import numpy as np
from Test import Test
from View import plot_tensors, create_video_from_plots
from Object import Object

# def get_random_action(state: list):
#     env_map = np.array(state[0])
#     goal_map = np.zeros_like(env_map[0, :, :])
#
#     all_object_locations = np.stack(np.where(env_map), axis=1)
#     goal_index = np.random.randint(low=0, high=all_object_locations.shape[0], size=())
#     goal_location = all_object_locations[goal_index, 1:]
#
#     # goal_location = all_object_locations[0, 1:] # ERASE THIS!!!!!
#     goal_map[goal_location[0], goal_location[1]] = 1
#     return goal_map


if __name__ == '__main__':
    utils = Utils()
    agent = DecisionMaking(params=utils.params)
    agent.generate_behavior()
    plot_tensors(utils.params.EPISODE_NUM,
                 agent.few_many_array,
                 agent.env_steps_tensor,
                 agent.mental_state_steps_tensor,
                 agent.states_params_steps_tensor,
                 agent.episode_step_num)
    # create_video_from_plots(utils.params)

    # environment = Environment(params=utils.params, few_many_objects=['few', 'few'], object_reappears=False)
    # index = 0
    # each_type_object_num = [4, 1]
    # object_type_num = 2
    # rewards = [[9., 6., 13., 17.], [11.]]
    # object_locations = [[(1, 1), (2, 7), (4, 5), (5, 1)], [(7, 0)]]
    #
    # environment.set_environment(agent_location=np.array([4, 6]), # work on this function and make it useful for debugging
    #                             mental_states=np.array([.83, 12.83]),
    #                             mental_states_slope=np.array([1., 1.]),
    #                             object_locations=object_locations,
    #                             object_rewards=rewards,
    #                             object_coefficients=np.array([[-1., 0.], [0., -1.]]))
    # episodes = 10
    # while True:
    #     environment.object_reappears = False
    #     goal_map = agent.take_action(environment=environment)
    #     print(goal_map)
    #     environment.object_reappears = True
    #     next_state, reward, _, _, _ = environment.step(goal_map=goal_map)
    #
    # view = Test(utils=utils)
    # view.get_goal_directed_actions()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
