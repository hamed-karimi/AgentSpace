import os
import torch
import numpy as np
import itertools

from DecisionMaking import DecisionMaking
from Environment import Environment
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)


def get_predefined_parameters(params, param_name):
    if param_name == 'all_mental_states':
        all_param = [[-10, -5, 0, 5, 10]] * params.OBJECT_TYPE_NUM
    elif param_name == 'all_object_rewards':
        # all_param = [[0, 4, 8, 12, 16, 20]] * num_object
        param_range = params.ENVIRONMENT_OBJECT_COEFFICIENT_RANGE
        all_param = np.expand_dims(np.linspace(param_range[0],
                                               param_range[1], num=min(param_range[1] - param_range[0] + 1, 4),
                                               dtype=int), axis=0).tolist() * params.OBJECT_TYPE_NUM
    elif param_name == 'all_mental_states_change':
        # all_param = [[0, 1, 2, 3, 4, 5]] * num_object
        param_range = params.MENTAL_STATES_SLOPE_RANGE
        all_param = np.expand_dims(np.linspace(param_range[0],
                                               param_range[1],
                                               num=min(param_range[1] - param_range[0] + 1, 4), dtype=int),
                                   axis=0).tolist() * params.OBJECT_TYPE_NUM
    else:
        print('no such parameters')
        return
    num_param = len(all_param[0]) ** params.OBJECT_TYPE_NUM
    param_batch = []
    for i, ns in enumerate(itertools.product(*all_param)):
        param_batch.append(list(ns))
    return param_batch


def make_rest_folder():
    res_folder = './Test'
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    return res_folder


class Test:
    def __init__(self, utils):
        self.params = utils.params
        self.res_folder = make_rest_folder()
        self.agent = DecisionMaking(params=self.params)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = self.params.HEIGHT
        self.width = self.params.WIDTH
        self.object_type_num = self.params.OBJECT_TYPE_NUM

        self.all_mental_states = get_predefined_parameters(self.params, 'all_mental_states')
        self.all_object_rewards = get_predefined_parameters(self.params, 'all_object_rewards')
        self.all_mental_states_change = get_predefined_parameters(self.params, 'all_mental_states_change')

        self.color_options = [[1, 0, .2], [0, .8, .2], [0, 0, 0]]
        self.goal_shape_options = ['*', 's', 'P', 'o', 'D', 'X']
        self.objects_color_name = ['red', 'green', 'black']  # 2: stay

    def get_figure_title(self, mental_states):
        title = '$n_{0}: {1:.2f}'.format('{' + self.objects_color_name[0] + '}', mental_states[0])
        for i in range(1, self.object_type_num):
            title += ", n_{0}: {1:.2f}$".format('{' + self.objects_color_name[i] + '}', mental_states[i])
        return title

    def get_object_shape_dictionary(self, object_locations, agent_location, each_type_object_num):
        shape_map = dict()
        st = 0
        for obj_type in range(self.object_type_num):
            at_type_object_num = each_type_object_num[obj_type]
            at_type_object_locations = object_locations[st:at_type_object_num + st, :]
            st = at_type_object_num
            for at_obj in range(len(at_type_object_locations)):
                key = tuple(at_type_object_locations[at_obj, :].tolist())
                shape_map[key] = self.goal_shape_options[at_obj]
        key = tuple(agent_location)
        shape_map[key] = '.'
        return shape_map

    def get_goal_location_from_goal_map(self, goal_map):
        goal_location = np.argwhere(goal_map)[0]
        return goal_location

    def next_environment(self):
        for view_id in range(15):
            few_many = [np.random.choice(['few', 'few']) for _ in range(self.params.OBJECT_TYPE_NUM)]
            environment = Environment(self.params, few_many_objects=few_many, object_reappears=False)
            environment.reset()
            state = environment.get_observation()
            for subplot_id, mental_state in enumerate(self.all_mental_states):
                for i in range(self.height):
                    for j in range(self.width):
                        environment.set_environment(agent_location=[i, j],
                                                    mental_states=mental_state,
                                                    mental_states_slope=state[2],
                                                    object_coefficients=state[3])
                        yield environment, subplot_id

    def get_goal_directed_actions(self):
        fig, ax = None, None
        row_num = 5
        col_num = 5
        for setting_id, output in enumerate(self.next_environment()):
            environment = output[0]
            subplot_id = output[1]
            object_locations, agent_location = environment.get_possible_goal_locations()
            agent_location = agent_location.squeeze()
            state = environment.get_observation()
            each_type_object_num = environment.each_type_object_num
            env_map = torch.Tensor(state[0]).unsqueeze(0)
            mental_states = state[1]
            mental_states_slope = state[2]
            object_coefficients = state[3]

            if setting_id % (col_num * row_num * self.width * self.height) == 0:
                fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))

            r = subplot_id // col_num
            c = subplot_id % col_num

            shape_map = self.get_object_shape_dictionary(object_locations, agent_location, each_type_object_num)

            proposed_goal_map = self.agent.take_action(environment=environment)
            goal_location = np.argwhere(proposed_goal_map)[0]

            if tuple(goal_location.tolist()) in shape_map.keys():
                selected_goal_shape = shape_map[tuple(goal_location.tolist())]
                goal_type = torch.where(env_map[0, :, goal_location[0], goal_location[1]])[0].min().item()
            else:
                selected_goal_shape = '_'
                goal_type = 0

            goal_type = 2 if goal_type == 0 else goal_type - 1
            size = 10 if selected_goal_shape == '.' else 50
            ax[r, c].scatter(agent_location[1], agent_location[0],
                             marker=selected_goal_shape,
                             s=size,
                             alpha=0.4,
                             facecolor=self.color_options[goal_type])

            if agent_location[0] == self.height - 1 and agent_location[1] == self.width - 1:
                ax[r, c].set_title(self.get_figure_title(mental_states.squeeze()), fontsize=10)

                st = 0
                for obj_type in range(self.object_type_num):
                    at_type_object_num = each_type_object_num[obj_type]
                    at_type_object_locations = object_locations[st:at_type_object_num + st, :]
                    st = at_type_object_num
                    for obj in range(each_type_object_num[obj_type]):
                        ax[r, c].scatter(at_type_object_locations[obj, :][1],
                                         at_type_object_locations[obj, :][0],
                                         marker=self.goal_shape_options[obj],
                                         s=200,
                                         edgecolor=self.color_options[obj_type],
                                         facecolor='none')
                ax[r, c].tick_params(length=0)
                ax[r, c].set(adjustable='box')
            if (setting_id + 1) % (col_num * row_num * self.width * self.height) == 0 or (setting_id + 1) % (
                    self.width * self.height) == 0:
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].invert_yaxis()
                plt.tight_layout(pad=0.1, w_pad=6, h_pad=1)

                fig.savefig('{0}/slope_{1:.2f}-{2:.2f}_coef_{3:.2f}-{4:.2f}-{5:.2f}-{6:.2f}.png'.format(self.res_folder,
                                                                                                        mental_states_slope[0],
                                                                                                        mental_states_slope[1],
                                                                                                        object_coefficients[0, 0],
                                                                                                        object_coefficients[0, 1],
                                                                                                        object_coefficients[1, 0],
                                                                                                        object_coefficients[1, 1]
                                                                                                        ))
                plt.close()

