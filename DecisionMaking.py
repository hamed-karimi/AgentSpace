import os.path
from copy import deepcopy
import operator
import torch
from Environment import Environment, get_distance_between_locations
import numpy as np
import math
from torch import nn


def init_empty_tensor(size: tuple):
    return torch.empty(size, dtype=torch.float64)


class DecisionMaking:
    def __init__(self, params):
        self.few_many_array = None
        self.episode_step_num = None
        self.env_steps_tensor = None
        self.states_params_steps_tensor = None
        self.mental_state_steps_tensor = None
        self.states_params_tensor = None
        self.mental_state_tensor = None
        self.env_tensor = None
        self.params = params
        self.horizon = self.params.TIME_HORIZON # self.params.OBJECT_HORIZON
        if self.params.DEVICE == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.params.DEVICE
        self.sigmoid = nn.Sigmoid()
        self.gamma = self.params.GAMMA
        self.init_data_tensors()
        self.use_estimated_transition = self.params.USE_ESTIMATED_TRANSITION
        self.min_object_time_init = 13
        self.min_object_time = self.min_object_time_init

    def imagine(self, environment: Environment,
                horizon,
                grabbed_goals,
                cum_reward,
                n_object_grabbed,
                time,
                n_rewarding_object,
                only_agent_location=False) -> dict:
        # We should count the number of staying steps to avoid infinite recursions
        object_locations, agent_location = environment.get_possible_goal_locations()
        goal_returns = dict()

        # if horizon >= self.horizon and n_rewarding_object > 0:  # at least one rewarding object for updating min time
        #     self.min_object_time = min(self.min_object_time, time)
        if horizon >= self.horizon: # and n_rewarding_object > 0:  # at least one rewarding object
            # print(time, grabbed_goals, end=' ')
            # print(cum_reward)
            return goal_returns
        # elif horizon >= self.horizon and time >= self.min_object_time:  # no objects, only staying
        #     print(time, grabbed_goals, end=' ')
        #     print(cum_reward)
        #     return goal_returns

        state = environment.get_observation()
        env_map = torch.Tensor(state[0]).unsqueeze(0)

        staying_goal = []
        if only_agent_location:
            all_goal_locations = deepcopy(agent_location)
        elif not np.all(object_locations == agent_location, axis=1).any():
            all_goal_locations = np.concatenate([object_locations, agent_location], axis=0)
            staying_goal.append(agent_location[0])
        else:
            all_goal_locations = deepcopy(object_locations)
        for obj in all_goal_locations:
            imagined_environment = deepcopy(environment)
            goal_map = torch.zeros_like(env_map[:, 0, :, :])
            goal_map[0, obj[0], obj[1]] = 1

            next_obs, pred_reward, _, _, info = imagined_environment.step(goal_map=goal_map.squeeze().numpy())
            next_mental_state = next_obs[1]

            imagined_environment.set_mental_state(next_mental_state)
            new_grabbed = deepcopy(grabbed_goals)
            new_grabbed.append(obj.tolist())
            future_goal_returns = self.imagine(environment=imagined_environment, horizon=horizon+info['dt'], #1,
                                               grabbed_goals=new_grabbed, cum_reward=self.gamma * pred_reward + cum_reward,
                                               n_object_grabbed=n_object_grabbed + int(info['object']),
                                               time=time + info['dt'],
                                               n_rewarding_object=n_rewarding_object + int(info['rewarding']),
                                               only_agent_location=only_agent_location)

            future_returns = 0 if not future_goal_returns else max(future_goal_returns.items(),
                                                                   key=operator.itemgetter(1))[1]
            goal_returns[tuple(obj)] = self.gamma**info['dt'] * (pred_reward + future_returns)

        return goal_returns

    def get_goal_return(self, environment: Environment) -> dict:
        # print('all objects: ', environment.get_possible_goal_locations())
        goal_returns = self.imagine(environment=deepcopy(environment),
                                    horizon=0,
                                    grabbed_goals=[],
                                    cum_reward=0,
                                    n_object_grabbed=0,
                                    time=0,
                                    n_rewarding_object=0)
        return goal_returns

    def take_action(self, environment: Environment):
        self.min_object_time = self.min_object_time_init
        goal_returns = self.get_goal_return(environment)
        best_goal_location = max(goal_returns.items(), key=operator.itemgetter(1))[0]
        goal_map = np.zeros((self.params.HEIGHT, self.params.WIDTH))
        goal_map[best_goal_location[0], best_goal_location[1]] = 1
        return goal_map

    def generate_behavior(self):
        for episode in range(int(self.params.EPISODE_NUM)):
            print(episode)
            few_many = [np.random.choice(['few', 'many']) for _ in range(self.params.OBJECT_TYPE_NUM)]
            self.few_many_array[episode] = ' '.join(few_many)
            environment = Environment(params=self.params, few_many_objects=few_many)
            state, _ = environment.reset()
            env_dict = environment.get_env_dict()
            for step in range(int(self.params.EPISODE_STEPS)):
                print(step, end=' ')
                environment.object_reappears = False
                goal_map = self.take_action(environment)
                environment.object_reappears = True
                next_state, reward, _, _, _ = environment.step(goal_map=goal_map)

                self.add_data_point(state, env_dict, episode, step)
                state = deepcopy(next_state)
                env_dict = environment.get_env_dict()
            print()
        self.save_tensors()

    # def load_transition_model(self) -> TransitionNet:
    #     transition_weights = torch.load(os.path.join(self.params.TRANSITION_MODEL, 'model.pt'),
    #                                     map_location=self.device)
    #     transition = TransitionNet(self.params, device=self.device)
    #     transition.load_state_dict(transition_weights)
    #     return transition

    def add_data_point(self, state: list, env_dict: dict, episode, step):
        env_map = torch.Tensor(state[0])
        mental_states = torch.Tensor(state[1])
        states_params = torch.Tensor(np.concatenate([state[2], state[3].flatten()]))
        for loc in env_dict.keys():
            env_map[loc[0]+1, loc[1], loc[2]] = deepcopy(env_dict[loc].reward)

        self.env_tensor[episode, step, :, :, :] = env_map
        self.mental_state_tensor[episode, step, :] = mental_states
        self.states_params_tensor[episode, step, :] = states_params

    def init_data_tensors(self):
        self.env_tensor = init_empty_tensor(size=(self.params.EPISODE_NUM,
                                                  self.params.EPISODE_STEPS,
                                                  self.params.OBJECT_TYPE_NUM + 1,
                                                  self.params.HEIGHT,
                                                  self.params.WIDTH))
        self.mental_state_tensor = init_empty_tensor((self.params.EPISODE_NUM,
                                                      self.params.EPISODE_STEPS,
                                                      self.params.OBJECT_TYPE_NUM))
        self.states_params_tensor = init_empty_tensor((self.params.EPISODE_NUM,
                                                       self.params.EPISODE_STEPS,
                                                       self.params.OBJECT_TYPE_NUM * 2 + self.params.OBJECT_TYPE_NUM))

        self.env_steps_tensor = init_empty_tensor((self.params.EPISODE_NUM,
                                                   self.params.EPISODE_ACTION_STEPS,
                                                   # EPISODE_ACTION_STEPS has to be larger than EPISODE_STEPS
                                                   self.params.OBJECT_TYPE_NUM + 1,
                                                   self.params.HEIGHT,
                                                   self.params.WIDTH))
        self.mental_state_steps_tensor = init_empty_tensor((self.params.EPISODE_NUM,
                                                            self.params.EPISODE_ACTION_STEPS,
                                                            self.params.OBJECT_TYPE_NUM))
        self.states_params_steps_tensor = init_empty_tensor((self.params.EPISODE_NUM,
                                                             self.params.EPISODE_ACTION_STEPS,
                                                             self.params.OBJECT_TYPE_NUM * 2 + self.params.OBJECT_TYPE_NUM))
        self.episode_step_num = torch.zeros((self.params.EPISODE_NUM, 1))
        self.few_many_array = np.zeros((self.params.EPISODE_NUM, ), dtype='<U9')

    def save_tensors(self):
        data_dir = './Data'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        for episode in range(int(self.params.EPISODE_NUM)):
            # state_0_ind = 0
            action_step = 0
            for step in range(int(self.params.EPISODE_STEPS) - 1):
                self.env_steps_tensor[episode, action_step, :, :, :] = self.env_tensor[episode, step, :, :, :].clone()
                self.mental_state_steps_tensor[episode, action_step, :] = self.mental_state_tensor[episode, step, :].clone()
                self.states_params_steps_tensor[episode, action_step, :] = self.states_params_tensor[episode, step, :].clone()
                action_step += 1
                for env_map, mental_state in self.next_step_state(episode=episode, state_0_ind=step,
                                                                  state_1_ind=step + 1):
                    self.env_steps_tensor[episode, action_step, :, :, :] = env_map
                    self.mental_state_steps_tensor[episode, action_step, :] = mental_state
                    self.states_params_steps_tensor[episode, action_step, :] = self.states_params_tensor[episode, step, :]
                    action_step += 1
            self.episode_step_num[episode] = action_step

        torch.save(self.env_tensor, os.path.join(data_dir, 'environments.pt'))
        torch.save(self.mental_state_tensor, os.path.join(data_dir, 'mental_states.pt'))
        torch.save(self.states_params_tensor, os.path.join(data_dir, 'states_params.pt'))

        torch.save(self.env_steps_tensor, os.path.join(data_dir, 'environments_steps.pt'))
        torch.save(self.mental_state_steps_tensor, os.path.join(data_dir, 'mental_states_steps.pt'))
        torch.save(self.states_params_steps_tensor, os.path.join(data_dir, 'states_params_steps.pt')) # [slope1, slope2, object coeffs (4)]

    def next_step_state(self, episode, state_0_ind, state_1_ind):
        env_map_0 = self.env_tensor[episode, state_0_ind, :, :, :]
        env_map_1 = self.env_tensor[episode, state_1_ind, :, :, :]
        mental_state = self.mental_state_tensor[episode, state_0_ind, :].clone()
        location_0 = torch.argwhere(env_map_0[0, :, :])
        location_1 = torch.argwhere(env_map_1[0, :, :])
        diagonal_steps, straight_steps = get_distance_between_locations(location_0[0, 0],
                                                                        location_0[0, 1],
                                                                        location_1[0, 0],
                                                                        location_1[0, 1])
        d = location_1 - location_0
        diagonal_path = torch.tensor(np.tile(np.sign([d[0, 0], d[0, 1]]), reps=[diagonal_steps, 1]))
        for diag_step in diagonal_path:
            d -= diag_step
        straight_path = torch.tensor(np.tile(np.sign([d[0, 0], d[0, 1]]), reps=[straight_steps, 1]))
        all_steps = torch.cat([diagonal_path, straight_path], dim=0)
        for step in range(all_steps.shape[0] - 1):
            location_0 = location_0 + all_steps[step, :]
            next_env_map = torch.zeros_like(env_map_0)
            next_env_map[0, location_0[0, 0], location_0[0, 1]] = 1
            next_env_map[1:, :, :] = env_map_0[1:, :, :]
            mental_state += torch.linalg.vector_norm(all_steps[step, :].float()) * self.states_params_tensor[episode,
                                                                                   state_0_ind, :2]
            yield next_env_map, mental_state


def check_dict(state, env_dict):
    env_map = torch.Tensor(state[0])
    all_obj = torch.argwhere(env_map[1:, :, :])
    for obj in all_obj:
        if tuple(obj.numpy()) not in env_dict.keys():
            print(obj, 'not in dict')
            return True
    return False

