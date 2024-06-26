import os.path
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox, TextArea)
import numpy as np
import torch
import cv2


def get_marker(layer: int, reward: str):
    markers = {
        # 1: '■',
        # 2: '▲',
        # 0: 'face',
        1: 'beverage',
        2: 'food'
    }
    if layer == 0:
        return './face.png'
    else:
        return './{0}/{1}.png'.format(markers[layer], reward)


def plot_env(env_map, show_reward=False):
    zoom = {
        0: .1,
        1: .01,
        2: .01
    }

    # Create a figure and axis
    fig, ax = plt.subplots(dpi=1152)

    # Iterate through the grid and plot the shapes
    for i in range(env_map.shape[1]):
        for j in range(env_map.shape[2]):
            layer = torch.where(env_map[:, i, j])[0]
            # print(value)
            if len(layer) > 0:  # and value[0].item() > 0:
                reward = str(env_map[layer[0].item(), i, j].int().item())
                marker = get_marker(layer[0].item(),
                                    reward)
                arr_img = plt.imread(marker, format='png')

                imagebox = OffsetImage(arr_img, zoom=zoom[layer[0].item()])
                imagebox.image.axes = ax

                ab = AnnotationBbox(imagebox, (j + .5, i + .5),
                                    frameon=False
                                    )

                ax.add_artist(ab)

                if layer[0].item() > 0 and show_reward:
                    text_box = TextArea(reward, textprops={'fontsize': 5})
                    ab = AnnotationBbox(text_box, (j + .2, i + .9),
                                        frameon=False)
                    ax.add_artist(ab)

    # Set axis limits and labels
    ax.set_xlim(0, env_map.shape[1])
    ax.set_ylim(0, env_map.shape[2])
    ax.set_xticks(np.arange(env_map.shape[1]))
    ax.set_yticks(np.arange(env_map.shape[2]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_box_aspect(aspect=1)

    # Show the plot
    plt.grid(True)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    # plt.show()
    return fig, ax


def plot_tensors(episode_num, few_many_array, env_tensor, mental_state_tensor,
                 states_params_tensor, step_num_tensor):
    base_dir = './Plots'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    episode_dict = {
        'few few': 0,
        'few many': 0,
        'many few': 0,
        'many many': 0,
    }
    for it1 in ['few', 'many']:
        for it2 in ['few', 'many']:
            path = os.path.join(base_dir, it1 + ' ' + it2)
            if not os.path.exists(path):
                os.mkdir(path)

    for episode in range(int(episode_num)):
        for step in range(int(step_num_tensor[episode])):
            save_dir = os.path.join(base_dir, few_many_array[episode])

            fig, ax = plot_env(env_tensor[episode, step, :, :, :], show_reward=True)
            fig.savefig('{0}/episode_{1}_step{2}_ms({3:.2f}_{4:.2f})_sl({5:.2f}_{6:.2f}).png'.format(save_dir,
                                                                                                     episode_dict[
                                                                                                         few_many_array[
                                                                                                             episode]],
                                                                                                     step,
                                                                                                     mental_state_tensor[
                                                                                                         episode, step, 0],
                                                                                                     mental_state_tensor[
                                                                                                         episode, step, 1],
                                                                                                     states_params_tensor[
                                                                                                         episode, step, 0],
                                                                                                     states_params_tensor[
                                                                                                         episode, step, 1]))

            # fig.savefig('{0}/episode_{1}_step{2}_ms({3:.2f}_{4:.2f})_sl({5:.2f}_{6:.2f})_rw({7:.2f}_{8:.2f}).png'.format(save_dir, episode_dict[few_many_array[episode]], step,
            #                                                                                                              mental_state_tensor[episode, step, 0],
            #                                                                                                              mental_state_tensor[episode, step, 1],
            #                                                                                                              states_params_tensor[episode, step, 0],
            #                                                                                                              states_params_tensor[episode, step, 1],
            #                                                                                                              -states_params_tensor[episode, step, 2],
            #                                                                                                              -states_params_tensor[episode, step, 5]))
            plt.close()
        episode_dict[few_many_array[episode]] += 1


def get_plots_file_info(it1: str, it2: str):
    plots_dir = './Plots'
    all_filenames = os.listdir(os.path.join(plots_dir, it1 + ' ' + it2))
    episode_step = np.array([[int(f.split('_')[1]), int(f.split('_')[2][4:])] for f in all_filenames])
    plots_filenames = []
    if len(episode_step) > 0:
        episode_num = np.unique(episode_step[:, 0]).shape[0]
        for episode in range(episode_num):
            step_num = episode_step[episode_step[:, 0] == episode, :].shape[0]
            episode_plots = []
            for step in range(step_num):
                for name in all_filenames:
                    if name.startswith('episode_{0}_step{1}_'.format(episode, step)):
                        episode_plots.append(name)
            plots_filenames.append(episode_plots)
    return plots_filenames


def create_video_from_plots(params):
    videos_dir = './Videos'
    if not os.path.exists('./Videos'):
        os.mkdir('./Videos')

    plots_dir = './Plots'
    for it1 in ['few', 'many']:
        for it2 in ['few', 'many']:
            few_many_path = os.path.join(videos_dir, it1 + ' ' + it2)
            if not os.path.exists(few_many_path):
                os.mkdir(few_many_path)

            plots_filename = get_plots_file_info(it1, it2)
            episode_num = len(plots_filename)
            for episode in range(episode_num):
                step_num = len(plots_filename[episode])
                frame_num = step_num  # params.SECONDS_PER_FRAME * step_num * 100
                video_array = np.zeros((frame_num, 480, 480, 3), dtype=np.uint8)

                video_writer = cv2.VideoWriter(os.path.join(few_many_path, 'episode_{0}.mp4'.format(episode)),
                                               cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps=1 / params.SECONDS_PER_FRAME, frameSize=(480, 480))
                for step in range(step_num):
                    frame_path = os.path.join(plots_dir, it1 + ' ' + it2, plots_filename[episode][step])
                    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)[:, 80:560, :]
                    # t1 = params.SECONDS_PER_FRAME * step * 100
                    # t2 = params.SECONDS_PER_FRAME * (step+1) * 100
                    video_array[step, :, :, :] = frame

                    # for ms in range(milliseconds):
                    cv2.putText(video_array[step, :, :, :],
                                'step {0}'.format(step).format(),
                                (200, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                thickness=2, color=(0, 0, 0))
                    video_writer.write(video_array[step, :, :, :])

                video_writer.release()
                # for frame in frames:
                #     vidwriter.write(frame)
