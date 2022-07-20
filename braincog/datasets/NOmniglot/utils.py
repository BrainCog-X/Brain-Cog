import torch
import threading
import numpy as np
import pandas
import os
from dv import AedatFile


class FunctionThread(threading.Thread):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.f(*self.args, **self.kwargs)


def integrate_events_to_frames(events, height, width, frames_num=10, data_type='event'):
    frames = np.zeros(shape=[frames_num, 2, height * width])

    # create j_{l}和j_{r}
    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)

    # split by time
    events['t'] -= events['t'][0]       # start with 0 timestamp
    assert events['t'][-1] > frames_num
    dt = events['t'][-1] // frames_num  # get length of each frame
    idx = np.arange(events['t'].size)
    for i in range(frames_num):
        t_l = dt * i
        t_r = t_l + dt
        mask = np.logical_and(events['t'] >= t_l, events['t'] < t_r)
        idx_masked = idx[mask]
        if len(idx_masked) == 0:
            j_l[i] = -1
            j_r[i] = -1
        else:
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1 if i < frames_num - 1 else events['t'].size

    for i in range(frames_num):
        if j_l[i] >= 0:
            x = events['x'][j_l[i]:j_r[i]]
            y = events['y'][j_l[i]:j_r[i]]
            p = events['p'][j_l[i]:j_r[i]]
            mask = []
            mask.append(p == 0)
            mask.append(np.logical_not(mask[0]))
            for j in range(2):
                position = y[mask[j]] * width + x[mask[j]]
                events_number_per_pos = np.bincount(position)
                frames[i][j][np.arange(events_number_per_pos.size)] += events_number_per_pos

        if data_type == 'frequency':
            if i < frames_num - 1:
                frames[i] /= dt
            else:
                frames[i] /= (dt + events['t'][-1] % frames_num)
        frames = frames.astype(np.float16)

    if data_type == 'event':
        frames = (frames > 0).astype(np.bool)
    else:
        frames = normalize_frame(frames, 'max')
    return frames.reshape((frames_num, 2, height, width))


def normalize_frame(frames: np.ndarray or torch.Tensor, normalization: str):
    eps = 1e-5
    for i in range(frames.shape[0]):
        if normalization == 'max':
            frames[i][0] = frames[i][0] / max(frames[i][0].max(), eps)
            frames[i][1] = frames[i][1] / max(frames[i][1].max(), eps)

        elif normalization == 'norm':
            frames[i][0] = (frames[i][0] - frames[i][0].mean()) / np.sqrt(max(frames[i][0].var(), eps))
            frames[i][1] = (frames[i][1] - frames[i][1].mean()) / np.sqrt(max(frames[i][1].var(), eps))

        elif normalization == 'sum':
            frames[i][0] = frames[i][0] / max(frames[i][0].sum(), eps)
            frames[i][1] = frames[i][1] / max(frames[i][1].sum(), eps)

        else:
            raise NotImplementedError
    return frames


def convert_events_dir_to_frames_dir(events_data_dir, frames_data_dir, suffix,
                                     frames_num=12, result_type='event', thread_num=1,
                                     compress=True):
    """
    Iterate through all event data in eventS_date_DIR and generate frame data files in frames_data_DIR
    """
    def read_function(file_name):
        return np.load(file_name, allow_pickle=True).item()

    def cvt_fun(events_file_list):
        for events_file in events_file_list:
            print(events_file)
            frames = integrate_events_to_frames(read_function(events_file), 260, 346, frames_num, result_type  )
            if compress:
                frames_file = os.path.join(frames_data_dir,
                                           os.path.basename(events_file)[0: -suffix.__len__()] + '.npz')
                np.savez_compressed(frames_file, frames)
            else:
                frames_file = os.path.join(frames_data_dir,
                                           os.path.basename(events_file)[0: -suffix.__len__()] + '.npy')
                np.save(frames_file, frames)

    # Obtain the path of the all files
    events_file_list = list_all_files(events_data_dir, '.npy')

    if thread_num == 1:
        cvt_fun(events_file_list)
    else:
        # Multithreading acceleration
        thread_list = []
        block = events_file_list.__len__() // thread_num
        for i in range(thread_num - 1):
            thread_list.append(FunctionThread(cvt_fun, events_file_list[i * block: (i + 1) * block]))
            thread_list[-1].start()
            print(f'thread {i} start, processing files index: {i * block} : {(i + 1) * block}.')
        thread_list.append(FunctionThread(cvt_fun, events_file_list[(thread_num - 1) * block:]))
        thread_list[-1].start()
        print(
            f'thread {thread_num} start, processing files index: {(thread_num - 1) * block} : {events_file_list.__len__()}.')
        for i in range(thread_num):
            thread_list[i].join()
            print(f'thread {i} finished.')


def convert_aedat4_dir_to_events_dir(root, train):
    kind = 'background' if train else "evaluation"
    originroot = root
    root = root + '/dvs_' + kind + '/'
    alphabet_names = [a for a in os.listdir(root) if a[0] != '.']  # get folder names

    for a in range(len(alphabet_names)):
        alpha_name = alphabet_names[a]

        for b in range(len(os.listdir(os.path.join(root, alpha_name)))):
            character_id = b + 1
            character_path = alpha_name + '/character' + num2str(character_id)
            print('Parsing %s \\ character%s ...' % (alpha_name, num2str(character_id)))

            file_path = os.path.join(root, character_path)
            aedat4_name = [a for a in os.listdir(file_path) if a[-4:] == 'dat4' and len(a) == 11][0]
            csv_name = [a for a in os.listdir(file_path) if a[-4:] == '.csv' and len(a) == 8][0]
            number = csv_name[:4]
            new_path = originroot + '/events_npy/' + kind + '/' + alpha_name + '/character' + num2str(character_id)
            if not os.path.exists(new_path):
                os.makedirs(new_path)

            start_end_timestamp = pandas.read_csv(os.path.join(file_path, csv_name)).values

            a_timestamp, a_polarity, a_x, a_y = [], [], [], []
            with AedatFile(os.path.join(file_path, aedat4_name)) as f:  # read aedat4
                for e in f['events']:
                    a_timestamp.append(e.timestamp)
                    a_polarity.append(e.polarity)
                    a_x.append(e.x)
                    a_y.append(e.y)

            for ii in range(20):  # each file has 20 samples
                name = str(number) + '_' + num2str(ii + 1) + '.npy'
                start_index = a_timestamp.index(start_end_timestamp[ii][1])
                end_index = a_timestamp.index(start_end_timestamp[ii][2])
                tmp = {'t': np.array(a_timestamp[start_index:end_index]),
                       'x': np.array(a_x[start_index:end_index]),
                       'y': np.array(a_y[start_index:end_index]),
                       'p': np.array(a_polarity[start_index:end_index])}
                np.save(os.path.join(new_path, name), tmp)


def num2str(idx):
    if idx < 10:
        return '0' + str(idx)
    return str(idx)


def list_all_files(root, suffix, getlen=False):
    '''
    List the path of all files under root, output a list
    '''
    file_list = []
    alphabet_names = [a for a in os.listdir(root) if a[0] != '.']  # get folder names
    idx = 0
    for a in range(len(alphabet_names)):
        alpha_name = alphabet_names[a]
        for b in range(len(os.listdir(os.path.join(root, alpha_name)))):
            character_id = b + 1
            character_path = os.path.join(root, alpha_name, 'character' + num2str(character_id))
            idx += 1
            for c in range(len(os.listdir(character_path))):
                fn_example = os.listdir(character_path)[c]
                if fn_example[-4:] == suffix:
                    file_list.append(os.path.join(character_path, fn_example))
    if getlen:
        return file_list, idx
    else:
        return file_list


def list_class_files(root, frames_kind_root, getlen=False, use_npz=False):
    '''
    index the generated samples,
    get dictionaries according to categories, each corresponding to a list,
    the list contain the address of the new file in fnum_x_dtype_x_npz_True
    '''
    file_list = {}
    alphabet_names = [a for a in os.listdir(root) if a[0] != '.']  # get folder names
    idx = 0
    for a in range(len(alphabet_names)):
        alpha_name = alphabet_names[a]
        for b in range(len(os.listdir(os.path.join(root, alpha_name)))):
            character_id = b + 1
            character_path = os.path.join(root, alpha_name, 'character' + num2str(character_id))
            file_list[idx] = []
            for c in range(len(os.listdir(character_path))):
                fn_example = os.listdir(character_path)[c]
                if use_npz:
                    fn_example = fn_example[:-1] + 'z'
                file_list[idx].append(os.path.join(frames_kind_root, fn_example))
            idx += 1
    if getlen:
        return file_list, idx
    else:
        return file_list
