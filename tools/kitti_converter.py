import argparse
import os
import shutil

from jmodt.config import TRAIN_SEQ_ID, VALID_SEQ_ID, SMALL_VAL_SEQ_ID, TEST_SEQ_ID

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--data_root', type=str, default='data/KITTI')
args = parser.parse_args()


def init_or_clear_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for f in os.listdir(path):
            f_path = os.path.join(path, f)
            os.remove(f_path)


def create_train_sample_data(input_root, output_root, init_or_clear_dirs=True, only_labels=False):
    res_training = os.path.join(output_root, 'training')

    res_calib = os.path.join(res_training, 'calib')
    res_image = os.path.join(res_training, 'image_2')
    res_label = os.path.join(res_training, 'label_2')
    res_lidar = os.path.join(res_training, 'velodyne')

    if init_or_clear_dirs:
        init_or_clear_dir(res_calib)
        init_or_clear_dir(res_image)
        init_or_clear_dir(res_label)
        init_or_clear_dir(res_lidar)

    in_training = os.path.join(input_root, 'training')

    sample_id = 0
    used_tid = 0  # start from tid = 1
    tid_dict = {}  # change sparse tid to compact tid for all sequences

    sample_to_real_frame = {}
    seq_to_sample = {}

    for seq in range(21):
        seq = '%04d' % seq
        tracking_image = os.path.join(in_training, 'image_02', seq)
        tracking_lidar = os.path.join(in_training, 'velodyne', seq)
        tracking_calib = os.path.join(in_training, 'calib', f'{seq}.txt')
        tracking_label = os.path.join(in_training, 'label_02', f'{seq}.txt')
        # get number of frames
        lidar_files = os.listdir(tracking_lidar)
        frames = [f.split('.')[0] for f in lidar_files]
        frames.sort()
        print('processing sequence', seq, ', length:', len(frames))

        label_dict = {}
        for frame in frames:
            label_dict[frame] = []
        with open(tracking_label, 'r') as f:
            lines = f.readlines()
            for line in lines:
                split_line = line.strip().split()
                frame_id_int = split_line[0]
                the_frame = frame_id_int.zfill(6)
                if the_frame not in label_dict:
                    continue
                tid = int(split_line[1])
                if tid != -1:
                    if f'{seq}_{tid}' not in tid_dict:
                        used_tid += 1
                        tid_dict[f'{seq}_{tid}'] = used_tid
                        tid = used_tid
                    else:
                        tid = tid_dict[f'{seq}_{tid}']
                obj = ""
                for s in split_line[2:]:
                    obj += s + " "
                obj += str(tid) + "\n"
                label_dict[the_frame].append(obj)

        for frame in frames:
            sample_str = str(sample_id).zfill(6)
            if not only_labels:
                assert os.path.isfile(os.path.join(tracking_image, f'{frame}.png'))
                shutil.copyfile(os.path.join(tracking_image, f'{frame}.png'),
                                os.path.join(res_image, f'{sample_str}.png'))
                assert os.path.isfile(os.path.join(tracking_lidar, f'{frame}.bin'))
                shutil.copyfile(os.path.join(tracking_lidar, f'{frame}.bin'),
                                os.path.join(res_lidar, f'{sample_str}.bin'))
                assert os.path.isfile(tracking_calib)
                shutil.copyfile(tracking_calib, os.path.join(res_calib, f'{sample_str}.txt'))
            with open(os.path.join(res_label, f'{sample_str}.txt'), 'w+') as f:
                f.writelines(label_dict[frame])
            sample_to_real_frame[sample_str] = (seq, frame)
            if seq in seq_to_sample.keys():
                seq_to_sample[seq].append(sample_str)
            else:
                seq_to_sample[seq] = [sample_str]
            sample_id += 1

    with open(os.path.join(res_training, 'sample2frame.txt'), 'w+') as f:
        for cur_sample_id in range(sample_id):
            cur_sample_id_str = str(cur_sample_id).zfill(6)
            cur_seq, cur_frame = sample_to_real_frame[cur_sample_id_str]
            f.write(f'{cur_sample_id_str} {cur_seq} {cur_frame}\n')

    with open(os.path.join(res_training, 'seq2sample.txt'), 'w+') as f:
        for seq in range(21):
            seq = '%04d' % seq
            f.write(f'{seq} ')
            sample_list = seq_to_sample[seq]
            for sample in sample_list:
                f.write(sample + ' ')
            f.write('\n')
    print(sample_id, used_tid)

    split_dir = os.path.join(output_root, 'ImageSets')
    os.makedirs(split_dir, exist_ok=True)

    training_seq_to_sample = {}
    with open(os.path.join(output_root, 'training', 'seq2sample.txt'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_split = line.split()
        training_seq_to_sample[line_split[0]] = line_split[1:]

    with open(os.path.join(split_dir, 'train.txt'), 'w+') as f:
        for seq in TRAIN_SEQ_ID:
            sample_list = training_seq_to_sample[seq]
            for i in range(len(sample_list) - 1):
                f.write(f'{sample_list[i]} {sample_list[i + 1]}\n')

    with open(os.path.join(split_dir, 'val.txt'), 'w+') as f:
        for seq in VALID_SEQ_ID:
            sample_list = training_seq_to_sample[seq]
            for sample in sample_list:
                f.write(f'{sample}\n')

    with open(os.path.join(split_dir, 'small_val.txt'), 'w+') as f:
        for seq in SMALL_VAL_SEQ_ID:
            sample_list = training_seq_to_sample[seq]
            for i in range(len(sample_list) - 1):
                f.write(f'{sample_list[i]} {sample_list[i + 1]}\n')


def create_test_sample_data(input_root, output_root, init_or_clear_dirs=True):
    out_test_dir = os.path.join(output_root, 'testing')

    res_calib = os.path.join(out_test_dir, 'calib')
    res_image = os.path.join(out_test_dir, 'image_2')
    res_lidar = os.path.join(out_test_dir, 'velodyne')

    if init_or_clear_dirs:
        init_or_clear_dir(res_calib)
        init_or_clear_dir(res_image)
        init_or_clear_dir(res_lidar)

    in_test_dir = os.path.join(input_root, 'testing')
    sample_id = 0
    sample_to_real_frame = {}
    seq_to_sample = {}

    for seq in TEST_SEQ_ID:
        tracking_image = os.path.join(in_test_dir, 'image_02', seq)
        tracking_lidar = os.path.join(in_test_dir, 'velodyne', seq)
        tracking_calib = os.path.join(in_test_dir, 'calib', f'{seq}.txt')
        # get number of frames
        lidar_files = os.listdir(tracking_lidar)
        frames = [f.split('.')[0] for f in lidar_files]
        frames.sort()
        print('processing sequence', seq, ', length:', len(frames))

        for frame in frames:
            sample_str = str(sample_id).zfill(6)
            assert os.path.isfile(os.path.join(tracking_image, f'{frame}.png'))
            shutil.copyfile(os.path.join(tracking_image, f'{frame}.png'),
                            os.path.join(res_image, f'{sample_str}.png'))
            assert os.path.isfile(os.path.join(tracking_lidar, f'{frame}.bin'))
            shutil.copyfile(os.path.join(tracking_lidar, f'{frame}.bin'),
                            os.path.join(res_lidar, f'{sample_str}.bin'))
            assert os.path.isfile(tracking_calib)
            shutil.copyfile(tracking_calib, os.path.join(res_calib, f'{sample_str}.txt'))
            sample_to_real_frame[sample_str] = (seq, frame)
            if seq in seq_to_sample.keys():
                seq_to_sample[seq].append(sample_str)
            else:
                seq_to_sample[seq] = [sample_str]
            sample_id += 1

    with open(os.path.join(out_test_dir, 'sample2frame.txt'), 'w+') as f:
        for cur_sample_id in range(sample_id):
            cur_sample_id_str = str(cur_sample_id).zfill(6)
            cur_seq, cur_frame = sample_to_real_frame[cur_sample_id_str]
            f.write(f'{cur_sample_id_str} {cur_seq} {cur_frame}\n')

    with open(os.path.join(out_test_dir, 'seq2sample.txt'), 'w+') as f:
        for seq in range(29):
            seq = '%04d' % seq
            f.write(f'{seq} ')
            sample_list = seq_to_sample[seq]
            for sample in sample_list:
                f.write(sample + ' ')
            f.write('\n')

    split_dir = os.path.join(output_root, 'ImageSets')
    os.makedirs(split_dir, exist_ok=True)

    testing_seq_to_sample = {}
    with open(os.path.join(output_root, 'testing', 'seq2sample.txt'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_split = line.split()
        testing_seq_to_sample[line_split[0]] = line_split[1:]

    with open(os.path.join(split_dir, 'test.txt'), 'w+') as f:
        for seq in TEST_SEQ_ID:
            sample_list = testing_seq_to_sample[seq]
            for sample in sample_list:
                f.write(f'{sample}\n')


if __name__ == '__main__':
    in_dir = os.path.join(args.data_root, 'tracking')
    out_dir = os.path.join(args.data_root, 'tracking_object')
    create_train_sample_data(input_root=in_dir, output_root=out_dir, init_or_clear_dirs=True, only_labels=False)
    create_test_sample_data(input_root=in_dir, output_root=out_dir, init_or_clear_dirs=True)
