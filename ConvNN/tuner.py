import random
import os
import subprocess

target_dir = './tuning_20190309/'
target_file = target_dir + 'summary.txt'
default_parameters = {
    'gpu': 'y',
    'epochs': 500,
    'mb_size': 64,
    'v_filter': 1,
    'h_filter': 3,
    'd_filter': None,
    'u_filter': '8 8 16 16 32 32 64 64',
    'forecast': 12,
    'history': 24,
    'offset': 6,
}


def append_to_file(filename, items):
    with open(filename, 'a+') as file:
        for item in items:
            file.write(item + '\n')


def generate_random_params(params):
    random.seed()

    binary = [2 ** num for num in range(1, 9)]
    odds = [num for num in range(40) if num % 2 == 1]
    hours = [num for num in range(73) if num % 3 == 0]

    i = random.randrange(len(binary))
    params['mb_size'] = binary[i]

    # only 1, 3, 5
    i = random.randrange(0, 3)
    params['v_filter'] = odds[i]

    # only 1, 3, ... , 27
    i = random.randrange(odds.index(1), odds.index(19))
    params['h_filter'] = odds[i]

    # forecast 8 - 24
    params['forecast'] = random.randrange(8, 25)

    # history 18 - 36
    params['history'] = random.randrange(18, 37)

    # offset 1 - 12, doing 9 for now
    params['offset'] = random.randrange(1, 9)

    # filter down needs to be less than 2h + f down to 16
    length = random.randrange(0, 3)
    lim = 2 * params['history'] + params['forecast']
    if length == 1:
        params['d_filter'] = '32'
    elif length == 2:
        params['d_filter'] = '32 16'

    # filter up needs to be from binary[8, 8, 16, 16, ...]
    # all dups = [num for num in binary[2:8] for _ in (0, 1)]
    i = random.randrange(3, 9)
    dups = [num for num in binary[2:i] for _ in (0, 1)]
    u_filter = ''
    for i in dups:
        u_filter = u_filter + str(i) + ' '
    params['u_filter'] = u_filter.strip()

    return params


def build_arg_array(params):
    arg_s = 'python ConvNN/train.py '
    arg_a = arg_s.strip().split(' ')
    for key, val in params.items():
        if val is None:
            continue
        arg_s += '--' + key + ' ' + str(val) + ' '
        arg_a.append('--' + key)
        arg_a.append(str(val))

    return arg_s, arg_a


if __name__ == '__main__':

    parameters = default_parameters

    arg_string, arg_array = build_arg_array(parameters)
    
    # summary = open('./summary.txt', 'w')
    for i in range(1000):
        # Fill in meta details
        run_id = str(i + 1).zfill(4)
        expmt = "EXPERIMENT " + run_id
        expmt_file = target_dir + run_id + '.expmt'

        # generate random params
        if i != 0:
            parameters = generate_random_params(parameters)
        parameters['expmt'] = run_id

        arg_string, arg_array = build_arg_array(parameters)

        detail = [expmt, expmt_file, str(parameters), arg_string]
        append_to_file(target_file, detail)

        print(expmt, end="\t")
        with open(expmt_file, 'w') as f:
            result = subprocess.run(arg_array, stdout=f, stderr=f)
        print('--> ', result.returncode)

        append_to_file(target_file, ['returncode:\t' + str(result.returncode), '------------------'])
