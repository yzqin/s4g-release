import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--obj_dir', '-o', type=str)
parser.add_argument('--wrl_dir', '-w', type=str)
parser.add_argument('--exec_path', '-e', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    input_dir = args.obj_dir
    output_dir = args.wrl_dir
    vhacd = args.exec_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in os.listdir(input_dir):
        if not file.startswith('.'):
            name = os.path.splitext(file)[0]
            input = os.path.abspath(os.path.join(input_dir, file))
            output = os.path.abspath(os.path.join(output_dir, name + '.wrl'))
            os.system("{} --input {} --output {} --resolution 1000000 --concavity 0.0001".format(vhacd, input, output))
