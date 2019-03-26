import argparse
parser = argparse.ArgumentParser( description="Train and save an image classification model.",
        usage="python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_units 3136 --epochs 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_directory', action="store",default='/home/workspace/aipnd-project')
parser.add_argument('--save_dir',action='store',default='/home/workspace/aipnd-project')
parser.add_argument('--arch',action='store',default='vgg16')
parser.add_argument('--learning_rate',action='store',default=0.003)
parser.add_argument('--hidden_size',action='store',default=[4096])
parser.add_argument('--epochs',action='store',default=6)
parser.add_argument('--gpu',action='store',default=True)
args = parser.parse_args()
print(args)