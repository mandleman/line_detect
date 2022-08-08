import argparse
parser = argparse.ArgumentParser(description='add video file path')

parser.add_argument('--input',type=str,default='car.mp4')

opt = parser.parse_args()

print(opt.input)
