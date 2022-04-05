
import sys

sys.path.append('/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid')
sys.path.append('/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/model')
sys.path.append('/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/preprocess')

import argparse
import time

import torch
from model.attack import AttackModel
from model.model_manager import load_model
from preprocess import preprocess
from util import printProgressBar
import logging

parser = argparse.ArgumentParser(description='Secure Covid Demo')
parser.add_argument('--image_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/demo/img/IM-0122-0001.jpeg',
                    type=str,
                    help='Path to load the image')
parser.add_argument('--label', default='no_covid', type=str, help='True label of the image')
parser.add_argument('--inside', default=False, type=bool, help='Is the data in the training data?')
parser.add_argument('--target_weight_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/temp/target/train_best_shadow_1649025547.7971604.pth',
                    type=str,
                    help='Path to load the trained model')
parser.add_argument('--covid_attack_weight_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/temp/five/covid_attack_1649110011.8203351.pth',
                    type=str,
                    help='Path to load the trained model')
parser.add_argument('--no_covid_attack_weight_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/temp/five/no_covid_attack_1649110470.614971.pth',
                    type=str,
                    help='Path to load the trained model')
args = parser.parse_args()


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
target = load_model(device, "covidnet", [])
target.load_state_dict(torch.load(args.target_weight_path, map_location=torch.device('cpu')))
target.eval()
img = preprocess.load_single_image(device, args.image_path)
res = target(img)
with torch.no_grad():
    y_test_pred = torch.log_softmax(res, dim=1)
    _, y_pred_tag = torch.max(y_test_pred, dim=1)
label = y_pred_tag[0]
true_label = ["no_covid", "covid"].index(args.label)
print()
# print("Running Target Model...")

items = list(range(0, 10))
l = len(items)
# Initial call to print 0% progress
printProgressBar(0, l, prefix = 'Running Target Model:', suffix = 'Complete', length = 50)
for i, item in enumerate(items):
    time.sleep(0.5)
    # Update Progress Bar
    printProgressBar(i + 1, l, prefix = 'Running Target Model:', suffix = 'Complete', length = 50)


print("Predicted Label: {}, True Label: {}".format(["Negative", "Positive"][label], ["Negative", "Positive"][true_label]))
print()

items = list(range(0, 5))
l = len(items)

printProgressBar(0, l, prefix = 'Running Attack Model:', suffix = 'Complete', length = 50)
for i, item in enumerate(items):
    time.sleep(0.5)
    # Update Progress Bar
    printProgressBar(i + 1, l, prefix = 'Running Attack Model:', suffix = 'Complete', length = 50)


attack = AttackModel(2, 64, 1)

if args.label.__eq__("no_covid"):
    attack.load_state_dict(torch.load(args.no_covid_attack_weight_path, map_location=torch.device('cpu')))
else:
    attack.load_state_dict(torch.load(args.covid_attack_weight_path, map_location=torch.device('cpu')))

attack = attack.to(device)

attack.eval()

y_test_pred.unsqueeze(0)

with torch.no_grad():
    attack_pred = attack(y_test_pred)
    y_pred_tag = torch.round(torch.sigmoid(attack_pred))
    attack_label = int(y_pred_tag[0])

print("{} Attack Model predicted the instance is {} the training data".format(["Negative", "Positive"][true_label], ["out of", "in"][attack_label]))
print("And the data is actually {} of the training data".format(["out of", "in"][int(args.inside)]))


