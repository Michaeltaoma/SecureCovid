import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
from preprocess import preprocess
import pandas as pd
import os
import torch
from scipy.interpolate import make_interp_spline, BSpline


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def prepare_name(df_dir):
    """

    :param df_dir: dir for csv
    :return: list of names
    """

    df = pd.read_csv(df_dir, sep=" ", header=None)
    return list(df[1])


def toFig(train_rec, val_rec, saved_path, fig_num, added_name=""):
    epoch = len(train_rec)
    plt.figure(fig_num)
    plt.plot(range(epoch), train_rec, label="Train")
    plt.plot(range(epoch), val_rec, label="Validation")
    plt.title(added_name)
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.legend(loc='upper right')
    plt.savefig(saved_path)


def toFig_smooth(train_rec, val_rec, saved_path, fig_num, metric, added_name=""):
    epoch = len(train_rec)
    plt.figure(fig_num)
    epoch_new = np.linspace(0, epoch - 1, 300)
    smooth_train = make_interp_spline(range(epoch), train_rec, k=5)
    smooth_val = make_interp_spline(range(epoch), val_rec, k=3)
    epoch_new_train = smooth_train(epoch_new)
    epoch_new_val = smooth_val(epoch_new)
    plt.plot(epoch_new, epoch_new_train, label="Train")
    plt.plot(epoch_new, epoch_new_val, label="Validation")
    plt.title(added_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend(loc='upper right')
    plt.savefig(saved_path)


def write_csv(data, name):
    """
    Args:
        data ():
        name ():
    """
    with open(name, 'w') as fout:
        for item in data:
            # print(item)
            fout.write(item)
            fout.write('\n')


def save_model(cpkt_dir, model, optimizer, loss, epoch, name):
    save_path = cpkt_dir
    make_dirs(save_path)

    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'loss': loss}
    name = os.path.join(cpkt_dir, name + '_checkpoint.pth.tar')
    print(name)
    torch.save(state, name)


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def visualize_model(device, model, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_handeled = 0
    ax = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_handeled += 1
                ax = plt.subplot(num_images // 2, 2, images_handeled)
                ax.axis('off')
                ax.set_title('Actual: {} predicted: {}'.format(class_names[labels[j].item()], class_names[preds[j]]))
                imshow(inputs.cpu().data[j], (5, 5))

                if images_handeled == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def imshow(inp, size=(30, 30), title=None):
    """Imshow for Tensor."""
    inp = inp.detach().numpy().transpose((1, 2, 0))
    mean = preprocess.mean_nums
    std = preprocess.std_nums
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=size)
    plt.imshow(inp)
    if title is not None:
        plt.title(title, size=30)
    plt.pause(0.001)  # pause a bit so that plots are updated


def toTxt(s, path):
    with open(path, "a+") as text_file:
        text_file.write(s)


def fromTxt(path):
    with open(path) as file:
        return file.readlines()


def toPickle(obj, path):
    with open(path, 'wb+') as handle:
        pickle.dump(obj, handle)


def fromPickle(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj
