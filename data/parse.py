import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle


def parse_table(df, shadow_size=100):
    all_pos = df.loc[df[2] == "positive"]
    all_pos = shuffle(all_pos)
    all_neg = df.loc[df[2] == "negative"]
    all_neg = shuffle(all_neg)

    res = {}
    all_pos_shadow = all_pos[:shadow_size]
    all_pos_target = all_pos[shadow_size:]

    all_neg_shadow = all_neg[:shadow_size]
    all_neg_target = all_neg[shadow_size:]

    res["target"] = {}
    res["target"]["pos"] = all_pos_target
    res["target"]["neg"] = all_neg_target

    res["shadow"] = {}
    res["shadow"]["pos"] = all_pos_shadow
    res["shadow"]["neg"] = all_neg_shadow

    return res


# train_data_root = Path(
#     "/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/data/cxr3/train.txt")
# test_data_root = Path(
#     "/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/data/cxr3/test.txt")
#
# csv_path = Path("/Users/michaelma/desktop/workspace/School/ubc/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/data/train_test")

# total_shadow_sizes = 500
# df = pd.read_csv(train_data_root, sep=" ", header=None)
# train_res = parse_table(df, total_shadow_sizes)
#
# for model in ["target", "shadow"]:
#     for classes in ["pos", "neg"]:
#         cur_csv_path = csv_path.joinpath("train_{}_{}_{}.csv".format(model, classes, total_shadow_sizes))
#         train_res[model][classes].to_csv(cur_csv_path, sep=' ', index=False)

# total_shadow_sizes = 200
# df = pd.read_csv(test_data_root, sep=" ", header=None)
# test_res = parse_table(df, total_shadow_sizes)
#
# for model in ["target", "shadow"]:
#     for classes in ["pos", "neg"]:
#         cur_csv_path = csv_path.joinpath("test_{}_{}_{}.csv".format(model, classes, total_shadow_sizes))
#         test_res[model][classes].to_csv(cur_csv_path, sep=" ", index=False)
