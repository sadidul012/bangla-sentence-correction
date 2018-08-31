import json
from random import shuffle, randint

with open("../config.json") as json_file:
    config = json.load(json_file)

report_point = config["report_point"]
dataset = "../"+config["save_dir"]+config["file_name"]


def shuffle_multiple_list(*ls):
    l =list(zip(*ls))

    shuffle(l)
    return zip(*l)


X = []
Y = []
with open(dataset, "r") as file:
    count = 1
    for line in file:
        line = line.strip()
        words = line.split(" ")
        length = len(words)

        if config["max_length"] >= length >= config["min_length"]:
            x = []
            y = []

            perturbation = randint(1, length)
            for _ in range(perturbation):
                incorrect = list(words)
                per_count = 0
                while per_count < config["max_perturbation"] and per_count < length:
                    i = randint(0, length-2)
                    incorrect[i], incorrect[i+1] = incorrect[i+1], incorrect[i]
                    per_count += 1

                for _ in range(randint(config["min_repetition"],config["max_repetition"])):
                    x.append(" ".join(incorrect) + "\n")
                    y.append(" ".join(words) + "\n")

            missing = randint(1, length)
            for _ in range(missing):
                incorrect = list(words)
                per_count = 0
                while per_count < config["max_missing"] and per_count < length:
                    i = randint(0, length - 2)
                    del incorrect[i]
                    incorrect.append("1")
                    per_count += 1

                for _ in range(randint(config["min_repetition"], config["max_repetition"])):
                    x.append(" ".join(incorrect) + "\n")
                    y.append(" ".join(words) + "\n")

            incorrect = []
            correct = []
            auto_comp = list(words)
            correct.append(auto_comp[0])
            for i in range(0, length - 1):
                incorrect.append(auto_comp[i])
                correct.append(auto_comp[i + 1])

                incorrect_full = list(incorrect)
                incorrect_full.append("1")
                for _ in range(randint(config["min_repetition"], config["max_repetition"])):
                    x.append(" ".join(incorrect_full) + "\n")
                    y.append(" ".join(correct) + "\n")

            X.extend(x)
            Y.extend(y)
            count += 1

        if count%report_point == 0:
            print("Perturbing {} done".format(count))


print("shuffling...")
X, Y = shuffle_multiple_list(X, Y)


def save_dataset(path, partition, data_x, data_y):
    incorrect_train_file = path + partition + ".incorrect.txt"
    correct_train_file = path + partition + ".correct.txt"

    file = open(incorrect_train_file, "w")
    file.close()
    file = open(correct_train_file, "w")
    file.close()

    with open(incorrect_train_file, "a") as incorrect, open(correct_train_file, "a") as correct:
        i = 0
        while i < len(data_x):
            incorrect.writelines(data_x[i:i+report_point])
            correct.writelines(data_y[i:i+report_point])

            i += report_point
            print("saved lines {}".format(i))


length = len(X)
start_index = length - int((length * config['dev_set']) / 100)
dev_X = X[start_index:]
X = X[:start_index]
dev_Y = Y[start_index:]
Y = Y[:start_index]

length = len(dev_X)
start_index = length - int((length * config['test_set']) / 100)
test_X = dev_X[start_index:]
dev_X = dev_X[:start_index]
test_Y = dev_Y[start_index:]
dev_Y = dev_Y[:start_index]


save_dataset("../"+config["dataset_path"], "train", X, Y)
save_dataset("../"+config["dataset_path"], "dev", dev_X, dev_Y)
save_dataset("../"+config["dataset_path"], "test", test_X, test_Y)
print("train dataset size {}".format(len(X)))
print("dev dataset size {}".format(len(dev_Y)))
print("test dataset size {}".format(len(test_X)))

print("done")