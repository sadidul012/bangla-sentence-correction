import matplotlib.pyplot as plt
import process
import time

def save_curve(xs, ys, labels, save_file, xlabel="Epoch", ylabel="Loss"):
    plt.clf()

    for i in range(len(labels)):
        plt.plot(xs[i], ys[i], label=labels[i])

    plt.xticks(xs[0])

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()

    plt.savefig("graph/"+save_file)

print("loading stat...")
train_stat = process.load_obj("train_stat")

tl_epoch = train_stat["train"]["epoch"]
vl_epoch = train_stat["valid"]["epoch"]
train_losses = train_stat["train"]["losses"]
validation_losses = train_stat["valid"]["losses"]
a_epoch = train_stat["accuracy"]["epoch"]
accuracies = train_stat["accuracy"]["accuracies"]
times = train_stat["time"]["times"]
t_epoch = train_stat["time"]["epoch"]

print("making graph...")
save_curve([tl_epoch, vl_epoch], [train_losses, validation_losses], ["Train", "Valid"], "error_graph.png")
save_curve([tl_epoch, vl_epoch, a_epoch], [train_losses, validation_losses, accuracies], ["Train", "Valid", "Accuracy"], "accuracy-and-error_graph.png", ylabel="Loss/Accuracy")
save_curve([a_epoch], [accuracies], ["Accuracy"], "accuracy_graph.png", ylabel="Accuracy")

total = 0
cdf = []
yt = []
for t in times:
    cdf.append(total)
    yt.append(time.strftime("%H:%M:%S", time.gmtime(t)))
    total += t

plt.clf()
plt.plot(t_epoch, times, label="Cumulative epoch time")
# plt.plot(t_epoch, times, label="Epoch time")
plt.xticks(t_epoch)
plt.yticks(times, yt)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Time", fontsize=12)
plt.legend()

plt.savefig("graph/time.png")
print("done")