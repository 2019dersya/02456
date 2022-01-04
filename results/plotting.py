import numpy as np
import matplotlib.pyplot as plt
import csv
import regex as re

def plot_train_gpu(file_path="training_loss_gpu.txt"):
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    print(len(lines))

    steps = []; losses = []; vnums=[]
    for i in range(len(lines)):
        vnum = re.findall(r"v\_num=(\d*)",lines[i])
        step = re.findall(r"\|\s+(\d*)\/\d+",lines[i])
        loss = re.findall(r"loss=([\w\.\-]*)",lines[i])
        if vnum and step and loss:
            vnums.append(int(vnum[0]))
            steps.append(int(vnum[0])*16602+float(step[0]))
            l = loss[0].split("e")
            if len(l)==1:
                l=float(l[0])
            elif len(l)==2:
                l=float(l[0])*10**float(l[1])
            losses.append(l)

    print("steps:", len(steps))
    print("losses:",len(losses))

    max_vnum = max(vnums)
    end_steps_vnum = {} # end of steps per epoch
    i=0
    for v in range(max_vnum+1):
        while vnums[i]==v and i<len(vnums)-1:
            i+=1
        end_steps_vnum[v]=int(steps[i])
        i+=1

    plt.scatter(steps,losses, label="Training loss", s=0.001, alpha=0.75)
    #plt.plot([0, steps[-1]+500],[losses[-1],losses[-1]], c='green', linestyle='dashed')
    #plt.annotate(str(losses[-1]),[-1.5,losses[-1]-0.15],c='green')
    print(str(losses[-1]))
    
    plt.plot([0]+[s for s in end_steps_vnum.values()],[0]+[0]*len(end_steps_vnum.values()), '-o', c='purple', linewidth=0.25, markersize=1, label="Epochs")

    plt.title("Training loss during 24 epochs, on 16602 training examples")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    return
#plot_train_gpu("training_loss_gpu.txt")

def plot_avg_train_gpu(file_path="training_loss_gpu.txt"):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    max_vnum = int(re.findall(r"v\_num=(\d*)",lines[-1])[0])

    total_losses = {v:0 for v in range(0, max_vnum+1)}; nb_losses={v:0 for v in range(0, max_vnum+1)}
    for i in range(len(lines)):
        vnum = re.findall(r"v\_num=(\d*)",lines[i])
        loss = re.findall(r"loss=([\w\.\-]*)",lines[i])
        if vnum and loss:
            v = int(vnum[0])
            l = loss[0].split("e")
            if len(l)==1:
                l=float(l[0])
            elif len(l)==2:
                l=float(l[0])*10**float(l[1])
            total_losses[v]+=l
            nb_losses[v]+=1

    avg_losses={}
    for v in total_losses.keys():
        avg_losses[v]=total_losses[v]/nb_losses[v]
    plt.plot(list(avg_losses.keys()), list(avg_losses.values()), '-o', linewidth=2, markersize=2, label="Average training loss")
    plt.plot([0, max_vnum],[avg_losses[max_vnum],avg_losses[max_vnum]], c='green', linestyle='dashed')
    plt.annotate(str(avg_losses[max_vnum])[:5],[0,avg_losses[max_vnum]+0.01],c='green')
    plt.title("Average training loss during 24 epochs, on 16602 training examples")
    plt.xlabel("Epoch")
    plt.xticks(list(avg_losses.keys()))
    #plt.ylim(bottom=0)
    plt.ylabel("Average Loss")
    plt.legend()
    plt.show()
    return avg_losses
#plot_avg_train_gpu()

def plot_validation(file_path="validation_loss_gpu.txt"):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    epochs = []; losses = []
    for i in range(len(lines)):
        epoch = re.findall(r"v_num=(\d*)",lines[i])
        loss = re.findall(r"loss=([\w\.\-]*)",lines[i])
        if epoch and loss:
            epochs.append(epoch[0])
            l = loss[0].split("e")
            if len(l)==1:
                l=float(l[0])
            elif len(l)==2:
                l=float(l[0])*10**float(l[1])
            losses.append(l)
    plt.plot(epochs, losses, 'o-', label="Validation loss", markersize=5)
    plt.plot([0, epochs[-1]],[losses[-1],losses[-1]], c='green', linestyle='dashed')
    plt.annotate(str(losses[-1]),[0,losses[-1]+0.01],c='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation loss after each epoch of training")
    plt.legend()
    plt.show()
    return epochs, losses
#plot_validation()

all_metrics = ["AP_IoU_050_095_all", "AR_IoU_050_095_all"]
def plot_evaluation(file_path="evaluation_iou_gpu.txt", metrics=all_metrics):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    eval_nums={}
    for i in range(0,len(lines),17): # 16 lines per evaluation
        eval_num = int(re.findall(r"Running evaluation (\d*)...",lines[i])[0])
        AP_IoU_050_095_all = float(re.findall(r"\] = ([\w\.\d]*)\n",lines[i+4])[0])
        AR_IoU_050_095_all = float(re.findall(r"\] = ([\w\.\d]*)\n",lines[i+12])[0])
        eval_nums[eval_num]={"AP_IoU_050_095_all":AP_IoU_050_095_all, "AR_IoU_050_095_all":AR_IoU_050_095_all}
    for metric in metrics:
        plt.plot([eval_num for eval_num in eval_nums.keys()], [eval_nums[eval_num][metric] for eval_num in eval_nums.keys()], 'o-', label=metric)
    plt.xlabel("Number of training epochs before evaluation")
    plt.ylabel("Evaluation")
    plt.title("Evaluation metrics after n epochs of training on GPU")
    plt.legend(loc='lower right')
    plt.show()
    return
#plot_evaluation()

def plot_train_val():
    avg_train_losses = plot_avg_train_gpu()
    epochs, val_losses = plot_validation()
    plt.plot(list(avg_train_losses.keys()), list(avg_train_losses.values()), '-o', linewidth=2, markersize=2, label="Average training loss")
    plt.plot(epochs, val_losses, 'o-', label="Validation loss", markersize=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.show()
    return
#plot_train_val()
