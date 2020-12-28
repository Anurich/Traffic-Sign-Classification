import torch
from dataset import read_data
import torch.nn as nn
from tqdm import tqdm
import os
import cv2
import argparse
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self,classes):
        super(model,self).__init__()
        self.no_class = classes
        # we will create  Yann LeCun’s  LeNet Architecture
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=1, padding=0 )
        self.avg_pool = nn.AvgPool2d(kernel_size=(2,2),stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5),stride=1,padding=0)
        self.linear = nn.Linear(120,84)
        self.output = nn.Linear(84,self.no_class)

    def forward(self,x):
        #first block

        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.avg_pool(x)

        #second block
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.avg_pool(x)

        #third block
        x = self.conv3(x)
        x = torch.tanh(x)

        #fourth block
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        output = self.output(x)
        return output


def plotting(val_loss,train_loss,val_acc,train_acc,epoch):
    fig,axs = plt.subplots(2)
    fig.tight_layout()
    axs[0].plot(epoch,val_loss,label='validation_loss')
    axs[0].plot(epoch,train_loss,label='train_loss')
    axs[0].set_title("validation vs training Loss")
    axs[0].legend()
    axs[1].plot(epoch,val_acc,label='validation_acc')
    axs[1].plot(epoch,train_acc,label='train_acc')
    axs[1].set_title("train vs validation Accuracy")
    axs[1].legend()
    plt.show()

def predictionPlotting(prediction,acc_pred):
    count =0
    test_folder = "FullIJCNN2013/"
    fig,ax = plt.subplots(4,8)
    fig.tight_layout()
    for i in range(4):
        for j in range(8):
            pred = prediction[count].item()
            if pred < 10:
                pred = "0"+str(pred)

            fileName = os.listdir(test_folder+""+str(pred))[0]
            img      = cv2.imread(test_folder+"/"+str(pred)+"/"+fileName)
            ax[i,j].imshow(img)
            count+=1
    fig.suptitle("Predicted images with Accuracy of "+str(acc_pred*100))
    plt.show()

def main():
    batch_size=32
    classes = 43
    iteration = 10
    # call the data
    train, test, val = read_data(batch_size)
    net = model(classes)
    #loss
    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    for epoch in tqdm(range(iteration)):
        total_loss = 0.0
        correct  = 0.0
        total  = 0
        for j, data in enumerate(train):
            img, label = data
            net.zero_grad()
            prediction = net(img)
            loss  = criterian(prediction,label.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total += label.size(0)
            _,predicted = torch.max(prediction.data,1)
            correct += (predicted == label).sum().item()
            if j%20 == 0 and j!=0:
                net.eval()
                total_val_loss = 0.0
                val_total = 0
                val_correct = 0
                with torch.no_grad():
                    for data in val:
                        valimg, valLabel= data
                        prediction_val = net(valimg)
                        loss_val       = criterian(prediction_val,valLabel)
                        total_val_loss += loss_val.item()
                        val_total += valLabel.size(0)
                        _,predicted_val = torch.max(prediction_val.data,1)
                        val_correct += (predicted_val == valLabel).sum().item()


                # here we print
                print("training loss after epoch {} is {} and accuracy is {} ".format(str(epoch),str(total_loss/20),str(correct/total)))

                print("validation loss after epoch {} is {} and accuracy is {}  ".format(str(epoch),str(total_val_loss/len(val)),str(val_correct/val_total)))
                if not os.path.isdir("weights/"):
                    os.mkdir("weights/")

                torch.save({
                    "epoch":epoch,
                    "model_state_dict":net.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "training_loss":total_loss/20,
                    "validation_loss":total_val_loss/len(val),
                    "training_accuracy":correct/total,
                    "validation_accuracy":val_correct/val_total
                },"weights/weight_after_step"+str(j)+"_"+str(epoch)+".pth")
                total_loss = 0.0
                correct = 0
                total   = 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Do you want to train/test")
    parser.add_argument("--train",type=str,help="Please enter yes in case you want to train and no if not ")
    args = parser.parse_args()
    if args.train.lower() == "yes":
        main()
    else:
        batch_size=32
        classes = 43
        _, test, _= read_data(batch_size)

        #load the weights
        all_weights = os.listdir("weights/")

        epoch = []
        val_loss =[]
        train_loss=[]
        val_acc  =[]
        train_acc =[]


        for weight in all_weights:
            epoch.append(weight.split("_")[-1].split(".")[0])
            weight_ech_epoch = torch.load("weights/"+weight)
            val_loss.append(weight_ech_epoch["validation_loss"])
            train_loss.append(weight_ech_epoch["training_loss"])
            val_acc.append(weight_ech_epoch["validation_accuracy"])
            train_acc.append(weight_ech_epoch["training_accuracy"])

        epoch = sorted(list(map(int,epoch)))
        plotting(val_loss,train_loss,val_acc,train_acc,epoch)
        # test calculation
        model_weight = torch.load("weights/"+all_weights[-1])
        net  = model(classes)
        net.eval()
        net.load_state_dict(model_weight["model_state_dict"])

        img_test, label_test = next(iter(test))
        test_prediction  = net(img_test)
        _,predicted  = torch.max(test_prediction.data,1)
        correct_test = (predicted == label_test).sum().item()
        predictionPlotting(predicted,correct_test/label_test.size(0))






