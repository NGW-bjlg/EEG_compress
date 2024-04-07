import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from FAE import AE2
from data_set import EEG
import numpy as np
import math
from scipy.stats import pearsonr

batchsz = 32
lr = 1e-3
epochs = 10

device = torch.device('cuda')
torch.manual_seed(1234)


# viz = visdom.Visdom()


def mse(y, y_pred):
    sq_error = (y-y_pred)**2
    sum_sq_error = torch.sum(sq_error)
    MSE = sum_sq_error / y.size(3) / y.size(0) / y.size(2)
    return MSE


def prd(y,y_pred):
    MSE = mse(y, y_pred)
    sum_y2 = torch.sum(y**2) / y.size(3) / y.size(0) / y.size(2)
    PRD = math.sqrt(MSE / sum_y2)
    return PRD * 100


def pcc(y,y_pred):
    x = torch.squeeze(y, 1)
    x_pred = torch.squeeze(y_pred, 1)
    PCC = 0
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            y1 = x[i:i+1, j:j+1, :]
            y2 = torch.squeeze(y1)
            y_pred1 = x_pred[i:i + 1, j:j+1, :]
            y_pred2 = torch.squeeze(y_pred1)
            y3 = y2.cpu().numpy()
            y_pred3 = y_pred2.cpu().numpy()
            pc = pearsonr(y3, y_pred3)
            PCC += pc[0]
    PCC = PCC/x.size(0)/x.size(1)
    return PCC


def psnr(y,y_pred):
    MSE = mse(y,y_pred)
    y = torch.abs(y)
    max_y = torch.max(y)
    PSNR = 20*math.log10(max_y / math.sqrt(MSE))
    return PSNR


def evalute(model, loader):
    model.eval()
    result_mse = 0
    result_prd = 0
    result_pcc = 0
    result_psnr = 0
    total = len(loader.dataset) / loader.batch_size

    for x, y in loader:
        x = x.float()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x_pred = model(x)
            x_mse = mse(x, x_pred)
            x_mse = x_mse.cpu().numpy()
            x_prd = prd(x, x_pred)
            x_pcc = pcc(x, x_pred)
            x_psnr = psnr(x, x_pred)
        result_mse += x_mse
        result_prd += x_prd
        result_pcc += x_pcc
        result_psnr += x_psnr
    total_mse = result_mse / total
    total_prd = result_prd / total
    total_pcc = result_pcc / total
    total_psnr = result_psnr / total

    return total_mse, total_prd, total_pcc, total_psnr

def main():

    for subject_num in range(10):
        f = open('result_FAE.txt', 'a')
        num = subject_num + 1
        subject = 'subject' + str(num)
        train_db = EEG(subject, mode='train')
        test_db = EEG(subject, mode='test')
        train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                                  num_workers=4)
        test_loader = DataLoader(test_db, batch_size=1)

        device = torch.device('cuda')
        model = AE2().to(device)
        criteon = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3)
        global_step = 0
        best_loss = 2
        for epoch in range(600):
            model.train()
            los = 0
            for batchidx, (x, _) in enumerate(train_loader):
                x = x.float()
                x = x.to(device)

                x_hat = model(x)
                loss = criteon(x_hat, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                los += loss
            los = los / batchidx
            scheduler.step()
            global_step += 1

            if epoch % 10 == 0:
                print(epoch, 'loss:', los.item())
                if los < best_loss:
                    best_loss = los
                    total_mse, total_prd, total_pcc, total_psnr = evalute(model, test_loader)
                    print('total_mse:', total_mse, 'total_prd:', total_prd, 'total_pcc:', total_pcc, 'total_psnr:', total_psnr)
        print(subject, 'total_mse:', total_mse, 'total_prd:', total_prd, 'total_pcc:', total_pcc, 'total_psnr:', total_psnr)
        result = subject + ' total_mse:' + str(total_mse) + ' total_prd:' + str(total_prd) + ' total_pcc:' + str(total_pcc) + ' total_psnr:'+str(total_psnr)+'\n'
        f.write(result)
        f.close()


if __name__ == '__main__':
    main()