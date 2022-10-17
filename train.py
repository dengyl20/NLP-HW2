import data_loader
import torch
import model
from config import Config
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设置随机种子
torch.manual_seed(0)
config = Config()
RNNmodel = model.Lstm().to(device)
optimizer = torch.optim.Adam(RNNmodel.parameters(), lr=Config.LR)
loss_func = torch.nn.CrossEntropyLoss() # 分类问题
# 定义学习率衰减点，训练到50%和75%时学习率缩小为原来的1/10
mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                             milestones=[config.EPOCH//2, config.EPOCH//4*3], gamma=0.1)

train_loss = []
valid_loss = []
min_valid_loss = np.inf
for i in range(config.EPOCH):
    total_train_loss = []    
    RNNmodel.train()   # 进入训练模式
    for step, (b_x, b_y) in enumerate(train_loader): #TODO
#         lr = set_lr(optimizer, i, EPOCH, LR)
        b_x = b_x.type(torch.FloatTensor).to(device)   # 
        b_y = b_y.type(torch.long).to(device)   # CrossEntropy的target是longtensor，且要是1-D，不是one hot编码形式
        prediction = RNNmodel(b_x) # RNNmodel output    # prediction (4, 72, 2)
#         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
#         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
        loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0]))         # 计算损失，target要转1-D，注意b_y不是one hot编码形式
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients
        total_train_loss.append(loss.item())
    train_loss.append(np.mean(total_train_loss )) # 存入平均交叉熵
  
    
    total_valid_loss = [] 
    RNNmodel.eval()
    for step, (b_x, b_y) in enumerate(valid_loader):
        b_x = b_x.type(torch.FloatTensor).to(device) 
        b_y = b_y.type(torch.long).to(device) 
        with torch.no_grad():
            prediction = RNNmodel(b_x) # RNNmodel output
#         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
#         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
        loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0]))         # calculate loss        
        total_valid_loss.append(loss.item())        
    valid_loss.append(np.mean(total_valid_loss))
    
    if (valid_loss[-1] < min_valid_loss):      
        torch.save({'epoch': i, 'model': RNNmodel, 'train_loss': train_loss,
                'valid_loss': valid_loss},'./LSTM.model') # 保存字典对象，里面'model'的value是模型
#         torch.save(optimizer, './LSTM.optim')     # 保存优化器      
        min_valid_loss = valid_loss[-1]
        
    # 编写日志
    log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCH,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      min_valid_loss,
                                                                      optimizer.param_groups[0]['lr'])
    mult_step_scheduler.step()  # 学习率更新
    # 服务器一般用的世界时，需要加8个小时，可以视情况把加8小时去掉
    print(str(datetime.datetime.now()+datetime.timedelta(hours=8)) + ': ')
    print(log_string)    # 打印日志
    log('./LSTM.log', log_string)   # 保存日志
