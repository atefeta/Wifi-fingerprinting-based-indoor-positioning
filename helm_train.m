function [TrAc,TestAc,Tr_time,Test_time,L_est_tst,L_tst_mat,...
    test_orgL,xx_tr] = helm_train(train_x,train_y,test_x,test_y,w1,w2,w3,s,C,bis)

tic
q=.001;
iter=50;
train_x = zscore(train_x')';
H1 = [train_x bis * ones(size(train_x,1),1)];%bb=0.1*ones(size(train_x,1),1);
clear train_x;
%% First layer RELM
A1 = H1 * w1;
A1 = mapminmax(A1);
clear b1;
beta1  =  sparse_elm_autoencoder(A1,H1,q,iter)';
clear A1;
T1 = H1 * beta1;
% fprintf(1,'Layer 1: Max Val of Output %f Min Val %f\n',max(T1(:)),min(T1(:)));
[T1,ps1]  =  mapminmax(T1',0,1);
T1 = T1';
clear H1;
%% Second layer RELM
H2 = [T1 bis * ones(size(T1,1),1)];
clear T1;

A2 = H2 * w2;
A2 = mapminmax(A2);
clear b2;
beta2 = sparse_elm_autoencoder(A2,H2,q,iter)';
clear A2;

T2 = H2 * beta2;
% fprintf(1,'Layer 2: Max Val of Output %f Min Val %f\n',max(T2(:)),min(T2(:)));

[T2,ps2] = mapminmax(T2',0,1);
T2 = T2';

clear H2;
%% Original ELM
H3 = [T2 bis * ones(size(T2,1),1)];
clear T2;

T3 = H3 * w3;
l3 = max(max(T3));
l3 = s/l3;
% fprintf(1,'Layer 3: Max Val of Output %f Min Val %f\n',l3,min(T3(:)));
%l3 =1;
T3 = logsig(T3 * l3); % logsig(n) = 1 / (1 + exp(-n))
clear H3;
%% Finsh Training
beta = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3'  *  train_y);
Tr_time = toc;
% disp('Training has been finished!');
% disp(['The Total Training Time is : ', num2str(Tr_time), ' seconds' ]);
%% Calculate the training accuracy
xx_tr = T3 * beta;
clear T3;

yy = result_tra(xx_tr);
train_yy = result_tra(train_y);
TrAc = length(find(yy == train_yy))/size(train_yy,1);
% disp(['Training Accuracy is : ', num2str(TrAc * 100), ' %' ]);
%% First layer feedforward
tic;
test_x = zscore(test_x')';
HH1 = [test_x bis * ones(size(test_x,1),1)];
clear test_x;
TT1 = HH1 * beta1;
TT1  =  mapminmax('apply',TT1',ps1)';
clear HH1;clear beta1;
%% Second layer feedforward
HH2 = [TT1 bis * ones(size(TT1,1),1)];
clear TT1;
TT2  =  HH2 * beta2;
TT2  =  mapminmax('apply',TT2',ps2)';
clear HH2;clear beta2;
%% Last layer feedforward
HH3 = [TT2 bis * ones(size(TT2,1),1)];
clear TT2;
TT3 = logsig(HH3 * w3 * l3);
clear HH3;clear b3;
L_tst_mat = TT3 * beta;%x(:,2)=test_y;
L_est_tst = result_tra(L_tst_mat);
test_orgL = result_tra(test_y);
TestAc = length(find(L_est_tst == test_orgL))/size(test_orgL,1);
clear TT3;
%% Calculate the testing accuracy
Test_time = toc;
% disp('Testing has been finished!');
% disp(['The Total Testing Time is : ', num2str(Test_time), ' seconds' ]);
% disp(['Testing Accuracy is : ', num2str(TestAc * 100), ' %' ]);
