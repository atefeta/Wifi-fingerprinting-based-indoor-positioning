function [tr,tst]=preprocessing_elm(Train_all,Test_all,Ap_num,pwd)
% (1) delition of ap column that dont heared
%Train_all=Tra_row;Test_all=Tst_row;Ap_num=ap_num;

min_tr=min(min(Train_all(:,1:Ap_num)));
max_tr=max(max(Train_all(:,1:Ap_num)));
min_ts=min(min(Test_all(:,1:Ap_num)));
max_ts=max(max(Test_all(:,1:Ap_num)));
ther=min(min_tr,min_ts);
% ther=-110;
idxtst=find(Test_all(:,1:Ap_num)==100);% non heared
Test_all(idxtst)=ther;
idxtr_r=find(Train_all(:,1:Ap_num)==100);
Train_all(idxtr_r)=ther;
% All+thr  all rssi between 0 and 1
Train_all(:,1:Ap_num)=Train_all(:,1:Ap_num)-ther;
Test_all(:,1:Ap_num)=Test_all(:,1:Ap_num)-ther;
idnan=isnan(Train_all);ID_nan=find(idnan==1);
%% train normalization  ----------------------------------------------------
ss=Train_all(:,1:Ap_num)-min(Train_all(:,1:Ap_num),[],2);
maxmi=max(Train_all(:,1:Ap_num),[],2)-min(Train_all(:,1:Ap_num),[],2);
id_zer=find(maxmi==0);
Train_all(:,1:Ap_num)=(ss./maxmi).^pwd;
%% test normalization   ----------------------------------------------------
ss=Test_all(:,1:Ap_num)-min(Test_all(:,1:Ap_num),[],2);
maxmi=max(Test_all(:,1:Ap_num),[],2)-min(Test_all(:,1:Ap_num),[],2);
Test_all(:,1:Ap_num)=(ss./maxmi).^pwd;
tst=Test_all;
tr=Train_all;
end