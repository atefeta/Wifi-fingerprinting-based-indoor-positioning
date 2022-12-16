function result= HELM_2L_All(RSS_all,Test_RSS,RSS_Mean,center,class_id,Loc_cor,ap_heard,...
        mean_id,K_scor_id_cls_f,Simil,Flag_set,Elm_Type,...
        cls_alg,cand_l,ActivationFunction,k_cand,N1,N2,N3,s,C,clas_center_floor,biass)
    
    
% RSS_all=RSS_ALL;
% Test_RSS=Tst_RSS;
% RSS_Mean=RSS_mean;

    
for cl=1:size(center,1)
    ff=find(class_id==cl);
    Loc_cor(ff,5)=center(cl,1);% 5: center id
    clear ff
end
%% ******** -------------------- All Train Labeling -----------------------
for jj=1:size(Loc_cor,1)
    idd=find(RSS_all(:,ap_heard+4)==jj);
    RSS_all(idd,ap_heard+5)=class_id(jj);      % RSS_all(:,ap_heard+5)== id of Affinity clustering
    clear idd
end
%% ******** -------------------- Test Labeling ----------------------------
[prob_class ,hard_class]=Test_Fuzzy_Label(Test_RSS(:,ap_heard+1:end),Loc_cor,cls_alg,cand_l);
class_unique_test=unique(hard_class);
for gg=1:size(prob_class,2)
    [~ ,ll]=max(prob_class{1,gg}(:,2));
    Class_prob_max(gg)=prob_class{1,gg}(ll,1);clear ll
end
Tarclus_tst_org=hard_class;
%Tar_tst_org= Class_prob_max;
%% ******** -------------------- Multi Label construction -----------------
cls_numAff=size(center,1);
train_x=RSS_all(:,1:ap_heard);     % Ns * Nf > Ns==sample number of train data, Nf, feature number
train_y=-ones(size(RSS_all,1),size(center,1)+4); % Ns * Class
[funiq, ~ ,jjm]=unique(RSS_all(:,ap_heard+3));
for i=1:size(RSS_all,1)
    train_y(i,RSS_all(i,ap_heard+5))=1;
    train_y(i,jjm(i)+cls_numAff)=1;
end
test_x=Test_RSS(:,1:ap_heard);       % Nt * Nf > Nt==sample number of test data
test_y=-ones(size(Test_RSS,1),size(center,1)+4);
[funiq, ~ ,Floor_tst]=unique(Test_RSS(:,ap_heard+3));
for i2=1:size(Test_RSS,1)
    test_y(i2,Tarclus_tst_org(1,i2))=1;
    test_y(i2,Floor_tst(i2)+cls_numAff)=1;
end             % Nt * Class
%Tar_tst= Class_prob_max;
clear Tar_tst
Tar=train_y';%RSS_all(:,ap_heard+5) aff
Tar_train=train_y;
%% ******** -------------------- HELM Algor. -------------------------------
ww1=2*rand(size(train_x,2)+1,N1)-1;
ww2=2*rand(N1+1,N2)-1;
ww3=2*rand(N2+1,N3)-1;
% vc=(2*rand(N2+1,N3)');
% training with HELM algorithim
[~,~,~,~,label_est,TY_test,tst_org,Y_tr] =...
    helm_train(train_x,Tar_train,test_x,test_y,ww1,ww2,ww3,s,C,biass);
Targ_tst=test_y';
Y_train=Y_tr';
% TY_test=TY_test';
[~ ,idx_tst_cls_est]=max(TY_test(:,1:cls_numAff)');        % est. class id of each sampel of test
[~ ,idx_tst_flr_est]=max(TY_test(:,cls_numAff+1:end)');  % est. floor id of each sampel of test
[~ ,tst_flr_org]=max( Targ_tst(cls_numAff+1:end,:)); % real. floor id of each sampel of test

[~ ,idx_tr_cls]=max(Y_train(1:cls_numAff,:));         % est. class id of each sampel of train
[~ ,idx_tr_floor]=max(Y_train(cls_numAff+1:end,:));   % est. floor id of each sampel of train
[~,tr_floor]=max(Tar(cls_numAff+1:end,:));
%% Accuracy Cluster and Floor Estimation -------------------------------------------
% train
C_cls = confusionmat( RSS_all(:,ap_heard+5),idx_tr_cls'); % C = confusionmat(group,grouphat)
C_tr = 100*sum(diag(C_cls))/sum(C_cls(:));
C_flr_train = confusionmat(tr_floor ,idx_tr_floor'); % C = confusionmat(group,grouphat)
F_tr = 100*sum(diag(C_flr_train))/sum(C_flr_train(:));
floor_tr(:,1)=tr_floor;
floor_tr(:,2)=idx_tr_floor;
% test
C_con = confusionmat( hard_class,idx_tst_cls_est'); % C = confusionmat(group,grouphat)
C_tst = 100*sum(diag(C_con))/sum(C_con(:));
C_flr_test = confusionmat(tst_flr_org ,idx_tst_flr_est'); % C = confusionmat(group,grouphat)
F_tst = 100*sum(diag(C_flr_test))/sum(C_flr_test(:));
floor_test(:,1)=tst_flr_org;
floor_test(:,2)=idx_tst_flr_est';
%% Coordinate Estimation -------------------------------------------------------------
if mean_id==0
    [Coordinate2d,Coordinate3d,Erro2D]=CoordinateEst_Soft(floor_test,idx_tst_cls_est',...
    RSS_all(:,ap_heard+1:end),Test_RSS(:,ap_heard+1:end),Test_RSS,RSS_all,k_cand,Simil,center,Flag_set,ap_heard);
else
    [Coordinate2d,Coordinate3d,Erro2D]=CoordinateEst_hard(floor_test,idx_tst_cls_est',...
    RSS_all(:,ap_heard+1:end),Test_RSS(:,ap_heard+1:end),Test_RSS,RSS_all,k_cand,Simil,center,Flag_set,ap_heard);
end

%% -------------------------------------------------------------------------------------
% ----------- ------------------------------------------Error 2d
dis2=(Coordinate2d(:,1:2)-Coordinate2d(:,3:4)).^2;
sum_dis2=sum(dis2,2);
Error2D=sqrt(sum_dis2);
max_error2d=max(Error2D);
min_error2d=min(Error2D);
RMSE_2D=sqrt(sum(sum_dis2)/size(sum_dis2,1));
[Xlim2 cumDis2]=CDF(max_error2d,Error2D);
% ----------- ------------------------------------------Error 3d
dis3=(Coordinate3d(:,1:3)-Coordinate3d(:,4:6)).^2;
sum_dis3=sum(dis3,2);
Error3D=sqrt(sum_dis3);max_error3=max(Error3D);
min_error3=min(Error3D);
[Xlim3 cumDis3]=CDF(max_error3,Error3D);
RMSE_3D=sqrt(sum(sum_dis3)/size(sum_dis3,1));
Mean_Error2D=mean(Error2D);
Mean_Error3D=mean(Error3D);
Median_2D=median(Error2D);
Median_3D=median(Error3D);
% ----------- ------------------------------------------X Error 
disX=(Coordinate2d(:,1)-Coordinate2d(:,3)).^2;
sum_disX=sum(disX,2);
Error2DX=sqrt(sum_disX);
max_errorX=max(Error2DX);
min_errorX=min(Error2DX);
RMSE_X=sqrt(sum(sum_disX)/size(sum_disX,1));
Mean_X=mean(Error2DX);
[Xlim2X cumDis2X]=CDF(max_errorX,Error2DX);
% ----------- ------------------------------------------Y Error 
disY=(Coordinate2d(:,2)-Coordinate2d(:,4)).^2;
sum_disY=sum(disY,2);
Error2DY=sqrt(sum_disY);
max_errorY=max(Error2DY);
min_errorY=min(Error2DY);
RMSE_Y=sqrt(sum(sum_disY)/size(sum_disY,1));
Mean_Y=mean(Error2DY);
[Xlim2Y cumDis2Y]=CDF(max_errorY,Error2DY);
%% coordinate estimation with true floor 
id_truefloor=find(Coordinate3d(:,3)==Coordinate3d(:,6));
Floor_accuracy=100*(size(id_truefloor,1)/size(Coordinate3d,1));
dif_2=(Coordinate3d(id_truefloor,1:3)-Coordinate3d(id_truefloor,4:6)).^2;
sum_dif_2=sum(dif_2 ,2);Error_tru_floor=sqrt(sum_dif_2);max_error_tf=max(Error_tru_floor);
[Xlimt, cumDist]=CDF(max_error_tf,Error_tru_floor);

RMSE_tf=sqrt(sum(sum_dif_2)/size(sum_dif_2,1));
Mean_Errortf=mean(Error_tru_floor);
%------------------------------------------------------- X Error 
disXt=(Coordinate2d(id_truefloor,1)-Coordinate2d(id_truefloor,3)).^2;
sum_disXtf=sum(disXt,2);Error2DXt=sqrt(sum_disXtf);max_errorXt=max(Error2DXt);
min_errorXtf=min(Error2DXt);
RMSE_Xtf=sqrt(sum(sum_disXtf)/size(sum_disXtf,1));

[Xlim2Xtf cumDis2Xtf]=CDF(max_errorXt,Error2DXt);
% ----------------------------------------------------- Y Error  
disYt=(Coordinate2d(id_truefloor,2)-Coordinate2d(id_truefloor,4)).^2;
sum_disYt=sum(disYt,2);Error2DYt=sqrt(sum_disYt);max_errorYt=max(Error2DYt);
min_errorYt=min(Error2DYt);
RMSE_Ytf=sqrt(sum(sum_disYt)/size(sum_disYt,1));
[Xlim2Yt cumDis2Yt]=CDF(max_errorYt,Error2DYt);
%% coordinate estimation with true cluster
clear id_truefloor;
flr_est=Coordinate3d(:,7);
id_trueCLS=find(flr_est==Tarclus_tst_org');
clst_accuracy=100*(size(id_trueCLS,1)/size(Coordinate3d,1));
dif_2c=(Coordinate3d(id_trueCLS,1:3)-Coordinate3d(id_trueCLS,4:6)).^2;
sum_dif_2c=sum(dif_2c ,2);Error_tru_cls=sqrt(sum_dif_2c);max_error_tc=max(Error_tru_cls);
[Xlimtc, cumDistc]=CDF(max_error_tc,Error_tru_cls);
RMSE_tc=sqrt(sum(sum_dif_2c)/size(sum_dif_2c,1));
Mean_tc=mean(Error_tru_cls);
%------------------------------------------------------- X Error 
disXtc=(Coordinate2d(id_trueCLS,1)-Coordinate2d(id_trueCLS,3)).^2;
sum_disXtc=sum(disXtc,2);Error2DXtc=sqrt(sum_disXtc);max_errorXtc=max(Error2DXtc);
min_errorXtc=min(Error2DXtc);
RMSE_Xtc=sqrt(sum(sum_disXtc)/size(sum_disXtc,1));
[Xlim2Xtc cumDis2Xtc]=CDF(max_errorXtc,Error2DXtc);
% ----------------------------------------------------- Y Error  
disYtc=(Coordinate2d(id_trueCLS,2)-Coordinate2d(id_trueCLS,4)).^2;
sum_disYtc=sum(disYtc,2);Error2DYtc=sqrt(sum_disYtc);max_errorYtc=max(Error2DYtc);
min_errorYtc=min(Error2DYtc);
RMSE_Ytc=sqrt(sum(sum_disYtc)/size(sum_disYtc,1));
[Xlim2Ytc cumDis2Ytc]=CDF(max_errorYtc,Error2DYtc);
%% result ===================================================================

result.error=[F_tr,C_tr,F_tst,C_tst,...
RMSE_2D,Median_2D,Mean_Error2D,RMSE_X,Mean_X,RMSE_Y,Mean_Y,...
RMSE_3D,Median_3D,Mean_Error3D,RMSE_tf,Mean_Errortf,RMSE_Xtf,RMSE_Ytf,...
RMSE_tc,Mean_tc,RMSE_Xtc,RMSE_Ytc]';
result.er2d=Error2D;
result.er3d=Error3D;
result.erX=Error2DX;
result.erY=Error2DY;
result.er_truF=Error_tru_floor;
result.er_trcls=Error_tru_cls;
result.test_F=[floor_test(:,1) floor_test(:,2)];
result.train_F=[floor_tr(:,1) floor_tr(:,2)];
result.train_C=[RSS_all(:,ap_heard+5) idx_tr_cls'];
result.test_C=[hard_class' Coordinate3d(:,end)];% Coordinate3d(:,end=7)>> class id
result.Cord2D=Coordinate2d;
result.Cord3D=Coordinate3d;


end
