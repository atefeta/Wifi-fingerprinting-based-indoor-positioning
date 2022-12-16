clc;close all;clear;
%% -------------------------- parameters -----------------------------------
pwd=2;                               % (normalized rssi)^pwd 
C=0.001; 
bias=1;                              % bias of ELM
N1=90;N2=30;N3=400;                  % number of ELM neurons
k_cand=7;                       
Flag_set=1;                          % selection of sampels from 1 == shared in estimated Cluster and Floor,   2 == Floor    3 == Cluster
finalSelection=1;                    % 1 ==  hard      0 == soft voitting
K_scor_id_cls_f=5;                   % this paremeter is for more exprements and dont effect on our proposed reported work
Similarity=1;                        % 1 cosin          2==eqlid.
cls_methode='Affinity';              % Specral  Affinity
K_Func='cosin';                      % gaussian poly invquad
dampf_convit_maxit=[.5 100 500];     % for clustering setup >> landa, itteration, maximum itteration
Elm_Type=1;                          %  1 >> classification
funct={'radbas','tribas','hardlim','sin','sigmoid'}; % ActivationFunctions
ActivationFunction= funct{1,5};
K_Spectral=10;                       % if spectral is selected, should be set
method='normalized';                 % 'normalized' % unnormalized 'symetric' >> for spectral clustering 
cls_alg=4;                           % 4: aff_class_id    6: Sp_class_id, in this study affinity is used >> 4
cand_l=1;
s=5.6;                               % for optimization step of sparse ELM_AE
rng('default')
%% ******** -------------------- Data Loading ---------------------------------------------------
% UTS Data
% path_to_data = 'E:....\data';
% adr='addresses of folder name of scripts';
% addpath(adr);
path_to_data = 'E:\Research_for_ThesiS\ATF_Simulation_Code\UJIindoorLoc\ATF_Simu\ATF\data';
% adr='E:\Research_for_ThesiS\ATF_Simulation_Code\UJIindoorLoc\ATF_Simu\ATF\M_atf\APC-2Lhelm\CoorEst_aff_2LHelm';
% addpath(adr);
load([path_to_data,'/Test_UTS'])
load([path_to_data,'/Training_UTS'])
Training_UTS(:,end)=Training_UTS(:,end)+4; % floor are set to locate in [1-16]
Test_UTS(:,end)=Test_UTS(:,end)+4;          
ap_heard0=size(Training_UTS,2)-3;
% Lower and Upper Band calculation for filtering ---------------------
[LB UB]=LBUB_filter(Training_UTS,Test_UTS,ap_heard0,1);
% normalization and filtering of weak and strong signals  --------------
[Train_RSS,Tst_RSS]=preprocessing_elms(Training_UTS(:,1:ap_heard0),Test_UTS(:,1:ap_heard0),ap_heard0,1);
Train_RSS(:,ap_heard0+1:ap_heard0+3)=Training_UTS(:,ap_heard0+1:end);
Tst_RSS(:,ap_heard0+1:ap_heard0+3)=Test_UTS(:,ap_heard0+1:end);
trx_id_column=find(all(Train_RSS(:,1:ap_heard0)==0,1)); % tr ap index of dont heared
Train_RSS(:,trx_id_column)=[];    % deleted ap index of dont heared in train 
Tst_RSS(:,trx_id_column)=[];      % deleted ap index of dont heared in test based train data
ap_heard=size(Train_RSS,2)-3;     % number of final heared AP 
% signal filtering based of obtained thresholds 
tr_strong=find(Train_RSS(:,1:ap_heard)>UB);Train_RSS(tr_strong)=1;
tst_strong=find(Tst_RSS(:,1:ap_heard)>UB);Tst_RSS(tst_strong)=1;
tr_week=find(Train_RSS(:,1:ap_heard)<=LB);Train_RSS(tr_week)=0;
tst_week=find(Tst_RSS(:,1:ap_heard)<=LB);Tst_RSS(tst_week)=0;
clear Training_UTS Test_UTS;
% Power Transform  applying -----------------------------------------------
Train_RSS(:,1:ap_heard)=Train_RSS(:,1:ap_heard).^pwd;
Tst_RSS(:,1:ap_heard)=Tst_RSS(:,1:ap_heard).^pwd;
%% Clustering Step ---------------------------------------------------------
%%%% find and split same Location for clustring implementation 
%%% unique reference points are participated in clustering 
[num,id,ic]=unique(Train_RSS(:,ap_heard+1:ap_heard+3), 'rows'); 
RSS_uni_Loc_split = splitapply( @(x){x},Train_RSS, ic ) ;
for i=1:size(num,1)
    RSS_uni_Loc_split{i,1}(:,ap_heard+4)=i;   
end
RSS_ALL=cell2mat(RSS_uni_Loc_split);
for ii=1:size(RSS_uni_Loc_split,1)
    RS_Mean_n(ii,1:ap_heard)=mean(RSS_uni_Loc_split{ii,1}(:,1:ap_heard),1);
    RS_Mean_n(ii,ap_heard+1:ap_heard+3)=RSS_uni_Loc_split{ii,1}(1,ap_heard+1:ap_heard+3);
    % averaging of heared rssi in same points to obtain one fingerprint in each point 
    M=RSS_uni_Loc_split{ii,1}(:,1:ap_heard);
    mean_nonZearo(ii,1:ap_heard) = sum(M, 1)./sum(M~=0, 1);
    mean_nonZearo(isnan(mean_nonZearo))=0;
    mean_nonZearo(ii,ap_heard+1:ap_heard+3) = RSS_uni_Loc_split{ii,1}(1,ap_heard+1:ap_heard+3);
end
train_coor= mean_nonZearo(:,ap_heard+1:ap_heard+3);
RSS_mean=[mean_nonZearo train_coor]; 
%% similarity matrix construction  and Clustering -------------------------
S_Matrix_UTS=Similarity_Mat(mean_nonZearo(:,1:ap_heard),K_Func);
clstr_uts =cluster_func(cls_methode,S_Matrix_UTS,dampf_convit_maxit,method,K_Spectral,train_coor);
% clear mean_nonZearo
% load('S_Matrix_UTS') % output of clustering on UTS
% load('clstr_UTS');
clstr_uts=clstr_uts;clear clstr_UTS
[center,~,class_id]=unique(clstr_uts.idx);
Loc_cor=train_coor;                 % x, y, floor
Loc_cor(:,4)=class_id;              % 1:4 >> x, y, floor, class id
az=[1:size(center,1)];
clas_center_floor=[az' center train_coor(center,3)];
for cl=1:size(center,1)
    ff=find(class_id==cl);
    Loc_cor(ff,5)=center(cl,1);     % 5: center id
    clear ff
end
% Training an Test steps
result_2L=HELM_2L_All(RSS_ALL,Tst_RSS,RSS_mean,center,class_id,Loc_cor,ap_heard,...
    finalSelection,K_scor_id_cls_f,Similarity,Flag_set,Elm_Type,...
    cls_alg,cand_l,ActivationFunction,k_cand,N1,N2,N3,s,C,clas_center_floor,bias);

%% Test cluster Accuracy --------------------------------------------------
tst_clusterconf=confusionmat(result_2L.test_C(:,1),result_2L.test_C(:,2));
cluster_tst = 100*sum(diag(tst_clusterconf))/sum(tst_clusterconf(:));
%% Train Floor Accuracy --------------------------
Ctr = confusionmat(result_2L.train_F(:,1),result_2L.train_F(:,2));
F_tr = 100*sum(diag(Ctr))/sum(Ctr(:));
%% Test Floor Accuracy ----------------------------------------------------
Ctst = confusionmat(result_2L.test_F(:,1),result_2L.test_F(:,2));
F_tst = 100*sum(diag(Ctst))/sum(Ctst(:));
C_cord3D=confusionmat(result_2L.Cord3D(:,3),result_2L.Cord3D(:,6));
F_tst_cor3D = 100*sum(diag(C_cord3D))/sum(C_cord3D(:));
%% Error 
Mean2D_2L=mean(result_2L.er2d);
Mean2D_trF=mean(result_2L.er_truF);
Mean3D=mean(result_2L.er3d);
Mean_trcls=mean(result_2L.er_trcls);
RMSE2D_2L=sqrt(sum(result_2L.er2d.^2)/size(result_2L.er2d,1));
%% 
Varname1l={'Floor_tr ','Floor_tst','MeanError','MeanError_trueFloor','RMSE'};
HELM_2L=[F_tr,F_tst,Mean2D_2L,Mean2D_trF,RMSE2D_2L]';
T1=table(HELM_2L,'RowNames',Varname1l)
%% CDF Plot -------------------------------------------------------------
% max_error2d=max(result_2L.er2d);
% [Xlim2 cumDis2]=CDF(max(result_2L.er2d),result_2L.er2d);
% figure()
% plot(Xlim2 ,cumDis2,'LineWidth',2);grid on
% xlabel('Error (m)');
% ylabel('CDF (probability)');
% legend('Proposed (UTS. dataset)')



