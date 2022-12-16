function [Loc_2d,Loc_3d,Erro2D]=CoordinateEst_Soft(test_F_est,Test_cluster,train_cor,...
    test_cor,Tst_RSS,RSStr_all,kCand,sim,Cntr,flag_set,ap_heard)


for tst_loc=1:size(test_cor,1) %tst_loc=tst_loc+1
    cls_tst_est=Test_cluster(tst_loc,1);
    test_xyz=test_cor(tst_loc,1:3);
    floor_tst_est=test_F_est(tst_loc,2);
    idx_cf=find(train_cor(:,end)==cls_tst_est & train_cor(:,3)==floor_tst_est);% co-cluster and floor
    idx_f=find(train_cor(:,3)==floor_tst_est); % co-floor
    idx_c=find(train_cor(:,end)==cls_tst_est); % co-cluster
    if size(idx_cf,1)==0
        idx_cf=idx_c;
    end
    loc_cond_cf=train_cor(idx_cf,1:3);
    loc_cond_f=train_cor(idx_f,1:3);
    loc_cond_c=train_cor(idx_c,1:3);
    if sim==1   % cosine similarity
        Cosin_sim_CF=Similarity_Cosin(Tst_RSS(tst_loc,1:ap_heard),RSStr_all(idx_cf,1:ap_heard));
        Cosin_sim_F=Similarity_Cosin(Tst_RSS(tst_loc,1:ap_heard),RSStr_all(idx_f,1:ap_heard));
        Cosin_sim_c=Similarity_Cosin(Tst_RSS(tst_loc,1:ap_heard),RSStr_all(idx_c,1:ap_heard));
        [~,cand_id_CF]=sort(Cosin_sim_CF,'descend');
        [~,cand_id_F]=sort(Cosin_sim_F,'descend');
        [~,cand_id_c]=sort(Cosin_sim_c,'descend');
    else
        % eqlid % 2== cosin
        dif_cf=(repmat(Tst_RSS(tst_loc,1:ap_heard),size(idx_cf,1),1)-RSStr_all(idx_cf,1:ap_heard)).^2;
        sqrt_sum_dif_CF=sqrt(sum(dif_cf,2));
        dif_F=(repmat(Tst_RSS(tst_loc,1:ap_heard),size(idx_f,1),1)-RSStr_all(idx_f,1:ap_heard)).^2;
        sqrt_sum_dif_F=sqrt(sum(dif_F,2));
        dif_c=(repmat(Tst_RSS(tst_loc,1:ap_heard),size(idx_c,1),1)-RSStr_all(idx_c,1:ap_heard)).^2;
        sqrt_sum_dif_c=sqrt(sum(dif_c,2));
        
        [~,cand_id_CF]=sort(sqrt_sum_dif_CF);
        [~,cand_id_F]=sort(sqrt_sum_dif_F);
        [~,cand_id_c]=sort(sqrt_sum_dif_c);
    end
    
    if flag_set==1 % same in c and  F
        kk=min(size(cand_id_CF,2),kCand);
        Loc_kcod2D=loc_cond_cf(cand_id_CF(1:kk),1:2);
        Loc_k3d=loc_cond_cf(cand_id_CF(1:kk),1:3);
        clear loc_cond_cf
    elseif flag_set==2 % same in F
        
        kk=min(size(cand_id_F,2),kCand);
        Loc_kcod2D=loc_cond_f(cand_id_F(1:kk),1:2);
        Loc_k3d=loc_cond_f(cand_id_F(1:kk),1:3);
        clear loc_cond_f
    elseif flag_set==3 % same in cluster
        
        kk=min(size(cand_id_c,2),kCand);
        Loc_kcod2D=loc_cond_c(cand_id_c(1:kk),1:2);
        Loc_k3d=loc_cond_c(cand_id_c(1:kk),1:3);
        clear loc_cond_c
        
    end
    Mean_Loc_2d =mean(Loc_kcod2D,1);
    difm2d=abs(Loc_kcod2D-Mean_Loc_2d);
    [~,id_minX]=min(difm2d(:,1));
    [~,id_minY]=min(difm2d(:,2));
    Loc_2d(tst_loc,1:2)=[Loc_kcod2D(id_minX,1) Loc_kcod2D(id_minY,2)];
    Loc_2d(tst_loc,3:4)=test_cor(tst_loc,1:2);
    Loc_2d(tst_loc,5)=cls_tst_est;
    Mean_Loc_3d=mean(Loc_k3d,1);
    difm3d=abs(Loc_k3d-Mean_Loc_3d);
    [~,id_min3X]=min(difm3d(:,1));
    [~,id_min3Y]=min(difm3d(:,2));
    [~,id_min3Z]=min(difm3d(:,3));
    Loc_3d(tst_loc,1:3)=[Loc_k3d(id_min3X,1) Loc_k3d(id_min3Y,2) Loc_k3d(id_min3Z,3)];
    Loc_3d(tst_loc,4:6)=test_cor(tst_loc,1:3);%
    Loc_3d(tst_loc,7)=cls_tst_est;
    tst_2_Loc_kcod2D=sqrt(sum((repmat(test_cor(tst_loc,1:2),kk,1)-Loc_kcod2D).^2,2));
    [er,id]=min(tst_2_Loc_kcod2D);clear tst_2_Loc_kcod2D
    Erro2D(tst_loc,1)=er;
    clear idxx cls data Cosin_sim Loc_k cand_id kk Loc_kcod2D Loc_k3d cls_head
    clear id_hed Cosin_sim_CF Cosin_sim_c Cosin_sim_F cand_id_c cand_id_F cand_id_CF
    clear idx_f idx_c idx_cf difm2d id_minX id_minY

end
end
