function [Loc_2d,Loc_3d,Erro2D_ideal]=CoordinateEst_hard(test_F_est,Tst_clus_est,train_cor,...
    test_cor,Tst_RSS,RSStr_all,kCand,sim,Cntr,flag_set,ap_heard)



for tst_loc=1:size(test_cor,1) %tst_loc=tst_loc+1
    cls_tst_est=Tst_clus_est(tst_loc,1);
    cls_head=Cntr(cls_tst_est); % sampel number that is center of cluster of cls_tst_est
    id_hed=find(train_cor(:,4)==cls_head);
    cls_head(1,2:4)=train_cor(id_hed(1),1:3); % id/X/Y/Z=1234
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
        [valcf,cand_id_CF]=sort(Cosin_sim_CF,'descend');
        [valf,cand_id_F]=sort(Cosin_sim_F,'descend');
        [valc,cand_id_c]=sort(Cosin_sim_c,'descend');
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
    
    if flag_set==1 % shared in cluster and  Floor
        kk=min(size(cand_id_CF,2),kCand);
        Loc_kcod2D=loc_cond_cf(cand_id_CF(1:kk),1:2);
        Loc_k3d=loc_cond_cf(cand_id_CF(1:kk),1:3);
        
    elseif flag_set==2 % shared in Floor
        
        kk=min(size(cand_id_F,2),kCand);
        Loc_kcod2D=loc_cond_f(cand_id_F(1:kk),1:2);
        Loc_k3d=loc_cond_f(cand_id_F(1:kk),1:3);
    elseif flag_set==3 % shared in cluster
        kk=min(size(cand_id_c,2),kCand);
        Loc_kcod2D=loc_cond_c(cand_id_c(1:kk),1:2);
        Loc_k3d=loc_cond_c(cand_id_c(1:kk),1:3);
        
    end
    tst_2_Loc_kcod2D=sqrt(sum((repmat(test_cor(tst_loc,1:2),kk,1)-Loc_kcod2D).^2,2));
    [er,id]=min(tst_2_Loc_kcod2D);clear tst_2_Loc_kcod2D
    Erro2D_ideal(tst_loc,1)=er; % best probalitic point can be selected
    
    Loc_2d(tst_loc,1:2)=mean(Loc_kcod2D,1);%median(Loc_kcod,1)
    Loc_2d(tst_loc,3:4)=test_cor(tst_loc,1:2);
    Loc_2d(tst_loc,5)=cls_tst_est;
    
    Loc_3d(tst_loc,1:3)=mean(Loc_k3d,1);
    Loc_3d(tst_loc,3)=round(Loc_3d(tst_loc,3));
    Loc_3d(tst_loc,4:6)=test_cor(tst_loc,1:3);%
    Loc_3d(tst_loc,7)=cls_tst_est;
    
    clear idxx cls data Cosin_sim Loc_k cand_id kk
    
end
end
