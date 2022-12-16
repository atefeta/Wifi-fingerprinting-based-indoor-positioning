function [Score_all,tst_class]=Test_Fuzzy_Label(Test_Loc,Train_Loc,cls_alg,cand)
% Test_Loc=Test_RSS(:,201:end);Train_Loc=Loc_cor;cls_alg=4;cand=6
% Test_Loc(:,3)= Test_Loc(:,3).*100;
% Train_Loc(:,3)= Train_Loc(:,3).*100;
Test_Loc(:,3)=(Test_Loc(:,3)-1).*2.5;
Train_Loc(:,3)=(Train_Loc(:,3)-1).*2.5;
for ts=1:size(Test_Loc,1)% ts=ts+1
    %floor=Test_Loc(ts,3);
    %f_id=find(Train_Loc(:,3)==floor);
    dif=repmat(Test_Loc(ts,:),size(Train_Loc,1),1)-Train_Loc(:,1:3);
    dif_2=dif.*dif;
    sum_dif_2=sum(dif_2,2);
    dist=sqrt(sum_dif_2);
    [val sort_idx]=sort(dist,'ascend');
    class_cond=Train_Loc(sort_idx(1:cand),cls_alg);
    [freq,class_uniq]=hist(class_cond,unique(class_cond));%i=2
    [idm mx]=max(freq);tst_class(ts)=class_uniq(mx);
    if cand==1 
        class_uniq=class_cond;
        freq=1;
    end
    MeanVal=mean(val(1:cand));
    for i=1:size(unique(class_cond),1)%i=i+1
        index=find(class_cond==class_uniq(i));
        scr(i,1)=class_uniq(i);
        kk=(freq(i)/cand);
        err_pro=val(index)./MeanVal;
        scr(i,2)=mean(kk.*err_pro);
        clear index kk err_pro
    end
Score_all{ts}=scr;
clear scr val idm mx freq class_uniq dist sum_dif_2 dif_2 dif
end


end
% [frq,cls]=hist(Train_Loc(:,6),unique(Train_Loc(:,6)));