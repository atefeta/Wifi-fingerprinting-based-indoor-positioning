function tst_class=TestLabel(Test_Loc,Train_Loc,cls_alg)
% Test_Loc=Test_RSS(:,201:end);Train_Loc=Loc_cor;
% Test_Loc(:,3)= Test_Loc(:,3).*100;
% Train_Loc(:,3)= Train_Loc(:,3).*100;
for ts=1:size(Test_Loc,1)% ts=ts+1

    %floor=Test_Loc(ts,3);
    %f_id=find(Train_Loc(:,3)==floor);
    dif=repmat(Test_Loc(ts,:),size(Train_Loc,1),1)-Train_Loc(:,1:3);
    dif_2=dif.*dif;
    sum_dif_2=sum(dif_2,2);
    dist=sqrt(sum_dif_2);
    [val sort_idx]=sort(dist,'ascend');
    class_cond=Train_Loc(sort_idx(1:5),cls_alg);
    [freq,class]=hist(class_cond,unique(class_cond));[idm mx]=max(freq);
    tst_class(ts)=class(mx);clear val sort_idx a b idm mx class_cond floor f_id
end


end
