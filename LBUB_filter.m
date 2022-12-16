function [LB UB]=LBUB_filter(Training_UTS,Test_UTS,ap_heard0,cc)
% pos and normalization 
[TrRSS,~]=preprocessing_elms(Training_UTS(:,1:ap_heard0),Test_UTS(:,1:ap_heard0),ap_heard0,1);
trx_id_column=find(all(TrRSS(:,1:ap_heard0)==0,1)); % tr ap index of dont heared
TrRSS(:,trx_id_column)=[];   % deleted ap index of dont heared in train based tr
ap_heard=size(TrRSS,2)-3;
% %% (2) After Normalization ------------------------------------------------
nonZ2=find(TrRSS(:,1:ap_heard)~=0);
%histogram(TrRSS(nonZ2));
Mean2=mean(TrRSS(nonZ2));
STD2=std(TrRSS(nonZ2));
UB=Mean2+2*cc*STD2;
LB=abs(Mean2-2*STD2);

end