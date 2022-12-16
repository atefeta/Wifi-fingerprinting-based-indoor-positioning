function CS=Similarity_Cosin(tst_i,tr_all)
%tst_i=Test_RSS(tst_loc,1:200);
% tr_all=RSS_all(idx,1:200);
for i=1:size(tr_all,1)
    x=tst_i;
    y=tr_all(i,:);
    xy   = dot(x,y);
    nx   = norm(x);
    ny   = norm(y);
    nxny = nx*ny;
    CS(1,i)   = xy/nxny;
end
end
