function clstr =cluster_func(cls_methode,S_Matrix,par,method,K,Loc_corr)
% ls_methode=cls_methode;par=dampf_convit_maxit;K=K_Spectral;Loc_corr=Loc_cor;
if(strcmpi(cls_methode,'Affinity'))
    
    p=median(S_Matrix)';
    [idx,netsim,ih,unconverged,dpsim,expref,S,R]=affinitypcluster(S_Matrix,p,...
        'dampfact',par(1,1),'convits',par(1,2),'maxits',par(1,3),'nonoise');
    % [idx,netsim,i,unconverged,dpsim,expref]
    exampl_id=unique(idx(:,end));
    clstr.idx=idx;
    clstr.net=netsim;
    clstr.dampsim=dpsim;
    clstr.exampref=expref;
    clstr.exampl_idx=exampl_id;
    clstr.i=ih;
    clstr.unconverged=unconverged;
    clstr.S=S;
    clstr.R=R;
    fprintf('Number of clusters: %d\n',length(unique(idx(:,end))));
    fprintf('Fitness (net similarity): %f\n',netsim);
    cls=0;
    for i1=unique(idx)'
        ii=find(idx(:,end)==i1); h=plot3(Loc_corr(ii,1),Loc_corr(ii,2),Loc_corr(ii,3),'o'); hold on;
        col=rand(1,3); set(h,'Color',col,'MarkerFaceColor',col);
        cls=cls+1;
        legendInfo{cls}=['Class ID' ,num2str(cls)];
        clear ii
        coler_m(cls,1:3)=col;
    end
    grid on
    title('Affinity Clustering')
    legend(legendInfo)
    hold on
    bb=0;
    for i1=unique(idx)'
        bb=bb+1;
        ii=find(idx(:,end)==i1);
        h=plot3(Loc_corr(ii,1),Loc_corr(ii,2),Loc_corr(ii,3),'o');
        hold on;
        xi1=Loc_corr(i1,1)*ones(size(ii));
        xi2=Loc_corr(i1,2)*ones(size(ii));
        xi3=Loc_corr(i1,3)*ones(size(ii));
        line([Loc_corr(ii,1),xi1]',[Loc_corr(ii,2),xi2]',[Loc_corr(ii,3),xi3]','Color',coler_m(bb,:));
        clear ii
    end
elseif(strcmpi(cls_methode,'Specral'))
    [clusters, evals, evects ] = spectral(S_Matrix, K, method);
    sampl_id=K;
    clstr.idx=clusters;
    clstr.eval=evals;
    clstr.evect=evects;
    for k=1:K
        kk=find(clusters==k);
        pl=plot3(Loc_corr(kk,1),Loc_corr(kk,2),Loc_corr(kk,3),'o');hold on
        col2=rand(1,3); set(pl,'Color',col2,'MarkerFaceColor',col2);
        legendInfo2{k}=['Class ID' ,num2str(k)];
        clear kk
    end
    grid on
    title('Spectral Clustering')
    legend(legendInfo2)
    
    
    
    
end
end