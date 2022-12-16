function SM=Similarity_Mat(data,K_Func)
PDIST = squareform(pdist(data,'euclidean'));
if(strcmpi(K_Func,'gaussian'))
    SIGMA=1;
    SM = exp(-PDIST/(2*SIGMA^2));
elseif(strcmpi(K_Func,'poly'))
    d=2;
    SM = (data*data')^d;
elseif(strcmpi(K_Func,'invquad'))
    a=1;b=2;
    SM = inv((PDIST + a*eye(size(PDIST)))^b);
elseif(strcmpi(K_Func,'euqlid'))
    SM = -PDIST;
elseif(strcmpi(K_Func,'cosin'))
    for i=1:size(data,1)
        for j=1:size(data,1)
            x=data(i,:);
            y=data(j,:);
            xy   = dot(x,y);
            nx   = norm(x);
            ny   = norm(y);
            nxny = nx*ny;
            SM(i,j)   = xy/nxny;
        end
    end
elseif(strcmpi(K_Func,'pearson'))
    [nrow,ncol] = size(data);
    x = mean(data);
    data = data-repmat(x,nrow,1);
    SM = ones(ncol,ncol);
    for i = 1:ncol-1
        x = data(:,i);
        X = sqrt(x'*x);
        for j = i+1:ncol
            y = data(:,j);
            xy = x'*y;
            Y = sqrt(y'*y);
            sim = xy/(X*Y);
            SM(i,j) = sim;
            SM(j,i) = sim;
        end
    end
elseif(strcmpi(K_Func,'make'))
    type = 2;nrow=size(data,1);
    if type == 1
        [Dist, dmax] = similarity_euclid(data,2);   %  pdist(data,'type');
    else
        Dist = 1-(1+similarity_pearson(data'))/2;
        dmax = 1;
    end
    nap = nrow*nrow-nrow;
    SM = zeros(nap,3);
    j = 1;
    for i=1:nrow
        for k = [1:i-1,i+1:nrow]
            SM(j,1) = i;
            SM(j,2) = k;
            SM(j,3) = -Dist(i,k);
            j = j+1;
        end
    end
end

end