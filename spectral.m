function[ clusters, evals, evects ] = spectral(W, K, method)
%Clusters data points by the spectral clustering algorithm.
%Author:Batuhan Toker #25981
%Written for CS512 HW #3 W=S_Matrix; 

%% Parameters
    %W:N × N similarity matrix
    %K:the number of clusters
    %method:the method to be used for the Laplacian, ‘normalized’ 
if ~exist('method', 'var') %If no specification for the method, it is defined
                           %method will be normalized
    method = 'normalized';
end
%% Function
D = diag(sum(W,2));
if strcmp(method ,'normalized')
    L = eye(length(W))-inv((D))*W;
end
if  strcmp(method, 'unnormalized')
    L = D-W;
end
if strcmp(method, 'symmetric')
    L=eye(length(W))-D^(-1/2)*W*D^(-1/2);
end
[vector,values] = eig(L,D);
[d,ind] = sort(diag(values));
eigvalues = values(ind,ind);
eigvector = vector(:,ind);
evects=eigvector(:,1:K);
evals=eigvalues(:,1:K);
clusters=kmeans(evects,K);
end
