function [ yrange1 cumDis]=CDF(max_err,data1)
step0 =0.0001;
x1 =0:step0:max_err; 
[y1, yrange1] = hist(data1, x1);
cumDis = (cumsum(y1))/(numel(data1));

end
