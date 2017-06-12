function[kernel,Hessian] = rbfkernel(X,y,gamma)
% Computes an rbf kernel matrix from the input data
% INPUT 
% X =  a matrix containing all training data as rows
% gamma = the kernel width; 
%
% OUTPUT 
% K = the rbf kernel matrix ( = exp(-gamma)*||Xi - Xj||^2) 

n = size(X,1)
for i=1:n
     K(i,i)=1;
     for j=1:i-1
         K(i,j) = (X(i,:)-X(j,:));
     	 Y(i,j) = (y(i,:) * y(j,:));
     end
 end

 kernel = exp(-gamma * (K * K'))
 yvalue = Y * Y'

 Hessian = yvalue * kernel