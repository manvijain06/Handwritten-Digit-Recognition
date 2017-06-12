function [ k ] = polykernel( alpha,beta,d,x,y )
% polykernel : uses the formula k(xi,xj) = [alpha * dot(xi,xj) + beta]^d
% 
% OUTPUT : It returns the value of the kernel function for the training data
n = size(X,1)
for i=1:n
     for j=1:i-1
        K(i,j) = (x(i,:) * x(j,:));
        Y(i,j) = (y)
    end      
end

Kernel = ((alpha * K + beta)^ d));
