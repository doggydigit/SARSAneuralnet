function [ R ] = QQ( W00,W01,W02 )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

V = [];
U = [];
u=-0.2;
for i = 1:4
    if (i>2.5)
        u=0.2;
    end
    for j = 1:4
        U(i,j) = u;
        if (W00(i,j) > W01(i,j) & W00(i,j) > W02(i,j))
            V(i,j) = -W00(i,j)*10;
            
        else
            if ((W01(i,j) > W00(i,j) & W01(i,j) > W02(i,j)))
                V(i,j) = 0;
                U(i,j) = 0;
            else
                V(i,j) = (W02(i,j)*10);
            end
        end
    end
end
R = quiver(U,V,0);
end

