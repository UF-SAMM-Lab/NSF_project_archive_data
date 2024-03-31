function [discrete] = Descritize_XYZ(grid_spacing, anylitical)
%The purpose of this fuction is to determine the closest node to each point
%in the given array.
        
        %---------------------------------------------------------
        %Calculate the lower bound gridpoint for each direction
        %---------------------------------------------------------
        Xl = anylitical.x - mod(anylitical.x,grid_spacing);
        Yl = anylitical.y - mod(anylitical.y,grid_spacing);
        Zl = anylitical.z - mod(anylitical.z,grid_spacing);
        
        %---------------------------------------------------------
        %Calculate the upper bound gridpoint for each direction
        %---------------------------------------------------------
        Xu = Xl + grid_spacing;
        Yu = Xl + grid_spacing;
        Zu = Xl + grid_spacing;
        
        %---------------------------------------------------------
        %Select the node that is closest
        %---------------------------------------------------------      
        %comapare is a binary indicator. If the real point is closer to the
        %upper point, a 1 will be returned
        compareX = Xl - anylitical.x < anylitical.x - Xu;
        discrete.x = Xl + compareX*grid_spacing;
        
        compareY = Yl - anylitical.y < anylitical.y - Yu;
        discrete.y = Yl + compareY*grid_spacing;
        
        compareZ = Zl - anylitical.z < anylitical.z - Zu;
        discrete.z = Zl + compareZ*grid_spacing;
           
end

