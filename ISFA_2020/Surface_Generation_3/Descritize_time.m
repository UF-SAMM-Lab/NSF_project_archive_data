function [discrete] = Descritize_time(time_step, anylitical)
%The purpose of this fuction is to determine the closest standard point
%in time in the given array.
        
        %---------------------------------------------------------
        %Calculate the lower bound time
        %---------------------------------------------------------
        Tl = anylitical - mod(anylitical,time_step);
        
        %---------------------------------------------------------
        %Calculate the upper bound time
        %---------------------------------------------------------
        Tu = Tl + time_step;
        
        %---------------------------------------------------------
        %Select the time that is closest
        %---------------------------------------------------------      
        %comapare is a binary indicator. If the real time is closer to the
        %upper bound time, a 1 will be returned
        compareT = Tl - anylitical < anylitical - Tu;
        discrete = Tl + compareT*time_step;

end

