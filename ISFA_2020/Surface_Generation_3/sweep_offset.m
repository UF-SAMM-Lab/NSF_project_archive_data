function [sweep] = sweep_offset(patches, safety_factor, norm, nbf_u, nbf_v)
%The purpose of this function is to create the two offset surfaces given
%the swept surface previously determined

%Generate swept volume by offsetting the points in normal direction
sweep.x = zeros(1,2*length(patches.x));
sweep.y = zeros(1,2*length(patches.y));
sweep.z = zeros(1,2*length(patches.z));
sweep.time = [patches.time patches.time];
sweep.segment = [patches.segment patches.segment];

first = 1;
for ii = 1:length(norm)
   last = first+nbf_u(ii)*nbf_v(ii);
   if isnan(norm(:,ii))  %NaN indicates that the link creating the patch has not moved.In this case it makes no sense to offset
      sweep.x(first:last) = patches.x(first:last);
      sweep.y(first:last) = patches.y(first:last);
      sweep.z(first:last) = patches.z(first:last);
   else
      offset = safety_factor(patches.joint(first));
      sweep.x(first:last) = patches.x(first:last)+norm(1,ii)*offset;
      sweep.y(first:last) = patches.y(first:last)+norm(2,ii)*offset;
      sweep.z(first:last) = patches.z(first:last)+norm(3,ii)*offset;
   end
   first = last;
end

first = 1;
for ii = 1:length(norm)
   skip = length(patches.x);
   last = first + nbf_u(ii)*nbf_v(ii);
   if isnan(norm(:,ii))  %NaN indicates that the link creating the patch has not moved.In this case it makes no sense to offset
      sweep.x(skip+first:skip+last) = patches.x(first:last);
      sweep.y(skip+first:skip+last) = patches.y(first:last);
      sweep.z(skip+first:skip+last) = patches.z(first:last);
   else
      offset = safety_factor(patches.joint(first));
      sweep.x(skip+first:skip+last) = patches.x(first:last)-norm(1,ii)*offset;
      sweep.y(skip+first:skip+last) = patches.y(first:last)-norm(2,ii)*offset;
      sweep.z(skip+first:skip+last) = patches.z(first:last)-norm(3,ii)*offset;
   end
   first = last;
end

end

