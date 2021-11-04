function CP = get_constrained_factor(factors, constraint)
    CP= factors;
    if nargin>1

        % Left-side Constraint
        if isfield(constraint,'left_constraint')
            CP= cellfun(@(x,y) x*y,constraint.left_constraint,CP , 'UniformOutput',false);
        end
        
        % Right-side Constraint
        if isfield(constraint,'right_constraint')
            CP= cellfun(@(x,y) x*y,CP,constraint.right_constraint, 'UniformOutput',false);

        end
    end
end

