function [] = computeAcceptanceProbabilities(accept)
% Computes and displays acceptance probabilities of RJ-MCMC run.
% Used to diagnose the mixing of the MCMC run.
update_steps = (accept(:,1) == 2);
update_hyp_steps = (accept(:,1) == 4);
birth_steps = (accept(:,1) == 1);
death_steps = (accept(:,1) == 3)|(accept(:,1) == -1);

update_acceptrate(1) = sum(accept(update_steps,2))/sum(update_steps);
update_acceptrate(2) = sum(accept(update_steps,3))/sum(update_steps);
update_acceptrate(3) = sum(accept(update_steps,4))/sum(update_steps);
update_acceptrate(4) = sum(accept(update_hyp_steps,5))/sum(update_hyp_steps);
update_acceptrate(5) = sum(accept(update_hyp_steps,6))/sum(update_hyp_steps);
birth_acceptrate = sum(accept(birth_steps,2)==1)/sum(birth_steps);
death_acceptrate = sum(accept(death_steps,2)==1)/sum(death_steps);

fprintf('Acceptance probabilities:\nbeta   = %g \nchi    = %g\nlambda = %g\ngamma  = %g\neta    = %g\nbirth  = %g\ndeath  = %g\n\n', [update_acceptrate, birth_acceptrate, death_acceptrate])
end