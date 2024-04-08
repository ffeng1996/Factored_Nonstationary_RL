"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch
import torch.nn as nn
import torch.optim as optim

from utils import helpers as utl


class SAC:
    def __init__(self,
                 args,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 policy_optimiser,
                 policy_anneal_lr,
                 train_steps,
                 optimiser_vae=None,
                 lr=None,
                 eps=None,
                 ):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        # coefficients for mixing the value and entropy loss
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # optimiser
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=0.99)
        self.optimiser_vae = optimiser_vae

        self.lr_scheduler_policy = None
        self.lr_scheduler_encoder = None
        if policy_anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler_policy = optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lam)
            if hasattr(self.args, 'rlloss_through_encoder') and self.args.rlloss_through_encoder:
                self.lr_scheduler_encoder = optim.lr_scheduler.LambdaLR(self.optimiser_vae, lr_lambda=lam)

    def update(self,
               policy_storage,
               encoder_s=None,
               encoder_r=None,
               rlloss_through_encoder=False,  # whether or not to backprop RL loss through encoder
               compute_vae_loss=None,  # function that can compute the VAE loss
               freeze=False
               ):

        # get action values
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]

        if rlloss_through_encoder:
            # re-compute encoding (to build the computation graph from scratch)
            utl.recompute_embeddings(policy_storage, encoder_s, encoder_r, sample=False, update_idx=0,
                                     detach_every=self.args.tbptt_stepsize if hasattr(self.args,
                                                                                      'tbptt_stepsize') else None)

        # update the normalisation parameters of policy inputs before updating
        self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)

        data_generator = policy_storage.feed_forward_generator(advantages, 1)
        for sample in data_generator:

            state_batch, belief_batch, task_batch, \
            actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, value_preds_batch, \
            return_batch, old_action_log_probs_batch, adv_targ, next_state_batch= sample

            if not rlloss_through_encoder:
                state_batch = state_batch.detach()
                if latent_sample_batch is not None:
                    latent_sample_batch = latent_sample_batch.detach()
                    latent_mean_batch = latent_mean_batch.detach()
                    latent_logvar_batch = latent_logvar_batch.detach()

            latent_batch = utl.get_latent_for_policy(args=self.args, latent_sample=latent_sample_batch,
                                                     latent_mean=latent_mean_batch, latent_logvar=latent_logvar_batch
                                                     )

            _,target_value, next_action_log_probs, next_dist_entropy = \
                self.actor_critic.evaluate_actions(state=next_state_batch, latent=latent_batch,
                                                   belief=belief_batch, task=task_batch,
                                                   action=actions_batch)
            next_q_value = return_batch + 0.99 * target_value

            excepted_value,_, action_log_probs, dist_entropy = \
                self.actor_critic.evaluate_actions(state=state_batch, latent=latent_batch,
                                                   belief=belief_batch, task=task_batch,
                                                   action=actions_batch)
            excepted_Q1,excepted_Q2=self.actor_critic.critic_sac(state=state_batch, latent=latent_batch,
                                                   belief=belief_batch, task=task_batch,
                                                   action=actions_batch)

            excepted_new_Q = torch.min(excepted_Q1, excepted_Q2)
            next_value = excepted_new_Q - action_log_probs

            value_loss = torch.nn.MSELoss()(excepted_value, next_value.detach()).mean()
            Q1_loss = torch.nn.MSELoss()(excepted_Q1, next_q_value.detach()).mean()  # J_Q
            Q2_loss = torch.nn.MSELoss()(excepted_Q2, next_q_value.detach()).mean()
            action_loss = (action_log_probs - excepted_new_Q).mean()  # according to original paper

            # --  UPDATE --

            # zero out the gradients
            self.optimiser.zero_grad()
            if rlloss_through_encoder:
                self.optimiser_vae.zero_grad()

            # compute policy loss and backprop

            loss = value_loss + Q1_loss + Q2_loss + action_loss

            # compute vae loss and backprop
            if rlloss_through_encoder:
                loss += self.args.vae_loss_coeff * compute_vae_loss()

            # compute gradients (will attach to all networks involved in this computation)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)
            if encoder_s is not None and encoder_r is not None and rlloss_through_encoder:
                nn.utils.clip_grad_norm_(encoder_s.parameters(), self.args.encoder_max_grad_norm)
                nn.utils.clip_grad_norm_(encoder_r.parameters(), self.args.encoder_max_grad_norm)

            # update
            self.optimiser.step()
            if rlloss_through_encoder:
                self.optimiser_vae.step()

        if (not rlloss_through_encoder) and (self.optimiser_vae is not None):
            for _ in range(self.args.num_vae_updates):
                compute_vae_loss(update=True)

        if self.lr_scheduler_policy is not None:
            self.lr_scheduler_policy.step()
        if self.lr_scheduler_encoder is not None:
            self.lr_scheduler_encoder.step()

        #update target
        for target_param, param in zip(self.actor_critic.critic_layers2.parameters(), self.actor_critic.critic_layers.parameters()):
            target_param.data.copy_(target_param * (1 - 0.01) + param * 0.01)
        for target_param, param in zip(self.actor_critic.critic_linear2.parameters(), self.actor_critic.critic_linear.parameters()):
            target_param.data.copy_(target_param * (1 - 0.01) + param * 0.01)

        return value_loss, action_loss, dist_entropy, loss

    def act(self, state, latent, belief, task, deterministic=False):
        return self.actor_critic.act(state=state, latent=latent, belief=belief, task=task, deterministic=deterministic)
