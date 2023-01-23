import gym
import argparse
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf

from policy.sac import SoftActorCritic
from policy.replay_buffer import ReplayBuffer

from model_est.min_n_suff_set import min_n_suff_set_state, min_n_suff_set_theta
from utils.hyper_param import default_hps


def infer_theta(change_point, n_epoch, hps, model, dh, save_p):
    N_data = 50 * change_point
    N_batches = int(np.floor(N_data / hps.batch_size))
    sign = 1
    for epoch in range(n_epoch):
        for idx in range(N_batches):
            step = model.sess.run(model.global_step)
            curr_learning_rate = \
                (hps.learning_rate - hps.min_learning_rate) * hps.decay_rate ** step + hps.min_learning_rate
            batch_obs, batch_action, batch_reward = dh.next_batch()
            action_init = np.zeros((hps.batch_size, 1, hps.action_size))
            batch_action_prev = np.concatenate((action_init, batch_action[:, :-1, :]), axis=1)
            reward_init = np.zeros((hps.batch_size, 1, hps.reward_size))
            batch_reward_prev = np.concatenate((reward_init, batch_reward[:, :-1, :]), axis=1)

            batch_obs = np.reshape(batch_obs, (hps.batch_size, hps.max_seq_len, 128, 128, 1))
            feed = {model.input_x: batch_obs,
                    model.input_a_prev: batch_action_prev,
                    model.input_a: batch_action,
                    model.input_r_prev: batch_reward_prev,
                    model.input_r: batch_reward,
                    model.lr: curr_learning_rate,
                    model.seq_length: hps.max_seq_len,
                    model.input_sign: sign}
            (total_loss,
             vae_loss, vae_r_obs_loss, vae_r_next_obs_loss, vae_r_reward_loss, vae_kl_loss, vae_causal_filter_loss,
             transition_loss, causal_filter_loss, state, _) \
                = model.sess.run([model.total_loss,
                                  model.vae_loss,
                                  model.r_obs_loss,
                                  model.r_next_obs_loss,
                                  model.r_reward_loss,
                                  model.kl_loss,
                                  model.vae_causal_filter_loss,
                                  model.transition_loss,
                                  model.causal_filter_loss,
                                  model.final_state,
                                  model.train_op], feed)
            if vae_loss < 50:
                sign = 0
            if step % 100 == 0:
                SSL_A, SSL_B, SSL_C, SSL_D, SSL_E, SSL_F, theta_o, theta_s, theta_r = model.sess.run(
                    [model.SSL_A, model.SSL_B, model.SSL_C,
                     model.SSL_D, model.SSL_E, model.SSL_F,
                     model.theta_o, model.theta_s, model.theta_r])
                print("theta_o:", theta_o)
                print("theta_r:", theta_r)
                print("theta_s:", theta_s)
            output_log = "step: %d (Epoch: %d idx: %d), " \
                         "total_loss: %.4f, " \
                         "vae_loss: %.4f, " \
                         "vae_r_obs_loss: %.4f, " \
                         "vae_r_next_obs_loss: %.4f, " \
                         "vae_r_reward_loss: %.4f, " \
                         "vae_kl_loss: %.4f, " \
                         "vae_causal_filter_loss: %.4f, " \
                         "transition_loss: %.4f, " \
                         "causal_filter_loss: %.4f," \
                         % (step,
                            epoch,
                            idx,
                            total_loss,
                            vae_loss, vae_r_obs_loss, vae_r_next_obs_loss,
                            vae_r_reward_loss, vae_kl_loss, vae_causal_filter_loss,
                            transition_loss,
                            causal_filter_loss)
            print(output_log)
            f = open(os.path.join(save_p, 'output.txt'), 'a')
            f.write(output_log + '\n')
            f.close()

            save_p_e = os.path.join(save_p, 'epochs')
            if not os.path.exists(save_p_e):
                os.makedirs(save_p_e)

            if step % 500 == 0:
                model.save_json(os.path.join(save_p_e, str(epoch) + '_' + str(step) + 'test.json'))

        model.save_json(os.path.join(save_p, 'test.json'))

tf.keras.backend.set_floatx('float64')

logging.basicConfig(level='INFO')

parser = argparse.ArgumentParser(description='SAC')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--env_name', type=str, default='Half-Cheetah-v3',
                    help='name of the gym environment with version')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--verbose', type=bool, default=False,
                    help='log execution details')
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch sample size for training')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to run backprop in an episode')
parser.add_argument('--start_steps', type=int, default=0,
                    help='number of global steps before random exploration ends')
parser.add_argument('--model_path', type=str, default='../data/models/',
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default=f'{str(datetime.utcnow().date())}-{str(datetime.utcnow().time())}',
                    help='name of the saved model')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--polyak', type=float, default=0.995,
                    help='coefficient for polyak averaging of Q network weights')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')



if __name__ == '__main__':
    args = parser.parse_args()
    hps = default_hps(args.env_name)

    #tf.random.set_seed(args.seed)
    writer = tf.summary.create_file_writer(args.model_path + args.model_name + '/summary')

    # Instantiate the environment.
    env = gym.make(args.env_name)
    env.seed(args.seed)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    theta_space = 20

    # Initialize Replay buffer.
    replay = ReplayBuffer(state_space, theta_space, action_space)

    # Initialize policy and Q-function parameters.
    sac = SoftActorCritic(action_space, writer,
                          learning_rate=args.learning_rate,
                          gamma=args.gamma, polyak=args.polyak)

    # Repeat until convergence
    global_step = 1
    episode = 1
    episode_rewards = []
    sr = np.load('sr.npz')
    ss = np.load('ss.npz')
    reduction_set_s = np.load('reduction_set_s.npz')

    while True:

        # Observe state
        current_state = env.reset()
        current_state *= min_n_suff_set_state(sr, ss)
        theta = infer_theta(episode, args.epochs, hps, sac, dh, args.model_path)
        theta *= min_n_suff_set_theta(reduction_set_s)

        step = 1
        episode_reward = 0
        done = False
        while not done:

            if args.render:
                env.render()

            if global_step < args.start_steps:
                if np.random.uniform() > 0.8:
                    action = env.action_space.sample()
                else:
                    action = sac.sample_action(tf.concat([current_state, theta], 1))
            else:
                action = sac.sample_action(tf.concat([current_state, theta], 1))

            # Execute action, observe next state and reward
            next_state, reward, done, _ = env.step(action)

            episode_reward +=  reward

            # Set end to 0 if the episode ends otherwise make it 1
            # although the meaning is opposite but it is just easier to mutiply
            # with reward for the last step.
            if done:
                end = 0
            else:
                end = 1

            if args.verbose:
                logging.info(f'Global step: {global_step}')
                logging.info(f'current_state: {current_state}')
                logging.info(f'action: {action}')
                logging.info(f'reward: {reward}')
                logging.info(f'next_state: {next_state}')
                logging.info(f'end: {end}')

            # Store transition in replay buffer
            replay.store(current_state, theta, action, reward, next_state, end)

            # Update current state
            current_state = next_state

            step += 1
            global_step += 1


        if (step % 1 == 0) and (global_step > args.start_steps):
            for epoch in range(args.epochs):

                # Randomly sample minibatch of transitions from replay buffer
                current_states, actions, rewards, next_states, ends = replay.fetch_sample(num_samples=args.batch_size)

                # Perform single step of gradient descent on Q and policy
                # network
                critic1_loss, critic2_loss, actor_loss, alpha_loss = sac.train(current_states, actions, rewards, next_states, ends)
                if args.verbose:
                    print(episode, global_step, epoch, critic1_loss.numpy(),
                          critic2_loss.numpy(), actor_loss.numpy(), episode_reward)

                with writer.as_default():
                    tf.summary.scalar("actor_loss", actor_loss, sac.epoch_step)
                    tf.summary.scalar("critic1_loss", critic1_loss, sac.epoch_step)
                    tf.summary.scalar("critic2_loss", critic2_loss, sac.epoch_step)
                    tf.summary.scalar("alpha_loss", alpha_loss, sac.epoch_step)

                sac.epoch_step += 1

                if sac.epoch_step % 1 == 0:
                    sac.update_weights()


        if episode % 1 == 0:
            sac.policy.save_weights(args.model_path + args.model_name + '/model')

        episode_rewards.append(episode_reward)
        # model estimation
        infer_theta(epochs, hps, sac, dh, save_p)
        episode += 1
        avg_episode_reward = sum(episode_rewards[-100:])/len(episode_rewards[-100:])

        print(f"Episode {episode} reward: {episode_reward}")
        print(f"{episode} Average episode reward: {avg_episode_reward}")
        with writer.as_default():
            tf.summary.scalar("episode_reward", episode_reward, episode)
            tf.summary.scalar("avg_episode_reward", avg_episode_reward, episode)
