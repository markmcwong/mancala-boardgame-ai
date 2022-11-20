from collections import deque
from Game import Game
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
import os
from agents import MinimaxAgent
from os.path import exists

action_available = 6
state_space_size = (14,)
train_episodes = 500  # An episode a full game
test_episodes = 100
epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start 
max_epsilon = 1  # You can't explore more than 100% of the time
min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
decay = 0.00001
learning_rate = 0.1  # Learning rate
discount_factor = 0.95
MIN_REPLAY_SIZE = 150


class DeepQLearningAgent():
    def __init__(self, is_training=True):
        self.model = None
        self.exploration_rate = epsilon
        self.is_training = is_training

    def get_exploration_rate(self, episode_number):
        if not self.is_training:
            return 0
        else:
            return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode_number)

    def create_model(self, state_shape, action_shape):
        with tf.device('/GPU:0'):
            init = tf.keras.initializers.HeUniform()
            model = keras.Sequential()
            model.add(keras.layers.Dense(24, input_shape=state_shape,
                                         activation='relu', kernel_initializer=init))
            model.add(keras.layers.Dense(
                64, activation='relu', kernel_initializer=init))
            model.add(keras.layers.Dense(
                128, activation='relu', kernel_initializer=init))
            model.add(keras.layers.Dense(
                64, activation='relu', kernel_initializer=init))
            model.add(keras.layers.Dense(action_shape,
                                         activation='linear', kernel_initializer=init))
            model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(
                lr=learning_rate), metrics=['accuracy'])

            self.model = model
            return model

    def exit_training_mode(self):
        self.is_training = False

    def policy(self, game):
        if (self.model is None):
            file_exists = exists('deepQLearning.h5')
            if (file_exists):
                self.model = tf.keras.models.load_model('deepQLearning.h5')
            else:
                self.main()
            self.exit_training_mode()

        board_reshaped = np.array(game.board).reshape(
            [1, state_space_size[0]])
        action_scores = self.model.predict(board_reshaped).flatten()
        actions = game.actions()
        action_scores = [action_scores[i]
                         if i in actions else -500 for i in range(len(action_scores))]

        return np.argmax(action_scores)

    def select_action(self, valid_actions, board, episode_number):
        rate = self.get_exploration_rate(episode_number)
        action = None

        if random.random() <= rate:
            # Explore
            action = random.choice(valid_actions)
        else:
            board_reshaped = np.array(board).reshape(
                [1, state_space_size[0]])
            predicted = self.model.predict(board_reshaped).flatten()
            action = np.argmax(predicted)

        return action

    def train(self, replay_memory, model, target_model, done):

        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64 * 2
        mini_batch = random.sample(replay_memory, batch_size)

        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = model.predict(current_states)

        new_current_states = np.array(
            [transition[3] for transition in mini_batch])
        future_qs_list = target_model.predict(new_current_states)

        X = []
        Y = []

        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * \
                    np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * \
                current_qs[action] + learning_rate * max_future_q

            X.append(observation)
            Y.append(current_qs)

        model.fit(np.array(X), np.array(Y),
                  batch_size=batch_size, verbose=0, shuffle=True)

    def get_new_state(self, curr_state, action, player_2):

        new_state = Game(curr_state.action(action).board, turn='y')
        reward = new_state.score(
            turn='x') - new_state.score(turn='y')
        done = new_state.is_over()

        if (not new_state.is_over()):
            player_2_board = new_state.action(player_2.policy(
                new_state))
            reward = player_2_board.score(
                turn='x') - player_2_board.score(turn='y')
            done = player_2_board.is_over()

        return new_state, reward, done

    def main(self):
        replay_memory = deque(maxlen=5000)
        print("start model initiation")

        model = self.create_model(state_space_size, action_available)
        target_model = self.create_model(state_space_size, action_available)
        target_model.set_weights(model.get_weights())

        steps_to_update_target_model = 0
        rewards_sum = 0

        player_2 = MinimaxAgent(9)

        print("start model training")
        for episode in range(train_episodes):
            total_training_rewards = 0
            state = Game()
            done = False
            while not done:
                steps_to_update_target_model += 1
                # 2. Explore using the Epsilon Greedy Exploration Strategy
                action = self.select_action(
                    state.actions(), state.board, episode)
                new_state, reward, done = self.get_new_state(
                    state, action, player_2)

                replay_memory.append(
                    [state.board, action, reward, new_state.board, done])

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0 or done:
                    self.train(replay_memory, model, target_model, done)

                state = new_state
                total_training_rewards += reward

                if done:
                    print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
                        total_training_rewards, episode, reward))
                    rewards_sum += reward

                    if steps_to_update_target_model >= 100:
                        # print('Copying main network weights to the target network weights')
                        target_model.set_weights(model.get_weights())
                        steps_to_update_target_model = 0
                    break

        print('average reward: ', rewards_sum/train_episodes)
        self.model = model
        model.save('deepQLearning.h5')
