from collections import deque
from Game import Game
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random

from agents import MinimaxAgent

action_available = 6
state_space_size = (14,)
train_episodes = 300  # An episode a full game
test_episodes = 50
epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start 
max_epsilon = 1  # You can't explore more than 100% of the time
min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
decay = 0.00001
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
        learning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=state_shape,
                                     activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(
            36, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(
            36, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(
            36, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(action_shape,
                                     activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(
            lr=learning_rate), metrics=['accuracy'])
        # model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(
        #     lr=learning_rate), metrics=['accuracy'])
        self.model = model
        return model

    def exit_training_mode(self):
        self.is_training = False

    def policy(self, game):
        if(self.model is None):
            # self.main()
            self.model = tf.keras.models.load_model('deepQLearning.h5')
            self.exit_training_mode()

        board_reshaped = np.array(game.board).reshape(
            [1, state_space_size[0]])
        action_scores = self.model.predict(board_reshaped).flatten()
        actions = game.actions()
        action_scores = [action_scores[i]
                         if i in actions else -100 for i in range(len(action_scores))]
        # print(action_scores, actions)
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
        learning_rate = 0.1  # Learning rate
        discount_factor = 0.8

        MIN_REPLAY_SIZE = 200

        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64 * 2
        mini_batch = random.sample(replay_memory, batch_size)
        # print(mini_batch[0])
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in mini_batch])
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

    def main(self):
        replay_memory = deque(maxlen=5000)
        
        print("start model initiation")
        model = self.create_model(state_space_size, action_available)
        target_model = self.create_model(state_space_size, action_available)
        steps_to_update_target_model = 0

        player_2 = MinimaxAgent(9)

        print("start model training")
        for episode in range(train_episodes):
            total_training_rewards = 0
            observation = Game()
            done = False
            while not done:
                steps_to_update_target_model += 1
                # 2. Explore using the Epsilon Greedy Exploration Strategy
                action = self.select_action(observation.actions(), observation.board, episode)
                
                new_observation = Game(observation.action(action).board, turn='y')
                reward = new_observation.score(
                    turn='x') - new_observation.score(turn='y')
                done = new_observation.is_over()

                if(not new_observation.is_over()):
                    player_2_board = new_observation.action(player_2.policy(
                        new_observation))
                    reward = player_2_board.score(
                        turn='x') - player_2_board.score(turn='y')
                    done = player_2_board.is_over()
                # print(observation.board, "action : ",
                #       action, new_observation.board, "reward : ", reward, observation.actions())

                replay_memory.append(
                    [observation.board, action, reward, player_2_board.board, done])

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0 or done:
                    self.train(replay_memory, model, target_model, done)

                observation = player_2_board
                total_training_rewards += reward

                if done:
                    print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))

                    if steps_to_update_target_model >= 100:
                        print('Copying main network weights to the target network weights')
                        target_model.set_weights(model.get_weights())
                        steps_to_update_target_model = 0
                    break
        
        self.model = model
        model.save('deepQLearning.h5')
