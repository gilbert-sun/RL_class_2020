class ValueEstimator():
    def __init__(self):
        with tf.variable_scope('value_estimator'):
            l_input = Input(shape=(NUM_STATES, ))
            l_dense = Dense(16, activation='relu')(l_input)
            state_value = Dense(1, activation='linear')(l_dense)
            value_network = Model(input=l_input, output=state_value)

            graph = self._build_graph(value_network)
            self.state, self.action, self.target, self.state_value, self.minimize, self.loss = graph

    def _build_graph(self, model):
        state = tf.placeholder(tf.float32)
        action = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        target = tf.placeholder(tf.float32, shape=(None))

        state_value = model(state)[0][0]
        loss = tf.squared_difference(state_value, target)

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE_VALUE)
        minimize = optimizer.minimize(loss)

        return state, action, target, state_value, minimize, loss

    def predict(self, sess, state):
        return sess.run(self.state_value, { self.state: [state] })

    def update(self, sess, state, target):
        feed_dict = {self.state:[state], self.target:target}
        _, loss = sess.run([self.minimize, self.loss], feed_dict)
        return loss

    def train(env, sess, policy_estimator, value_estimator, num_episodes, gamma=1.0):

        Step = collections.namedtuple("Step", ["state", "action", "reward"])
        last_100 = np.zeros(100)

        # comment this out for recording
        env = gym.wrappers.Monitor(env, 'cartpole_', force=True)

        for i_episode in range(1, num_episodes + 1):
            state = env.reset()

            episode = []

            while True:
                action_probs = policy_estimator.predict(sess, state)[0]
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(action)

                episode.append(Step(state=state, action=action, reward=reward))

                if done:
                    break

                state = next_state

            loss_p_list = []
            loss_v_list = []

            failed = len(episode) < 500

            for t, step in enumerate(episode):
                total_return = sum(gamma ** i * t2.reward for i, t2 in enumerate(episode[t:]))
                baseline_value = value_estimator.predict(sess, step.state)
                advantage = total_return - baseline_value

                # Update our value estimator. Only update it when the episode failed to use only the accurate value.
                if failed:
                    loss_v = value_estimator.update(sess, step.state, total_return)
                loss_p = policy_estimator.update(sess, step.state, step.action, advantage)

                loss_p_list.append(loss_p)
                loss_v_list.append(loss_v)

            total_reward = sum(e.reward for e in episode)
            last_100[i_episode % 100] = total_reward
            last_100_avg = sum(last_100) / (i_episode if i_episode < 100 else 100)
            avg_loss_p = sum(loss_p_list) / len(loss_p_list)
            avg_loss_v = sum(loss_v_list) / len(loss_v_list)
            print('episode %s p: %s v: %s reward: %d last 100: %f' % (
            i_episode, avg_loss_p, avg_loss_v, total_reward, last_100_avg))

            if last_100_avg >= env.spec.reward_threshold:
                break

        return