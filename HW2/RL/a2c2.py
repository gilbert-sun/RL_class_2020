class PolicyEstimator():
    def __init__(self):
        with tf.variable_scope('PolicyEstimator'):
            l_input = Input(shape=(NUM_STATES, ))
            l_dense = Dense(20, activation='relu')(l_input)
            action_probs = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
            model = Model(inputs=[l_input], outputs=[action_probs])

            self.state, self.action, self.target, self.action_probs, self.minimize, self.loss = self._build_graph(model)

    def _build_graph(self, model):
        state = tf.placeholder(tf.float32)
        action = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        target = tf.placeholder(tf.float32, shape=(None))

        action_probs = model(state)
        log_prob = tf.log(tf.reduce_sum(action_probs * action))
        loss = -log_prob * target

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        minimize = optimizer.minimize(loss)

        return state, action, target, action_probs, minimize, loss

    def predict(self, sess, state):
        return sess.run(self.action_probs, { self.state: [state] })

    def update(self, sess, state, action, target):
        feed_dict = {self.state:[state], self.target:target, self.action:to_categorical(action, NUM_ACTIONS)}
        _, loss = sess.run([self.minimize, self.loss], feed_dict)
        return loss

def train(env, sess, estimator_policy, num_episodes, gamma=1.0):

    Step = collections.namedtuple("Step", ["state", "action", "reward"])
    last_100 = np.zeros(100)

    ## comment this out for recording
    env = gym.wrappers.Monitor(env, 'cartpole_', force=True)

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()

        episode = []

        while True:
            action_probs = estimator_policy.predict(sess, state)[0]
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            episode.append(Step(state=state, action=action, reward=reward))

            if done:
                break

            state = next_state

        loss_list = []

        for t, step in enumerate(episode):
            target = sum(gamma ** i * t2.reward for i, t2 in enumerate(episode[t:]))
            loss = estimator_policy.update(sess, step.state, step.action, target)
            loss_list.append(loss)

        # log
        total_reward = sum(e.reward for e in episode)
        last_100[i_episode % 100] = total_reward
        last_100_avg = sum(last_100) / (i_episode if i_episode < 100 else 100)
        avg_loss = sum(loss_list) / len(loss_list)
        print('episode %s avg_loss %s reward: %d last 100: %f' % (i_episode, avg_loss, total_reward, last_100_avg))

        if last_100_avg >= env.spec.reward_threshold:
            break

    return