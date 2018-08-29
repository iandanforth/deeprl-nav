class Peel(object):
    def __init__(self, unity_env):
        """
        A wrapper for the Bananas Unity Environment provided by the Udacity
        Deep Learning Nanodegree. Provides compatability with the OpenAI
        Gym framework.
        """
        self.env = unity_env
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size

        self.initial_state = self.reset()
        self.state_size = len(self.initial_state)

    @staticmethod
    def _get_state(env_info):
        state = env_info.vector_observations[0]
        return state

    @staticmethod
    def _get_reward(env_info):
        reward = env_info.rewards[0]
        return reward

    @staticmethod
    def _get_done(env_info):
        done = env_info.local_done[0]
        return done

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode)[self.brain_name]
        return self._get_state(env_info)

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = self._get_state(env_info)
        reward = self._get_reward(env_info)
        done = self._get_done(env_info)
        return (next_state, reward, done, env_info)
