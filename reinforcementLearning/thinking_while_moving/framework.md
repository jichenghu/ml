# RL Framework


## Index

1. [��ؿ�ܴ��뵼�������](#codes_review) \
   [1.1 �ؼ���ѵ������](#key_train) \
   [1.2 ѵ������Ĺؼ�����](#key_steps_of_train)
2. []()


## 1. ��ؿ�ܴ��뵼������� <span id="codes_review"> </span>

�������ǱȽ�2�ݴ��룺
1. ycl ����
2. sfujim ���룺https://github.com/sfujim/TD3

��ܴ����Ӳ�������ǣ�
1. ��ͬʱʹ�ö��ֲ�ͬ�㷨
2. �ܽ���ʹ�ö��ֲ�ͬ�㷨
3. ������ѡȡ��ǰ�Ĳ����㷨
4. �ܶԲ���ʵʩ����(the strategy of policies choice)

### 1.1 �ؼ���ѵ������ <span id="key_train"> </span>

ycl ���� algorithm/td3.py ln61:
```python
    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            noise=torch.normal(0, self.policy_noise, size=action.size()).clamp(
                -self.noise_clip, self.noise_clip).to(self.device)

            next_action = (self.target_model.policy(next_obs) + noise).clamp(
                -1, 1)
            target_Q1, target_Q2 = self.target_model.value(
                next_obs, next_action)
            target_Q = torch.minimum(target_Q1, target_Q2)

            target_Q = reward + (1 - terminal) * self.gamma * target_Q

        cur_Q1, cur_Q2 = self.model.value(obs, action)
        critic_loss = F.mse_loss(cur_Q1, target_Q) + F.mse_loss(cur_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
```

sfujim ���� TD3.py ln104:
```python
	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

�����ǹؼ��Ķ�����1����ȡ `next_action`; 2) ����Ŀ��� `Q-value`����Ȼ������Ҫ�Ĳ��衣
ycl ���� algorithm/td3.py ln66:
```python
            next_action = (self.target_model.policy(next_obs) + noise).clamp(
                -1, 1)
            target_Q1, target_Q2 = self.target_model.value(
                next_obs, next_action)
```

sfujim ���� TD3.py ln116:
```python
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
```
������ǽ�����׷���������ؼ����衣����֪�������� actor-critic algorithm ��ؼ�����������
���μ��Ҹ��˵ļ����ĵ��� 7.1.2 Stochastic Actor-Critic Algorithms �йص�˵������
actor_target �������ǰ���ǲ����ݶȣ����õ��ǻغ�(episode)���¡�
critic_target ��ǰ������ q-learning ��ֵ�������ܵ������¡�

**��������Ҫ��������Ū��� critic_target �� critic ����֮��Ĺ�ϵ��**



[Back to index](#index)


### 1.2 ѵ������Ĺؼ����� <span id="key_steps_of_train"> </span>




[Back to index](#index)


