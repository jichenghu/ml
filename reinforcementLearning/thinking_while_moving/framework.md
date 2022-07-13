# RL Framework


## Index

1. [相关框架代码导读与分析](#codes_review) \
   [1.1 关键的训练代码](#key_train) \
   [1.2 训练代码的关键步骤](#key_steps_of_train)
2. []()


## 1. 相关框架代码导读与分析 <span id="codes_review"> </span>

这里我们比较2份代码：
1. ycl 代码
2. sfujim 代码：https://github.com/sfujim/TD3

框架代码的硬性需求是：
1. 能同时使用多种不同算法
2. 能交叉使用多种不同算法
3. 能智能选取当前的策略算法
4. 能对策略实施策略(the strategy of policies choice)

### 1.1 关键的训练代码 <span id="key_train"> </span>

ycl 代码 algorithm/td3.py ln61:
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

sfujim 代码 TD3.py ln104:
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

下面是关键的二步：1）获取 `next_action`; 2) 计算目标的 `Q-value`。显然是最重要的步骤。
ycl 代码 algorithm/td3.py ln66:
```python
            next_action = (self.target_model.policy(next_obs) + noise).clamp(
                -1, 1)
            target_Q1, target_Q2 = self.target_model.value(
                next_obs, next_action)
```

sfujim 代码 TD3.py ln116:
```python
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
```
因此我们接下来追踪这两个关键步骤。我们知道这里是 actor-critic algorithm 最关键的两个步骤
（参见我个人的技术文档的 7.1.2 Stochastic Actor-Critic Algorithms 有关的说明）。
actor_target 这个函数前身是策略梯度，采用的是回合(episode)更新。
critic_target 的前身则是 q-learning 的值函数，能单步更新。

**下面我们要从理论上弄清楚 critic_target 与 critic 二者之间的关系。**



[Back to index](#index)


### 1.2 训练代码的关键步骤 <span id="key_steps_of_train"> </span>




[Back to index](#index)


