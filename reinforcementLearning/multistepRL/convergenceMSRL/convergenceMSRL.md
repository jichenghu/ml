# Convergence Analysis of Multistep RL Algorithm

[paper in 《计算机与数字工程》](./convergence.pdf)

***

Agent 与环境交互，如何自动地学习到最佳策略使 Rewards 最大。

强化学习不告诉 Agent 如何正确决策，而是由 Agent 自身在与环境交互过程中通过学习获取模型知识并优化决策。
步骤如下：

1) 获取环境状态 s
2) 针对当前环境状态，依据当前策略确定动作 a
3) 执行 a 后环境状态由 s 迁移到 s*, 计算 Rewards 值 R
4) 将上述 {s, a, s*, R} 反馈给 Agent 作为学习样本
5) 循环 1-4 的学习过程，直到满足收敛判定

## 1 Terminology


### 1.1 Markov 过程

**[DEF]** 设 $ \lbrace X(t), t \in T \rbrace $ 是定义在概率空间 $ (\Omega, F, P) $ 上，取值于 $ E $ 中的随机过程。
如果对任意正整数 $ n $，$ t_1 < t_2 < \cdots < t_n < t_{n+1} $， $ t_k \in T (k=1, 2, \dots, n+1) $
及状态 $ x_1 $, $ x_2 $, $ \dots $, $ x_n $, $ x_{n+1} \in E $，都有

$$
P \lbrace X(t_{n+1}) \leq x_{n+1} \vert X(t_1) = x_1, X(t_2) = x_2, \dots, X(t_n) = x_n  \rbrace 
= P \lbrace X(t_{n+1} \leq x_{n+1} \vert X(t_n) = x_n \rbrace
$$

则称过程 $ X(t) , t\in T $ 为 Markov 过程。

**[注1]** 下一个状态的产生只和当前状态有关，与历史无关(即无后效性)。

**[注2]** 用条件分布函数表示等价于 $ F \left( x_{n+1}; t_{n+1} \vert x_1, x_2, \dots, x_n; t_1, t_2, \dots, t_n  \right) 
= F\left( x_{n+1}; t_{n+1} \vert x_n ; t_n \right) $

**[注3]** 用条件分布密度表示等价于 $ f \left( x_{n+1}; t_{n+1} \vert x_1, x_2, \dots, x_n; t_1, t_2, \dots, t_n  \right) 
= f\left( x_{n+1}; t_{n+1} \vert x_n ; t_n \right) $

### 1.2 Markov 链

**[DEF]** 设 $ \lbrace X(n), n = 0, 1, 2, \dots \rbrace $ 是定义在概率空间 $ (\Omega, F, P) $ 上
取值于可数集 $E$ 上的随机序列。如果对任意非负整数 $n$ 及状态 $ i_0, i_1, i_2, \dots, i_{n+1} \in E $，
都有 $ P \lbraceX_{n+1}=i_{n+1} \vert X_0 = i_0 , X_1 = i_1, \dots, X_n = i_n \rbrace = P \lbrace X_{n+1} = i_{n+1} \vert X_n = i_n \rbrace $
则称随机序列 $ \lbrace X(n), n = 0, 1, 2, \dots \rbrace $ 为 Markov 链，记为 
$ \lbrace X(n), n \geq 0 \rbrace $

### 1.3 Markov Decision Process













