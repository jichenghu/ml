# Convergence Analysis of Multistep RL Algorithm

[paper in ������������ֹ��̡�](./convergence.pdf)

***

Agent �뻷������������Զ���ѧϰ����Ѳ���ʹ Rewards ���

ǿ��ѧϰ������ Agent �����ȷ���ߣ������� Agent �������뻷������������ͨ��ѧϰ��ȡģ��֪ʶ���Ż����ߡ�
�������£�

1) ��ȡ����״̬ s
2) ��Ե�ǰ����״̬�����ݵ�ǰ����ȷ������ a
3) ִ�� a �󻷾�״̬�� s Ǩ�Ƶ� s*, ���� Rewards ֵ R
4) ������ {s, a, s*, R} ������ Agent ��Ϊѧϰ����
5) ѭ�� 1-4 ��ѧϰ���̣�ֱ�����������ж�

## 1 Terminology


### 1.1 Markov ����

**[DEF]** �� $ \lbrace X(t), t \in T \rbrace $ �Ƕ����ڸ��ʿռ� $ (\Omega, F, P) $ �ϣ�ȡֵ�� $ E $ �е�������̡�
��������������� $ n $��$ t_1 < t_2 < \cdots < t_n < t_{n+1} $�� $ t_k \in T (k=1, 2, \dots, n+1) $
��״̬ $ x_1 $, $ x_2 $, $ \dots $, $ x_n $, $ x_{n+1} \in E $������

$$
P \lbrace X(t_{n+1}) \leq x_{n+1} \vert X(t_1) = x_1, X(t_2) = x_2, \dots, X(t_n) = x_n  \rbrace 
= P \lbrace X(t_{n+1} \leq x_{n+1} \vert X(t_n) = x_n \rbrace
$$

��ƹ��� $ X(t) , t\in T $ Ϊ Markov ���̡�

**[ע1]** ��һ��״̬�Ĳ���ֻ�͵�ǰ״̬�йأ�����ʷ�޹�(���޺�Ч��)��

**[ע2]** �������ֲ�������ʾ�ȼ��� $ F \left( x_{n+1}; t_{n+1} \vert x_1, x_2, \dots, x_n; t_1, t_2, \dots, t_n  \right) 
= F\left( x_{n+1}; t_{n+1} \vert x_n ; t_n \right) $

**[ע3]** �������ֲ��ܶȱ�ʾ�ȼ��� $ f \left( x_{n+1}; t_{n+1} \vert x_1, x_2, \dots, x_n; t_1, t_2, \dots, t_n  \right) 
= f\left( x_{n+1}; t_{n+1} \vert x_n ; t_n \right) $

### 1.2 Markov ��

**[DEF]** �� $ \lbrace X(n), n = 0, 1, 2, \dots \rbrace $ �Ƕ����ڸ��ʿռ� $ (\Omega, F, P) $ ��
ȡֵ�ڿ����� $E$ �ϵ�������С����������Ǹ����� $n$ ��״̬ $ i_0, i_1, i_2, \dots, i_{n+1} \in E $��
���� $ P \lbraceX_{n+1}=i_{n+1} \vert X_0 = i_0 , X_1 = i_1, \dots, X_n = i_n \rbrace = P \lbrace X_{n+1} = i_{n+1} \vert X_n = i_n \rbrace $
���������� $ \lbrace X(n), n = 0, 1, 2, \dots \rbrace $ Ϊ Markov ������Ϊ 
$ \lbrace X(n), n \geq 0 \rbrace $

### 1.3 Markov Decision Process













