# History of Reinforcement Learning

<http://incompleteideas.net/book/ebook/node12.html>

***

在强化学习发展的历史长河中有着两条主线，二者均历经久远且都硕果累累。
各自独立发展并最终交织在一起演变成现代意义下的强化学习。强化学习在其
发展过程中一直是机器学习研究的热点之一，特别是在其数学基础于上世纪
八十年代后期至九十年代早期取得一些突破性研究进展之后，相关理论研究
和引用研究都获得了快速推进。

**一条主线研究通过试错来进行学习**，其发源于动物学习的心理学研究。
这条主线贯穿在人工智能的一些早期研究工作中并在上世纪八十年代初期
造就了强化学习的再度复兴。

**另一条主线采用值函数和动态规划来研究最优控制问题的解**。大部分
时期该主线并未涉及学习的机制。

尽管两条主线各自独领风骚，但围绕时间差分方法（可用于三连星游戏）进行
研究的第三条小线亦频频显山露水。三条溪水终于在上世纪八十年代后期汇聚
在一起成长为现代意义下的强化学习研究领域。

## 最优控制理论

The thread focusing on trial-and-error learning is the one with 
which we are most familiar and about which we have the most to say 
in this brief history. Before doing that, however, we briefly 
discuss the optimal control thread.

最优控制理论于二十世纪五十年代后期被提出，用于设计一个控制器来驱使
动态系统在时间轴上的行为测度趋向最小化。Richard Bellman 等人在上世纪
五十年代中期提出了一种解决方案，他们采用了推广流行于十九世纪的
Hamilton 与 Jacobi 理论的方法。这一方法使用了动态系统状态与值函数
的概念，或称“最优回馈函数（optimal return function）”，来定义一个
泛函方程，现在被广为人知的 Bellman 方程。通过对该方程的求解来解决最
优控制问题的此类方法成为动态规划(Bellman, 1957a)。而在(Bellman 
1957b)中引入了最优控制问题的离散随机版本，亦称马尔可夫决策过程（
Markovian decision processes，MDP），并于 1960 年 Ron Howard 设计
出 MDP 的策略叠代方法。所有这一切成为施行现代强化学习理论与算法的重要
元素。

最优控制理论描述了一个受控动力学系统或运动过程，好的设计应该是能够从
一类控制方案中寻找最优的那个，使系统的运动在指定时间状态范围内获
得最优性能，减少随机行为（不受控行为）。例如对于长跑运动员，如何
确定在马拉松比赛过程中体力的分配，使平均速度最快，获得最好的名次。
显然这样的问题往往是有数不清的解，而有一个人创造了提出了一种方法，
为最优控制问题的解决方案奠定了基础——Richard Bellman。

## 动态规划

Dynamic programming is widely considered the only feasible way of 
solving general stochastic optimal control problems. It suffers 
from what Bellman called "the curse of dimensionality," meaning 
that its computational requirements grow exponentially with the 
number of state variables, but it is still far more efficient 
and more widely applicable than any other general method. Dynamic 
programming has been extensively developed since the late 1950s, 
including extensions to partially observable MDPs (surveyed by 
Lovejoy, 1991), many applications (surveyed by White, 1985, 1988, 
1993), approximation methods (surveyed by Rust, 1996), and 
asynchronous methods (Bertsekas, 1982, 1983). Many excellent 
modern treatments of dynamic programming are available (e.g., 
Bertsekas, 1995; Puterman, 1994; Ross, 1983; and Whittle, 1982, 
1983). Bryson (1996) provides an authoritative history of optimal 
control.

In this book, we consider all of the work in optimal control also 
to be, in a sense, work in reinforcement learning. We define 
reinforcement learning as any effective way of solving 
reinforcement learning problems, and it is now clear that these 
problems are closely related to optimal control problems, 
particularly those formulated as MDPs. Accordingly, we must 
consider the solution methods of optimal control, such as dynamic 
programming, also to be reinforcement learning methods. Of course, 
almost all of these methods require complete knowledge of the 
system to be controlled, and for this reason it feels a little 
unnatural to say that they are part of reinforcement learning. 
On the other hand, many dynamic programming methods are 
incremental and iterative. Like learning methods, they gradually 
reach the correct answer through successive approximations. As we 
show in the rest of this book, these similarities are far more 
than superficial. The theories and solution methods for the cases 
of complete and incomplete knowledge are so closely related that 
we feel they must be considered together as part of the same 
subject matter.

Let us return now to the other major thread leading to the modern 
field of reinforcement learning, that centered on the idea of 
trial-and-error learning. This thread began in psychology, where 
"reinforcement" theories of learning are common. Perhaps the first 
to succinctly express the essence of trial-and-error learning was 
Edward Thorndike. We take this essence to be the idea that actions 
followed by good or bad outcomes have their tendency to be 
reselected altered accordingly. In Thorndike's words:

Of several responses made to the same situation, those which are 
accompanied or closely followed by satisfaction to the animal 
will, other things being equal, be more firmly connected with the 
situation, so that, when it recurs, they will be more likely to 
recur; those which are accompanied or closely followed by 
discomfort to the animal will, other things being equal, have 
their connections with that situation weakened, so that, when it 
recurs, they will be less likely to occur. The greater the 
satisfaction or discomfort, the greater the strengthening or 
weakening of the bond. (Thorndike, 1911, p. 244)
Thorndike called this the "Law of Effect" because it describes 
the effect of reinforcing events on the tendency to select 
actions. Although sometimes controversial (e.g., see Kimble, 
1961, 1967; Mazur, 1994), the Law of Effect is widely regarded 
as an obvious basic principle underlying much behavior (e.g., 
Hilgard and Bower, 1975; Dennett, 1978; Campbell, 1960; Cziko, 
1995).

The Law of Effect includes the two most important aspects of what 
we mean by trial-and-error learning. First, it is selectional, 
meaning that it involves trying alternatives and selecting among 
them by comparing their consequences. Second, it is associative, 
meaning that the alternatives found by selection are associated 
with particular situations. Natural selection in evolution is a 
prime example of a selectional process, but it is not associative. 
Supervised learning is associative, but not selectional. It is the 
combination of these two that is essential to the Law of Effect 
and to trial-and-error learning. Another way of saying this is 
that the Law of Effect is an elementary way of combining search 
and memory: search in the form of trying and selecting among many 
actions in each situation, and memory in the form of remembering 
what actions worked best, associating them with the situations in 
which they were best. Combining search and memory in this way is 
essential to reinforcement learning.

In early artificial intelligence, before it was distinct from 
other branches of engineering, several researchers began to 
explore trial-and-error learning as an engineering principle. 
The earliest computational investigations of trial-and-error 
learning were perhaps by Minsky and by Farley and Clark, both 
in 1954. In his Ph.D. dissertation, Minsky discussed 
computational models of reinforcement learning and described his 
construction of an analog machine composed of components he 
called SNARCs (Stochastic Neural-Analog Reinforcement 
Calculators). Farley and Clark described another neural-network 
learning machine designed to learn by trial and error. In the 
1960s the terms "reinforcement" and "reinforcement learning" were 
used in the engineering literature for the first time (e.g., 
Waltz and Fu, 1965; Mendel, 1966; Fu, 1970; Mendel and McClaren, 
1970). Particularly influential was Minsky's paper "Steps Toward 
Artificial Intelligence" (Minsky, 1961), which discussed several 
issues relevant to reinforcement learning, including what he 
called the credit assignment problem: How do you distribute 
credit for success among the many decisions that may have been 
involved in producing it? All of the methods we discuss in this 
book are, in a sense, directed toward solving this problem.

The interests of Farley and Clark (1954; Clark and Farley, 1955) 
shifted from trial-and-error learning to generalization and 
pattern recognition, that is, from reinforcement learning to 
supervised learning. This began a pattern of confusion about the 
relationship between these types of learning. Many researchers 
seemed to believe that they were studying reinforcement learning 
when they were actually studying supervised learning. For 
example, neural network pioneers such as Rosenblatt (1962) and 
Widrow and Hoff (1960) were clearly motivated by reinforcement 
learning--they used the language of rewards and punishments--but 
the systems they studied were supervised learning systems 
suitable for pattern recognition and perceptual learning. Even 
today, researchers and textbooks often minimize or blur the 
distinction between these types of learning. Some modern 
neural-network textbooks use the term "trial-and-error" to 
describe networks that learn from training examples because 
they use error information to update connection weights. This is 
an understandable confusion, but it substantially misses the 
essential selectional character of trial-and-error learning.

Partly as a result of these confusions, research into genuine 
trial-and-error learning became rare in the the 1960s and 1970s. 
In the next few paragraphs we discuss some of the exceptions and 
partial exceptions to this trend.

One of these was the work by a New Zealand researcher named John 
Andreae. Andreae (1963) developed a system called STeLLA that 
learned by trial and error in interaction with its environment. 
This system included an internal model of the world and, later, 
an "internal monologue" to deal with problems of hidden state 
(Andreae, 1969a). Andreae's later work (1977) placed more 
emphasis on learning from a teacher, but still included trial 
and error. Unfortunately, his pioneering research was not well 
known, and did not greatly impact subsequent reinforcement 
learning research.

More influential was the work of Donald Michie. In 1961 and 1963 
he described a simple trial-and-error learning system for 
learning how to play tic-tac-toe (or naughts and crosses) called 
MENACE (for Matchbox Educable Naughts and Crosses Engine). It 
consisted of a matchbox for each possible game position, each 
matchbox containing a number of colored beads, a different 
color for each possible move from that position. By drawing a 
bead at random from the matchbox corresponding to the current 
game position, one could determine MENACE's move. When a game 
was over, beads were added to or removed from the boxes used 
during play to reinforce or punish MENACE's decisions. Michie 
and Chambers (1968) described another tic-tac-toe reinforcement 
learner called GLEE (Game Learning Expectimaxing Engine) and a 
reinforcement learning controller called BOXES. They applied 
BOXES to the task of learning to balance a pole hinged to a 
movable cart on the basis of a failure signal occurring only 
when the pole fell or the cart reached the end of a track. 
This task was adapted from the earlier work of Widrow and Smith 
(1964), who used supervised learning methods, assuming 
instruction from a teacher already able to balance the pole. 
Michie and Chambers's version of pole-balancing is one of the 
best early examples of a reinforcement learning task under 
conditions of incomplete knowledge. It influenced much later 
work in reinforcement learning, beginning with some of our own 
studies (Barto, Sutton, and Anderson, 1983; Sutton, 1984). 
Michie has consistently emphasized the role of trial and error 
and learning as essential aspects of artificial intelligence 
(Michie, 1974).

Widrow, Gupta, and Maitra (1973) modified the LMS algorithm of 
Widrow and Hoff (1960) to produce a reinforcement learning rule 
that could learn from success and failure signals instead of from training examples. They called this form of learning "selective bootstrap adaptation" and described it as "learning with a critic" instead of "learning with a teacher." They analyzed this rule and showed how it could learn to play blackjack. This was an isolated foray into reinforcement learning by Widrow, whose contributions to supervised learning were much more influential.

Research on learning automata had a more direct influence on the trial-and-error thread leading to modern reinforcement learning research. These are methods for solving a nonassociative, purely selectional learning problem known as the -armed bandit by analogy to a slot machine, or "one-armed bandit," except with  levers (see Chapter 2). Learning automata are simple, low-memory machines for solving this problem. Learning automata originated in Russia with the work of Tsetlin (1973) and has been extensively developed since then within engineering (see Narendra and Thathachar, 1974, 1989). Barto and Anandan (1985) extended these methods to the associative case.

John Holland (1975) outlined a general theory of adaptive systems based on selectional principles. His early work concerned trial and error primarily in its nonassociative form, as in evolutionary methods and the -armed bandit. In 1986 he introduced classifier systems, true reinforcement learning systems including association and value functions. A key component of Holland's classifier systems was always a genetic algorithm, an evolutionary method whose role was to evolve useful representations. Classifier systems have been extensively developed by many researchers to form a major branch of reinforcement learning research (e.g., see Goldberg, 1989; Wilson, 1994), but genetic algorithms--which by themselves are not reinforcement learning systems--have received much more attention.

The individual most responsible for reviving the trial-and-error thread to reinforcement learning within artificial intelligence was Harry Klopf (1972, 1975, 1982). Klopf recognized that essential aspects of adaptive behavior were being lost as learning researchers came to focus almost exclusively on supervised learning. What was missing, according to Klopf, were the hedonic aspects of behavior, the drive to achieve some result from the environment, to control the environment toward desired ends and away from undesired ends. This is the essential idea of trial-and-error learning. Klopf's ideas were especially influential on the authors because our assessment of them (Barto and Sutton, 1981a) led to our appreciation of the distinction between supervised and reinforcement learning, and to our eventual focus on reinforcement learning. Much of the early work that we and colleagues accomplished was directed toward showing that reinforcement learning and supervised learning were indeed different (Barto, Sutton, and Brouwer, 1981; Barto and Sutton, 1981b; Barto and Anandan, 1985). Other studies showed how reinforcement learning could address important problems in neural network learning, in particular, how it could produce learning algorithms for multilayer networks (Barto, Anderson, and Sutton, 1982; Barto and Anderson, 1985; Barto and Anandan, 1985; Barto, 1985, 1986; Barto and Jordan, 1987).

We turn now to the third thread to the history of reinforcement learning, that concerning temporal-difference learning. Temporal-difference learning methods are distinctive in being driven by the difference between temporally successive estimates of the same quantity--for example, of the probability of winning in the tic-tac-toe example. This thread is smaller and less distinct than the other two, but it has played a particularly important role in the field, in part because temporal-difference methods seem to be new and unique to reinforcement learning.

The origins of temporal-difference learning are in part in animal learning psychology, in particular, in the notion of secondary reinforcers. A secondary reinforcer is a stimulus that has been paired with a primary reinforcer such as food or pain and, as a result, has come to take on similar reinforcing properties. Minsky (1954) may have been the first to realize that this psychological principle could be important for artificial learning systems. Arthur Samuel (1959) was the first to propose and implement a learning method that included temporal-difference ideas, as part of his celebrated checkers-playing program. Samuel made no reference to Minsky's work or to possible connections to animal learning. His inspiration apparently came from Claude Shannon's (1950) suggestion that a computer could be programmed to use an evaluation function to play chess, and that it might be able to to improve its play by modifying this function on-line. (It is possible that these ideas of Shannon's also influenced Bellman, but we know of no evidence for this.) Minsky (1961) extensively discussed Samuel's work in his "Steps" paper, suggesting the connection to secondary reinforcement theories, both natural and artificial.

As we have discussed, in the decade following the work of Minsky and Samuel, little computational work was done on trial-and-error learning, and apparently no computational work at all was done on temporal-difference learning. In 1972, Klopf brought trial-and-error learning together with an important component of temporal-difference learning. Klopf was interested in principles that would scale to learning in large systems, and thus was intrigued by notions of local reinforcement, whereby subcomponents of an overall learning system could reinforce one another. He developed the idea of "generalized reinforcement," whereby every component (nominally, every neuron) views all of its inputs in reinforcement terms: excitatory inputs as rewards and inhibitory inputs as punishments. This is not the same idea as what we now know as temporal-difference learning, and in retrospect it is farther from it than was Samuel's work. On the other hand, Klopf linked the idea with trial-and-error learning and related it to the massive empirical database of animal learning psychology.

Sutton (1978a, 1978b, 1978c) developed Klopf's ideas further, particularly the links to animal learning theories, describing learning rules driven by changes in temporally successive predictions. He and Barto refined these ideas and developed a psychological model of classical conditioning based on temporal-difference learning (Sutton and Barto, 1981a; Barto and Sutton, 1982). There followed several other influential psychological models of classical conditioning based on temporal-difference learning (e.g., Klopf, 1988; Moore et al., 1986; Sutton and Barto, 1987, 1990). Some neuroscience models developed at this time are well interpreted in terms of temporal-difference learning (Hawkins and Kandel, 1984; Byrne, Gingrich, and Baxter, 1990; Gelperin, Hopfield, and Tank, 1985; Tesauro, 1986; Friston et al., 1994), although in most cases there was no historical connection. A recent summary of links between temporal-difference learning and neuroscience ideas is provided by Schultz, Dayan, and Montague (1997).

Our early work on temporal-difference learning was strongly influenced by animal learning theories and by Klopf's work. Relationships to Minsky's "Steps" paper and to Samuel's checkers players appear to have been recognized only afterward. By 1981, however, we were fully aware of all the prior work mentioned above as part of the temporal-difference and trial-and-error threads. At this time we developed a method for using temporal-difference learning in trial-and-error learning, known as the actor-critic architecture, and applied this method to Michie and Chambers's pole-balancing problem (Barto, Sutton, and Anderson, 1983). This method was extensively studied in Sutton's (1984) Ph.D. dissertation and extended to use backpropagation neural networks in Anderson's (1986) Ph.D. dissertation. Around this time, Holland (1986) incorporated temporal-difference ideas explicitly into his classifier systems. A key step was taken by Sutton in 1988 by separating temporal-difference learning from control, treating it as a general prediction method. That paper also introduced the TD($\lambda $) algorithm and proved some of its convergence properties.

As we were finalizing our work on the actor-critic architecture in 1981, we discovered a paper by Ian Witten (1977) that contains the earliest known publication of a temporal-difference learning rule. He proposed the method that we now call tabular TD(0) for use as part of an adaptive controller for solving MDPs. Witten's work was a descendant of Andreae's early experiments with STeLLA and other trial-and-error learning systems. Thus, Witten's 1977 paper spanned both major threads of reinforcement learning research--trial-and-error learning and optimal control--while making a distinct early contribution to temporal-difference learning.

Finally, the temporal-difference and optimal control threads were fully brought together in 1989 with Chris Watkins's development of Q-learning. This work extended and integrated prior work in all three threads of reinforcement learning research. Paul Werbos (1987) contributed to this integration by arguing for the convergence of trial-and-error learning and dynamic programming since 1977. By the time of Watkins's work there had been tremendous growth in reinforcement learning research, primarily in the machine learning subfield of artificial intelligence, but also in neural networks and artificial intelligence more broadly. In 1992, the remarkable success of Gerry Tesauro's backgammon playing program, TD-Gammon, brought additional attention to the field. Other important contributions made in the recent history of reinforcement learning are too numerous to mention in this brief account; we cite these at the end of the individual chapters in which they arise.


## References

Bellman, 1957a
Bellman, R.E. (1957a).
Dynamic Programming.
Princeton University Press, Princeton, NJ.

Bellman, 1957b
Bellman, R.E. (1957b).
A Markov decision process.
Journal of Mathematical Mech., 6:679 - 684.
