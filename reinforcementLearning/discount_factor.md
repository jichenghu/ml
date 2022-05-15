# Understanding the role of the discount factor in reinforcement learning

<https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning>

***

I'm teaching myself about reinforcement learning, and trying to understand the concept of 
discounted reward. So the reward is necessary to tell the system which state-action pairs are 
good, and which are bad. But what I don't understand is why the discounted reward is necessary. 
Why should it matter whether a good state is reached soon rather than later?

I do understand that this is relevant in some specific cases. For example, if you are using 
reinforcement learning to trade in the stock market, it is more beneficial to make profit sooner 
rather than later. This is because having that money now allows you to do things with that money 
now, which is more desirable than doing things with that money later.

But in most cases, I don't see why the discounting is useful. For example, let's say you wanted a 
robot to learn how to navigate around a room to reach the other side, where there are penalties if it 
collides with an obstacle. If there was no discount factor, then it would learn to reach the other 
side perfectly, without colliding with any obstacles. It may take a long time to get there, but it 
will get there eventually.

But if we give a discount to the reward, then the robot will be encouraged to reach the other side 
of the room quickly, even if it has to collide with objects along the way. This is clearly not a 
desirable outcome. Sure, you want the robot to get to the other side quickly, but not if this means 
that it has to collide with objects along the way.

So my intuition is that any form of discount factor, will actually lead to a sub-optimal solution. 
And the choice of the discount factor often seems arbitrary -- many methods I have seen simply set 
it to 0.9. This appears to be very naive to me, and seems to give an arbitrary trade-off between the 
optimum solution and the fastest solution, whereas in reality this trade-off is very important.

Please can somebody help me to understand all this? Thank you :)



