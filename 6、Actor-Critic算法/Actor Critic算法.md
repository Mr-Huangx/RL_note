Actor-Critic算法是一种结合`**策略梯度**`和`**时序差分学习**`的强化学习方法。actor即一个策略，通过学习策略获得更高的回报。critic即价值函数，对当前策略进行evaluation。借助critic可以实现单步参数更新，不需要等回合结束才进行。其中出名的：

+ A2C（Advantage actor-critic，A2C）算法。
+ 异步Advantage actor-critic算法。

# 策略梯度回顾
我们通过计算在某个state，采取某个action的概率。然后计算从s采取a之后所能获得的累计奖励（discounted return），以此来计算梯度。如果为正，则增大该动作的概率，如果为负，则减小该状态的概率。

问题：累计奖励是一个变量，他是由agent和env交互获得的，交互本身就是随机的，因此每次采样的结果可能都不同。如果采样次数过少，可能会采到较差的结果，导致训练很差。

# DQN回顾
Q：我们能不能让整个训练过程变得稳定，能不能直接估测随机变量G的期望呢？

我们直接用一个网络去估测在状态$ s $采取动作$ a $时 $ G $的期望值。如果这样是可行的，那么在随后的训练中我们就用期望值代替采样的值，这样就会让训练变得更加稳定。

Q：怎么使用期望代替采样的值呢？

基于value based的方法：DQN。DQN有两种函数，分别计算：state value、action value。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767409632326-aedf2a9b-fb4e-4bb2-b8bd-cc26c05bad05.png)

# Actor Critic算法
将DQN和PG算法结合起来，用DQN辅助PG的训练。

具体而言，就是将原本PG中计算discounted return和baseline一项替换成DQN获取，使得通过采样获取discounted return的方差变小，让模型训练更加稳定的效果。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767410000488-e2e8be4f-5bb5-42cc-89cd-87d393cb2c63.png)

**advantage actor-critic**算法

问题：通过上面的替换，我们需要两个网络，一个计算Q，一个计算V，估计不准的风险翻了两倍，怎么办？

进一步妥协，我们只估计V，用V来表示Q值。即：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767410226352-d7b1940d-4b6a-4d50-8051-cb416aa8c9f3.png)

上面是V和 Q的转换，但仍然比较麻烦，需要计算一个期望，我们更进一步简化，讲期望移除：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767410373452-60e9819e-aeb5-4df3-869a-41c925e9d870.png)

有了上式，我们就可以取代原本的discounted return和baseline，得到：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767410434757-5278b450-9483-4e8e-b62b-3e760bc93a09.png)

相较于最开始的PG算法，我们只引入了一个随机变量r。但相较于原本的G，是一个 非常小的值。因此，也使得模型训练能更加稳定。

## 训练过程
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767410721497-678a4bdd-7ff4-441e-9088-f9f6b5e3805c.png)

算法流程

最开始随机初始化一个policy $ \pi $,使用policy让agent和环境产生交互，获取对应的经验。

+ 对于PG算法，会直接使用数据进行策略更新；
+ 对于Actor-Critic算法，其首先学习value function（可以使用蒙特卡洛、TD算法【推荐】），然后讲学习到的value function带入PG算法，更新策略。
+ 有了新的policy，重复上面的步骤。

## 训练Actor-critic算法技巧
### 需要估计两个网络
+ 价值函数网络V
    - V输入一个状态S，输出对应的价值
+ 策略网络P
    - 输入为一个S，如果action是离散的，输出为一个动作的分布；如果是连续的，输出一个连续的向量。
+ 下图以离散动作为例子：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767424051091-1410c5bd-e1cb-4ae0-bd41-af57e3b529d9.png)

前几层共享参数并不是必须的，但确实可以的。以玩雅达利游戏为例子，输入都是图像，因此前期可以通过卷积神经网络来处理他们，把图像抽象成high level的信息。然后再输入不同的网络获取action和价值。

### 需要探索机制
由于`**critic**`的存在，因此我们希望agent尝试不同的动作，才会把环境探索的比较好，从而得到比较好的结果。

# 异步Advantage actor-critic算法
A2C的问题时训练比较慢，如何提高训练速度呢？

多开几个进程，同时进行学习。流程如下：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767424485703-78504111-6859-4035-bb49-a94bc63a3334.png)

+ step1 : 一开始拥有一个全局网络（global network）。包括policy network和value function network。两个network是绑定在一起的。
+ step2 : 使用多进程进行运行。每个一个进程把global network的参数复制下来。
+ step3 : actor和env进行交互，每个actor进行交互的时候，都手机比较多样的数据。
+ step4 : 交互完成，计算梯度。计算梯度之后，用梯度去更新全局网络的参数。（该过程是将梯度回传给中央的控制中心，中央的控制中心会用这个梯度来更新原来的参数）

<font style="color:#DF2A3F;">问题</font>：step4如果其他worker更新了中央控制中心的参数，此时再进行更新，会不会有问题？

<font style="color:#74B602;">A</font>：没问题，A3C允许使用“过期参数”更新global network。原因是使用的是`**异步SGD**`，梯度更新近似一致，经验上是稳定的。（其实也是训练方便）。与此同时，学习率是比较小的，不会对参数造成太大的影响。梯度本身也是带有噪音的估计，本身就不精确，所以这里有点偏差其实还好。



# 问题
## 简述下A3C，另外A3C是online还是offline policy？
A3C是在A2C的基础上增加了一个异步的概念，即异步训练，旨在使用并行训练的方式，加快模型的训练。其为同策略算法。

## Actor-Critic算法的优缺点
+ 相对于value based的方法，actor - critic算法应用了策略梯度的技巧，使得它能处理连续动作的场景。Q learning对于这件事比较难
+ 相较于policy based的方法，其引入了critic，使得模型可以进行单步更新，而不是回合更新，提升了训练效率。

## Actor-Critic算法中，critic起到了什么作用？
evaluate当前决策的好坏。结合策略模型，当critic认为该动作有益时，策略就更新参数，以增大该动作出现的概率，反之减少该动作出现的概率。

换句话说，critc的存在是辅助policy的学习，提升policy训练的稳定和速度。



