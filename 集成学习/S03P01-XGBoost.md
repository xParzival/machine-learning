# XGboost
XGboost是现在效果最好，应用最广泛的boosting方法。相比GBDT它加入了正则化项、对缺失数据做处理等优点，让它非常流行
## 基本原理
XGboost算法的思想跟一般的boosting算法相同，也是一个利用前向分步算法的加法模型，每一轮迭代生成一棵回归树使当前损失函数最小，经过多次迭代最小化整体损失函数。与GBDT不同的是，XGboost在最小化目标中添加了正则化项。设共经历$T$轮迭代，集成分类器表达为
$$
f(x)=f_T(x)=\sum_{t=1}^TT(x;\theta_t)
$$
可以看到，某个样本$x_i$的预测值是其在每一轮决策树$T(x_i;\theta_t)$上预测值的总和，所以XGboost也相当于每一轮去减少预测值与真实目标值之间的偏差，通过一轮一轮的迭代趋近于真实值
目标函数定义为损失函数加上正则化项，于是在每一轮
$$
\begin{aligned}
Obj(t)&=L(y,f_t(x))+\Omega(f_t(x))\\
&=\sum_{i=1}^NL(y_i,f_t(x_i))+\Omega(f_t(x))
\end{aligned}
$$
直接最小化目标函数当然是可以的，但是这样就和非集成学习的方法无异。而我们要用boosting思想来解决问题，也就是要找一个方法把之前的迭代与当前轮的迭代联系起来，XGboost采用将损失函数进行二阶泰勒展开的方式。根据泰勒公式有
$$
f(x+\Delta x)\approx f(x)+f^{'}(x)\Delta x+{1\over2}f^{''}(x)\Delta x^2
$$
于是损失函数可以展开为
$$
\begin{aligned}
L(y_i,f_t(x_i))&=L(y_i,f_{t-1}(x_i)+T(x_i;\theta_t))\\
&\approx L(y_i,f_{t-1}(x_i))+\frac{\partial L(y,f(x))}{\partial f(x)}\bigg|_{f(x)=f_{t-1}(x_i)}T(x_i;\theta_t)+{1\over2}\frac{\partial^2L(y,f(x))}{\partial^2f(x)}\bigg|_{f(x)=f_{t-1}(x_i)}T^2(x_i;\theta_t)
\end{aligned}
$$
令
$$
\begin{gathered}
g_{ti}=\frac{\partial L(y,f(x))}{\partial f(x)}\bigg|_{f(x)=f_{t-1}(x_i)}\\
h_{ti}=\frac{\partial^2L(y,f(x))}{\partial^2f(x)}\bigg|_{f(x)=f_{t-1}(x_i)}
\end{gathered}
$$
目标函数可写为
$$
Obj(t)=\sum_{i=1}^NL(y_i,f_{t-1}(x_i))+\sum_{i=1}^Ng_{ti}T(x_i;\theta_t)+\sum_{i=1}^N{1\over2}h_{ti}T^2(x_i;\theta_t)+\Omega(f_t(x))
$$
对于正则项，它是当前已经生成的所有树正则化项的总和，即
$$
\begin{aligned}
\Omega(f_t(x))&=\sum_{s=1}^t\Omega(T(x;\theta_s))\\
&=\sum_{s=1}^{t-1}\Omega(T(x;\theta_s))+\Omega(T(x;\theta_t))
\end{aligned}
$$
对于第$t$轮训练完成的回归树，设其有$M$个叶结点，每个叶结点记作$R_{tm}$，对应的取值为$c_{tm}$，即
$$
T(x;\theta_t)=\sum_{m=1}^Mc_{tm}I(x\in R_{tm})
$$
正则化项就是树的复杂度，用以降低过拟合，我们希望树的结构越简单越好。那么一方面希望叶结点个数$M$少；一方面希望叶结点对应的取值$c_{tm}$小。为什么$c_{tm}$要尽量小呢？其一，泰勒展开式近似的条件就是$\Delta x$足够小，反应到损失函数中就是$c_{tm}$足够小；其二，$c_{tm}$小本身就能一定程度降低过拟合，比如某个样本目标值为1，那么可以前一棵树拟合值为10001，后一棵树拟合值为-10000，按这种方法理论上任意值都可以找到完美拟合方式，于是可以学习到在训练集上准确率100%的集成学习器，显然过拟合严重，而限制$c_{tm}$大小可以降低这种风险。因此一棵树的正则化项可以表达为
$$
\Omega(T(x;\theta_t))={1\over2}\lambda\sum_{m=1}^Mc_{tm}^2+\gamma M
$$
$0\lt\lambda,\gamma\lt1$，是用于控制我们对树复杂度的容忍程度的参数
因为前$t-1$棵树已经训练完成了，因此其损失以及其正则化项都是常数，因此第$t$轮的目标函数为
$$
\begin{aligned}
Obj(t)&=\sum_{i=1}^Ng_{ti}T(x_i;\theta_t)+\sum_{i=1}^N{1\over2}h_{ti}T^2(x_i;\theta_t)+{1\over2}\lambda\sum_{m=1}^Mc_{tm}^2+\gamma M+\text{constant}\\
&=\sum_{i=1}^Ng_{ti}\sum_{m=1}^Mc_{tm}I(x_i\in R_{tm})+\sum_{i=1}^N{1\over2}h_{ti}\sum_{m=1}^Mc_{tm}^2I(x_i\in R_{tm})+{1\over2}\lambda\sum_{m=1}^Mc_{tm}^2+\gamma M+\text{constant}\\
&=\sum_{m=1}^M\sum_{i=1}^Ng_{ti}c_{tm}I(x_i\in R_{tm})+\sum_{m=1}^M\sum_{i=1}^N{1\over2}h_{ti}c_{tm}^2I(x_i\in R_{tm})+{1\over2}\lambda\sum_{m=1}^Mc_{tm}^2+\gamma M+\text{constant}\\
&=\sum_{m=1}^M\sum_{x_i\in R_{tm}}(g_{ti}c_{tm}+{1\over2}h_{ti}c_{tm}^2)+{1\over2}\lambda\sum_{m=1}^Mc_{tm}^2+\gamma M+\text{constant}
\end{aligned}
$$
令$G_{tm}=\sum\limits_{x_i\in R_{tm}}g_{ti},H_{tm}=\sum\limits_{x_i\in R_{tm}}h_{ti}$，即回归树某个叶结点样本的一阶导数或二阶导数之和。因为是最小化，所以可省去常数项，则
$$
\begin{aligned}
Obj(t)&=\sum_{m=1}^M(G_{tm}c_{tm}+{1\over2}H_{tm}c_{tm}^2)+{1\over2}\lambda\sum_{m=1}^Mc_{tm}^2+\gamma M\\
&=\sum_{m=1}^M[G_{tm}c_{tm}+({1\over2}H_{tm}+{1\over2}\lambda)c_{tm}^2]+\gamma M
\end{aligned}
$$
要求使目标函数最小的$c_{tm}$，那么对$c_{tm}$求导令其为零
$$
\begin{gathered}
\frac{\partial Obj(t)}{\partial c_{tm}}=G_{tm}+(H_{tm}+\lambda)c_{tm}=0\\
\Rightarrow c_{tm}=-\frac{G_{tm}}{H_{tm}+\lambda}
\end{gathered}
$$
于是我们得到了回归树每个叶结点的拟合值，这就是XGboost每一轮的回归树拟合目标。将其带入目标函数得
$$
Obj(t)=-{1\over2}\sum_{m=1}^M\frac{G_{tm}^2}{H_{tm}+\lambda}+\gamma M
$$
至此我们已经得到了回归树最终叶结点的结果
## 回归树分裂算法
因为建树的目的是最小化目标函数，显然回归树的结果会影响目标函数值，上述推导已经找到了能使目标函数最小的叶结点结果，但是还有一个因素能够影响目标函数值，即回归树的结构，回归树的结构决定了会如何就将样本分裂为叶结点。于是下面需要思考如何确定树结构即如何建树。又因为目标函数上式已经给出了计算方式，那么自然想到学习回归树以优化目标函数的方式进行，即每次分裂结点目的都是使目标函数值减小，所以上式又称为回归树的结构分数，反映了回归树结构的好坏
具体来讲，对于回归树中某一个结点$R$，我们要确定分不分裂，传统回归树的分裂指标一般是平方损失函数，在XGboost算法中分裂指标即结构分数，即分裂前后结构分数的变化，如果结构分数下降则分裂，否则不分裂。类似传统回归树，对于数据的每一个特征按取值排序，然后对每一个特征的每一个取值都做如下计算：如果以该特征的该取值为分裂点，设分为左子结点$R_l$和右子结点$R_r$，因为只考虑这一个结点的分裂情况，其余结点都不变，所以分裂前后结构分数的变化也只跟这一个结点有关。则分裂前结构分数为
$$
Obj_1=-{1\over2}\frac{(G_l+G_r)^2}{H_l+H_r+\lambda}+\gamma
$$
分裂后结构分数为
$$
Obj_2=-{1\over2}\left[\frac{G_l^2}{H_l+\lambda}+\frac{G_r^2}{H_r+\lambda}\right]+2\gamma
$$
所以分裂前后结构分数的变化为
$$
\begin{aligned}
\text{Gain}&=Obj_1-Obj_2\\
&={1\over2}\left[\frac{G_l^2}{H_l+\lambda}+\frac{G_r^2}{H_r+\lambda}-\frac{(G_l+G_r)^2}{H_l+H_r+\lambda}\right]-\gamma
\end{aligned}
$$
遍历所有特征的所有取值，选择使$\text{Gain}$最大的特征及其对应取值作为分裂条件。可以看到结构分数也可以作为特征重要性的指标，$\text{Gain}$越大的特征对样本的影响更大。另外，并不是每次都需要分裂，如果$\text{Gain}\gt0$，说明分裂这个结点使目标函数变小，那么可以分裂，反之若$\text{Gain}\lt0$则不分裂。实际使用中一般设定一个$\text{Gain}$的阈值，大于这个阈值就分裂结点，反之不作分裂
### 回归树分裂近似算法
当数据量非常大的时候，采用刚才的建树方法需要的内存非常大，因为需要存储太多一阶和二阶导数值，因此改进出了近似算法，对于每个特征，只考察分位点可以减少计算复杂度。首先根据特征分布的分位数提出候选划分点，然后将连续型特征映射到由这些候选点划分的桶中，然后再找到所有区间的最佳分裂点
对于特征某个特征，根据其分布确定$l$个候选切分点$S=\{s_1,s_2,\cdots,s_l\}$，然后将样本对应到各切分区间中，再对$S$中的每个切分点贪心搜索最低结构分数
分位点的选取有两种策略：
1. global：学习每棵树前就提出候选切分点，并在每次分裂时都采用这种分割
2. local：每次分裂前将重新提出候选切分点

显然local策略需要的计算复杂度更高，但是global的精度不如local。但是当global分位点取得较为精细时也可以达到和local相似的精度。根据实践经验，在分位数取值合理的情况下，分位数策略可以获得与贪心算法相同的精度
另一个问题是分位点的具体计算方法，用如下序列为例来介绍
给定一组数据
$$
\text{input}:14,19,3,15,4,6,1,13,13,7,11,8,4,5,15,2
$$
排序后的序列为
$$
\text{sort}:1,2,3,4,4,5,6,7,8,11,13,13,14,15,15,19
$$
那么这个序列长度为$N=16$，可写出这个序列的rank
$$
\text{rank}:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
$$
#### 分位点(quantile)
1. 分位点
   $\phi$分位点为$\text{rank}=\lfloor\phi\times N\rfloor$的元素。比如上述序列的0.25分位点为$\text{rank}=0.25\times16=4$的元素，即4
2. 近似分位点
   给定一个误差$\varepsilon$，近似$\phi$分位点为一组元素，这些元素的rank值满足$\text{rank}\in[\lfloor\phi\times N\rfloor-N\varepsilon,\lfloor\phi\times N\rfloor+N\varepsilon]$，近似分位点可以表达为$\varepsilon$-approximate $\phi$ quantile。比如上述序列的0.1-approximate 0.5 quantile为$\text{rank}\in[6.4,9.6]$即$\text{rank}=\{7,8,9\}$的元素，含义就是允许误差在0.1时近似分位点为7，8，9，选择这三个值作为0.5分位点都算正确
3. quantile summary
   以上两种分位点计算方法必须先对数据进行排序，而如果我们的内存不足以让全部数据排序时，就需要使用$\varepsilon$-approximate quantile summary。$\varepsilon$-approximate quantile summary 是一种数据结构，该数据结构能够以$\varepsilon N$的精度计算任意的分位查询，大致思路如下：不用一次性存入所有数据，而是用一些内部排序的元组存入部分数据，这些元组记录数据的值以及位置信息，通过这些信息就可以支持计算分位点
   总之运用$\varepsilon$-approximate quantile summary，无论多大的数据，我们只要给定查询的rank值，就可以得到误差在$\varepsilon$以内的近似分位点
4. 加权分位点
   加权分位点即序列中的每个元素有相应的权重，那么其加权rank值就是累计的权重，此时的rank值不再是整数。不加权的rank即权重为1的加权rank
#### XGBoost加权形式
观察XGBoost的目标函数
$$
\begin{aligned}
Obj(t)&=\sum_{i=1}^NL(y_i,f_{t-1}(x_i))+\sum_{i=1}^Ng_{ti}T(x_i;\theta_t)+\sum_{i=1}^N{1\over2}h_{ti}T^2(x_i;\theta_t)+\Omega(f_t(x))\\
&=\sum_{i=1}^N\left(g_{ti}T(x_i;\theta_t)+{1\over2}h_{ti}T^2(x_i;\theta_t)\right)+\Omega(f_t(x))+\text{constant}\\
&=\sum_{i=1}^N{1\over2}h_{ti}\left(T^2(x_i;\theta_t)+2\frac{g_{ti}}{h_{ti}}T(x_i;\theta_t)\right)+\Omega(f_t(x))+\text{constant}\\
&={1\over2}\sum_{i=1}^Nh_{ti}\left(T(x_i;\theta_t)+\frac{g_{ti}}{h_{ti}}\right)^2+\Omega(f_t(x))+\text{constant}
\end{aligned}
$$
由上式可以看出，损失函数可以视作目标值为$-\cfrac{g_{ti}}{h_{ti}}$且以$h_{ti}$为样本权重的加权平方误差。因此可以应用加权分位点的方法搜索最优分裂点
设$N$个训练样本第$k$维特征与其对应的二阶导数表示为
$$
D_k=\{(x_{1k},h_{t1}),(x_{2k},h_{t2}),\cdots,(x_{Nk},h_{tN})\}
$$
根据加权分位数的定义，我们可以写出一个计算rank值的函数
$$
r_k(z)=\frac{\sum\limits_{(x,h)\in D_k,x\lt z}h}{\sum\limits_{(x,h)\in D_k}h}
$$
即小于$z$的特征值对应的二阶导数之和占总二阶导数之和的比例，也就是二阶导数归一化后的权重
根据这个排序就可以应用quantile summary计算出一组分位点$S=\{s_{k1},s_{k2},\cdots,s_{kl}\}$，满足
$$
|r_k(s_{kj})-r_k(s_{k(j+1)})|\lt\delta
$$
其中$s_{k1}=\min\limits_ix_{ik},s_{kl}=\max\limits_ix_{ik}$，$\delta$是采样率，即每个分位点之间rank值间距至少为$\delta$，也就是每个分桶的比例约为$\delta$，也意味着最终会得到${1\over\delta}$个分桶
##### Weighted Quantile Sketch
对于每个样本都有相同权重的问题，有quantile sketch算法解决上述寻找分位点组$S=\{s_{k1},s_{k2},\cdots,s_{kl}\}$的问题，但是对于加权数据，quantile sketch算法无法使用，XGBoost采用的是改进后的Weighted Quantile Sketch算法，具体做法参考XGBoost论文的补充部分
## 稀疏感知
实际工程中一般会出现输入值稀疏的情况。比如数据的缺失、one-hot编码都会造成输入数据稀疏。XGBoost在构建树的结点过程中只考虑非缺失值的数据遍历，而为每个结点增加了一个缺省方向，当样本相应的特征值缺失时，可以被归类到缺省方向上，最优的缺省方向可以从数据中学到
至于如何学到缺省值的分支，其实很简单，在利用非缺失值分裂结点之后，分别枚举特征缺失的样本归为左右分支后的增益，选择增益最大的方向作为最优缺省方向。如果在训练时特征没有缺失，而预测数据中缺失，那么默认归为右子树
## 并行学习
XGboost算法总体来说是无法并行的，因为前向分步算法的缘故，导致必须进行串行计算。但是可以进行局部并行化，在树生成过程中，最耗时的一个步骤就是在每次寻找最佳分裂点时都需要对特征的值进行排序。而XGBoost在训练之前会根据特征对数据进行排序，然后保存排序后的索引
通过顺序访问排序后特征的索引，方便进行切分点的查找。多个特征之间互不干涉，可以使用多线程同时对不同的特征进行切分点查找，即特征的并行化处理。在对节点进行分裂时需要选择增益最大的特征作为分裂，这时各个特征的增益计算可以同时进行，这样就能够局部实现分布式或者多线程计算
## XGboost总结
XGboost是一种工程性的集成学习算法框架，实际上基分类器并不一定要是回归树，也支持线性模型。但是线性模型的线性累加本质仍然是一个线性模型，所以一般只用回归树
* 过拟合预防方法：
  1. 算法本身添加了正则化项，已经有一定预防过拟合的能力
  2. 学习率(Shrinkage)：每次迭代后将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。这和GBDT类似
  3. 列采样：借鉴了随机森林的做法，支持列抽样，在迭代时只使用一部分特征来构建模型，不仅能降低过拟合，还能减少计算
  4. 子采样：在迭代时按比例抽取一部分样本来构建模型
  5. 剪枝：对基决策树进行剪枝显然可以降低过拟合风险

XGboost算法优点：
1. 精度高：因为引入了损失函数的二阶导数，其精度有很大提升。加入学习率也能一定程度提高精度
2. 灵活性高：可以自定义损失函数，只要损失函数二阶可导；基学习器不仅支持回归树，还可以使用线性学习器，但是不常用
3. 过拟合：加入正则化项有效预防过拟合
4. 部分并行：增加算法效率，显著减少训练时间
5. 稀疏处理：对缺失值即稀疏的特征有处理方法，可以支持稀疏数据

XGboost算法缺点：
1. 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集
2. 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存