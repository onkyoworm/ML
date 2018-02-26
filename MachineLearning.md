数据收集、数据清洗--> 特征选取--> 模型选择--> 效果验证</br>
其实最难的是如何把生活难题变成机器学习问题</br>
k近邻</br>
常用算法： `Brute Force、 K-D Tree、 Ball Tree`</br>
Brute Force</br>
计算预测样本和所有训练样本的距离 然后计算最小的K个(所以叫k近邻啊=。=) 然后做多数表决 做出预判</br>
优点是样本量少 特征少时好用 但实际中往往是特征数 样本非常多的情况下会非常耗时间</br>
K-D Tree</br>

Ball Tree</br>

在使用k近邻检测异常操作时 进行全量比较要比选取最频繁 最不频繁的命令进行测试的结果要好 焱哥书上所言</br>
但是根据win上结果 对Masquerading数据集进行全部测试(20个文件) 会发现有些结果不如对最(不)频繁进行测试好 (小密圈中也指出在win上或ubuntu上面跑sklearn的时候 FreqDist()好像是没有默认排序的 这么搞笑的吗??????) 仅以记录</br>
大概流程</br>
清洗 数据集里面每一百条作为一个序列 所以把文件分为多个`list`的组合 其中通过`FreqDist().keys()`获取排序后的结果(list来的) 然后通过`set`获取最(不)频繁 (当初有个疑惑 为什么这里要用`set` 一开始以为是为了确保命令之间不重复 但是后来看了后文觉得应该是为后面KNN标量化做准备)</br>
获取特征值 通过之前清洗后获取到的`cmd_list, max, min `然后取`cmd_list`中每个`list`的最频繁和最不频繁和对应的`max`, `min`进行`set`的交集再取交集长度 及可以计算出重叠程度 最后计算每个`list`的长度 返回`[f1, f2, f3]`就是特征值</br>
模型 通过给定的`label`文件()标出正常 异常 通过一个`list`进行记录</br>
    `y=[0]*50+labels`主要是创建一个150长的`list` 因为15000/100 那么多个  `x_train y_train主要是训练数据 x_test y_test主要是测试数据`</br>
训练 	
	
	neigh = KNeighborsClassifier
	(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)
	neigh.fit(x_train, y_train) 进行训练 x为训练数据 y为目标值
	neigh.predict() 通过训练值进行预测(基于上一步的fit)
	np.mean() 计算矩阵均值
另一个则是用全部特征向量进行比较的</br>
数据清洗这里同样是100个为一份 分成150个 只不过不需要用`fdist[0:50] fdist[-50:]`来获取最(不)频繁的命令了 </br>
特征值这里最点睛的一笔就是		
	
	if dist[i] in cmd_list:
		v[i] +=1
通过`FreqDist().keys()`获取到所有排序后的命令 然后检测每一个是否在另外的`cmd_list`中 最后再把结果放到`list`中</br>
后面基本大同小异</br>
</br>
</br>
检测`Rootkit`</br>
可以先看看要用到的KDD99数据集 [kdd99](http://blog.csdn.net/com_stu_zhang/article/details/6987632)</br>
因为数据集本来就已经是挺好的了 所以清洗起来很方便 最后放入到`list`里面就可以了</br>
特征这里的话因为是数据集 所以有说明的 具体是先判断是否是`rootkit` `normal` 以及是否是`telnet` 然后再获取对应的tcp连接内容特征 </br>
注意就是`x1[9:21]`实际获取的内容是哪一段 以及最后特征值要`float`化(为什么?? 测试结果上来看 并没有不同) 还有就是为什么只需要匹配`( x1[41] in ['rootkit.','normal.'] ) and ( x1[2] == 'telnet' )`这个 </br>
后来回去想了想 这次主要检测的是`rootkit` 现在是在做特征匹配 要做的是发现出`rootkit` 以及`normal` 两种情况下的特征区别 因此所以才只单纯的做这两个</br>
最后模型那里还是用的是`KNeighborsClassifier()` 不过和上面不同的是没有调用`*.fit()` 而是直接调用了同样是`sklearn`里面的`cross_validation.cross_val_score()`进行测试</br>
	
	sklearn.cross_validation.cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’)
使用的则是`cross_validation.cross_val_score(clf, x, y, n_jobs=-1, cv=10)`</br>
    `n_jobs` 居然是CPU使用个数 -1为全部</br>
    `cv` 可以简单的认为是几折 </br>
    `estimator` 其实就是用那个算法来fit(上面提到的问题)</br>
    `x,y` 分别就是需要训练 预测的数据
</br>
</br>
</br>
k近邻检测webshell 这个就很有意思了</br>
一开始以为所有文件里面记录的都是文件名字 然后处理方式和上面异常操作类似 都是进行匹配 或者是需要人为的去操作进行分类 然后输出发现都是数字 就很懵逼了 后来打开文件看了看 发现已经是调用顺序了 = = 就说怎么这些数据哪里来的 明明就没有相关操作! 还有一点就是 兜哥`import`了两次`os`库…… 居然没有发现……</br>
另外一个点就是兜哥代码里面没有做win的适配 导致结果有问题 还要改一改=。=</br>
下面讲一下代码</br>
清洗方面主要功夫是放在读取文件上 内容的话如上面说所 已经是调用的序列了</br>
主要用到的函数有两个`load_training` 以及`load_webshell`</br>
函数内部主要是通过`os`模块进行获取文件名 细节上是使用`os.listdir`获取文件名字 `os.path.join(path, dir)`来拼接成完整的路径 最后通过`os.path.isfile()`来判断是否是一个文件</br>
稍微不同的是 `training_file` 和` webshell_files` 的返回值不一样 两者返回的 `x`都是文件里面的内容 不同的是`training`里面返回的`y`是纯0 而`webshell`里面返回的是纯1 当时思考了为什么要这样做 看了看后面使用得是十折交叉验证 就明白了 当时还有一个不明白的地方就是为什么使用的是`词集模型特征化` 代码如下		
	
	x = x1+x2
	y = y1+y2
	vectorizer = CountVectorizer(min_df=1)
	x=vectorizer.fit_transform(x)
	x=x.toarray()
然后回去看了看`词袋` 和`词集`之间的区别:</br>
词袋是词集基础上加了频率的维度 也就是说词袋不仅关心是否出现 还关心出现了多少次 而词集只是关心有没有 </br>
这里我们要做的只是检测是否是`webshell` 所以只关心有没有 而如果检测`webshell做了什么` `哪种类型的webshell` 我想可能就要用到词袋模型了 这个学完后可以好好研究一下</br>
回来继续讲讲代码</br>
	
	sklearn.feature_extraction.text.CountVectorizer
	(input=’content’, encoding=’utf-8’, decode_error=’strict’, strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern=’(?u)\b\w\w+\b’, ngram_range=(1, 1), analyzer=’word’, max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class ‘numpy.int64’>)
这里用到的是`min_df` 主要是频率小于这个 是不会被当作关键词的</br>
    `vectorizer.fit_transform()`是对词汇进行学习并返回一个矩阵</br>
另外一个类似功能是`fit`</br>
问了问兜哥 `fit`只是训练的过程 不会对结果进行转换 所以要使用`fit_transform` 对数组进行转换</br>
后面为什么要用`x=x.toarray()`是个细节问题 刚转换出来的是稀疏矩阵 要显式转换为一个矩阵</br>
	
	vectorizer = CountVectorizer(min_df=1)
    x=vectorizer.fit_transform(x)
    x=x.toarray()

    clf = KNeighborsClassifier(n_neighbors=3)
    scores=cross_validation.cross_val_score(clf, x, y, n_jobs=-1, cv=10)
    print scores
    print np.mean(scores)
</br>
后面的话基本就是常规套路了</br>
随后补充knn的相关</br>
</br>
</br>
</br>
接下来是决策树和随机森林了</br>
决策树其实是非常好理解的一个东西 就像人思考的时候遇到多种条件组合作为判断一样 那什么是随机森林 首先森林是由树组成的 所以里面肯定是需要用到决策树的 那随机是怎么理解呢 打个比方 判断东、西方人 可以通过身高 体重 头发颜色 眼睛颜色进行判断 一个决策树来判断东西方人可以按照身高-->体重-->头发颜色-->眼睛颜色进行判决 另外一个树可以是体重-->身高-->眼睛颜色-->头发颜色来进行判断 当然还有其他的组合 所以说随机就是判决条件的顺序的随机</br>
那么最后怎么判断随机森林结果呢 是通过输出类别的众数来决定的</br>
[决策树](http://blog.csdn.net/l18930738887/article/details/47686519) [随机森林](https://segmentfault.com/a/1190000007463203)</br>
书上的话主要是用决策树检测`POP3`以及`FTP`的暴力破解 随机森林的话也是计算了`FTP`暴力破解</br>
先讲一下`POP3`暴力破解</br>
    `POP3`也是一个用来发送邮件的协议 相关的还有`IMAP`以及`SMTP`</br>
    `POP3` `IMAP`之间的区别主要在客户端 `SMTP` 基于`tcp/ip` 注重的是是否登陆</br>
代码分析</br>
数据清洗这里和之前的`检测rootkit`类似 可能因为用的都是`kdd99`数据集的原因</br>
变化的则是取的值略微不同了 这里用的是`x1[41]`是否是`guess_passwd`以及`normal`还有就是判断`x1[2]`是否是`pop_3`</br>
最终构成的向量里面用到的是`x1 = [x1[0]] + x1[4:8]+x1[22:30]` 分别为`x1[0] 连接持续时间` `x1[4:8] 字节数 是否同一个端口 错误分段数量 加急包个数` `x1[22:30] 基于时间的网络流量统计`</br>
主体程序如下</br>
	
	v=load_kdd99("../data/kddcup99/corrected")
    x,y=get_guess_passwdandNormal(v)
    clf = tree.DecisionTreeClassifier()
    print  cross_validation.cross_val_score(clf, x, y, n_jobs=-1, cv=10)
这里和前面所讲的`rootkit`检测基本一样 变化的只是knn算法变成了 决策树`tree.DecisionTreeClassifier()`算法</br>
另一个则是决策树检测`FTP暴力破解了`</br>
同样 也是基于`ADFA_LD`数据集进行的测试因此函数结构 函数调用都和上面的`webshell`检测处十分的类似</br>
    `dirlist()`函数读取文件名字以及路径 最后再拼接在一起 加载训练数据的函数为`load_adfa_training_files()` 加载爆破数据的是`load_adfa_hydra_ftp_files()`函数 同时 训练数据里`y`记为`0` 爆破数据里`y`记为`1`</br>
核心代码是基本不变的		
	
	x1,y1=load_adfa_training_files("../data/ADFA-LD/Training_Data_Master/")
    x2,y2=load_adfa_hydra_ftp_files("../data/ADFA-LD/Attack_Data_Master/")

    x=x1+x2
    y=y1+y2
    #print x
    vectorizer = CountVectorizer(min_df=1)
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    #print y
    clf = tree.DecisionTreeClassifier()
    print  cross_validation.cross_val_score(clf, x, y, n_jobs=-1, cv=10)
	print np.mean(cross_validation.cross_val_score(clf, x, y, n_jobs=-1, cv=10))
看完之后不由得产生一个问题 `是不是数据本身决定了使用哪种算法?`</br>
虽然只是看了一点点 但是就出现了一种感觉 同一种数据样本下 对数据的清洗以及特征值的选取是十分的类似</br>
先带着这个问题 继续看下去</br>
