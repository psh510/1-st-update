import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tensorflow as tf

gpu_config = tf.ConfigProto(device_count={'GPU':1}) #GPU는 한번 사용
gpu_config.gpu_options.allow_growth = #True 메모리가 필요할때 사용
gpu_config.gpu_options.per_process_gpu_memory_fraction = #메모리의 충돌을 50프로로 제한

sess = tf.InteractiveSession(config=gpu_config)
mu = 0.8
sigma=0.1
num_bins = 100
num_samples = 1000000

class GenerativeNetwork:   # Generative(생성자)를 클래스로 구성
    dim_z=1 # z->1차원
    dim_g=1 # g->1차원
    n_hidden = 10 #hidden layer -> 10으로 설정
    learning_rate = 1e-1
    
    def __init__(self):   #초기 설정
        
        rand_uni = tf.random_uniform_initializer(-1e1,1e1) #W와 b를 임의적으로 초기 설정
        
        self.z_input = tf.placeholder(tf.float32,shape = [None,self.dim_z],name = "z-input")
        self.g_target = tf.placeholder(tf.float32,shape=[None,self.dim_g])
        self.W0 = tf.Variable(rand_uni([self.dim_z,self.n_hidden])) #첫번째 Weight 설정
        self.b0 = tf.Variable(rand_uni([self.n_hidden]))
        self.W1 = tf.Variable(rand_uni([self.n_hidden,self.dim_g]))  #두번째 Weight 설정
        self.b1 = tf.Variable(rand_uni([self.dim_g]))
        
        temp =  tf.nn.sigmoid(tf.matmul(self.z_input,self.W0)+self.b0)
        self.g = tf.nn.sigmoid(tf.matmul(temp,self.W1)+self.b1)
        self.loss = tf.losses.mean_squared_error(self.g,self.g_target)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def generate(self,z_i):  #feed_dict를 통하여 z_i가 들어가 run
        z_i = np.reshape(z_i,[-1,self.dim_z])
        g_i = sess.run([self.g],feed_dict={self.z_input : z_i})
        return g_i[0]
    
    def train(self,z_i,g_i):  #train 과정
        error, _ = sess.run([self.loss,self.opt],feed_dict={self.z_input : z_i,self.g_target:g_i})
        return error


class Discriminator:    #Discriminator(판별자)를 class로 구성
    dim_x = 1 # z->1차원
    dim_d = 1 # d->1차원
    num_hidden_neurons = 10
    learning_rate = 0.1
    
    x_input = tf.placeholder(tf.float32,shape=[None,dim_x],name="x_input")
    d_target = tf.placeholder(tf.float32,shape=[None,dim_d],name="d_target")

    rand_uni = tf.random_uniform_initializer(-1e-2,1e-2)  #W와 b를 임의적으로 초기 설정

    W0 = tf.Variable(rand_uni([dim_x,num_hidden_neurons])) #첫번째 Weight를 num_hidden_neurons에 대해 매김
    b0 = tf.Variable(rand_uni([num_hidden_neurons]))
    W1 = tf.Variable(rand_uni([num_hidden_neurons,dim_d])) #두번째 Weight를 첫번째 과정과 변수가 역으로 들어가 매김
    b1 = tf.Variable(rand_uni([dim_d]))

    
    def __init__(self): #초기 dim과 loss와 optimizer 설정
        
        self.d = self.getNetwork(self.x_input)
        self.loss = tf.losses.mean_squared_error(self.d,self.d_target)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,)
        
    def getNetwork(self,input):
        temp = tf.nn.tanh(tf.matmul(input,self.W0)+self.b0)
        return tf.nn.sigmoid(tf.matmul(temp,self.W1)+self.b1)
    
    def discriminate(self,x_i): #feed_dict로부터 input을 받아 d_i return
        d_i = sess.run([self.d],feed_dict={self.x_input:x_i})
        return d_i[0]
    
    def train(self,x_i,d_i): #Discriminator의 train과정
        error,_ = sess.run([self.loss,self.opt],feed_dict={self.x_input:x_i,self.d_target:d_i})
        return error


def draw(x,z,g,D):   # x,z,g,D를 도표로 시각적으로 보여줌   
    #draw histogram
    bins = np.linspace(0,1,num_bins)
    px, _ = np.histogram(x,bins=bins, density = True)
    pz, _ = np.histogram(z,bins=bins, density = True)
    pg,_ = np.histogram(g,bins=bins, density= True)

    v = np.linspace(0,1,len(px))

    v_i = np.reshape(v,(len(v),D.dim_x))
    db = D.discriminate(v_i)
    db = np.reshape(db,len(v))

    l = plt.plot(v,px,'b--',linewidth=1)
    l = plt.plot(v,pz,'r--',linewidth=1)
    l = plt.plot(v,pg,'g--',linewidth=1)
    l = plt.plot(v,db,'k--',linewidth=1)


    plt.title('1D GAN Test')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.show()



print("Hello! GAN!")
sess.run(tf.global_variables_initializer())  #초기 설정
tf.global_variables_initializer().run()
x = np.random.normal(mu,sigma,num_samples)
z = np.random.uniform(0,1,num_samples)
g = np.ndarray(num_samples)

G = GenerativeNetwork()
D = Discriminator()   #각 변수들을 x,z,g,G,D로 설정

# data를 생성
x_i = np.reshape(x,(num_samples,D.dim_x)) 
z_i = np.reshape(z,(num_samples,G.dim_z))

sess.run(tf.global_variables_initializer())
g_i = G.generate(z_i)
g = np.reshape(g_i,(num_samples))

d_x_i = np.ndarray(shape=(num_samples,D.dim_x))
d_x_i.fill(1.0)

d_g_i = np.ndarray(shape = (num_samples,D.dim_x))
d_g_i.fill(0.0)



draw(x,z,g,D)



#Generator로 부터 discrimination을 학습시키는 과정
D_from_g = D.getNetwork(G.g)    # G로쿠터 판별
loss_g = tf.reduce_mean(-tf.log(D_from_g)) # G의 입장에서는 G로쿠터 생성된 D가->1로 판별하는 것 유리
loss_d = tf.reduce_mean(-tf.log(D.d)-tf.log(1-D_from_g)) # D의 입장에서는 G로부터 생성된 것이 0으로 판별하는 것 유리

opt_g = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_g) # G의 입장에서 Gradientoptimizer
opt_d = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_d)# D의 입장에서 Grdientoptimizer



# 좋은 판별을 위하여 Generator를 미리 학습시켜 놓는 과정
print("Pre-traning generator for good distribution")
for tr in range(0,1000,1):
    G.train(z_i,z_i) # uniform generation
    if tr % 100 ==0:
        print("G error=",G.train(z_i,z_i))
    
g_i = G.generate(z_i)
g = np.reshape(g_i,(num_samples))
draw(x,z,g,D)

print("Pre-train Discriminator")
for tr in range(0,500,1):
    D.train(x_i,d_x_i)
    D.train(g_i,d_g_i)
    if tr % 100 ==0:
        print(D.train(x_i,d_x_i))
        print(D.train(g_i,d_g_i))
        
draw(x,z,g,D)


# G와 D를 for안에서 동시에 학습 되는 과정
for tr in range(0,10000,1):        
    # generate g from z again to respond the training of Generator
    g_i = G.generate(z_i)
    g = np.reshape(g_i,(num_samples))

    # 진짜 data로부터 Disriminarot 학습
    D.train(x_i,d_x_i)
    D.train(g_i,d_g_i)
    
    # GAN을 UPdate하는 과정
    sess.run([loss_g, opt_g],feed_dict={G.z_input:z_i})
    sess.run([loss_d, opt_d],feed_dict={D.x_input:x_i,G.z_input:z_i})
    
    if tr%1000==0:
        error_g,_ = sess.run([loss_g, opt_g],feed_dict={G.z_input:z_i})
        error_d,_ = sess.run([loss_d, opt_d],feed_dict={D.x_input:x_i,G.z_input:z_i})
        print(error_g,error_d)
        
        
# Generator를 훈련시킨후의 g_i의 발생
g_i = G.generate(z_i)
g = np.reshape(g_i,(num_samples))
draw(x,z,g,D)


