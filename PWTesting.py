import torch.tensor as tensor
import matplotlib.pyplot as plt
from ParzenWindow import ParzenWindow
import torch.optim as opt
import torch.nn as nn

#ParzenWindows estimating a Gaussian Distribution
criterion = nn.MSELoss()

def trainon(xs,ys,e,net):
    optimizer.zero_grad()
    s = "E: "+str(e)+" "
    for x in range(len(xs)):
        out = net(xs[x])
        s += " "+str(ys[x].item()-out.item())
        loss = criterion(out,ys[x])
        loss.backward()
    optimizer.step()
    print(s)

def test(range,step,net):
    xs = []
    ys = []
    x = -range
    while x <= range:
        out = net(tensor([x]))
        xs.append(x)
        ys.append(out.item())
        x += step
    return xs,ys

print("-----FIRST-----")
#First case: Two simple positive numbers(unnormalized)
x1 = tensor([1.5])
y1 = tensor([1.0])
x2 = tensor([3.5])
y2 = tensor([0.6])

xd = [x1,x2]
yd = [y1,y2]

net = nn.Sequential(ParzenWindow(1,1))
optimizer = opt.SGD(net.parameters(),lr=0.005)

for e in range(0):#6000
    trainon(xd,yd,e,net)

xs,ys = test(4.0,0.1,net)

plt.figure(1)
plt.plot(xs,ys)
plt.title("Test 1")
#plt.savefig('./Plots/test_1.png',bbox_inches='tight')

print("-----SECOND-----")
#Second case: More complexity 4 positive numbers(unnormalized)
x3 = tensor([2.5])
y3 = tensor([0.0])
x4 = tensor([3.8])
y4 = tensor([0.1])

xd.extend([x3,x4])
yd.extend([y3,y4])

net = nn.Sequential(ParzenWindow(1,4),ParzenWindow(4,4),ParzenWindow(4,1))
optimizer = opt.SGD(net.parameters(),lr=0.01)

for e in range(10000):
    trainon(xd,yd,e,net)

xs,ys = test(4.0,0.1,net)

plt.figure(2)
plt.plot(xs,ys)
plt.title("Test 2")
#plt.savefig('./Plots/test_2.png',bbox_inches='tight')

print("-----THIRD-----")
#Third case: Added 3 negative numbers -3,-5,-6(unnormalized)
x5 = tensor([-3.0])
y5 = tensor([0.1])
x6 = tensor([-5.0])
y6 = tensor([0.7])
x7 = tensor([-6.0])
y7 = tensor([0.3])

xd.extend([x5,x6,x7])
yd.extend([y5,y6,y7])

net = nn.Sequential(ParzenWindow(1,6),ParzenWindow(6,6),ParzenWindow(6,1))
optimizer = opt.SGD(net.parameters(),lr=0.01)

for e in range(15000):
    trainon(xd,yd,e,net)

xs,ys = test(6.0,0.1,net)

plt.figure(3)
plt.plot(xs,ys)
plt.title("Test 3")
plt.savefig('./Plots/test_3.png',bbox_inches='tight')

print("-----FOURTH-----")
#Fourth case: A little more complexity
x8 = tensor([.0])
y8 = tensor([0.05])

xd.append(x8)
yd.append(y8)

net = nn.Sequential(ParzenWindow(1,6),ParzenWindow(6,6),ParzenWindow(6,1))
optimizer = opt.SGD(net.parameters(),lr=0.01)

for e in range(20000):
    trainon(xd,yd,e,net)

xs,ys = test(6.0,0.1,net)

plt.figure(4)
plt.plot(xs,ys)
plt.title("Test 4")
plt.savefig('./Plots/test_4.png',bbox_inches='tight')

print("-----FIFTH-----")
#Fifth case: Big positive number: 30
x9 = tensor([30.0])
y9 = tensor([0.45])

xd.append(x9)
yd.append(y9)

net = nn.Sequential(ParzenWindow(1,6),ParzenWindow(6,6),ParzenWindow(6,1))
optimizer = opt.SGD(net.parameters(),lr=0.01)

for e in range(50000):
    trainon(xd,yd,e,net)

xs,ys = test(30.0,0.5,net)

plt.figure(5)
plt.plot(xs,ys)
plt.title("Test 5")
plt.savefig('./Plots/test_5.png',bbox_inches='tight')

print("-----SIXTH-----")
#Fifth case: Big negative number: -45.5
#Couldn't solve it with smaller lr, more layers, more units, ReLU activations, linear layers or more epochs.
#This is the only solution that worked. Effectively normalization.
x10 = tensor([-45.5])
y10 = tensor([0.82])

xd.append(x10)
yd.append(y10)

net = nn.Sequential(nn.Linear(1,3),nn.Sigmoid(),ParzenWindow(3,6),ParzenWindow(6,6),ParzenWindow(6,1))
optimizer = opt.SGD(net.parameters(),lr=0.01)

for e in range(50000):
    trainon(xd,yd,e,net)

xs,ys = test(48.0,0.5,net)

plt.figure(6)
plt.plot(xs,ys)
plt.title("Test 6")
#plt.savefig('./Plots/test_6.png',bbox_inches='tight')
