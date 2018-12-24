import torch.tensor as tensor
import matplotlib.pyplot as plt
from ParzenWindow import ParzenWindow
import torch.optim as opt
import torch.nn as nn

criterion = nn.MSELoss()

def trainon(xs,ys,e,net,norm):
    optimizer.zero_grad()
    s = "E: "+str(e)+" "
    for x in range(len(xs)):
        out = net(xs[x]/norm)
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
x3 = tensor([2.5])
y3 = tensor([0.0])
x4 = tensor([3.8])
y4 = tensor([0.1])
x5 = tensor([-3.0])
y5 = tensor([0.1])
x6 = tensor([-5.0])
y6 = tensor([0.7])
x7 = tensor([-6.0])
y7 = tensor([0.3])
x8 = tensor([.0])
y8 = tensor([0.05])
x9 = tensor([30.0])
y9 = tensor([0.45])
x10 = tensor([-45.5])
y10 = tensor([0.82])

#First Case: No big numbers
print("-----FIRST-----")
xd = [x1,x2,x3,x4,x5,x6,x7,x8]
yd = [y1,y2,y3,y4,y5,y6,y7,y8]

net = nn.Sequential(ParzenWindow(1,6),ParzenWindow(6,6),ParzenWindow(6,1))
optimizer = opt.SGD(net.parameters(),lr=0.006)

for e in range(0):#30000
    trainon(xd,yd,e,net,6.0)

xs,ys = test(1.0,0.01,net)

plt.figure(1)
plt.plot(xs,ys)
plt.title("Test 1")
#plt.savefig('./Plots/test_1.png',bbox_inches='tight')
plt.show()

#Second Case: Added one big number
#Doesn't converge under normalized values
print("-----SECOND-----")
xd.append(x9)
yd.append(y9)

net = nn.Sequential(ParzenWindow(1,6),nn.ReLU(),ParzenWindow(6,6),nn.ReLU(),ParzenWindow(6,1))
optimizer = opt.SGD(net.parameters(),lr=0.01)

for e in range(50000):
    trainon(xd,yd,e,net,30.0)

xs,ys = test(1.0,0.01,net)

plt.figure(2)
plt.plot(xs,ys)
plt.title("Test 2")
#plt.savefig('./Plots/test_2.png',bbox_inches='tight')
plt.show()

#Third Case: Added one more big number
#Same as above
print("-----THIRD-----")
xd.append(x10)
yd.append(y10)

net = nn.Sequential(ParzenWindow(1,6),ParzenWindow(6,6),ParzenWindow(6,1))
optimizer = opt.SGD(net.parameters(),lr=0.01)

for e in range(50000):
    trainon(xd,yd,e,net,46.0)

xs,ys = test(1.0,0.01,net)

plt.figure(3)
plt.plot(xs,ys)
plt.title("Test 3")
#plt.savefig('./Plots/test_3.png',bbox_inches='tight')
plt.show()