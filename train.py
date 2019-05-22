import os
import torch
import torch.distributed as dist
import torch.utils.data
from dask.distributed import Client
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style="darkgrid")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self._network = torch.nn.Sequential(
            torch.nn.Linear(5,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,2),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self._network(x)
    
def getDataset():
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=1000, centers=2, n_features=5,
                  random_state=42)
    return X, y

def run(X, y, rank, size, backend='gloo'):
    """
    Initialize the distributed environment
    """
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "23456"
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)

    ## Init distributed
    print("Waiting other workers...")
    dist.init_process_group(
        init_method="env://",
        backend='gloo'
    )

    device = torch.device('cpu')
    net = Net().to(device)
    if device == 'cpu':
        net = torch.nn.parallel.DistributedDataParallelCPU(net)
    else:
        # device == 'cuda'
        net = torch.nn.parallel.DistributedDataParallel(net)
    
    data = torch.tensor(X, dtype=torch.float32)
    target = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(data, target)
    
    train_loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=32
    )
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    loss_function = torch.nn.CrossEntropyLoss()
    
    iteration_loss=[]
    
    for epoch in range(10):
        for data, target in train_loader:
            output = net.forward(data)
            loss = loss_function(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            iteration_loss.append(loss.item())
        print('Rank %s ,Loss: %s'  % (rank, loss.item()))
    
    return {'worker':rank, 'loss':iteration_loss}

if __name__=='__main__':
    world_size = 2
    client = Client(local_dir="/tmp/test")
    
    X, y = getDataset()
    X_workers = [x for x in np.split(X, world_size)]
    y_workers = [y for y in np.split(y, world_size)]
    
    rank = [client.scatter(rank) for rank in range(world_size)]
    world_size = [client.scatter(world_size) for _ in range(world_size)]
    X = [client.scatter(x) for x in X_workers]
    y = [client.scatter(y) for y in y_workers]
    
    futures = client.map(run, X, y, rank, world_size)
    history = client.gather(futures)

    fig, ax = plt.subplots()
    for h in history:
        ax.plot(h['loss'], label='worker %s'%h['worker'])
	ax.set_title('Training loss')
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Loss')
	ax.legend()
    fig.savefig('training_loss.pdf')
