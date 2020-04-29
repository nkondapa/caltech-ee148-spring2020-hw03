from main import Net
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle as pkl
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.cm
from matplotlib.lines import Line2D


def mark_mistakes(model, device, test_loader):
    model.eval()  # Set the model to inference mode

    mistake_inds = []
    cm = np.zeros(shape=(11, 11))
    predicted_expected_tuples = []
    with torch.no_grad():  # For the inference step, gradient is not computed
        for c, dt_tuple in enumerate(test_loader):
            print(c)
            data, target = dt_tuple
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            cm[target, pred] += 1
            if pred.flatten() != target:
                mistake_inds.append(c)
                predicted_expected_tuples.append((pred.flatten(), target))

    return mistake_inds, predicted_expected_tuples, cm


def visualize_mistakes(mistake_inds, pe_tuples, test_dataset, num_mistakes=9):
    fig = plt.figure()
    plt.title('Mistaken Numbers')
    for i in range(num_mistakes):
        ind = mistake_inds[i]
        ax = plt.subplot(3, 3, i + 1)
        ax.title.set_text('predicted=' + str(int(pe_tuples[i][0][0])) + ', target=' + str(int(pe_tuples[i][1][0])))
        if i > 0:
            ax.axis('off')
        plt.imshow(test_dataset.data[ind, :])
    fig.tight_layout()
    plt.show()


def visualize_kernels(model):
    params = list(model.parameters())
    l1 = params[0]
    l1 = l1.to(torch.device('cpu')).detach().numpy()
    fig = plt.figure()
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        ax.title.set_text('kernel=' + str(i + 1))
        if i > 0:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.imshow(np.squeeze(l1[i, :]), vmin=0, vmax=1, cmap='binary')
        if i == 0:
            plt.colorbar()
    fig.tight_layout()
    plt.show()


def confusion_matrix(cm):
    precision = np.diag(cm) / np.sum(cm, 0)
    recall = np.diag(cm) / np.sum(cm, 1)
    acc = 1 - np.sum(np.sum(cm, 0) - np.diag(cm)) / np.sum(cm)
    cm[-1, :] = precision
    cm[:, -1] = recall
    cm[-1, -1] = acc
    lol = np.matrix(np.array(cm[0:-1, 0:-1], dtype=int), dtype=str).tolist()
    row_labels = ['expected ' + str(i) for i in list(range(10))]
    col_labels = ['predicted ' + str(i) for i in list(range(10))]
    table = plt.table(lol, loc='center', rowLabels=row_labels, colLabels=col_labels, fontsize=50)
    table.auto_set_font_size(False)
    plt.axis('off')
    # plt.show()

    plt.figure()
    col_labels = list(range(10))
    row_labels = ['precision', 'recall']
    pr = np.zeros(shape=(2, 10))
    pr[0, :] = precision[:-1]
    pr[1, :] = recall[:-1]
    pr = np.around(pr, 5)
    table = plt.table(pr, loc='center', rowLabels=row_labels, colLabels=col_labels, fontsize=50)
    table.auto_set_font_size(False)
    plt.axis('off')
    plt.show()


def get_layer_outputs(model, device, test_loader):
    model.eval()  # Set the model to inference mode
    outputs = []

    def hook_fn(self, input, output):
        outputs.append(np.array(output.to(torch.device('cpu'))))

    model.fc1.register_forward_hook(hook_fn)
    mistake_inds = []
    cm = np.zeros(shape=(11, 11))
    predicted_expected_tuples = []
    with torch.no_grad():  # For the inference step, gradient is not computed
        for c, dt_tuple in enumerate(test_loader):
            print(c)
            data, target = dt_tuple
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            cm[target, pred] += 1
            if pred.flatten() != target:
                mistake_inds.append(c)
                predicted_expected_tuples.append((pred.flatten(), target))

    return outputs


def t_sne_visualization(outputs, targets):
    output_arr = np.squeeze(np.array(outputs))
    cmap = matplotlib.cm.get_cmap('tab10')
    tsne = TSNE(n_components=2)
    trans_output = tsne.fit_transform(output_arr)
    plt.scatter(trans_output[:, 0], trans_output[:, 1], c=cmap(targets))
    leg_obs = [Line2D([0], [0], marker='o', color='w', label='Scatter',
                      markerfacecolor=cmap(i), markersize=7) for i in range(10)]
    plt.legend(leg_obs, list(range(10)), loc='upper right')
    plt.xlabel('t-SNE 1')
    plt.xlabel('t-SNE 2')
    plt.title('t-SNE of output before final layer')
    plt.savefig('./t-sne_vis.png')


def find_nearby_images(outputs, data):
    output_arr = np.squeeze(np.array(outputs))
    np.random.seed(10)
    sel_inds = np.random.randint(0, len(outputs), 4)
    plt.figure()
    count = 1
    for si in sel_inds:
        arr = output_arr[si, :]
        distances = np.linalg.norm((arr - output_arr), axis=1)
        nn_inds = np.argpartition(distances, 9)[:9]
        # nn_inds = np.delete(nn_inds, np.where(nn_inds==si))
        sort_inds = list(zip(*sorted(zip(distances[nn_inds], range(9)), key=lambda x: x[0], reverse=False)))[1]
        nn_inds = nn_inds[list(sort_inds)]

        for c, ni in enumerate(nn_inds):
            ax = plt.subplot(4, 9, count)
            if c > 0:
                ax.axis('off')
            if count == 1:
                ax.set_title('Selected Image')
            if count == 2:
                ax.set_title('Nearby Images ->')
            img_arr = np.array(data.data[ni, :].reshape(28, 28).to(torch.device('cpu')))
            plt.imshow(img_arr)
            count += 1
    plt.show()


def run():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set the test model
    model = Net().to(device)
    model.load_state_dict(torch.load('./mnist_model_epoch=14.pt'))

    test_dataset = datasets.MNIST('../data', train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, **kwargs)

    path = './model_mistakes.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            mistake_inds, pe_tuples, cm = pkl.load(f)
    else:
        with open(path, 'wb') as f:
            mistake_inds, pe_tuples, cm = mark_mistakes(model, device, test_loader)
            pkl.dump((mistake_inds, pe_tuples, cm), f)

    # visualize_mistakes(mistake_inds, pe_tuples, test_dataset)
    # visualize_kernels(model)
    # confusion_matrix(cm)

    path2 = './final_layer_outputs.pkl'
    if os.path.exists(path2):
        with open(path2, 'rb') as f:
            fl_outputs = pkl.load(f)
    else:
        with open(path2, 'wb') as f:
            fl_outputs = get_layer_outputs(model, device, test_loader)
            pkl.dump(fl_outputs, f)
    # t_sne_visualization(fl_outputs, test_dataset.targets)
    find_nearby_images(fl_outputs, test_dataset)

run()
