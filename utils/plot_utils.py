import matplotlib.pyplot as plt


def plot_similarity(posterior_similarity, metrics=0):
    distance_metric_list_name = ['cosine', 'euclidean', 'correlation', 'chebyshev', 'braycurtis', 'canberra',
                                 'cityblock', 'sqeuclidean']
    plt.figure(figsize=(10, 7))
    plt.plot(
        posterior_similarity[:, metrics], color='tab:blue', linestyle='-',
        label=f'{distance_metric_list_name[metrics]}'
    )

    x = list(range(posterior_similarity.shape[0]))
    labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(x, labels)

    plt.xlabel('perterbation rate')
    plt.ylabel('Similarity')
    plt.legend()
    plt.savefig(os.path.join('/content/drive/MyDrive/dl/link_steal_project',
                             f'5_layer_distance_3-{distance_metric_list_name[metrics]}.png'))

    plt.show()


def plot_similarity_sum(posterior_similarity):
    plt.figure(figsize=(10, 7))
    plt.plot(
        posterior_similarity.sum(dim=1), color='tab:blue', linestyle='-',
        label='sum of all similarity metrics'
    )
    x = list(range(posterior_similarity.shape[0]))
    # labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # plt.xticks(x, labels)

    plt.xlabel('perterbation rate')
    plt.ylabel('Similarity')
    plt.legend()
    plt.show()


def plot_similarity_compare(posterior_similarity1, posterior_similarity2, target_distance):
    plt.figure(figsize=(10, 7))
    plt.plot(
        posterior_similarity1[:, target_distance], color='tab:blue', linestyle='-',
        label='posterior_similarity1'
    )

    plt.plot(
        posterior_similarity2[:, target_distance], color='tab:red', linestyle='-',
        label='posterior_similarity2'
    )

    x = list(range(posterior_similarity1.shape[0]))
    labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(x, labels)

    plt.xlabel('perterbation rate')
    plt.ylabel('Similarity')
    plt.legend()
    plt.show()
    # plt.savefig(os.path.join('outputs', name+'_accuracy.png'))


def plot_influence(influence):
    influence = influence.norm(dim=1)
    plt.figure(figsize=(10, 7))
    plt.plot(
        influence, color='tab:blue', linestyle='-',
        label=f'influence score'
    )

    x = list(range(influence.shape[0]))
    # labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # plt.xticks(x, labels)

    plt.xlabel('perterbation rate')
    plt.ylabel('Similarity')
    plt.legend()
    plt.savefig(os.path.join('/content/drive/MyDrive/dl/link_steal_project', f'5_layer_sage_distance_3_influence.png'))

    plt.show()


def plot_different_distance_similarity(posterior_similarity_list, metrics=0):
    distance_metric_list_name = ['cosine', 'euclidean', 'correlation', 'chebyshev', 'braycurtis', 'canberra',
                                 'cityblock', 'sqeuclidean']
    plt.figure(figsize=(10, 7))

    for i in range(len(posterior_similarity_list)):
        plt.plot(
            posterior_similarity_list[i][:, metrics], linestyle='-',
            label=f'distance={i + 1}'
        )

    x = list(range(posterior_similarity_list[0].shape[0]))
    labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(x, labels)

    plt.xlabel('perterbation rate')
    plt.ylabel(distance_metric_list_name[metrics])
    plt.legend()
    plt.savefig(os.path.join('/content/drive/MyDrive/dl/link_steal_project',
                             f'5_layer_compare_similarity_3-{distance_metric_list_name[metrics]}.png'))
    plt.show()


def plot_multiple_similarity(posterior_similarity_list, metrics=0):
    distance_metric_list_name = ['cosine', 'euclidean', 'correlation', 'chebyshev', 'braycurtis', 'canberra',
                                 'cityblock', 'sqeuclidean']
    plt.figure(figsize=(10, 7))

    for i in range(len(posterior_similarity_list)):
        plt.plot(
            posterior_similarity_list[i][:, metrics], linestyle='-',
        )

    x = list(range(posterior_similarity_list[0].shape[0]))
    labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(x, labels)

    plt.xlabel('perterbation rate')
    plt.ylabel(distance_metric_list_name[metrics])
    plt.legend()
    plt.savefig(os.path.join('/content/drive/MyDrive/dl/link_steal_project',
                             f'5_layer_compare_similarity_3-{distance_metric_list_name[metrics]}.png'))
    plt.show()


def plot_scatter_plot(inp):
    # Creating a new figure
    plt.figure(dpi=100)
    for i in range(6):
        plt.scatter(inp[i][:, 0], inp[i][:, 1], label=i)

    # Adding details to the plot
    plt.title('posterior similarity')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    # Displaying the plot
    plt.grid()
    plt.show()
