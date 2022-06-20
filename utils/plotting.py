import matplotlib.pyplot as plt

def plot_segments(signals, title):
    """visualize the heartbeats extracted above"""
    plt.figure(figsize=(12, 12))
    col_num = 5
    row_num = 5
    signal_nums = 25
    k = 13
    for i in range(signal_nums):
        plt.subplot(row_num, col_num, i+1)
        plt.plot(signals[i + k*signal_nums]) # pay attention to the range
        # plt.title(anns[i + k*signal_nums])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(title, size=20) 
    plt.show()