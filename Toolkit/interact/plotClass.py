import matplotlib.pyplot as plt
from object.objectHelper import ObjectHelper
import numpy as np
import librosa

class PlotManager():
    def __init__(self, row=1, col=1, size=[7,3], type="melspectrogram"):
        self.row = row
        self.col = col
        self.figure = None
        self.type_dict = {'rawaudio':self.plot_rawaudio, 'melspectrogram': self.plot_spectrogram, 'plot_numpy': self.plot_numpy}
        self.plotMethod = self.getPlotMethod(type)
        self.data_list = [] #별도 리스트를 생성하고 싶지 않은 경우

    def getPlotMethod(self, type, dim=2):
        return self.type_dict[type]

    def preparePlot(self):
        plt.clf()
        
        fig, axs = plt.subplots(self.row, self.col, figsize=(10, 6), constrained_layout=True)
        return fig, axs

    def drawPlot(self, plot_list, save_as_file=None):
        self.figure, axs = self.preparePlot()

        if not ObjectHelper.is_NestedIterable(axs):
            axs = [[axs]]

        for content_dict in plot_list:
            if content_dict is None:
                continue
                
            pos_1 = content_dict['position'][0]
            pos_2 = content_dict['position'][1]

            self.plotMethod(axs[pos_1][pos_2], content_dict['content'], title = content_dict['title'])
        plt.show(block=True)
        if save_as_file is not None:
            plt.savefig(save_as_file)

    def plot_spectrogram(self, axis, content, title=None, ylabel="freq_bin"):
        axis.set_title(title or "Spectrogram")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("frame")
        im = axis.imshow(librosa.power_to_db(content), origin="lower", aspect="auto")
        self.figure.colorbar(im, ax=axis)

    def plot_rawaudio(self, axis, content, title=None, ylabel="db"):
        content = content.t().numpy()
        axis.set_title(title or "rawaudio")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("time")
        im = axis.plot(content)

    def plot_numpy(self, axis, content, title=None, ylabel="db"):
        axis.set_title(title or "rawaudio")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("time")
        im = axis.plot(content)

    def draw_2D(self, axis, content, title=None, ylabel="-", xlabel="-",):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel(xlabel)
        im = axis.imshow(content)
        self.figure.colorbar(im, ax=axis)