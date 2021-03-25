import sys
from io import StringIO
from multiprocessing import Pipe, Process
from threading import Thread
from typing import List

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# use other matplotlib backend, PyQt steals focus from other apps
#matplotlib.use("Tkagg")


def plotter(pipe: Pipe, max_iter: int):

    gen = [0]
    fitness = [0]

    def receiver(pipe):
        nonlocal gen
        nonlocal fitness

        while True:
            gen, fitness = pipe.recv()

    plt.ion()

    axis = plt.gca()
    axis.set_xlim([0, max_iter])
    axis.set_xlabel("Generation")
    axis.set_ylabel("Fitness function")

    Thread(target=receiver, args=(pipe,), daemon=True).start()

    while True:
        if gen and fitness:
            plt.plot(gen, fitness)
            axis.set_title(f"Fitness evolution. min={min(fitness)}")
            plt.draw()
            plt.pause(0.1)
            plt.clf()
        else:
            return


class ProgressBar(StringIO):
    def __init__(self, max_iter: int, headers: List[str], plot: bool = True):
        super().__init__(None, "\n")

        self._pbar = tqdm(total=max_iter, ncols=200)
        self._headers = ["neval"] + list(headers)
        self._last_iter = 0
        self._max_iter = max_iter

        self.gen = []
        self.fitness = []
        self.couter = 0
        self.pipe, child_pipe = Pipe(duplex=True)
        #Process(target=plotter, args=(child_pipe, max_iter)).start()

    def __exit__(self, type, value, traceback):
        self._pbar.close()

    def write(self, s: str):

        # if string is empty do not print
        if not s.strip():
            return
        elif "Couldnt cross" in s:
            return

        # try to parse data in line
        try:
            gen, *data = s.split()
        except ValueError:
            print("\r", s, " " * (200 - len(s)), file=sys.stderr)
            return
        # if successful pass data to description in progressbar
        else:
            stats = {}
            for header, value in zip(self._headers, data):
                stats[header] = value

            self._pbar.set_postfix(ordered_dict=stats, refresh=False)

        # try to get generation number
        try:
            n = int(gen)
        except ValueError:
            print("\r", s, " " * (200 - len(s)), file=sys.stderr)
            return
        # if successful update progressbar
        else:
            diff = n - self._last_iter
            self._last_iter = n
            self._pbar.update(n=diff)

            self.gen.append(n)
            self.fitness.append(float(stats["min"]))

            if self.couter > 100:
                #self.pipe.send([self.gen, self.fitness])
                self.couter = 0
            else:
                self.couter += 1

            if n >= self._max_iter:
                #self.pipe.send([None, None])
                pass
