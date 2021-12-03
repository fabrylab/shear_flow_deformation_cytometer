#!/usr/bin/env python3
from multiprocessing import Process, Queue
import sys
import logging
import traceback
import inspect


STOP = "STOP"
SHUTDOWN = "SHUTDOWN"
SHUTDOWN_LAST = "SHUTDOWN_LAST"

class _STOP: pass
class _SHUTDOWN: pass
class _SHUTDOWN_LAST: pass

STOP = _STOP()
SHUTDOWN = _SHUTDOWN()
SHUTDOWN_LAST = _SHUTDOWN_LAST()

log = None


class Task:
    def __init__(self, id, fn, inputQueue, outputQueue, multiplicity):
        self.id = id
        self.fn = fn
        self.inputQueue = inputQueue
        self.outputQueue = outputQueue
        self.multiplicity = multiplicity

    def start(self):
        self.process = Process(target=self.main, args=(self.inputQueue, self.outputQueue))
        self.process.start()

    def main(self, inputQueue, outputQueue):
        self.inputQueue = inputQueue
        self.outputQueue = outputQueue

        if inspect.isfunction(self.fn):
            logger = logging.getLogger(str(self.id) + ":" +
                                       self.fn.__name__)
        else:
            logger = logging.getLogger(str(self.id) + ":" +
                                       type(self.fn).__name__)
        global log
        log = lambda a: logger.debug(a)

        try:
            if hasattr(self.fn, "init"):
                self.fn.init()

            log("Running")

            while True:
                input = self.inputQueue.get()
                log("Input is {}".format(input))
                if isinstance(input, _SHUTDOWN): break
                if isinstance(input, _SHUTDOWN_LAST):
                    self.outputQueue.put(STOP)
                    break
                if isinstance(input, _STOP):
                    for i in range(self.multiplicity - 1):
                        self.inputQueue.put(SHUTDOWN)
                    self.inputQueue.put(SHUTDOWN_LAST)
                    continue

                if isinstance(input, tuple):
                    result = self.fn(*input)
                else:
                    result = self.fn(input)
                if inspect.isgenerator(result):
                    for x in result:
                        if isinstance(x, _STOP):
                            self.inputQueue.put(STOP)
                            break
                        self.outputQueue.put(x)
                else:
                    if isinstance(result, _STOP):
                        self.inputQueue.put(STOP)
                    else:
                        self.outputQueue.put(result)

            log("Shutting down")
            if hasattr(self.fn, "shutdown"):
                self.fn.shutdown()

        except KeyboardInterrupt:
            pass
        except Exception:
            print("For {}".format(self.fn))
            raise


class Pipeline:
    def __init__(self, batch_size=100):
        self.tasks = []
        self.inputQueue = Queue(1)
        self.outputQueue = Queue(1)
        self.batch_size = batch_size
        self.nextId = 1

    def run(self, arg=None):

        for task in self.tasks:
            task.start()

        self.inputQueue.put(arg)
        while True:
            x = self.outputQueue.get()
            if isinstance(x, _STOP): break

    def add(self, fn, fanOut=1):
        inputQueue = self.inputQueue
        outputQueue = self.outputQueue
        if len(self.tasks):
            inputQueue = Queue(self.batch_size)
            for task in self.tasks:
                if task.outputQueue == self.outputQueue:
                    task.outputQueue = inputQueue

        for i in range(fanOut):
            task = Task(self.nextId, fn, inputQueue, outputQueue, fanOut)
            self.nextId += 1
            self.tasks.append(task)


















class Task2:
    def __init__(self, id, fn, inputQueue, outputQueue, multiplicity):
        self.id = id
        self.fn = fn
        self.inputQueue = inputQueue
        self.outputQueue = outputQueue
        self.multiplicity = multiplicity

    def start(self):
        self.process = Process(target=self.main, args=(self.inputQueue, self.outputQueue))
        self.process.start()

    def main(self, inputQueue, outputQueue):
        self.inputQueue = inputQueue
        self.outputQueue = outputQueue

        if inspect.isfunction(self.fn):
            logger = logging.getLogger(str(self.id) + ":" +
                                       self.fn.__name__)
        else:
            logger = logging.getLogger(str(self.id) + ":" +
                                       type(self.fn).__name__)
        global log
        log = lambda a: logger.debug(a)

        try:
            if hasattr(self.fn, "init"):
                self.fn.init()

            log("Running")

            while True:
                input = self.inputQueue.get()
                log("Input is {}".format(input))
                if isinstance(input, _SHUTDOWN): break
                if isinstance(input, _SHUTDOWN_LAST):
                    self.outputQueue.put(STOP)
                    break
                if isinstance(input, _STOP):
                    for i in range(self.multiplicity - 1):
                        self.inputQueue.put(SHUTDOWN)
                    self.inputQueue.put(SHUTDOWN_LAST)
                    continue

                if isinstance(input, tuple):
                    result = self.fn(*input)
                else:
                    result = self.fn(input)
                if inspect.isgenerator(result):
                    for x in result:
                        if isinstance(x, _STOP):
                            self.inputQueue.put(STOP)
                            break
                        self.outputQueue.put(x)
                else:
                    if isinstance(result, _STOP):
                        self.inputQueue.put(STOP)
                    else:
                        self.outputQueue.put(result)

            log("Shutting down")
            if hasattr(self.fn, "shutdown"):
                self.fn.shutdown()

        except KeyboardInterrupt:
            pass
        except Exception:
            print("For {}".format(self.fn))
            raise


class Pipeline2:
    def __init__(self, batch_size=None):
        self.tasks = []
        self.inputQueue = Queue(1)
        self.outputQueue = Queue(1)
        self.nextId = 1

    def run(self, arg=None):

        for fn in self.tasks:
            if hasattr(fn, "init"):
                fn.init()

        def pipe_outputs(result, index):
            if index < len(self.tasks):
                if inspect.isgenerator(result):
                    for res in result:
                        if isinstance(res, _STOP):
                            return
                        if isinstance(res, tuple):
                            pipe_outputs(self.tasks[index](*res), index+1)
                        else:
                            pipe_outputs(self.tasks[index](res), index+1)
                else:
                    if isinstance(result, _STOP):
                        return
                    if isinstance(result, tuple):
                        pipe_outputs(self.tasks[index](*result), index + 1)
                    else:
                        pipe_outputs(self.tasks[index](result), index + 1)
            else:
                if inspect.isgenerator(result):
                    for res in result:
                        pass

        pipe_outputs(arg, 0)

    def add(self, fn, fanOut=1):
        self.tasks.append(fn)