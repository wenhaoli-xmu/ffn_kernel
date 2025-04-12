from abc import abstractmethod
from profiler import WallTime
from pygments.console import colorize


class BaseSearch:
    def __init__(self, inputs):
        self.inputs = inputs

    @abstractmethod
    def get_configs(self):
        ...

    @abstractmethod
    def benchmark_object(self, *args):
        ...

    def search(self):
        stack = [(0,) * len(self.get_configs())]
        visited = {}
        c_to_config = {}
        length = [len(x) for x in self.get_configs().values()]

        while len(stack) > 0:
            c = stack.pop()
            if c not in visited:
                config_args = []
                for idx, v in zip(c, self.get_configs().values()):
                    config_args.append(v[idx])
                config_args = tuple(config_args)
                c_to_config[c] = config_args

                inputs = self.inputs + config_args

                wt = WallTime(name=f'{c}', cuda=0)
                for _ in range(3):
                    with wt:
                        self.benchmark_object(*inputs)

                visited.update({c: min(wt.time)})
                print(c_to_config[c], min(wt.time))

                for i in range(len(c)):
                    _c = list(c)
                    if _c[i] - 1 >= 0:
                        _c[i] -= 1
                    if tuple(_c) not in visited:
                        stack.append(tuple(_c))

                    _c = list(c)
                    if _c[i] + 1 < length[i]:
                        _c[i] += 1
                    if tuple(_c) not in visited:
                        stack.append(tuple(_c))

        min_key = None
        min_value = 1_000_000_000
        for key, value in visited.items():
            if value < min_value:
                min_key = key
                min_value = value

        print(colorize('green', f"optimum: {c_to_config[min_key]} | {min_value}s"))
        return c_to_config[min_key]