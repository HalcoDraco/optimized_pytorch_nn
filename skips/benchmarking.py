class Bmk:
    """
    Benchmarking utility class.
    
    Methods
    -------
    set(work: bool)
        Configure whether benchmarking is active.
    __call__(msg: str = "")
        Record a timestamp with an optional message.
    report()
        Print the benchmark report.
    reset()
        Reset the recorded timestamps.
    """

    def __init__(self):
        """Initialize the benchmark utility."""
        self.__work: bool = None
        self.__times: list[tuple[float, str]] = []
    
    def set(self, work: bool):
        if self.__work is None:
            self.__work = work
        else:
            raise RuntimeError("Benchmark has already been configured.")
    
    def __call__(self, msg: str = ""):
        if self.__work is None:
            raise RuntimeError("Benchmark has not been configured yet.")
        elif self.__work:
            self.__times.append((time(), msg))

    def report(self):
        if self.__work:
            print("Benchmark report:")
            for i in range(1, len(self.__times)):
                start_time, start_msg = self.__times[i-1]
                end_time, end_msg = self.__times[i]
                elapsed = end_time - start_time
                print(f"  {start_msg} -> {end_msg}: {elapsed:.4f} seconds")
            print("")
            print(f"Total time: {self.__times[-1][0] - self.__times[0][0]:.4f} seconds")
    
    def reset(self):
        self.__times = []