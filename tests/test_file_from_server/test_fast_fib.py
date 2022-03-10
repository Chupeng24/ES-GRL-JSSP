import time
from fast_fib import fib

start = time.time()
result = fib(40)
end = time.time()
print(f'斐波拉契数列第40项为：{result}，耗时：{end - start}秒')